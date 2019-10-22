//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>

#include "classification_sample.h"
#include <file_reader.h>

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

std::vector<std::string> readLabelsFromFile(const std::string& labelFileName) {
    std::vector<std::string> labels;

    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            trim(strLine);
            labels.push_back(strLine);
        }
    }
    return labels;
}

Blob::Ptr deQuantize(const Blob::Ptr &quantBlob, float scale, uint8_t zeroPoint) {
  const TensorDesc quantTensor = quantBlob->getTensorDesc();
  SizeVector dims = quantTensor.getDims();
  size_t batchSize = dims.at(0);
  slog::info << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << slog::endl;
  const size_t Count = quantBlob->size() / batchSize;
  const size_t ResultsCount = Count > 1000 ? 1000 : Count;
  dims[1] = ResultsCount;
  const TensorDesc outTensor = TensorDesc(
      InferenceEngine::Precision::FP32,
      dims,
      quantTensor.getLayout());
  slog::info << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << slog::endl;
  Blob::Ptr outputBlob = make_shared_blob<float>(outTensor);
  outputBlob->allocate();
  float *outRaw = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
  const uint8_t *quantRaw = quantBlob->cbuffer().as<const uint8_t *>();

  for (size_t pos = 0; pos < outputBlob->size(); pos++) {
    outRaw[pos] = (quantRaw[pos] - zeroPoint) * scale;
  }
  return outputBlob;
}


/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::string imageFileName = FLAGS_i;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;
        Core ie;

        if (FLAGS_p_msg) {
            ie.SetLogCallback(error_listener);
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read blob Generated by MCM Compiler ----------------------------------
        std::string binFileName = FLAGS_m;
        slog::info << "Loading blob:\t" << binFileName << slog::endl;

        ExecutableNetwork importedNetwork = ie.ImportNetwork(binFileName, "KMB", {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        Blob::Ptr inputImageBlob;
        vpu::KmbPlugin::utils::fromBinaryFile(imageFileName, inputImageBlob);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Create infer request -------------------------------------------------
        InferenceEngine::InferRequest inferRequest = importedNetwork.CreateInferRequest();
        slog::info << "CreateInferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        std::string firstInputName = inputInfo.begin()->first;
        /** Creating input blob **/
        Blob::Ptr inputBlob = inferRequest.GetBlob(firstInputName.c_str());
        if (!inputBlob) {
            throw std::logic_error("Cannot get input blob from inferRequest");
        }

        /** Filling input tensor with images. **/
        if (inputBlob->size() < inputImageBlob->size()) {
            throw std::logic_error("Input blob doesn't have enough memory to fit input image");
        }

        auto blobData = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::copy_n(inputImageBlob->buffer().as<uint8_t*>(), inputImageBlob->size(), blobData);

        inferRequest.Infer();
        slog::info << "inferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();
        if (outputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 output");

        std::string firstOutputName = outputInfo.begin()->first;

        Blob::Ptr outputBlob = inferRequest.GetBlob(firstOutputName.c_str());
        if (!outputBlob) {
            throw std::logic_error("Cannot get output blob from inferRequest");
        }

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels = readLabelsFromFile(labelFileName);

        auto inputInfoItem = *inputInfo.begin();
        Blob::Ptr input_blob = inferRequest.GetBlob(inputInfoItem.first.c_str());

        const SizeVector inputDims = input_blob->getTensorDesc().getDims();
        size_t batchSize = inputDims.at(0);

        std::vector<std::string> imageNames = { imageFileName };
        const size_t maxNumOfTop = 10;
        const size_t resultsCount = outputBlob->size() / batchSize;
        const size_t printedResultsCount = resultsCount > maxNumOfTop ? maxNumOfTop : resultsCount;

        // de-Quantization
        uint8_t zeroPoint = static_cast<uint8_t>(FLAGS_z);
        float scale = static_cast<float>(FLAGS_s);
        slog::info<< "zeroPoint" << zeroPoint << slog::endl;
        slog::info<< "scale" << scale << slog::endl;

        Blob::Ptr dequantOut = deQuantize(outputBlob, scale, zeroPoint);

        ClassificationResult classificationResult(dequantOut, imageNames,
                                                  batchSize, printedResultsCount,
                                                  labels);
        classificationResult.print();

        std::fstream outFile;
        outFile.open("/output.dat", std::ios::in | std::ios::out | std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(outputBlob->buffer(), outputBlob->size());
        }
        outFile.close();
    }
    catch (const std::exception& error) {
        slog::err << "" << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
