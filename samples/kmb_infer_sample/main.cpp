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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>

#include "infer_app.h"

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

    if (FLAGS_head.empty()) {
        throw std::logic_error("Parameter -head is not set");
    }

    if (FLAGS_tail.empty()) {
        throw std::logic_error("Parameter -tail is not set");
    }

    return true;
}

bool readBinaryFile(std::string input_binary, std::string& data) {
    std::ifstream in(input_binary, std::ios_base::binary | std::ios_base::ate);

    size_t sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    data.resize(sizeFile);
    bool status = false;
    if (in.good()) {
        in.read(&data.front(), sizeFile);
        status = true;
    }
    return status;
}

struct layerDesc {
    std::string type;
};

std::vector<layerDesc> parseSpecialProcessing(const std::string& filename) {
    std::vector<layerDesc> stages;
    return stages;
}

void processLayers(const std::vector<layerDesc>& layers, Blob::Ptr src, Blob::Ptr dst) {
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

        // --------------------------- 2. Read parts of network and blob Generated by MCM Compiler ----------------------------------

        /** Path to preprocessing layers **/
        std::string pathToHeadLayers = FLAGS_head;

        /** Path to postprocessing layers **/
        std::string pathToTailLayers = FLAGS_tail;

        std::vector<layerDesc> preProcess = parseSpecialProcessing(pathToHeadLayers);
        std::vector<layerDesc> postProcess = parseSpecialProcessing(pathToTailLayers);

        std::string binFileName = FLAGS_m;
        slog::info << "Loading blob:\t" << binFileName << slog::endl;

        InferenceEngine::ExecutableNetwork importedNetwork = ie.ImportNetwork(binFileName, "KMB", {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        InferenceEngine::ResponseDesc response;
        ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        std::string imageData;
        if (!readBinaryFile(imageFileName, imageData)) {
            slog::info << "failed to read " << imageFileName << slog::endl;
            throw std::logic_error("Valid input images were not found!");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. PreProcessing calculation ---------------------------------------------

        Blob::Ptr inputNetworkBlob;   // FP16
        Blob::Ptr afterProcessingBlob;  // U8

        processLayers(preProcess, inputNetworkBlob, afterProcessingBlob);


        // Input Processing


        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        auto inferRequest = importedNetwork.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /** Creating input blob **/
        Blob::Ptr kmbInputBlob = inferRequest.GetBlob(inputInfo.begin()->first);

        auto kmbInput = kmbInputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        auto afterProcessingData = afterProcessingBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

        std::copy_n(afterProcessingData, afterProcessingBlob->byteSize(), kmbInput);

        inferRequest.Infer();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();
        if (outputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 output");

        Blob::Ptr outputKMBBlob = inferRequest.GetBlob(outputInfo.begin()->first);

        Blob::Ptr output_blob;  // FP16

        processLayers(postProcess, outputKMBBlob, output_blob);

        // PostProcessing blobs
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
