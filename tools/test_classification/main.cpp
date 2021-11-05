// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>
#include <algorithm>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>

#include <sys/stat.h>

#include "test_classification.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    std::vector<std::string> allowedPrecision = {"u8", "fp16", "fp32"};
    if (!FLAGS_ip.empty()) {
        // input precision is u8, fp16 or fp32 only
        if (std::find(allowedPrecision.begin(), allowedPrecision.end(), FLAGS_ip ) == allowedPrecision.end() )
            throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
        std::transform(FLAGS_ip.begin(),FLAGS_ip.end(),FLAGS_ip.begin(), ::toupper);
    }

    if (!FLAGS_op.empty()) {
        // output precision is u8, fp16 or fp32 only
        if (std::find(allowedPrecision.begin(), allowedPrecision.end(), FLAGS_op ) == allowedPrecision.end() )
            throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
        std::transform(FLAGS_op.begin(),FLAGS_op.end(),FLAGS_op.begin(), ::toupper);
    }

    return true;
}

template <typename T> void writeToFile(std::vector<T>& input, const std::string& dst) {
    std::ofstream dumper(dst, std::ios_base::binary);
    dumper.write(reinterpret_cast<char *>(&input[0]), input.size()*sizeof(T));
    dumper.close();
}

void dumpBlob(const Blob::Ptr& inputBlobPtr, const std::string& dst) {
    std::ofstream dumper(dst, std::ios_base::binary);
    if (dumper.good()) {
        dumper.write(inputBlobPtr->cbuffer().as<char *>(), inputBlobPtr->byteSize());
        // std::cout << "byte size: " << inputBlobPtr->byteSize();
    }
    dumper.close();
}

template <class T_data>
std::vector<T_data> generateSequence(std::size_t dataSize, const std::string& dtype) {
    std::vector<T_data> result(dataSize);
    if (dtype == "U8") {
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = (T_data)i;
    } else if (dtype == "FP32") {
        float LO = -10, HI = 10;
        float nummax = RAND_MAX;
        for (std::size_t i = 0; i < result.size(); ++i)
             result[i] = LO + (static_cast <float> (std::rand()) /(nummax/(HI-LO)));
    }
    return result;
}

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;

        Core ie;

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = std::make_shared<Extension>(FLAGS_l);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Printing device version **/
        slog::info << ie.GetVersions(FLAGS_d) << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        bool multiInput = false;
        if (inputInfo.size() > 1){
            multiInput = true;
            slog::info << "\t multiple inputs detected" << slog::endl;
            slog::info << "\t inputs feeding order: ";
            for (auto& inputInfoItem : inputInfo) {
                slog::info << inputInfoItem.second->name() << ",";
            }
            slog::info << slog::endl;
        }

        // input precision
        Precision inPrecision;
        std::vector<SizeVector> dims;
        std::vector<size_t> totalSizes;
        if (!FLAGS_ip.empty()) {
            if (FLAGS_ip == "FP16")
                inPrecision = Precision::FP16;
            else if (FLAGS_ip == "FP32")
                inPrecision = Precision::FP32;
            else
                inPrecision = Precision::U8;
        } else
            inPrecision = Precision::U8;

        for (auto& inputInfoItem : inputInfo) {
            inputInfoItem.second->setPrecision(inPrecision);
            // input layout
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4)
                inputInfoItem.second->setLayout(Layout::NCHW);
            else if (inputInfoItem.second->getTensorDesc().getDims().size() == 1)
                inputInfoItem.second->setLayout(Layout::C);
            else
                inputInfoItem.second->setLayout(Layout::NC);

            dims.push_back(inputInfoItem.second->getTensorDesc().getDims());
            totalSizes.push_back(std::accumulate(begin(dims.back()), end(dims.back()), 1, std::multiplies<size_t>()));
        }

        // validImageNames is needed for classificationresult function. keeping it for now
        std::vector<std::string> validImageNames = {};
        std::vector<std::shared_ptr<unsigned char>> imagesData = {};
        std::vector<std::vector<uint8_t>> inputSeqs_u8;
        bool inputChannelMajor = false;

        // parsing the inputs
        std::vector<std::string> inputNames;
        if (!multiInput) inputNames.push_back(FLAGS_i);
        else {
            std::stringstream inputNameStream(FLAGS_i);
            while(inputNameStream.good()) {
                std::string inputNameItem;
                getline(inputNameStream, inputNameItem, ',');
                inputNames.push_back(inputNameItem);
            }
            if (inputNames.size() != inputInfo.size())
                throw std::logic_error("Inputs number doesn't match the required inputs.");
        }

        // generate an input or use provided
        if (FLAGS_i.empty()) {
            // auto generate an input
            slog::info << "No image provided, generating a random input..." << slog::endl;
            validImageNames.push_back("Autogenerated");
            for (int index = 0; index < totalSizes.size(); index++) {
                inputSeqs_u8.push_back((generateSequence<uint8_t>(totalSizes[index], "U8")));
            }
        } else {
            // Only consider the first image input.
            validImageNames.push_back(inputNames[0]);
            for (int inputIndex = 0; inputIndex < inputNames.size(); inputIndex++) {
                std::string inputName = inputNames[inputIndex];
                size_t totalSize = totalSizes[inputIndex];
                std::vector<size_t> inputDims = dims[inputIndex];
                std::vector<uint8_t> inputSeq_u8;
                if (fileExt(inputName).compare("dat") == 0 || fileExt(inputName).compare("bin") == 0) {
                    // use a binary input.dat or .bin
                    slog::info << "Using provided binary input..." << slog::endl;
                    std::ifstream file(inputName, std::ios::in | std::ios::binary);
                    if (!file.is_open())
                        throw std::logic_error("Input: " + inputName + " cannot be read!");
                    file.seekg(0, std::ios::end);
                    size_t total = file.tellg() / sizeof(uint8_t);
                    if (inPrecision == Precision::FP32)
                        total = file.tellg() / sizeof(float);
                    if (total != totalSize) {
                        // number of entries doesn't match, either not U8 or from different network
                        throw std::logic_error("Input contains " + std::to_string(total) + " entries," +
                            "which doesn't match expected dimensions: " + std::to_string(totalSize));
                    }
                    file.seekg(0, std::ios::beg);
                    inputSeq_u8.resize(total);
                    file.read(reinterpret_cast<char *>(&inputSeq_u8[0]), total * sizeof(uint8_t));
                    inputChannelMajor = inputDims.size()==4 ? true : false;

                } else {
                    // use a provided image
                    slog::info << "Using provided image..." << slog::endl;
                    FormatReader::ReaderPtr reader(inputName.c_str());
                    if (reader.get() == nullptr) {
                        throw std::logic_error("Image: " + inputName + " cannot be read!");
                    }

                    /** Store image data **/
                    std::shared_ptr<unsigned char> data(
                            reader->getData(inputDims[3],inputDims[2]));
                    if (data != nullptr) {
                        imagesData.push_back(data);
                    }
                    // store input in vector for processing later
                    for (size_t i = 0; i < totalSize; ++i) {
                        inputSeq_u8.push_back(static_cast<uint8_t>(imagesData.at(0).get()[i]));
                    }
                }
                inputSeqs_u8.push_back(inputSeq_u8);
            }
        }
        
        // Output precision
        OutputsDataMap outputInfo(network.getOutputsInfo());
        if (!FLAGS_op.empty()) {
            Precision prc = Precision::U8;
            if (FLAGS_op == "FP16") prc = Precision::FP16;
            else if (FLAGS_op == "FP32") prc = Precision::FP32;
            else prc = Precision::U8;

            // possibly multiple outputs
            for (auto outputInfoIt=outputInfo.begin(); outputInfoIt!=outputInfo.end(); ++outputInfoIt){
                outputInfoIt->second->setPrecision(prc);
                if (outputInfoIt->second->getDims().size() == 2) {
                    outputInfoIt->second->setLayout(InferenceEngine::Layout::NC);
                } else {
                    outputInfoIt->second->setLayout(InferenceEngine::Layout::NHWC);
                }
            }
        }

        /** Setting batch size using image count **/
        network.setBatchSize(1);
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        InferRequest inferRequest = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        auto item = inputInfo.begin();
        for (int inputIndex = 0; inputIndex < inputInfo.size(); inputIndex++) {
            Blob::Ptr inputBlob = inferRequest.GetBlob(item->first);
            SizeVector blobDims = inputBlob->getTensorDesc().getDims();
            size_t totalSize = totalSizes[inputIndex];
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = 1;
            size_t image_size = 1;
            if (blobDims.size() == 4) {  // image input
                num_channels = blobDims[1];
                image_size = blobDims[3] * blobDims[2];
            }
            else {  // unknown kind of input
                num_channels = 1;
                image_size = totalSize;
            }
            // ASSUMPTION inputSeq_u8 is Z-Major (NHWC), BGR
            std::vector<uint8_t> input_nchw_rgb(num_channels * image_size);
            std::vector<uint8_t> input_nchw_bgr(num_channels * image_size);
            std::vector<uint8_t> input_nhwc_rgb(num_channels * image_size);
            std::vector<uint8_t> input_nhwc_bgr(num_channels * image_size);

            // write to the buffer if the input is not image format
            // For non image input, only supportU8 precision
            if (blobDims.size() != 4) {
                std::cout << inputNames[inputIndex] << std::endl;
                std::ifstream file(inputNames[inputIndex], std::ios::in | std::ios::binary);
                if (!file.is_open())
                    throw std::logic_error("Input: " + inputNames[inputIndex] + " cannot be read!");
                file.seekg(0, std::ios::end);
                size_t total = file.tellg() / sizeof(float);
                if (total != totalSize) {
                    // number of entries doesn't match, either not FP32 or from different network
                    throw std::logic_error("Input contains " + std::to_string(total) + " entries, " +
                                           "which doesn't match expected dimensions: " + std::to_string(totalSize));
                }
                file.seekg(0, std::ios::beg);
                std::vector<float> inputSeq_fp32;
                inputSeq_fp32.resize(total);
                file.read(reinterpret_cast<char *>(&inputSeq_fp32[0]), total * sizeof(float));
                auto data = inputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                if (data == nullptr) {
                    throw std::logic_error("input blob buffer is null");
                }
                for (size_t pid = 0; pid < total; pid++) {
                    data[pid] = inputSeq_fp32[pid];
                }
                item++;
                continue;
            }

            if (inPrecision == Precision::U8) {
                auto data = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
                auto inputSeq_u8 = inputSeqs_u8[inputIndex];
                if (data == nullptr) {
                    throw std::logic_error("input blob buffer is null");
                }
                if(inputChannelMajor) // Input is Channel Major, BGR
                {
                    for (size_t i = 0; i < inputSeq_u8.size(); ++i){
                        if(!FLAGS_r) // Keep Channel Major BGR
                            data[i] = inputSeq_u8[i];
                        input_nchw_bgr[i] = inputSeq_u8[i];
                        // Channel Major RGB
                        if(i < image_size) { // B
                            if(FLAGS_r)
                                data[i] = inputSeq_u8.at(i+(image_size*2));
                            input_nchw_rgb[i] = inputSeq_u8.at(i+(image_size*2));
                        }
                        else if ( i > image_size * 2) { // R
                            if(FLAGS_r)
                                data[i+(image_size*2)] = inputSeq_u8.at(i);
                            input_nchw_rgb[i+(image_size*2)] = inputSeq_u8.at(i);
                        }
                        else { // G
                            if(FLAGS_r)
                                data[i] = inputSeq_u8.at(i);
                            input_nchw_rgb[i] = inputSeq_u8.at(i);
                        }
                    }
                    //Create Z-Major BGR for KMB (may be required later)
                    for (size_t ch = 0; ch < num_channels; ++ch)
                    {
                        for (size_t pid = 0; pid < image_size; pid++)
                        {
                            input_nhwc_bgr[pid*num_channels + pid] = inputSeq_u8.at(ch*image_size + pid);
                        }
                    }

                    // Create Z-Major RGB. Use Z-Major BGR, just swap channels into RGB
                    for (size_t i = 1; i <  input_nhwc_bgr.size(); i=i+3)
                    {
                        input_nhwc_rgb[i-1] =  input_nhwc_bgr.at(i+1); // R <- B
                        input_nhwc_rgb[i] =  input_nhwc_bgr.at(i); // G
                        input_nhwc_rgb[i+1] = input_nhwc_bgr.at(i-1); // B <- R
                    }
                }
                else // Input is Z-Major, BGR. From bmp image, for example.
                {
                    for(size_t pid = 0; pid < inputSeq_u8.size(); pid++)
                        input_nhwc_bgr[pid] = inputSeq_u8[pid];
                    for (size_t pid = 0; pid < image_size; pid++) {
                        /** Iterate over all channels to create channel major input **/
                        for (size_t ch = 0; ch < num_channels; ++ch) {
                            int swap_ch = (num_channels-1) - ch;
                            if(!FLAGS_r) // Create channel major, BGR input for CPU Plugin
                                data[ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            input_nchw_bgr[ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            if(FLAGS_r) // Create channel major, RGB input for CPU Plugin
                                data[swap_ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                            input_nchw_rgb[swap_ch * image_size + pid] = inputSeq_u8.at(pid*num_channels + ch);
                        }
                    }
                    // Keep Z-major, just swap channels into RGB
                    for (size_t i = 1; i < inputSeq_u8.size(); i=i+3) {
                        input_nhwc_rgb[i-1] = inputSeq_u8.at(i+1); // R <- B
                        input_nhwc_rgb[i] = inputSeq_u8.at(i); // G
                        input_nhwc_rgb[i+1] = inputSeq_u8.at(i-1); // B <- R
                    }
                }
                writeToFile<uint8_t>(input_nhwc_bgr, "./input_cpu_nhwc_bgr.bin");
                writeToFile<uint8_t>(input_nhwc_rgb, "./input_cpu_nhwc_rgb.bin");
                writeToFile<uint8_t>(input_nchw_rgb, "./input_cpu_nchw_rgb.bin");
                writeToFile<uint8_t>(input_nchw_bgr, "./input_cpu_nchw_bgr.bin");
             }
            // TODO Update to create all 4 options for input
            else if (inPrecision == Precision::FP32) {
                std::ifstream file(inputNames[inputIndex], std::ios::in | std::ios::binary);
                if (!file.is_open())
                    throw std::logic_error("Input: " + inputNames[inputIndex] + " cannot be read!");
                file.seekg(0, std::ios::end);
                size_t total = file.tellg() / sizeof(float);
                if (total != totalSize) {
                    // number of entries doesn't match, either not FP32 or from different network
                    throw std::logic_error("Input contains " + std::to_string(total) + " entries, " +
                                           "which doesn't match expected dimensions: " + std::to_string(totalSize));
                }
                file.seekg(0, std::ios::beg);
                std::vector<float> inputSeq_fp32;
                inputSeq_fp32.resize(total);
                file.read(reinterpret_cast<char *>(&inputSeq_fp32[0]), total * sizeof(float));
                auto data = inputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        data[ch * image_size + pid] = inputSeq_fp32.at(pid*num_channels + ch);
                        // std::cout << " data[" << (ch * image_size + pid) << "] = " << data[ch * image_size + pid];
                    }
                }
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nhwc_bgr.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nhwc_rgb.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nchw_bgr.bin");
                writeToFile<float>(inputSeq_fp32, "./input_cpu_nchw_rgb.bin");
            }
            item++;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        size_t numIterations = 1;
        size_t curIteration = 0;
        std::condition_variable condVar;

        inferRequest.SetCompletionCallback(
                [&] {
                    curIteration++;
                    slog::info << "Completed " << curIteration << " async request execution" << slog::endl;
                    if (curIteration < numIterations) {
                        /* here a user can read output containing inference results and put new input
                           to repeat async request again */
                        inferRequest.StartAsync();
                    } else {
                        /* continue sample execution after last Asynchronous inference request execution */
                        condVar.notify_one();
                    }
                });

        /* Start async request for the first time */
        slog::info << "Start inference (" << numIterations << " asynchronous executions)" << slog::endl;
        inferRequest.StartAsync();

        /* Wait all repetitions of the async request */
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        condVar.wait(lock, [&]{ return curIteration == numIterations; });

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing " << outputInfo.size() << " output blob" << ((outputInfo.size() > 1) ? "s" : "") << slog::endl;

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
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

        std::vector<Blob::Ptr> outputBlob;
        for (auto outputInfoIt=outputInfo.begin(); outputInfoIt!=outputInfo.end(); ++outputInfoIt){
            outputBlob.push_back(inferRequest.GetBlob(outputInfoIt->first));
        }

        for (unsigned i = 0 ; i < outputBlob.size(); i++)
        {
            /** Validating -nt value **/
            const size_t resultsCnt = outputBlob[i]->size() / batchSize;
            if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
                slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                          << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt << slog::endl;
                FLAGS_nt = resultsCnt;
            }

            // save the results file for validation
            if (i == 0) dumpBlob(outputBlob[i], "./output_cpu.bin"); //TODO: remove when validator updated
            dumpBlob(outputBlob[i], "./output_cpu" + std::to_string(i) + ".bin");
            ClassificationResult classificationResult(outputBlob[i], validImageNames,
                                                      batchSize, FLAGS_nt,
                                                      labels);
            if (outputBlob.size() > 1)
                std::cout << "Output " << i << ":" << std::endl;
            classificationResult.print();

            std::string results_filename;
            if (outputBlob.size() == 1)
                results_filename = "./inference_results.txt";
            else
                results_filename = "./inference_results" + std::to_string(i) + ".txt";
            std::ofstream f(results_filename);
            auto topK = classificationResult.getResults();
            for(auto i = topK.begin(); i != topK.end(); ++i) {
                f << *i << '\n';
            }

        }


        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}

