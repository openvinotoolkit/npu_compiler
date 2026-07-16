//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/except.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

DEFINE_string(m, "", "[Required] Path to the XML model or pre-compiled blob");
DEFINE_validator(m, [](const char* flagname, const std::string& value) {
    if (value.empty()) {
        std::cerr << "Error: the path to the model / blob must be provided via the -" << flagname << " argument"
                  << std::endl;
        return false;
    }
    if (!std::filesystem::exists(value)) {
        std::cerr << "Error: the specified model / blob file does not exist: " << value << std::endl;
        return false;
    }
    return true;
});

DEFINE_string(d, "", "[Required] The target device for which the model will be compiled (e.g. CPU, NPU)");
DEFINE_validator(d, [](const char* flagname, const std::string& value) {
    if (value.empty()) {
        std::cerr << "Error: the target device must be provided via the -" << flagname << " argument" << std::endl;
        return false;
    }
    return true;
});
DEFINE_string(i, "",
              "[Optional] Path(s) to the input tensor file(s), separated by comma. If not set, randomly generated "
              "values will be used.");
DEFINE_string(o, "",
              "[Optional] Path to the directory where outputs will be saved. If not set, the current directory will be "
              "used.");
DEFINE_string(c, "", "[Optional] Path to the configuration file that will be passed to the plugin");
DEFINE_string(ip, "", "[Optional] Specifies precision for all input layers of the network");
DEFINE_string(op, "", "[Optional] Specifies precision for all output layers of the network.");
DEFINE_string(iop, "",
              "[Optional] Specifies precision for input and output layers by name.\n"
              "Example: -iop \"input:FP16, output:FP16\".\n"
              "Notice that quotes are required.\n"
              "Overwrites precision from ip and op options for specified layers.");
DEFINE_string(il, "", "[Optional] Specifies layout for all input layers of the network");
DEFINE_string(ol, "", "[Optional] Specifies layout for all output layers of the network.");
DEFINE_string(iol, "",
              "[Optional] Specifies layout for input and output layers by name.\n"
              "Example: -iol \"input:NCHW, output:NHWC\".\n"
              "Notice that quotes are required.\n"
              "Overwrites layout from il and ol options for specified layers.");
DEFINE_string(iml, "", "[Optional] Specifies model layout for all input layers of the network");
DEFINE_string(oml, "", "[Optional] Specifies model layout for all output layers of the network.");
DEFINE_string(ioml, "",
              "[Optional] Specifies model layout for input and output tensors by name.\n"
              "Example: -ioml \"input:NCHW, output:NHWC\".\n"
              "Notice that quotes are required.\n"
              "Overwrites layout from il and ol options for specified layers.");
DEFINE_bool(pc, false, "[Optional] Print performance counters (per-layer execution statistics) after inference.");

namespace {

bool isModelPrecompiled(std::string_view filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) {
        return false;
    }
    const auto ext = filename.substr(pos + 1);
    return ext == "blob" || ext == "net";
}

std::vector<std::string> splitStringList(std::string_view str, char delim) {
    if (str.empty()) {
        return {};
    }
    std::istringstream istr(std::string(str), std::ios_base::in);
    std::vector<std::string> result;
    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }
        result.emplace_back(std::move(elem));
    }
    return result;
}

// Note: the original implementation is from the compile_tool, found in the OpenVINO project
void configurePrePostProcessing(std::shared_ptr<ov::Model>& model, const std::string& ip, const std::string& op,
                                const std::string& iop, const std::string& il, const std::string& ol,
                                const std::string& iol, const std::string& iml, const std::string& oml,
                                const std::string& ioml) {
    const auto parseArgMap = [&](std::string argMap) -> std::map<std::string, std::string> {
        argMap.erase(std::remove_if(argMap.begin(), argMap.end(), ::isspace), argMap.end());
        const auto pairs = splitStringList(argMap, ',');
        std::map<std::string, std::string> parsedMap;
        for (auto&& pair : pairs) {
            const auto lastDelimPos = pair.find_last_of(':');
            auto key = pair.substr(0, lastDelimPos);
            auto value = pair.substr(lastDelimPos + 1);
            OPENVINO_ASSERT(lastDelimPos != std::string::npos && !key.empty() && !value.empty(),
                            "Invalid key/value pair ", pair, ". Expected <layer_name>:<value>");
            parsedMap[std::move(key)] = std::move(value);
        }
        return parsedMap;
    };

    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    const auto inputs = model->inputs();
    const auto outputs = model->outputs();

    if (!ip.empty()) {
        auto type = ov::element::Type(ip);
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).tensor().set_element_type(type);
        }
    }
    if (!op.empty()) {
        auto type = ov::element::Type(op);
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).tensor().set_element_type(type);
        }
    }
    if (!iop.empty()) {
        const auto userPrecisionsMap = parseArgMap(iop);
        for (auto&& item : userPrecisionsMap) {
            const auto& tensorName = item.first;
            const auto type = ov::element::Type(item.second);

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensorName)) {
                    preprocessor.input(i).tensor().set_element_type(type);
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensorName)) {
                        preprocessor.output(i).tensor().set_element_type(type);
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input / output with tensor name: ", tensorName);
        }
    }

    if (!il.empty()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).tensor().set_layout(ov::Layout(il));
        }
    }
    if (!ol.empty()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).tensor().set_layout(ov::Layout(ol));
        }
    }
    if (!iol.empty()) {
        const auto userPrecisionsMap = parseArgMap(iol);
        for (auto&& item : userPrecisionsMap) {
            const auto& tensorName = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensorName)) {
                    preprocessor.input(i).tensor().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensorName)) {
                        preprocessor.output(i).tensor().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensorName);
        }
    }

    if (!iml.empty()) {
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).model().set_layout(ov::Layout(iml));
        }
    }

    if (!oml.empty()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).model().set_layout(ov::Layout(oml));
        }
    }

    if (!ioml.empty()) {
        const auto userPrecisionsMap = parseArgMap(ioml);
        for (auto&& item : userPrecisionsMap) {
            const auto& tensorName = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensorName)) {
                    preprocessor.input(i).model().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensorName)) {
                        preprocessor.output(i).model().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensorName);
        }
    }

    model = preprocessor.build();
}

// Note: the original implementation is from the compile_tool, found in the OpenVINO project
std::map<std::string, std::string> parseConfigFile(char comment = '#') {
    std::map<std::string, std::string> config;

    std::ifstream file(FLAGS_c);
    OPENVINO_ASSERT(file.is_open() || FLAGS_c.empty(),
                    "[ERROR] Configuration file " + FLAGS_c + " cannot be opened. " +
                            "Check if the file path is correct and that the file exists");

    if (file.is_open()) {
        std::string option;
        while (std::getline(file, option)) {
            if (option.empty() || option[0] == comment) {
                continue;
            }
            size_t spacePos = option.find_first_of(" \t\n\r");
            OPENVINO_ASSERT(spacePos != std::string::npos, "Failed to find a space separator in "
                                                           "provided plugin config option: " +
                                                                   option);

            std::string key = option.substr(0, spacePos);

            std::string value{};
            size_t valueStart = option.find_first_not_of(" \t\n\r", spacePos);
            OPENVINO_ASSERT(valueStart != std::string::npos, "An invalid config parameter value detected, "
                                                             "it mustn't be empty: " +
                                                                     option);
            size_t valueEnd = option.find_last_not_of(" \t\n\r");
            value = option.substr(valueStart, valueEnd - valueStart + 1);

            config[key] = std::move(value);
        }
    }
    return config;
}

template <typename T>
using uniformDistribution = typename std::conditional<
        std::is_floating_point<T>::value, std::uniform_real_distribution<T>,
        typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;

// Fills a sub-byte (u4/i4) tensor with random values.
// 4-bit types are packed as two nibbles per byte, so tensor.data<T>() is unavailable.
// The raw byte buffer is filled directly via tensor.data() (void*), with each byte
// holding two independently sampled 4-bit values: low nibble in bits [3:0], high in [7:4].
void fillRandom4bit(ov::Tensor& tensor, bool isSigned) {
    std::mt19937 gen(std::mt19937::default_seed);
    const size_t byteSize = tensor.get_byte_size();
    auto* data = static_cast<uint8_t*>(tensor.data());
    if (isSigned) {
        // i4: range [-8, 7], stored as 4-bit two's complement
        std::uniform_int_distribution<int32_t> distribution(-8, 7);
        for (size_t i = 0; i < byteSize; i++) {
            const uint8_t lo = static_cast<uint8_t>(distribution(gen)) & 0xF;
            const uint8_t hi = static_cast<uint8_t>(distribution(gen)) & 0xF;
            data[i] = static_cast<uint8_t>((hi << 4) | lo);
        }
    } else {
        // u4: range [0, 15]
        std::uniform_int_distribution<uint32_t> distribution(0, 15);
        for (size_t i = 0; i < byteSize; i++) {
            const uint8_t lo = static_cast<uint8_t>(distribution(gen));
            const uint8_t hi = static_cast<uint8_t>(distribution(gen));
            data[i] = static_cast<uint8_t>((hi << 4) | lo);
        }
    }
}

template <typename T, typename T2>
void fillRandom(ov::Tensor& tensor, T randMin = std::numeric_limits<uint8_t>::min(),
                T randMax = std::numeric_limits<uint8_t>::max()) {
    std::mt19937 gen(std::mt19937::default_seed);
    size_t tensorSize = tensor.get_size();
    OPENVINO_ASSERT(
            tensorSize != 0,
            "Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference");
    auto data = tensor.data<T>();
    uniformDistribution<T2> distribution(randMin, randMax);
    for (size_t i = 0; i < tensorSize; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }
}

// The original implementation is from the common testing utils, found in the OpenVINO project
inline void fillTensorRandom(ov::Tensor tensor) {
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        fillRandom<float, float>(tensor);
        break;
    case ov::element::f64:
        fillRandom<double, double>(tensor);
        break;
    case ov::element::f16:
        // ov::float16 is the correct pointer-representable type for f16 tensors;
        // using short/int16_t would be rejected by the OpenVINO tensor verifier.
        fillRandom<ov::float16, float>(tensor);
        break;
    case ov::element::i32:
        fillRandom<int32_t, int32_t>(tensor);
        break;
    case ov::element::i64:
        fillRandom<int64_t, int64_t>(tensor);
        break;
    case ov::element::u8:
        // uniform_int_distribution<uint8_t> is not allowed in the C++17
        // standard and vs2017/19
        fillRandom<uint8_t, uint32_t>(tensor);
        break;
    case ov::element::i8:
        // uniform_int_distribution<int8_t> is not allowed in the C++17 standard
        // and vs2017/19
        fillRandom<int8_t, int32_t>(tensor, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        break;
    case ov::element::u4:
        // u4/i4 are sub-byte types; packed nibble filling is required.
        fillRandom4bit(tensor, false);
        break;
    case ov::element::i4:
        // u4/i4 are sub-byte types; packed nibble filling is required.
        fillRandom4bit(tensor, true);
        break;
    case ov::element::u16:
        fillRandom<uint16_t, uint16_t>(tensor);
        break;
    case ov::element::i16:
        fillRandom<int16_t, int16_t>(tensor);
        break;
    case ov::element::boolean:
        fillRandom<uint8_t, uint32_t>(tensor, 0, 1);
        break;
    default:
        OPENVINO_THROW("Input type is not supported for a tensor");
    }
}

std::vector<ov::Tensor> prepareInputTensors(std::string_view inputFiles, const ov::CompiledModel& compiledModel) {
    std::vector<ov::Tensor> inputTensors;
    const auto& modelInputs = compiledModel.inputs();

    if (inputFiles.empty()) {
        std::cout << "-- generating random input data" << std::endl;
        for (const auto& input : modelInputs) {
            ov::Tensor tensor = ov::Tensor(input.get_element_type(), input.get_shape());
            fillTensorRandom(tensor);
            inputTensors.push_back(tensor);
        }
    } else {
        // Read input data from the given files
        const auto inputList = splitStringList(inputFiles, ',');
        OPENVINO_ASSERT(inputList.size() == modelInputs.size(), "The number of provided input files (",
                        inputList.size(), ") does not match the number of model inputs (", modelInputs.size(), ")");
        for (size_t i = 0; i < modelInputs.size(); i++) {
            std::cout << "-- reading input data from file: " << inputList[i] << std::endl;
            const auto& input = modelInputs[i];
            const auto& inputFile = inputList[i];
            std::ifstream inputStream(inputFile, std::ios_base::binary | std::ios_base::in);
            OPENVINO_ASSERT(inputStream.is_open(), "Cannot open input file ", inputFile);
            auto tensor = ov::Tensor(input.get_element_type(), input.get_shape());
            inputStream.read(reinterpret_cast<char*>(tensor.data()),
                             static_cast<std::streamsize>(tensor.get_byte_size()));
            inputTensors.push_back(tensor);
        }
    }

    OPENVINO_ASSERT(inputTensors.size() == modelInputs.size(), "Number of input tensors (", inputTensors.size(),
                    ") does not match the number of model inputs (", modelInputs.size(), ")");
    return inputTensors;
}

void dumpOutputs(std::string_view outputDir, ov::InferRequest& inferRequest) {
    auto outputDirPath = std::filesystem::current_path();
    if (!outputDir.empty()) {
        if (!std::filesystem::is_directory(outputDir)) {
            OPENVINO_ASSERT(std::filesystem::create_directory(outputDir), "Cannot create output directory ", outputDir);
        }
        outputDirPath = std::filesystem::path(outputDir);
    }

    const auto compiledModel = inferRequest.get_compiled_model();
    for (size_t i = 0; i < compiledModel.outputs().size(); i++) {
        const auto output = compiledModel.outputs()[i];
        auto name = output.get_any_name();
        std::replace_if(
                name.begin(), name.end(),
                [](unsigned char c) {
                    return !std::isalnum(c);
                },
                '_');
        std::ostringstream fileName;
        fileName << i << '_' << name << ".bin";

        const auto outputPath = outputDirPath / std::filesystem::path(fileName.str());
        std::cout << "-- writing output " << i << " to file: " << outputPath.string() << std::endl;

        std::ofstream outputFile(outputPath, std::ios_base::binary | std::ios_base::out);
        OPENVINO_ASSERT(outputFile.is_open(), "Cannot open output file ", outputPath.string());
        const auto outputTensor = inferRequest.get_tensor(output);
        outputFile.write(reinterpret_cast<const char*>(outputTensor.data()),
                         static_cast<std::streamsize>(outputTensor.get_byte_size()));
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        gflags::ParseCommandLineFlags(&argc, &argv, true);

        bool modelPrecompiled = isModelPrecompiled(FLAGS_m);

        std::cout << "Parsing configuration file" << std::endl;
        auto configs = parseConfigFile();

        // When performance counters are requested, enable profiling at compile time.
        // This must be set before compile_model / import_model; without it,
        // get_profiling_info() returns empty results.
        if (FLAGS_pc) {
            configs[ov::enable_profiling.name()] = "YES";
        }

        ov::Core core;
        ov::CompiledModel compiledModel;
        if (modelPrecompiled) {
            std::cout << "Importing blob" << std::endl;
            std::ifstream modelStream(FLAGS_m, std::ios_base::binary | std::ios_base::in);
            OPENVINO_ASSERT(modelStream.is_open(), "Cannot open model file ", FLAGS_m);
            compiledModel = core.import_model(modelStream, FLAGS_d, {configs.begin(), configs.end()});
        } else {
            std::cout << "Reading model" << std::endl;
            auto model = core.read_model(FLAGS_m);

            std::cout << "Configuring model pre / post processing" << std::endl;
            configurePrePostProcessing(model, FLAGS_ip, FLAGS_op, FLAGS_iop, FLAGS_il, FLAGS_ol, FLAGS_iol, FLAGS_iml,
                                       FLAGS_oml, FLAGS_ioml);

            std::cout << "Compiling model" << std::endl;
            compiledModel = core.compile_model(model, FLAGS_d, {configs.begin(), configs.end()});
        }

        std::cout << "Preparing input tensors" << std::endl;
        auto inputTensors = prepareInputTensors(FLAGS_i, compiledModel);

        std::cout << "Running inference" << std::endl;
        auto inferRequest = compiledModel.create_infer_request();
        // set_input_tensors() only works for single-input models; use the indexed
        // overload to support models with an arbitrary number of inputs.
        for (size_t i = 0; i < inputTensors.size(); i++) {
            inferRequest.set_input_tensor(i, inputTensors[i]);
        }
        inferRequest.infer();

        if (FLAGS_pc) {
            // Print per-layer profiling data: execution status, real wall-clock time,
            // CPU time (both in microseconds), and the backend execution type.
            std::cout << "Performance counters:" << std::endl;
            const auto perfCounts = inferRequest.get_profiling_info();
            for (const auto& info : perfCounts) {
                std::cout << std::left << std::setw(60) << info.node_name << " status: " << std::setw(12) <<
                        [&] {
                            switch (info.status) {
                            case ov::ProfilingInfo::Status::EXECUTED:
                                return "EXECUTED";
                            case ov::ProfilingInfo::Status::NOT_RUN:
                                return "NOT_RUN";
                            case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
                                return "OPTIMIZED_OUT";
                            default:
                                return "UNKNOWN";
                            }
                        }()
                          << " real_time: " << std::setw(10)
                          << std::chrono::duration_cast<std::chrono::microseconds>(info.real_time).count() << " us"
                          << " cpu_time: " << std::setw(10)
                          << std::chrono::duration_cast<std::chrono::microseconds>(info.cpu_time).count() << " us"
                          << " exec_type: " << info.exec_type << std::endl;
            }
        }

        std::cout << "Dumping outputs" << std::endl;
        dumpOutputs(FLAGS_o, inferRequest);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown / internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
