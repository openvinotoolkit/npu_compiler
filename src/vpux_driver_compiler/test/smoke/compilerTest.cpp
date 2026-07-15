//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/Support/CommandLine.h>
#include "utils.hpp"
#include "vcl_api.hpp"
#include "vcl_utils.hpp"

static constexpr char helpMessage[] = "Optional. Print the usage message.";

static constexpr char modelMessage[] = "Required. Path to the XML model.";

static constexpr char targetDeviceMessage[] =
        "Required. Specify a target device for which executable network will be compiled. NPU, NPU.4000 ...";

static constexpr char outputMessage[] = "Optional. Path to the output file. Default value: \"<model_xml_file>.blob\".";

static constexpr char logLevelMessage[] = "Optional. Log level to show log content";

static constexpr char configMessage[] = "Optional. Path to the configuration file.";

static constexpr char inputsPrecisionMessage[] = "Optional. Specifies precision for all input layers of the network.";

static constexpr char outputsPrecisionMessage[] = "Optional. Specifies precision for all output layers of the network.";

static constexpr char iopMessage[] =
        "Optional. Specifies precision for input and output layers by name.\n"
        "                                             Example: -iop \"input:FP16, output:FP16\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites precision from ip and op options for specified "
        "layers.";

static constexpr char inputsLayoutMessage[] = "Optional. Specifies layout for all input layers of the network.";

static constexpr char outputsLayoutMessage[] = "Optional. Specifies layout for all output layers of the network.";

static constexpr char iolMessage[] =
        "Optional. Specifies layout for input and output layers by name.\n"
        "                                             Example: -iol \"input:NCHW, output:NHWC\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites layout from il and ol options for specified layers.";

static constexpr char inputsModelLayoutMessage[] =
        "Optional. Specifies model layout for all input layers of the network.";

static constexpr char outputsModelLayoutMessage[] =
        "Optional. Specifies model layout for all output layers of the network.";

static constexpr char iomlMessage[] =
        "Optional. Specifies model layout for input and output tensors by name.\n"
        "                                             Example: -ionl \"input:NCHW, output:NHWC\".\n"
        "                                             Notice that quotes are required.\n"
        "                                             Overwrites layout from il and ol options for specified layers.";

static const char shapeMessage[] =
        "Optional. Set shape for model input. For example, \"input1[1,3,224,224],input2[1,4]\" or \"[1,3,224,224]\""
        " in case of one input size. This parameter affects model input shape and can be dynamic."
        " For dynamic dimensions use symbol `?` or '-1'. Ex. [?,3,224,224]."
        " For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].";

static const char overrideModelBatchSize[] = "Optional. Enforce a model to be compiled for batch size.";

static const char api1HelpMessage[] = "Optional. Choose to show the help message for compilerTest api 1.0.";

static llvm::cl::opt<std::string> inputModelFile("m");
static llvm::cl::opt<std::string> outputFile("o");
static llvm::cl::opt<std::string> configFile("c");
static llvm::cl::opt<std::string> deviceTarget("d");
static llvm::cl::opt<std::string> logLevel("log_level");
static llvm::cl::opt<bool> helpInfo("h", llvm::cl::init(false));
static llvm::cl::opt<std::string> inputPrecision("ip");
static llvm::cl::opt<std::string> inputLayout("il");
static llvm::cl::opt<std::string> outputPrecision("op");
static llvm::cl::opt<std::string> outputLayout("ol");
static llvm::cl::opt<std::string> inputOutputPrecision("iop");
static llvm::cl::opt<std::string> inputOutputLayout("iol");
static llvm::cl::opt<std::string> inputModelLayout("iml");
static llvm::cl::opt<std::string> outputModelLayout("oml");
static llvm::cl::opt<std::string> inputOutputModelLayout("ioml");
static llvm::cl::opt<std::string> inputShape("shape");
static llvm::cl::opt<uint32_t> batchSize("overrideModelBatchSize", llvm::cl::init(1));
static llvm::cl::opt<bool> api1HelpInfo("api1", llvm::cl::init(false));

namespace {
std::vector<std::string> splitStringList(const std::string& str, char delim) {
    if (str.empty()) {
        return {};
    }

    std::istringstream istr(str);

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

std::map<std::string, std::string> parseArgMap(std::string argMap) {
    argMap.erase(std::remove_if(argMap.begin(), argMap.end(), ::isspace), argMap.end());

    const auto pairs = splitStringList(argMap, ',');

    std::map<std::string, std::string> parsedMap;
    for (auto&& pair : pairs) {
        const auto lastDelimPos = pair.find_last_of(':');
        auto key = pair.substr(0, lastDelimPos);
        auto value = pair.substr(lastDelimPos + 1);

        if (lastDelimPos == std::string::npos || key.empty() || value.empty()) {
            throw std::invalid_argument("Invalid key/value pair " + pair + ". Expected <layer_name>:<value>");
        }

        parsedMap[std::move(key)] = std::move(value);
    }

    return parsedMap;
}
}  // namespace

using supported_type_t = std::unordered_map<std::string, ov::element::Type>;
ov::element::Type getType(std::string value, const supported_type_t& supported_precisions) {
    std::transform(value.begin(), value.end(), value.begin(), ::toupper);

    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}

ov::element::Type getType(const std::string& value) {
    static const supported_type_t supported_types = {
            {"FP32", ov::element::f32}, {"f32", ov::element::f32},      {"FP16", ov::element::f16},
            {"f16", ov::element::f16},  {"BF16", ov::element::bf16},    {"bf16", ov::element::bf16},
            {"U64", ov::element::u64},  {"u64", ov::element::u64},      {"I64", ov::element::i64},
            {"i64", ov::element::i64},  {"U32", ov::element::u32},      {"u32", ov::element::u32},
            {"I32", ov::element::i32},  {"i32", ov::element::i32},      {"U16", ov::element::u16},
            {"u16", ov::element::u16},  {"I16", ov::element::i16},      {"i16", ov::element::i16},
            {"U8", ov::element::u8},    {"u8", ov::element::u8},        {"I8", ov::element::i8},
            {"i8", ov::element::i8},    {"BOOL", ov::element::boolean}, {"boolean", ov::element::boolean},
    };

    return getType(value, supported_types);
}

bool isFP32(const ov::element::Type& type) {
    return type == ov::element::f32;
}

void configurePrePostProcessing(std::shared_ptr<ov::Model>& model, const std::string& ip, const std::string& op,
                                const std::string& iop, const std::string& il, const std::string& ol,
                                const std::string& iol, const std::string& iml, const std::string& oml,
                                const std::string& ioml) {
    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    const auto inputs = model->inputs();
    const auto outputs = model->outputs();

    if (!ip.empty()) {
        auto type = getType(ip);
        for (size_t i = 0; i < inputs.size(); i++) {
            preprocessor.input(i).tensor().set_element_type(type);
        }
    }

    if (!op.empty()) {
        auto type = getType(op);
        for (size_t i = 0; i < outputs.size(); i++) {
            preprocessor.output(i).tensor().set_element_type(type);
        }
    }

    if (!iop.empty()) {
        const auto user_precisions_map = parseArgMap(iop);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;
            const auto type = getType(item.second);

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).tensor().set_element_type(type);
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).tensor().set_element_type(type);
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
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
        const auto user_precisions_map = parseArgMap(iol);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).tensor().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).tensor().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
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
        const auto user_precisions_map = parseArgMap(ioml);
        for (auto&& item : user_precisions_map) {
            const auto& tensor_name = item.first;

            bool tensorFound = false;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].get_names().count(tensor_name)) {
                    preprocessor.input(i).model().set_layout(ov::Layout(item.second));
                    tensorFound = true;
                    break;
                }
            }
            if (!tensorFound) {
                for (size_t i = 0; i < outputs.size(); i++) {
                    if (outputs[i].get_names().count(tensor_name)) {
                        preprocessor.output(i).model().set_layout(ov::Layout(item.second));
                        tensorFound = true;
                        break;
                    }
                }
            }
            OPENVINO_ASSERT(tensorFound, "Model doesn't have input/output with tensor name: ", tensor_name);
        }
    }

    model = preprocessor.build();
}

inline std::string fileNameNoExt(const std::string& filePath) {
    auto pos = filePath.rfind('.');
    if (pos == std::string::npos) {
        return filePath;
    }
    return filePath.substr(0, pos);
}

static void showUsage() {
    std::cout << "compilerTest [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:                             " << std::endl;
    std::cout << "    -h                                       " << helpMessage << std::endl;
    std::cout << "    -m                           <value>     " << modelMessage << std::endl;
    std::cout << "    -d                           <value>     " << targetDeviceMessage << std::endl;
    std::cout << "    -o                           <value>     " << outputMessage << std::endl;
    std::cout << "    -log_level                   <value>     " << logLevelMessage << std::endl;
    std::cout << "    -c                           <value>     " << configMessage << std::endl;
    std::cout << "    -ip                          <value>     " << inputsPrecisionMessage << std::endl;
    std::cout << "    -op                          <value>     " << outputsPrecisionMessage << std::endl;
    std::cout << "    -iop                        \"<value>\"    " << iopMessage << std::endl;
    std::cout << "    -il                          <value>     " << inputsLayoutMessage << std::endl;
    std::cout << "    -ol                          <value>     " << outputsLayoutMessage << std::endl;
    std::cout << "    -iol                        \"<value>\"    " << iolMessage << std::endl;
    std::cout << "    -iml                         <value>     " << inputsModelLayoutMessage << std::endl;
    std::cout << "    -oml                         <value>     " << outputsModelLayoutMessage << std::endl;
    std::cout << "    -ioml                       \"<value>\"    " << iomlMessage << std::endl;
    std::cout << "    -shape                       <value>     " << shapeMessage << std::endl;
    std::cout << "    -overrideModelBatchSize   <value>     " << overrideModelBatchSize << std::endl;
    std::cout << "    -api1                     <value>     " << api1HelpMessage << std::endl;
    std::cout << std::endl;
}

static bool parseCommandLine(int* argc, char*** argv) {
    llvm::cl::ParseCommandLineOptions(*argc, *argv, "CommandLine Options\n");

    if (helpInfo) {
        showUsage();
        return false;
    }

    if (api1HelpInfo) {
        std::cout << "compilerTest usage for api 1.0:" << std::endl;
        std::cout << "compilerTest usage:\n\tcompilerTest net.xml weight.bin output.net" << std::endl;
        std::cout << "compilerTest usage:\n\tcompilerTest net.xml weight.bin output.net $configFile" << std::endl;
        return false;
    }

    if (inputModelFile.empty()) {
        throw std::invalid_argument("Path to model xml file is required");
    }

    if (deviceTarget.empty()) {
        throw std::invalid_argument("Target platform is required");
    }

    return true;
}

std::map<std::string, std::string> parseConfigFile(const std::string& configFile, char comment = '#') {
    std::map<std::string, std::string> config;

    std::ifstream file(configFile);
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

std::string getFileNameFromPath(const std::string& path,
#if defined(_WIN32)
                                const std::string& sep = "\\") {
#else
                                const std::string& sep = "/") {
#endif
    const auto pos = path.rfind(sep);
    if (std::string::npos == pos) {
        return path;
    } else {
        return path.substr(pos + 1);
    }
}

using TimeDiff = std::chrono::milliseconds;

int main(int argc, char* argv[]) {
    try {
        if (argc == 1 || VCLTest::hasOption("-h", argc, argv)) {
            showUsage();
            return -1;
        }

        if (VCLTest::hasOption("-m", argc, argv)) {
            std::cout << "Parsing command-line arguments" << std::endl;
            if (!parseCommandLine(&argc, &argv)) {
                return EXIT_SUCCESS;
            }

            std::cout << "Using compilerTest api2.0 to compile" << std::endl;
            TimeDiff loadNetworkTimeElapsed{0};

            const auto& version = ov::get_openvino_version();
            std::cout << version.description << " version ......... ";
            std::cout << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH
                      << std::endl;

            std::cout << "Build ........... ";
            std::cout << version.buildNumber << std::endl;
            if (outputFile.empty()) {
                outputFile = fileNameNoExt(inputModelFile.getValue()) + ".blob";
            }

            std::cout << "1. Reading model" << std::endl;
            ov::Core core;
            std::shared_ptr<ov::Model> model = core.read_model(inputModelFile.getValue());

            std::cout << "2. Performing reshape" << std::endl;
            auto inputsInfo = std::const_pointer_cast<ov::Model>(model)->inputs();
            VCLTest::InputsInfo infoMap;
            VCLTest::reshape(std::move(inputsInfo), infoMap, model, inputShape.getValue(), batchSize.getValue());

            if (inputShape.empty()) {
                VCLTest::setModelBatch(model, batchSize.getValue());
            }

            std::cout << "3. Configuring model pre & post processing" << std::endl;
            configurePrePostProcessing(model, inputPrecision.getValue(), outputPrecision.getValue(),
                                       inputOutputPrecision.getValue(), inputLayout.getValue(), outputLayout.getValue(),
                                       inputOutputLayout.getValue(), inputModelLayout.getValue(),
                                       outputModelLayout.getValue(), inputOutputModelLayout.getValue());

            std::cout << "4. Printing Updated Input and Output Info from model" << std::endl;
            VCLTest::printInputAndOutputsInfoShort(*model);

            std::cout << "5. Parse Configuration" << std::endl;
            std::map<std::string, std::string> compilerConfig;
            if (!configFile.empty()) {
                compilerConfig = parseConfigFile(configFile.getValue());
            }

            if (compilerConfig.find("NPU_PLATFORM") == compilerConfig.end()) {
                compilerConfig.emplace("NPU_PLATFORM", VCLTest::getPlatform(deviceTarget.getValue()));
            }

            if (compilerConfig.find("LOG_LEVEL") == compilerConfig.end()) {
                if (!logLevel.empty()) {
                    compilerConfig.emplace("LOG_LEVEL", logLevel.getValue());
                } else {
                    compilerConfig.emplace("LOG_LEVEL", "LOG_NONE");
                }
            }

            vcl_result_t status = VCL_RESULT_SUCCESS;
            std::string allocatedBlobName;
            int result = VCLTest::getName(outputFile.getValue(), allocatedBlobName);
            if (result) {
                return result;
            }

            std::cout << "6. Compile model with VCL" << std::endl;
            auto timeBeforeLoadNetwork = std::chrono::steady_clock::now();
            status = VCLTest::simulateVclCompilerAllocator(compilerConfig, model, allocatedBlobName.c_str());

            if (status != VCL_RESULT_SUCCESS) {
                VCLTest::printErrorMessage(status);
                return (int)status;
            }
            loadNetworkTimeElapsed =
                    std::chrono::duration_cast<TimeDiff>(std::chrono::steady_clock::now() - timeBeforeLoadNetwork);
            std::cout << "Done. LoadNetwork time elapsed: " << loadNetworkTimeElapsed.count() << " ms" << std::endl;
        } else {
            std::cout << "Using compilerTest api1.0 to compile" << std::endl;
            if (argc != 4 && argc != 5) {
                std::cout << "compilerTest usage:\n\tcompilerTest net.xml weight.bin output.net" << std::endl;
                std::cout << "compilerTest usage:\n\tcompilerTest net.xml weight.bin output.net $configFile"
                          << std::endl;
                return -1;
            }

            std::string allocatedBlobFileName;
            std::string inputFileName(argv[3]);
            int res = VCLTest::getName(inputFileName, allocatedBlobFileName);
            if (res) {
                return res;
            }

            vcl_result_t status = VCL_RESULT_SUCCESS;
            status = VCLTest::simulateVclCompilerAllocatorOldVersion(argc, argv, allocatedBlobFileName.c_str());
            if (status != VCL_RESULT_SUCCESS) {
                VCLTest::printErrorMessage(status);
                return (int)status;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
