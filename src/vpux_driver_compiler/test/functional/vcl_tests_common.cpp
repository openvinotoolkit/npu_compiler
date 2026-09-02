//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_tests_common.hpp"

#include <llvm/Support/JSON.h>

#include <regex>

namespace VCLTest {

uint8_t* allocateBlob(uint64_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(std::calloc(static_cast<size_t>(size), sizeof(uint8_t)));

    if (ptr == nullptr) {
        throw std::runtime_error("Memory allocation failed in allocateBlob!");
    }

    return ptr;
}

void deallocateBlob(uint8_t* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("Pointer is nullptr in deallocateBlob!");
    }

    free(ptr);
}

uint8_t* allocateBlob2(vcl_allocator2_t*, uint64_t size) {
    return allocateBlob(size);
}

void deallocateBlob2(vcl_allocator2_t*, uint8_t* ptr) {
    deallocateBlob(ptr);
}

vcl_result_t VCLTestsCommon::initModelData(const char* netName, const char* weightName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_version_info_t version;
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_ERROR;
    vcl_device_desc_t deviceDesc = getDeviceDesc();
    if (deviceDesc.deviceID == 0u) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    vcl_compiler_handle_t compiler = nullptr;
    ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, nullptr);
    if (ret) {
        std::cerr << "Failed to create compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        std::cerr << "Failed to query compiler props! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        version.major = compilerProp.version.major;
        version.minor = compilerProp.version.minor;
        vclCompilerDestroy(compiler);
    }

    /// Read buffer, add net.xml
    std::ifstream xmlFileStream(netName, std::ios::binary);
    if (!xmlFileStream.is_open()) {
        std::cerr << "Cannot open file " << netName << std::endl;
        return VCL_RESULT_ERROR_IO;
    }
    xmlFileStream.seekg(0, xmlFileStream.end);
    uint64_t xmlSize = xmlFileStream.tellg();
    xmlFileStream.seekg(0, xmlFileStream.beg);

    /// Read weights size, add weight.bin
    std::ifstream weightFileStream(weightName, std::ios::binary);

    if (!weightFileStream.is_open()) {
        std::cerr << "Cannot open file " << weightName << std::endl;
        return VCL_RESULT_ERROR_IO;
    }

    weightFileStream.seekg(0, weightFileStream.end);
    uint64_t weightsSize = weightFileStream.tellg();

    /// Prepare modelIRData for compiler
    uint32_t numberOfInputData = 2;
    modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    modelIR.resize(modelIRSize);

    uint64_t offset = 0;
    memcpy(modelIR.data(), &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR.data() + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR.data() + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR.data() + offset;
    xmlFileStream.read(reinterpret_cast<char*>(xmlData), xmlSize);

    if (xmlFileStream.fail()) {
        std::cerr << "Short read on network buffer!" << std::endl;
        return VCL_RESULT_ERROR_IO;
    }

    offset += xmlSize;
    memcpy(modelIR.data() + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = nullptr;
    if (weightsSize != 0) {
        weights = modelIR.data() + offset;
        weightFileStream.seekg(0, weightFileStream.beg);
        weightFileStream.read(reinterpret_cast<char*>(weights), weightsSize);
        if (weightFileStream.fail()) {
            std::cerr << "Short read on weights file!" << std::endl;
            return VCL_RESULT_ERROR_IO;
        }
    }
    return VCL_RESULT_SUCCESS;
}

std::shared_ptr<ov::Model> VCLTestsCommon::createSimpleModel() {
    /**
     * This example shows how to create ov::Model
     * Parameter--->Multiply--->Add--->Result
     *     Constant---'          /
     *              Constant---'
     */

    /// Create opset3::Parameter operation with static shape
    auto data = std::make_shared<ov::opset3::Parameter>(ov::element::f16, ov::Shape{3, 2});

    auto mul_constant = ov::opset3::Constant::create(ov::element::f16, ov::Shape{1}, {1.5});
    auto mul = std::make_shared<ov::opset3::Multiply>(data, mul_constant);

    auto add_constant = ov::opset3::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
    auto add = std::make_shared<ov::opset3::Add>(mul, add_constant);

    /// Create opset3::Result operation
    auto res = std::make_shared<ov::opset3::Result>(mul);

    /// Create the OpenVINO model
    return std::make_shared<ov::Model>(ov::ResultVector{std::move(res)}, ov::ParameterVector{std::move(data)});
}

std::string VCLTestsCommon::getTestModelsBasePath() {
    /// User shall set POR_PATH to show the location of model pacakge
    if (const auto envVar = std::getenv("POR_PATH")) {
        return envVar;
    }
    return {};
}

std::string VCLTestsCommon::getCidToolPath() {
    /// User shall set CID_TOOL to show the location of config files
    if (const auto envVar = std::getenv("CID_TOOL")) {
        return envVar;
    }
    std::cerr << "CID_TOOL empty! You need to export CID_TOOL to load config!" << std::endl;
    return {};
}

void VCLTestsCommon::postProcessNetOptions(const std::string& device) {
    netOptions += std::string{" NPU_PLATFORM=\""} + device + std::string{"\""};
    netOptions += std::string{" DEVICE_ID=\""} + std::string{"NPU."} + device + std::string{"\""};
}

vcl_device_desc_t VCLTestsCommon::getDeviceDesc() const {
    static const std::unordered_map<std::string, vcl_device_desc_t> deviceDescs = {
            {"3720", {sizeof(vcl_device_desc_t), 0x7D1D, 3, 2}},
            {"4000", {sizeof(vcl_device_desc_t), 0x643E, 3, 6}},
            {"5010", {sizeof(vcl_device_desc_t), 0xB03E, 3, 3}},
            {"5020", {sizeof(vcl_device_desc_t), 0xFD3E, 3, 1}},
    };
    const auto& netInfo = std::get<0>(GetParam());
    const auto& device = netInfo.at("device");
    const auto it = deviceDescs.find(device);
    if (it == deviceDescs.end()) {
        ADD_FAILURE() << "Unknown device \"" << device << "\": fix the test parameters.";
        return {sizeof(vcl_device_desc_t), 0u, 0u, 0u};
    }
    return it->second;
}

vcl_result_t VCLTestsCommon::createCompiler(vcl_compiler_handle_t& compiler, vcl_log_handle_t& logHandle,
                                            vcl_log_level_t debugLevel) {
    vcl_compiler_desc_t compilerDesc{};
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = debugLevel;
    vcl_device_desc_t deviceDesc = getDeviceDesc();
    if (deviceDesc.deviceID == 0u) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    compiler = nullptr;
    const auto ret = vclCompilerCreate(&compilerDesc, &deviceDesc, &compiler, &logHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create compiler! Result:0x", ret);
        return ret;
    }
    if (debugLevel >= VCL_LOG_INFO) {
        vcl_compiler_properties_t props;
        if (const auto propRet = queryCompilerProperties(compiler, props); propRet != VCL_RESULT_SUCCESS) {
            (void)destroyCompiler(compiler);
            compiler = nullptr;
            return propRet;
        }
        std::stringstream ss;
        ss << std::this_thread::get_id();
        const std::string threadId = ss.str();
        std::cout << "############################################" << std::endl;
        std::cout << threadId << " Current compiler info:" << std::endl;
        std::cout << threadId << " ID: " << props.id << std::endl;
        std::cout << threadId << " Version:" << props.version.major << "." << props.version.minor << std::endl;
        std::cout << threadId << "\tSupported opsets:" << props.supportedOpsets << std::endl;
        std::cout << "############################################" << std::endl;
    }
    return ret;
}

vcl_result_t VCLTestsCommon::destroyCompiler(vcl_compiler_handle_t compiler) {
    const auto ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to destroy compiler! Result:0x", ret);
    }
    return ret;
}

vcl_result_t VCLTestsCommon::queryCompilerProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t& props) {
    const auto ret = vclCompilerGetProperties(compiler, &props);
    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
    }
    return ret;
}

IRInfoTestType VCLTestsCommon::readJson2Vec(const std::string& fileName) {
    /// The map that contains model name and it configuration
    IRInfoTestType irInfos;
    std::ifstream configFile(fileName);
    if (!configFile.good()) {
        std::cerr << "File with models configuration is not opened correctly!" << std::endl;
        return irInfos;
    }
    std::string line;
    std::size_t lineNo{};
    const std::string enabledKey = "enabled";
    const std::string defaultEnabledStr = "false";
    while (std::getline(configFile, line)) {
        std::unordered_map<std::string, std::string> irInfo;
        auto expectedRes = llvm::json::parse(line);
        llvm::json::Object res{};
        if (!expectedRes) {
            std::cerr << "Provided config contains error in JSON syntax! File: " << fileName << ", line: " << lineNo
                      << std::endl;
            return irInfos;
        } else {
            auto resValue = expectedRes.get();
            if (resValue.getAsObject() == nullptr) {
                std::cerr << "Provided config contains error in a line: " << lineNo
                          << ". Test expects JSON object here." << std::endl;
                return irInfos;
            }
            res = *resValue.getAsObject();
        }

        for (auto it = res.begin(); it != res.end(); ++it) {
            irInfo.emplace(it->first.str(), it->second.getAsString().value());
        }

        /// If the test config is enabled
        bool enabled;
        auto find = irInfo.find(enabledKey);
        std::istringstream valueOfEnabled;
        if (find != irInfo.end()) {
            /// Store the user setting
            valueOfEnabled.str(find->second);
        } else {
            /// Skip this test config by default
            std::cout << "Not found enabled entry, using false as its default\n";
            valueOfEnabled.str(defaultEnabledStr);
        }
        valueOfEnabled >> std::boolalpha >> enabled;

        if (enabled) {
            /// Only add a config to test if we enable it in config file
            irInfos.push_back(std::move(irInfo));
        }

        ++lineNo;
    }
    return irInfos;
}

void VCLTestsCommon::SetUp() {
    auto ir = GetParam();
    auto netInfo = std::get<0>(ir);
    /// Model name
    const std::string net = netInfo.at("network");
    /// Model build flags
    netOptions = netInfo.at("info");
    /// Add device info to build flags
    postProcessNetOptions(netInfo.at("device"));
    if (net != "simple_function") {
        /// Load model data from disk
        std::string PATH = getTestModelsBasePath();
        if (PATH.empty()) {
            GTEST_SKIP() << "POR_PATH is empty. Skipping this test";
        }
        /// The relative path of model to model package
        const std::string interPath = netInfo.at("path");
        std::string netName = PATH + "/" + interPath + "/" + net + ".xml";
        std::string weightName = PATH + "/" + interPath + "/" + net + ".bin";
        std::cout << "netName:" << netName << " weightName:" << weightName << std::endl;

#if defined(_WIN32)
        bool netExist = PathFileExists(netName.c_str());
        EXPECT_EQ(netExist, true) << "The network " << netName.c_str() << " does not exist!" << std::endl;
#else
        struct stat buffer;
        int netExist = stat(netName.c_str(), &buffer);
        EXPECT_EQ(netExist, 0) << "The network " << netName.c_str() << " does not exist!" << std::endl;
#endif

        EXPECT_EQ(initModelData(netName.c_str(), weightName.c_str()), VCL_RESULT_SUCCESS);
    } else {
        std::shared_ptr<ov::Model> model = createSimpleModel();

        // Get input/output names dynamically.
        // In VCLTestsCommon::createSimpleModel, the created simple network only has one
        // input/output, so no need to iterate through iterators.
        std::string inputName = model->get_parameters().at(0)->get_friendly_name();
        std::string outputName = model->get_results().at(0)->get_input_node_ptr(0)->get_friendly_name();

        // Update netOptions by replacing input/output placeholders with dynamic names.
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_IN_PLACEHOLDER"), inputName);
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_OUT_PLACEHOLDER"), outputName);

        std::string mOutXmlPath = "./simple_func_multiple.xml";
        std::string mOutBinPath = "./simple_func_multiple.bin";

        ov::pass::Manager passManager;
        passManager.register_pass<ov::pass::Serialize>(mOutXmlPath, mOutBinPath);
        passManager.run_passes(std::move(model));

        EXPECT_EQ(initModelData(mOutXmlPath.c_str(), mOutBinPath.c_str()), VCL_RESULT_SUCCESS);

        int xmlStatus = std::remove(mOutXmlPath.c_str());
        int binStatus = std::remove(mOutBinPath.c_str());
        if (xmlStatus == 0 && binStatus == 0) {
            std::cout << "Temp IRs" << mOutXmlPath << " and " << mOutBinPath << " deleting succeeded!" << std::endl;
        } else {
            std::cerr << "Temp IRs" << mOutXmlPath << " and " << mOutBinPath << " deleting failed!" << std::endl;
        }
    }
}

}  // namespace VCLTest
