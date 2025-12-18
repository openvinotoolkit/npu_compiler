//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "npu_driver_compiler.h"
#include "openvino/core/layout.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/runtime/core.hpp"
#include "serialize_utils.hpp"
#include "vcl_metadata.hpp"

void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) {
    size_t constantId = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            ov::RTMap& runtimeInfoMap = node->get_rt_info();
            const auto& weightlessCacheAttrIt =
                    runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

            const std::string constantIdString = std::to_string(constantId++);
            if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
            }
        }
    }
}

void printErrorMessage(vcl_result_t status) {
    const char* message = nullptr;
    switch (status) {
    case VCL_RESULT_ERROR_OUT_OF_MEMORY:
        message = "VCL_RESULT_ERROR_OUT_OF_MEMORY";
        break;

    case VCL_RESULT_ERROR_INVALID_ARGUMENT:
        message = "VCL_RESULT_ERROR_INVALID_ARGUMENT";
        break;

    case VCL_RESULT_ERROR_INVALID_NULL_HANDLE:
        message = "VCL_RESULT_ERROR_INVALID_NULL_HANDLE";
        break;

    case VCL_RESULT_ERROR_IO:
        message = "VCL_RESULT_ERROR_IO";
        break;

    case VCL_RESULT_ERROR_INVALID_IR:
        message = "VCL_RESULT_ERROR_INVALID_IR";
        break;

    case VCL_RESULT_ERROR_UNKNOWN:
        message = "VCL_RESULT_ERROR_UNKNOWN";
        break;

    default:
        message = "UNKNOWN_ERROR";
    }
    std::cerr << "Error message: " << message << std::endl;
}

void getLastError(vcl_log_handle_t logHandle) {
    /// Get latest error info
    if (logHandle != nullptr) {
        size_t logSize = 0;
        vcl_result_t logRet = vclLogHandleGetString(logHandle, &logSize, nullptr);
        if (logRet != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to get size of error message" << std::endl;
        } else if (logSize == 0) {
            std::cerr << "No error during compilation" << std::endl;
        } else {
            try {
                std::vector<char> log(logSize);
                logRet = vclLogHandleGetString(logHandle, &logSize, log.data());
                if (logRet != VCL_RESULT_SUCCESS) {
                    std::cerr << "Failed to get content of error message" << std::endl;
                } else {
                    std::cerr << "The last error: " << log.data() << std::endl;
                }
            } catch (const std::bad_alloc&) {
                std::cerr << "Failed to allocate memory to store error log!" << std::endl;
            }
        }
    }
}

/// Print error info to pass coverity scanner
void printErrorInfo(const std::string& errorStr, vcl_result_t ret) {
    std::ios::fmtflags originalFormat = std::cerr.flags();
    std::cerr << errorStr << std::hex << ret << std::endl;
    std::cerr.flags(originalFormat);
}

bool configureDeviceDesc(vcl_device_desc_t& deviceDesc, uint32_t platform) {
    char* envDeviceDescEmpty = getenv("VCL_SETTING_DEVICEDESC_EMPTY");
    if (!envDeviceDescEmpty) {
        std::cout << "Use the complete device description" << std::endl;
        deviceDesc = {sizeof(vcl_device_desc_t), platform, 3, 5};
        return true;
    } else {
        std::cout << "Use the empty device description" << std::endl;
        deviceDesc = {};
        return false;
    }
}

using supported_platform = std::unordered_map<std::string, uint32_t>;
using supported_vcl_log_level = std::unordered_map<std::string, vcl_log_level_t>;
vcl_result_t getVclCompiler(const std::map<std::string, std::string>& buildConfig, vcl_compiler_handle_t* compiler,
                            vcl_compiler_properties_t* compilerProp, vcl_log_handle_t* logHandle) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    ret = vclGetVersion(&compilerVersion, &profilingVersion);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(*logHandle);
        printErrorInfo("Failed to get version! Result:  0x", ret);
        return ret;
    }
    std::cout << "\n############################################\n" << std::endl;
    std::cout << "Library VCL API version info:" << std::endl
              << "Compiler version:" << compilerVersion.major << "." << compilerVersion.minor << std::endl
              << "Profiling version:" << profilingVersion.major << "." << profilingVersion.minor << std::endl;
    std::cout << "\n############################################\n" << std::endl;

    /// Control if we save error log or output to terminal
    char* saveErrorLog = getenv("VCL_SAVE_ERROR");
    uint32_t platform;
    vcl_log_level_t debugLevel;

    static const supported_platform supported_platforms = {
            {"3720", 0x7D1D},
            {"4000", 0x643E},
            {"5010", 0xB03E},
    };

    const auto supportPlatform = supported_platforms.find(buildConfig.at("NPU_PLATFORM"));
    platform = (supportPlatform != supported_platforms.end()) ? supportPlatform->second : 0x643E;

    static const supported_vcl_log_level supported_vcl_log_levels = {
            {"LOG_NONE", VCL_LOG_NONE}, {"LOG_ERROR", VCL_LOG_ERROR}, {"LOG_WARNING", VCL_LOG_WARNING},
            {"LOG_INFO", VCL_LOG_INFO}, {"LOG_DEBUG", VCL_LOG_DEBUG}, {"LOG_TRACE", VCL_LOG_TRACE},
    };
    const auto supportLogLevel = supported_vcl_log_levels.find(buildConfig.at("LOG_LEVEL"));
    debugLevel = (supportLogLevel != supported_vcl_log_levels.end()) ? supportLogLevel->second : VCL_LOG_INFO;

    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = debugLevel;

    vcl_device_desc_t deviceDesc;
    bool hasDeviceDesc = configureDeviceDesc(deviceDesc, platform);
    vcl_device_desc_t* deviceDescPtr = hasDeviceDesc ? &deviceDesc : nullptr;
    if (saveErrorLog == nullptr) {
        ret = vclCompilerCreate(&compilerDesc, deviceDescPtr, compiler, nullptr);
    } else {
        ret = vclCompilerCreate(&compilerDesc, deviceDescPtr, compiler, logHandle);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(*logHandle);
        printErrorInfo("Failed to create compiler! Result: 0x", ret);
        return ret;
    }

    ret = vclCompilerGetProperties(*compiler, compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(*logHandle);
        printErrorInfo("Failed to query compiler props! Result: 0x", ret);
        return ret;
    }
    std::cout << "\n############################################\n\n";
    std::cout << " Current compiler info:\n"
              << " ID: " << compilerProp->id << "\n"
              << " Version: " << compilerProp->version.major << "." << compilerProp->version.minor << "\n"
              << "\tSupported opsets: " << compilerProp->supportedOpsets << "\n";
    std::cout << "\n############################################\n\n";

    return VCL_RESULT_SUCCESS;
}

std::string configMapToString(const std::map<std::string, std::string>& config) {
    if (config.empty()) {
        return "";
    }

    std::ostringstream oss;
    oss << "--config ";

    for (const auto& pair : config) {
        oss << pair.first << "=\"" << pair.second << "\" ";
    }

    return oss.str();
}

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

vcl_result_t saveVclAllocatorBlob(vcl_compiler_handle_t& compiler, vcl_executable_desc_t& exeDesc,
                                  const char* blobFileName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_allocator_t allocator;
    allocator.allocate = allocateBlob;
    allocator.deallocate = deallocateBlob;
    uint8_t* blob = nullptr;
    uint64_t size = 0;

    ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);

    if (ret != VCL_RESULT_SUCCESS || blob == nullptr || size == 0) {
        printErrorInfo("Failed to create executable handle! Result: 0x", ret);
        if (blob != nullptr) {
            allocator.deallocate(blob);
        }
        return ret;
    }

    std::ofstream outFile(blobFileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "Cannot open " << blobFileName << ", skip dump!" << std::endl;
        allocator.deallocate(blob);
        return VCL_RESULT_ERROR_IO;
    }

    outFile.write(reinterpret_cast<const char*>(blob), size);
    if (!outFile) {
        std::cerr << "Short write to " << blobFileName << ", the file is invalid!" << std::endl;
        ret = VCL_RESULT_ERROR_IO;
    } else {
        std::cout << "The output name: " << blobFileName << std::endl;
    }

    outFile.close();
    if (!outFile) {
        std::cerr << "Failed to close " << blobFileName << std::endl;
        ret = VCL_RESULT_ERROR_IO;
    }

    allocator.deallocate(blob);

    return ret;
}

struct vcl_allocator_vector_2 : vcl_allocator2_t {
    vcl_allocator_vector_2(): vcl_allocator2_t{vector_allocate, vector_deallocate} {
    }

    static uint8_t* vector_allocate(vcl_allocator2_t* allocator, uint64_t size) {
        vcl_allocator_vector_2* vecAllocator = static_cast<vcl_allocator_vector_2*>(allocator);
        auto newVec = std::make_shared<std::vector<uint8_t>>();
        newVec->resize(size);
        uint8_t* ptr = newVec->data();
        vecAllocator->m_vector.emplace_back(newVec);
        return ptr;
    }

    static void vector_deallocate(vcl_allocator2_t* allocator, uint8_t* ptr) {
        vcl_allocator_vector_2* vecAllocator = static_cast<vcl_allocator_vector_2*>(allocator);
        auto it = std::find_if(vecAllocator->m_vector.begin(), vecAllocator->m_vector.end(),
                               [ptr](const std::shared_ptr<std::vector<uint8_t>>& vec_ptr) {
                                   return vec_ptr->data() == ptr;
                               });

        if (it != vecAllocator->m_vector.end()) {
            vecAllocator->m_vector.erase(it);
        }
    }

    std::vector<std::shared_ptr<std::vector<uint8_t>>> m_vector;
};

vcl_result_t saveVclAllocatorBlobWS(vcl_compiler_handle_t& compiler, vcl_executable_desc_t& exeDesc,
                                    const char* blobFileName, const std::shared_ptr<ov::Model>& model) {
    std::cout << "  VCL step: Save VCL allocator blob." << std::endl;
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_allocator_vector_2 allocator;

    ret = vclAllocatedExecutableCreateWSOneShot(compiler, exeDesc, &allocator);

    if (ret != VCL_RESULT_SUCCESS) {
        printErrorInfo("Failed to create executable handle with WS! Result: 0x", ret);
        return ret;
    }

    if (allocator.m_vector.empty()) {
        printErrorInfo("Executable creation with WS returned success but no blobs were allocated! Result: 0x", ret);
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    std::ofstream outFile(blobFileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "Cannot open " << blobFileName << ", skip dump!" << std::endl;
        return VCL_RESULT_ERROR_IO;
    }

    size_t blobIndex = 0;
    // A prime number used as a seed for the hash calculation.
    constexpr std::uint32_t HASH_SEED = 1171117u;
    std::uint32_t totalResult = HASH_SEED;
    totalResult = ((totalResult << 7) + totalResult);
    uint64_t totalBlobSize = 0;

    auto writeToStream = [&](const uint8_t* blobRawPtr, uint64_t blobSize) -> uint64_t {
        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            std::cerr << "Blob size is too large to be represented on a std::streamsize!" << std::endl;
            return 0;
        }
        outFile.write(reinterpret_cast<const char*>(blobRawPtr), static_cast<std::streamsize>(blobSize));

        if (!outFile) {
            std::cerr << "Write blob to stream failed. Blob is broken!" << std::endl;
            return 0;
        }

        std::uint32_t result = HASH_SEED;
        for (const uint8_t* it = blobRawPtr; it != blobRawPtr + blobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }
        totalResult += result;

        std::stringstream str;
        if (blobIndex == 0) {
            str << "Main blob size " << blobSize << ", hash " << std::hex << result;
        } else {
            str << "Init part " << blobIndex << " blob size " << blobSize << ", hash " << std::hex << result;
        }
        std::cout << str.str() << std::endl;

        size_t alignedSize = vcl::utils::align_size_to_standard_page_size(blobSize);
        size_t paddingSize = alignedSize - blobSize;
        if (paddingSize > 0) {
            std::vector<char> padding(paddingSize, 0);
            outFile.write(padding.data(), paddingSize);
            if (!outFile) {
                std::cerr << "Write padding to " << blobFileName << " failed, the file is invalid!" << std::endl;
                return 0;
            }
        }
        return alignedSize;
    };

    // By convention, first write the main part
    const auto& mainBlobPtr = allocator.m_vector.back();
    uint64_t mainBlobSize = writeToStream(mainBlobPtr->data(), mainBlobPtr->size());
    if (mainBlobSize == 0) {
        outFile.close();
        return VCL_RESULT_ERROR_IO;
    }
    totalBlobSize += mainBlobSize;
    blobIndex++;

    // Then the init schedules
    std::optional<std::vector<uint64_t>> initBlobSizes;
    if (allocator.m_vector.size() > 1) {
        initBlobSizes = std::vector<uint64_t>();
        for (size_t i = 0; i < allocator.m_vector.size() - 1; ++i) {
            const auto& initBlobPtr = allocator.m_vector.at(i);
            uint64_t initBlobSize = writeToStream(initBlobPtr->data(), initBlobPtr->size());
            if (initBlobSize == 0) {
                outFile.close();
                return VCL_RESULT_ERROR_IO;
            }
            initBlobSizes->push_back(initBlobSize);
            totalBlobSize += initBlobSize;
            blobIndex++;
        }
    }

    // Append metadata to the blob file
    std::optional<std::vector<ov::Layout>> inputLayouts = std::vector<ov::Layout>();
    std::optional<std::vector<ov::Layout>> outputLayouts = std::vector<ov::Layout>();
    for (const auto& node : model->get_parameters()) {
        inputLayouts->push_back(node->get_layout());
    }
    for (const auto& node : model->get_results()) {
        outputLayouts->push_back(node->get_layout());
    }

    // The batch size is not available in this context, using a default value.
    const std::optional<int64_t> batchSize = std::nullopt;

    vcl::Metadata<vcl::CURRENT_METADATA_VERSION> metadata(totalBlobSize, vcl::CURRENT_OPENVINO_VERSION, initBlobSizes,
                                                          batchSize, inputLayouts, outputLayouts);
    metadata.write(outFile);

    std::cout << "The output name: " << blobFileName << std::endl;
    std::stringstream str;
    str << "Total blob size (with padding): " << totalBlobSize << ", hash: " << std::hex << totalResult;
    std::cout << str.str() << std::endl;

    outFile.close();

    return ret;
}

vcl_result_t simulateVclCompilerAllocator(std::map<std::string, std::string>& buildConfig,
                                          const std::shared_ptr<ov::Model>& model, const char* blobFileName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    std::cout << "  VCL step: Create compiler." << std::endl;
    struct CompilerHandle {
        vcl_compiler_handle_t handle;

        CompilerHandle() noexcept: handle(nullptr) {
        }

        ~CompilerHandle() {
            if (handle) {
                vclCompilerDestroy(handle);
            }
        }

        CompilerHandle(const CompilerHandle&) = default;
        CompilerHandle& operator=(const CompilerHandle&) = default;

        CompilerHandle(CompilerHandle&& other) noexcept: handle(other.handle) {
            other.handle = nullptr;
        }

        CompilerHandle& operator=(CompilerHandle&& other) noexcept {
            if (this != &other) {
                if (handle) {
                    vclCompilerDestroy(handle);
                }
                handle = other.handle;
                other.handle = nullptr;
            }
            return *this;
        }
    } compiler;

    vcl_log_handle_t logHandle = nullptr;
    vcl_compiler_properties_t compilerProp;
    ret = getVclCompiler(buildConfig, &compiler.handle, &compilerProp, &logHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }
    auto it = buildConfig.find("NPU_WEIGHTLESS_BLOB");
    if (it != buildConfig.end() && it->second == "YES") {
        storeWeightlessCacheAttribute(model);
    }

    std::cout << "  VCL step: Serialize IR." << std::endl;
    std::string logLevelStr =
            (buildConfig.find("LOG_LEVEL") != buildConfig.end()) ? buildConfig["LOG_LEVEL"] : "LOG_NONE";
    SerializedIR irSerializer;
    ret = serializeIR(model, compilerProp.version, compilerProp.supportedOpsets, compiler.handle, irSerializer,
                      logLevelStr);
    if (ret != VCL_RESULT_SUCCESS) {
        return ret;
    }

    /// Test query network, create query handle first
    std::cout << "  VCL step: Test Query Network." << std::endl;
    struct QueryHandle {
        vcl_query_handle_t handle;
        QueryHandle() noexcept: handle(nullptr) {
        }
        ~QueryHandle() {
            if (handle) {
                vclQueryNetworkDestroy(handle);
            }
        }

        QueryHandle(const QueryHandle&) = default;
        QueryHandle& operator=(const QueryHandle&) = default;

        QueryHandle(QueryHandle&& other) noexcept: handle(other.handle) {
            other.handle = nullptr;
        }

        QueryHandle& operator=(QueryHandle&& other) noexcept {
            if (this != &other) {
                if (handle) {
                    vclQueryNetworkDestroy(handle);
                }
                handle = other.handle;
                other.handle = nullptr;
            }
            return *this;
        }
    } query;

    std::string configStr = configMapToString(buildConfig);
    vcl_query_desc_t desc = {irSerializer.second.get(), irSerializer.first, configStr.c_str(), configStr.size()};

    ret = vclQueryNetworkCreate(compiler.handle, desc, &query.handle);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to query network! Result: 0x", ret);
        return ret;
    }

    std::vector<uint8_t> layerRawData;
    uint64_t layerSize = 0;
    /// First time calling vclQueryNetwork, layerRawData is nullptr, get layerSize
    ret = vclQueryNetwork(query.handle, nullptr, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to get size of query result! Result: 0x", ret);
        return ret;
    }

    /// layerRawData should be allocated with layerSize
    try {
        layerRawData.resize(layerSize);
    } catch (const std::bad_alloc&) {
        getLastError(logHandle);
        std::cerr << "Failed to allocate memory to store layer info!" << std::endl;
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }

    /// Second time calling vclQueryNetwork, copy queryResultString to layerRawData
    ret = vclQueryNetwork(query.handle, layerRawData.data(), &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to get data of query result!", ret);
        return ret;
    }

    /// Print the whole layerRawData
    std::cout << "  Print layerRawData as the result string of query: " << std::endl;
    std::cout.write(reinterpret_cast<const char*>(layerRawData.data()), layerSize);
    std::cout << std::endl;

    /// Destroy query network handle
    ret = vclQueryNetworkDestroy(query.handle);
    query.handle = nullptr;
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to destroy query handle!", ret);
        return ret;
    }

    std::cout << "  VCL step: Compile Network." << std::endl;
    std::string buildFlags;
    /// useIndicesForIOMetadata is set `true` as default.
    const bool useIndices = true;
    buildFlags += serializeIOInfo(model, useIndices);
    buildFlags += " ";
    buildFlags += configStr;
    vcl_executable_desc_t exeDesc = {irSerializer.second.get(), irSerializer.first, buildFlags.c_str(),
                                     buildFlags.size()};

    /// Get compiled blob
    if (it != buildConfig.end() && it->second == "YES") {
        std::cout << "  VCL step: Compile Network with WS." << std::endl;
        ret = saveVclAllocatorBlobWS(compiler.handle, exeDesc, blobFileName, model);
    } else {
        std::cout << "  VCL step: Compile Network without WS." << std::endl;
        ret = saveVclAllocatorBlob(compiler.handle, exeDesc, blobFileName);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to save Blob! Result: 0x", ret);
        return ret;
    }

    ret = vclCompilerDestroy(compiler.handle);
    compiler.handle = nullptr;
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        printErrorInfo("Failed to destroy compiler! Result: 0x", ret);
        return VCL_RESULT_ERROR_IO;
    }

    return ret;
}

vcl_result_t simulateVclCompilerAllocatorOldVersion(int argc, char** argv, const char* blobFileName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_version_info_t compilerVersion;
    vcl_version_info_t profilingVersion;
    ret = vclGetVersion(&compilerVersion, &profilingVersion);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to get version! Result:%x\n", ret);
        return ret;
    } else {
        std::printf("\n############################################\n\n");
        std::printf("Library VCL API version info:\n");
        std::printf("Compiler version:%d.%d\n", compilerVersion.major, compilerVersion.minor);
        std::printf("Profiling version:%d.%d\n", profilingVersion.major, profilingVersion.minor);
        std::printf("\n############################################\n\n");
    }

    /// Control if we save error log or output to terminal
    char* saveErrorLog = getenv("VCL_SAVE_ERROR");
    vcl_compiler_desc_t compilerDesc;
    compilerDesc.version.major = VCL_COMPILER_VERSION_MAJOR;
    compilerDesc.version.minor = VCL_COMPILER_VERSION_MINOR;
    compilerDesc.debugLevel = VCL_LOG_INFO;

    vcl_compiler_handle_t compiler = nullptr;
    vcl_log_handle_t logHandle = nullptr;
    vcl_device_desc_t deviceDesc;
    bool hasDeviceDesc = configureDeviceDesc(deviceDesc, 0x643e);
    vcl_device_desc_t* deviceDescPtr = hasDeviceDesc ? &deviceDesc : nullptr;
    if (saveErrorLog == nullptr) {
        ret = vclCompilerCreate(&compilerDesc, deviceDescPtr, &compiler, NULL);
    } else {
        ret = vclCompilerCreate(&compilerDesc, deviceDescPtr, &compiler, &logHandle);
    }

    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to create compiler! Result:%x\n", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to query compiler props! Result: %x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        std::printf("\n############################################\n\n");
        std::printf("Current compiler info:\n");
        std::printf("ID: %s\n", compilerProp.id);
        std::printf("Version:%d.%d\n", compilerProp.version.major, compilerProp.version.minor);
        std::printf("\tSupported opsets:%d\n", compilerProp.supportedOpsets);
        std::printf("\n############################################\n\n");
    }

    /// Read buffer, add net.xml
    size_t bytesRead = 0;
    char* netName = argv[1];
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        std::printf("Cannot open file %s\n", netName);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    long fileXmlSize = ftell(fpN);
    if (fileXmlSize < 0) {
        std::printf("Ftell method returns failure.\n");
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t xmlSize = (uint64_t)fileXmlSize;
    fseek(fpN, 0, SEEK_SET);

    /// Read weights size, add weight.bin
    char* weightName = argv[2];
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        std::printf("Cannot open file %s\n", weightName);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    long fileWeightsSize = ftell(fpW);
    if (fileWeightsSize < 0) {
        std::printf("Ftell method returns failure.\n");
        fclose(fpN);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    uint64_t weightsSize = (uint64_t)fileWeightsSize;

    /// Init modelIR
    vcl_version_info_t version = compilerProp.version;
    uint32_t numberOfInputData = 2;
    uint64_t modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    uint8_t* modelIR = (uint8_t*)malloc(modelIRSize);
    if (!modelIR) {
        std::printf("Failed to alloc memory for IR!\n");
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    uint64_t offset = 0;
    memcpy(modelIR, &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR + offset;
    bytesRead = fread(xmlData, 1, xmlSize, fpN);
    if ((uint64_t)bytesRead != xmlSize) {
        std::printf("Short read on network buffer!!!\n");
        free(modelIR);
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    int cret = fclose(fpN);
    if (cret) {
        std::printf("Failed to close %s. Result:%d\n", netName, cret);
        free(modelIR);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return (vcl_result_t)cret;
    }

    offset += xmlSize;
    memcpy(modelIR + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR + offset;
        fseek(fpW, 0, SEEK_SET);
        bytesRead = fread(weights, 1, weightsSize, fpW);
        if ((uint64_t)bytesRead != weightsSize) {
            std::printf("Short read on weights file!!!\n");
            free(modelIR);
            fclose(fpW);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
    }
    cret = fclose(fpW);
    if (cret) {
        std::printf("Failed to close %s. Result:%d\n", weightName, cret);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return (vcl_result_t)cret;
    }

    /// The options are for googlenet-v1
    char defaultOptions[] = "";
    char* newOptions = NULL;
    uint64_t configSize = 0;
    if (argc == 5) {
        char* configFile = argv[4];
        FILE* fpC = fopen(configFile, "rb");
        if (!fpC) {
            std::printf("Cannot open file %s\n", configFile);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        fseek(fpC, 0, SEEK_END);
        long fileConfigSize = ftell(fpC);
        if (fileConfigSize < 0) {
            std::printf("Ftell method returns failure.\n");
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        configSize = (uint64_t)fileConfigSize;
        fseek(fpC, 0, SEEK_SET);
        newOptions = (char*)malloc(configSize);
        if (!newOptions) {
            std::printf("Failed to alloc memory for options\n");
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        bytesRead = fread(newOptions, 1, configSize, fpC);
        if ((uint32_t)bytesRead != configSize) {
            std::printf("Short read on config file buffer!!!\n");
            free(newOptions);
            fclose(fpC);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        cret = fclose(fpC);
        if (cret) {
            std::printf("Failed to close %s. Result:%d\n", configFile, cret);
            free(newOptions);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return (vcl_result_t)cret;
        }
    }

    /// Test query network, create query handle first
    vcl_query_handle_t query = NULL;
    if (newOptions) {
        vcl_query_desc_t desc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    } else {
        vcl_query_desc_t desc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclQueryNetworkCreate(compiler, desc, &query);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        getLastError(logHandle);
        std::printf("Failed to query network! Result:%x\n", ret);
        free(newOptions);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint8_t* layerRawData = NULL;
    uint64_t layerSize = 0;
    /// First time calling vclQueryNetwork, layerRawData is nullptr, get layerSize
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to get size of query result! Result:%x\n", ret);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// layerRawData should be allocated with layerSize
    layerRawData = (uint8_t*)malloc(layerSize);
    if (layerRawData == NULL) {
        std::printf("Failed to malloc memory to store layer info!\n");
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    /// Second time calling vclQueryNetwork, copy queryResultString to layerRawData
    ret = vclQueryNetwork(query, layerRawData, &layerSize);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to get data of query result! Result:%x\n", ret);
        free(layerRawData);
        vclQueryNetworkDestroy(query);
        vclCompilerDestroy(compiler);
        return ret;
    }
    /// Print the whole layerRawData
    std::printf("Print layerRawData as the result string of query: \n%.*s", (int)layerSize, layerRawData);
    std::printf("\n");
    /// Destroy query network handle
    ret = vclQueryNetworkDestroy(query);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to destroy query handle! Result:%x\n", ret);
        free(layerRawData);
        vclCompilerDestroy(compiler);
        return ret;
    }
    free(layerRawData);
    query = NULL;

    vcl_allocator_t allocator;
    allocator.allocate = allocateBlob;
    allocator.deallocate = deallocateBlob;

    uint8_t* blob = NULL;
    uint64_t size = 0;
    if (argc != 5) {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, defaultOptions, sizeof(defaultOptions)};
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
    } else {
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, newOptions, configSize};
        ret = vclAllocatedExecutableCreate(compiler, exeDesc, &allocator, &blob, &size);
        free(newOptions);
    }
    free(modelIR);
    if (ret != VCL_RESULT_SUCCESS || blob == NULL || size == 0) {
        getLastError(logHandle);
        std::printf("Failed to create executable handle! Result:%x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }

    FILE* fpB = fopen(blobFileName, "wb");
    if (!fpB) {
        std::printf("Can not open %s, skip dump!\n", blobFileName);
    } else {
        uint64_t bytesWrite = fwrite(blob, 1, size, fpB);
        if (bytesWrite != size) {
            std::printf("Short write to %s, the file is invalid!\n", blobFileName);
        }
        int cret = fclose(fpB);
        if (cret) {
            std::printf("Failed to close %s. Result:%d\n", blobFileName, cret);
        } else {
            std::printf("The output name:%s\n", blobFileName);
        }
    }
    allocator.deallocate(blob);

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        std::printf("Failed to destroy compiler! Result:%x\n", ret);
        return ret;
    }
    return ret;
}
