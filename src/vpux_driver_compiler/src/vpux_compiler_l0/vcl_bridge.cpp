//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_bridge.cpp
 * @brief The bridge from L0 driver compiler to user API
 */

#include "npu_driver_compiler.h"

#include "vcl_compiler.hpp"
#include "vcl_executable.hpp"
#include "vcl_profiling.hpp"
#include "vcl_query_network.hpp"
#include "vpux/utils/ov/config.hpp"
#include "vpux/utils/ov/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>

using namespace vpux;

namespace {

template <typename Allocator>
class VCLBlobAllocator : public BlobAllocator {
public:
    explicit VCLBlobAllocator(Allocator* allocator): allocator_(allocator) {
        static_assert(std::is_same_v<Allocator, const vcl_allocator_t> || std::is_same_v<Allocator, vcl_allocator2_t>,
                      "VCLBlobAllocator only supports vcl_allocator_t and vcl_allocator2_t");
    }

    uint8_t* allocate(Byte size) override {
        if constexpr (std::is_same_v<Allocator, const vcl_allocator_t>) {
            return allocator_->allocate(static_cast<uint64_t>(size.count()));
        } else {
            return allocator_->allocate(allocator_, static_cast<uint64_t>(size.count()));
        }
    }

    void deallocate(uint8_t* ptr) override {
        if constexpr (std::is_same_v<Allocator, const vcl_allocator_t>) {
            allocator_->deallocate(ptr);
        } else {
            allocator_->deallocate(allocator_, ptr);
        }
    }

private:
    Allocator* allocator_;
};

// outside of the extern "C" block to allow C++ templates
template <typename Allocator>
vcl_result_t allocatedExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc, Allocator* allocator,
                                       uint8_t** blob, uint64_t* size, vcl_executable_handle_t* executable = nullptr) {
    if (executable != nullptr) {
        *executable = nullptr;
    }
    if (!compiler || !allocator || !blob || !size || !desc.modelIRData) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    *blob = nullptr;
    *size = 0;

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    auto& vclLogger = pCompiler->getLogger();

    try {
        // NetworkMetadata is part of the result, but unused in VCL
        // it'd just get destroyed at function call here
        VCLBlobAllocator vclAllocator{allocator};
        auto result = pCompiler->importNetwork(desc, vclAllocator);
        *blob = result.compiledNetwork.ptr;
        *size = result.compiledNetwork.size;
        if (executable != nullptr) {
            *executable = reinterpret_cast<vcl_executable_handle_t>(new (
                    std::nothrow) VPUXDriverCompiler::VPUXExecutableL0(result.metadata.compatibilityString, vclLogger));
        }
    } catch (const InvalidIrError& error) {
        vclLogger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_IR;
    } catch (const std::exception& error) {
        vclLogger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception! Can't compile model!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t allocatedExecutableCreateWSOneShot(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                vcl_allocator2_t* allocator,
                                                vcl_executable_handle_t* executable = nullptr) {
    if (executable != nullptr) {
        *executable = nullptr;
    }
    if (!compiler || !allocator || !desc.modelIRData) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    auto& vclLogger = pCompiler->getLogger();

    try {
        // NetworkMetadata is part of the result, but unused in VCL
        // it'd just get destroyed at function call here
        VCLBlobAllocator vclAllocator{allocator};
        auto result = pCompiler->importNetworkWSOneShot(desc, vclAllocator);
        if (result.empty()) {
            vclLogger->warning("Compiler successfully returned but the blob list is empty!");
        } else if (executable != nullptr) {
            // WS one-shot results contain Init schedules first and the Main schedule last.
            // The compatibility string is expected to be identical for all schedules.
            *executable =
                    reinterpret_cast<vcl_executable_handle_t>(new (std::nothrow) VPUXDriverCompiler::VPUXExecutableL0(
                            result.back()->metadata.compatibilityString, vclLogger));
            if (*executable == nullptr) {
                return VCL_RESULT_ERROR_OUT_OF_MEMORY;
            }
        }
    } catch (const InvalidIrError& error) {
        vclLogger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_IR;
    } catch (const std::exception& error) {
        vclLogger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception! Can't compile model!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return VCL_RESULT_SUCCESS;
}

std::optional<LogLevel> getLogLevel(vcl_log_level_t level) {
    // Note OV does not have LOG_FATAL
    switch (level) {
    case VCL_LOG_NONE:
        return LogLevel::None;
    case VCL_LOG_ERROR:
        return LogLevel::Error;
    case VCL_LOG_WARNING:
        return LogLevel::Warning;
    case VCL_LOG_INFO:
        return LogLevel::Info;
    case VCL_LOG_DEBUG:
        return LogLevel::Debug;
    case VCL_LOG_TRACE:
        return LogLevel::Trace;
    default:
        return std::nullopt;
    }
}

std::optional<LogLevel> getLogLevelFromEnv(VPUXDriverCompiler::VCLLogger& log) {
    const auto envVar = vpux::OV::LOG_LEVEL::envVar();
    assert(!envVar.empty());
    if (const auto logLevelEnv = std::getenv(envVar.data())) {
        try {
            ov::log::Level level;
            VPUX_THROW_UNLESS(std::stringstream{logLevelEnv} >> level, "Invalid value '{0}' for ov::log::level",
                              logLevelEnv);
            return vpux::getLogLevel(level);
        } catch (const std::runtime_error&) {
            log.warning("Failed to parse {0}", envVar);
        }
    }
    return std::nullopt;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

DLLEXPORT vcl_result_t vclGetVersion(vcl_version_info_t* compilerVersion, vcl_version_info_t* profilingVersion) {
    if (!compilerVersion || !profilingVersion) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    compilerVersion->major = VCL_COMPILER_VERSION_MAJOR;
    compilerVersion->minor = VCL_COMPILER_VERSION_MINOR;
    profilingVersion->major = VCL_PROFILING_VERSION_MAJOR;
    profilingVersion->minor = VCL_PROFILING_VERSION_MINOR;

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerCreate(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc,
                                         vcl_compiler_handle_t* compiler, vcl_log_handle_t* logHandle) {
    /// Saves latest error messages, output other messages to terminal
    const auto saveErrorLog = (logHandle != nullptr);
    auto vclLogger = std::shared_ptr<VPUXDriverCompiler::VCLLogger>(
            new VPUXDriverCompiler::VCLLogger("NPU_VCL", LogLevel::Error, saveErrorLog));

    if (compilerDesc == nullptr || compiler == nullptr) {
        vclLogger->outputError("Null argument to create compiler!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (const auto logLevel = getLogLevel(compilerDesc->debugLevel)) {
        vclLogger->setLevel(logLevel.value());
    } else {
        vclLogger->outputError("Invalid debug level in compiler descriptor!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // OV_NPU_LOG_LEVEL takes precedence if present
    if (const auto envLogLevel = getLogLevelFromEnv(*vclLogger)) {
        vclLogger->setLevel(envLogLevel.value());
    }

    /// Create compiler
    try {
        auto pCompiler = std::make_unique<VPUXDriverCompiler::VPUXCompilerL0>(compilerDesc, deviceDesc, vclLogger);
        vclLogger->info("Current compiler ID: {0}", pCompiler->getCompilerProp().id);
        *compiler = reinterpret_cast<vcl_compiler_handle_t>(pCompiler.release());
        /// Return logger to save error msg, pass the handle here
        if (logHandle != nullptr) {
            *logHandle = reinterpret_cast<vcl_log_handle_t>(vclLogger.get());
        }
    } catch (const std::exception& error) {
        vclLogger->outputError(formatv("Failed to create compiler:\n{0}", error.what()));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception during compiler creation!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerGetProperties(vcl_compiler_handle_t compiler, vcl_compiler_properties_t* properties) {
    if (!properties || !compiler) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    *properties = pCompiler->getCompilerProp();
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclQueryNetworkCreate(vcl_compiler_handle_t compiler, vcl_query_desc_t desc,
                                                         vcl_query_handle_t* query) {
    /// Format of modelIRData is defined in L0 adaptor
    /// The modelIRData is parsed into model data and weights info
    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    auto& vclLogger = pCompiler->getLogger();

    if (!desc.modelIRData) {
        vclLogger->outputError("Invalid IR buffer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (!desc.modelIRSize) {
        vclLogger->outputError("Invalid IR size!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /// Query which layers of the model are supported by current compiler
    VPUXDriverCompiler::VPUXQueryNetworkL0* pQueryNetwork = nullptr;
    pQueryNetwork = new VPUXDriverCompiler::VPUXQueryNetworkL0(vclLogger);
    if (auto ret = pCompiler->queryNetwork(desc, pQueryNetwork); ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to query network!");
        delete pQueryNetwork;
        return ret;
    }

    *query = reinterpret_cast<vcl_query_handle_t>(pQueryNetwork);
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclQueryNetwork(vcl_query_handle_t query, uint8_t* queryResult, uint64_t* size) {
    if (query == nullptr || size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXQueryNetworkL0* pvq = reinterpret_cast<VPUXDriverCompiler::VPUXQueryNetworkL0*>(query);
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    if (queryResult == nullptr) {
        /// First time calling vclQueryNetwork, get size of queryResultString
        ret = pvq->getQueryResultSize(size);
    } else {
        /// Second time calling vclQueryNetwork, get data of queryResultString
        ret = pvq->getQueryString(queryResult, *size);
    }
    return ret;
}

DLLEXPORT vcl_result_t vclQueryNetworkDestroy(vcl_query_handle_t query) {
    if (query != nullptr) {
        VPUXDriverCompiler::VPUXQueryNetworkL0* pvq = reinterpret_cast<VPUXDriverCompiler::VPUXQueryNetworkL0*>(query);
        delete pvq;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                           vcl_executable_handle_t* executable) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    if (!compiler || !executable || !desc.modelIRData) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    auto& vclLogger = pCompiler->getLogger();

    /// Use compiler to compile model and store the result blob
    std::pair<VPUXDriverCompiler::VPUXExecutableL0*, vcl_result_t> status;
    try {
        status = pCompiler->importNetwork(desc);
    } catch (const InvalidIrError& error) {
        vclLogger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_IR;
    } catch (const std::exception& error) {
        vclLogger->outputError(error.what());
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        vclLogger->outputError("Internal exception! Can't compile model!");
        ret = VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (status.second != VCL_RESULT_SUCCESS || ret != VCL_RESULT_SUCCESS) {
        /// Release memory if we failed to compile model
        if (status.first != nullptr) {
            delete status.first;
        }
        *executable = nullptr;
        vclLogger->outputError("Failed to create executable");
        return status.second;
    }
    /// Get blob from compiled result and store in executable
    VPUXDriverCompiler::VPUXExecutableL0* pExecutable = status.first;
    if (pExecutable != nullptr) {
        ret = pExecutable->serializeNetwork();
        if (ret != VCL_RESULT_SUCCESS) {
            delete pExecutable;
            *executable = nullptr;
            vclLogger->outputError("Failed to get compiled network");
            return ret;
        }
        /// Return the executable which holds the blob
        *executable = reinterpret_cast<vcl_executable_handle_t>(pExecutable);
    } else {
        vclLogger->outputError("Failed to get blob from compiled result");
        ret = VCL_RESULT_ERROR_UNKNOWN;
    }
    return ret;
}

DLLEXPORT vcl_result_t vclAllocatedExecutableCreate4(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                     vcl_allocator2_t* allocator, uint8_t** blobBuffer,
                                                     uint64_t* blobSize, vcl_executable_handle_t* executable) {
    return allocatedExecutableCreate(compiler, desc, allocator, blobBuffer, blobSize, executable);
}

DLLEXPORT vcl_result_t vclAllocatedExecutableCreate2(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                     vcl_allocator2_t* allocator, uint8_t** blob, uint64_t* size) {
    return allocatedExecutableCreate(compiler, desc, allocator, blob, size);
}

DLLEXPORT vcl_result_t vclAllocatedExecutableCreate(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                    vcl_allocator_t const* allocator, uint8_t** blob, uint64_t* size) {
    return allocatedExecutableCreate(compiler, desc, allocator, blob, size);
}

DLLEXPORT vcl_result_t vclAllocatedExecutableCreateWSOneShot(vcl_compiler_handle_t compiler, vcl_executable_desc_t desc,
                                                             vcl_allocator2_t* allocator) {
    return allocatedExecutableCreateWSOneShot(compiler, desc, allocator);
}

DLLEXPORT vcl_result_t VCL_APICALL vclAllocatedExecutableCreateWSOneShot2(vcl_compiler_handle_t compiler,
                                                                          vcl_executable_desc_t desc,
                                                                          vcl_allocator2_t* allocator,
                                                                          vcl_executable_handle_t* executable) {
    if (executable == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return allocatedExecutableCreateWSOneShot(compiler, desc, allocator, executable);
}

DLLEXPORT vcl_result_t vclExecutableGetSerializableBlob(vcl_executable_handle_t executable, uint8_t* blobBuffer,
                                                        uint64_t* blobSize) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    if (!blobSize || !executable) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXExecutableL0* pExecutable =
            reinterpret_cast<VPUXDriverCompiler::VPUXExecutableL0*>(executable);
    auto& vclLogger = pExecutable->getLogger();

    if (!blobBuffer) {
        /// When we call this function the first time, shall pass empty pointer to blob buffer and return the size of
        /// blob. User will use the size to alloc memory to store the blob
        ret = pExecutable->getNetworkSize(blobSize);
    } else {
        /// When we call this function the second time, the value of blobSize shall be the result of first call.
        /// Store the real blob data to the passed buffer
        ret = pExecutable->exportNetwork(blobBuffer, *blobSize);
    }
    if (ret != VCL_RESULT_SUCCESS) {
        vclLogger->outputError("Failed to get blob");
    }
    return ret;
}

DLLEXPORT vcl_result_t VCL_APICALL vclExecutableGetCompatibilityString(vcl_executable_handle_t executable,
                                                                       char* compatibilityString,
                                                                       uint64_t* compatibilityStringSize) {
    if (!executable || !compatibilityStringSize) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXExecutableL0* pExecutable =
            reinterpret_cast<VPUXDriverCompiler::VPUXExecutableL0*>(executable);

    const auto& compatString = pExecutable->getCompatibilityString();
    if (compatString.empty()) {
        *compatibilityStringSize = 0;
        return VCL_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    auto size = compatString.size() + 1;  // +1 for null terminator
    auto bufferSize = *compatibilityStringSize;
    *compatibilityStringSize = size;
    if (compatibilityString != nullptr) {
        if (bufferSize < size) {
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        const auto* cstr = compatString.c_str();
        std::copy(cstr, cstr + size, compatibilityString);
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclExecutableDestroy(vcl_executable_handle_t executable) {
    if (executable) {
        VPUXDriverCompiler::VPUXExecutableL0* pExecutable =
                reinterpret_cast<VPUXDriverCompiler::VPUXExecutableL0*>(executable);
        delete pExecutable;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclCompilerDestroy(vcl_compiler_handle_t compiler) {
    if (compiler) {
        VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
        delete pCompiler;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingCreate(p_vcl_profiling_input_t profilingInput,
                                                      vcl_profiling_handle_t* profilingHandle,
                                                      vcl_log_handle_t* logHandle) {
    if (!profilingInput || !profilingHandle) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VCLLogger* vclLogger = nullptr;
    if (logHandle != nullptr) {
        /// Create logger which saves latest error messages, output other messages to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("NPU_VCL", LogLevel::Error, true);
    } else {
        /// Create logger which output all message to terminal
        vclLogger = new VPUXDriverCompiler::VCLLogger("NPU_VCL", LogLevel::Error, false);
    }

    VPUXDriverCompiler::VPUXProfilingL0* profHandle =
            new (std::nothrow) VPUXDriverCompiler::VPUXProfilingL0(profilingInput, vclLogger);
    if (!profHandle) {
        vclLogger->outputError("Failed to create profiler");
        delete profHandle;
        delete vclLogger;
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }

    *profilingHandle = reinterpret_cast<vcl_profiling_handle_t>(profHandle);

    /// Create logger to save error msg, pass the handle here
    if (logHandle != nullptr) {
        *logHandle = reinterpret_cast<vcl_log_handle_t>(vclLogger);
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclGetDecodedProfilingBuffer(vcl_profiling_handle_t profilingHandle,
                                                                vcl_profiling_request_type_t requestType,
                                                                p_vcl_profiling_output_t profilingOutput) {
    if (!profilingHandle || !profilingOutput) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXProfilingL0* prof = reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
    VPUXDriverCompiler::VCLLogger* vclLogger = prof->getLogger();
    switch (requestType) {
    case VCL_PROFILING_LAYER_LEVEL:
        return prof->getLayerInfo(profilingOutput);
    case VCL_PROFILING_TASK_LEVEL:
        return prof->getTaskInfo(profilingOutput);
    case VCL_PROFILING_RAW:
        return prof->getRawInfo(profilingOutput);
    default:
        vclLogger->outputError("Request type is not supported.");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingDestroy(vcl_profiling_handle_t profilingHandle) {
    if (profilingHandle) {
        VPUXDriverCompiler::VPUXProfilingL0* pvp =
                reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
        VPUXDriverCompiler::VCLLogger* vclLogger = pvp->getLogger();
        delete vclLogger;
        delete pvp;
    }
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t VCL_APICALL vclProfilingGetProperties(vcl_profiling_handle_t profilingHandle,
                                                             vcl_profiling_properties_t* properties) {
    if (!profilingHandle || !properties) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VPUXProfilingL0* pvp = reinterpret_cast<VPUXDriverCompiler::VPUXProfilingL0*>(profilingHandle);
    *properties = pvp->getProperties();
    return VCL_RESULT_SUCCESS;
}

DLLEXPORT vcl_result_t vclLogHandleGetString(vcl_log_handle_t logHandle, size_t* logSize, char* log) {
    if (!logHandle) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    VPUXDriverCompiler::VCLLogger* vclLogger = reinterpret_cast<VPUXDriverCompiler::VCLLogger*>(logHandle);
    return vclLogger->getString(logSize, log);
}

DLLEXPORT vcl_result_t VCL_APICALL vclGetCompilerSupportedOptions(vcl_compiler_handle_t compiler, char* result,
                                                                  uint64_t* size) {
    if (compiler == nullptr || size == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    vcl_result_t ret = VCL_RESULT_ERROR_UNKNOWN;

    if (result == nullptr) {
        /// First time calling vclGetCompilerSupportedOptions, get size of queryResultString
        ret = pCompiler->getSupportedOptionsSize(size);
    } else {
        /// Second time calling vclGetCompilerSupportedOptions, get data of queryResultString
        ret = pCompiler->getSupportedOptions(result, *size);
    }
    return ret;
}

DLLEXPORT vcl_result_t VCL_APICALL vclGetCompilerIsOptionSupported(vcl_compiler_handle_t compiler, const char* option,
                                                                   const char* value) {
    if (compiler == nullptr || option == nullptr) {
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    VPUXDriverCompiler::VPUXCompilerL0* pCompiler = reinterpret_cast<VPUXDriverCompiler::VPUXCompilerL0*>(compiler);
    auto& vclLogger = pCompiler->getLogger();

    try {
        if (pCompiler->isOptionValueSupported(option, value)) {
            return VCL_RESULT_SUCCESS;
        }
    } catch (const std::exception& error) {
        vclLogger->outputError(formatv("Failed to check option support:\n{0}", error.what()));
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    return VCL_RESULT_ERROR_UNSUPPORTED_FEATURE;
};
}
