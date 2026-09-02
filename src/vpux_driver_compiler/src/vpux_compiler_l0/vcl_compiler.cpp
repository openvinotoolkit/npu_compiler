//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_compiler.hpp"
#include "npu_driver_compiler.h"
#include "vcl_executable.hpp"
#include "vcl_query_network.hpp"

#include <openvino/openvino.hpp>
#include <openvino/runtime/iplugin.hpp>

#include "vpux/compiler/icompiler.hpp"
#include "vpux/compiler/version.hpp"
#include "vpux/utils/ov/compat_string_check.hpp"
#include "vpux/utils/ov/options.hpp"

#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_set>

#define xstr(s) str(s)
#define str(s) #s

using namespace vpux;

namespace {

/// Compiler version contains the info of code commit, compiler API version
constexpr const char* COMPILER_VERSION =
        xstr(DRIVER_COMPILER_ID) "." xstr(VCL_COMPILER_VERSION_MAJOR) "." xstr(VCL_COMPILER_VERSION_MINOR);

enum class Platform : uint64_t {
    NPU3720 = 3720,
    NPU4000 = 4000,
    NPU5010 = 5010,
    NPU5020 = 5020,
};

Platform getPlatform(const vpux::OV::Config& config) {
    const std::string platform = vpux::OV::Platform::standardize(config.get<vpux::OV::PLATFORM>());
    if (platform == vpux::OV::Platform::NPU3720) {
        return Platform::NPU3720;
    } else if (platform == vpux::OV::Platform::NPU4000) {
        return Platform::NPU4000;
    } else if (platform == vpux::OV::Platform::NPU5010) {
        return Platform::NPU5010;
    } else if (platform == vpux::OV::Platform::NPU5020) {
        return Platform::NPU5020;
    } else {
        VPUX_THROW("Unsupported NPU platform");
    }
}

[[nodiscard]] bool setPlatformAndDeviceId(uint32_t deviceID, vpux::OV::Config& config) {
    switch (deviceID) {
    case 0x7D1D:  // MeteorLake (MTL-P, MTL-H)
    case 0xAD1D:  // ArrowLake (ARL)
        config.update({{vpux::OV::PLATFORM::key().data(), "3720"}});
        config.update({{vpux::OV::DEVICE_ID::key().data(), "3720"}});
        break;
    case 0x643E:  // LunarLake (LNL)
        config.update({{vpux::OV::PLATFORM::key().data(), "4000"}});
        config.update({{vpux::OV::DEVICE_ID::key().data(), "4000"}});
        break;
    case 0xB03E:  // PantherLake Mobile (PTL-P)
        config.update({{vpux::OV::PLATFORM::key().data(), "5010"}});
        config.update({{vpux::OV::DEVICE_ID::key().data(), "5010"}});
        break;
    case 0xFD3E:  // Wildcatlake (WCL)
        config.update({{vpux::OV::PLATFORM::key().data(), "5020"}});
        config.update({{vpux::OV::DEVICE_ID::key().data(), "5020"}});
        break;
    default:
        return false;
    }

    return true;
}

void setPlatformAndDeviceIdT(uint32_t deviceID, vpux::OV::Config& config) {
    if (!setPlatformAndDeviceId(deviceID, config)) {
        VPUX_THROW("Unsupported device ID: {0}", deviceID);
    }
}

/**
 * @brief Checks if a compiled model's runtime requirements are compatible with the current environment
 *        represented by a serialized string
 * @param compatibilityString the string capturing the runtime requirements
 * @param config a reference to NPUConfig containing plugin config options
 *        including config options related to compilation
 * @return bool representing compatibility status: true if compatibility check passed and false if not
 */
bool checkRuntimeRequirements(std::string_view compatibilityString, const vpux::OV::Config& config) {
    Logger log("vpux-compiler", config.get<vpux::OV::LOG_LEVEL>());

    if (compatibilityString.empty()) {
        log.warning("Empty compatibility string provided for runtime requirements check");
        return false;
    }

    log.info("Checking runtime requirements for string: '{0}'", compatibilityString);

    const auto platformID = static_cast<uint64_t>(getPlatform(config));
    const auto numOfTiles = [&] {
        VPUX_THROW_WHEN(!config.has<vpux::OV::MAX_TILES>(),
                        "MAX_TILES config is required for compatibility string validation");
        return config.get<vpux::OV::MAX_TILES>();
    }();

    auto reqs = compat::parseCompatibilityString(compatibilityString);

    return reqs.platformId == platformID && reqs.numTiles <= numOfTiles;
}

// Cannot use std::async as MSVC recycles the underlying thread
template <typename F>
auto runInWorkerThreadSync(F&& f) -> typename std::invoke_result<F>::type {
    std::packaged_task<typename std::invoke_result<F>::type()> task(std::forward<F>(f));
    auto result = task.get_future();
    std::thread(std::move(task)).join();
    return result.get();
}

// This option is already defined in newer openvino version
// Defining it here to avoid an openvino update in compiler
struct RUNTIME_REQUIREMENTS_MET final : vpux::OV::OptionBase<RUNTIME_REQUIREMENTS_MET, bool> {
    static std::string_view key() {
        return "RUNTIME_REQUIREMENTS_MET";
    }

    static bool defaultValue() {
        return false;
    }

    static vpux::OV::OptionMode mode() {
        return vpux::OV::OptionMode::CompileTime;
    }
};

}  // namespace

namespace VPUXDriverCompiler {

VPUXCompilerL0::VPUXCompilerL0(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc,
                               std::shared_ptr<VCLLogger> vclLogger)
        : _options(std::make_shared<vpux::OV::OptionsDesc>()),
          _compilerLoader(std::make_unique<CompilerLoader>()),
          _compilerDesc(*compilerDesc),
          _isDeviceDescEmpty(false),
          _logger(std::move(vclLogger)) {
    // Treat a zero device ID as an empty descriptor.
    if (deviceDesc && deviceDesc->deviceID > 0) {
        _deviceDesc = *deviceDesc;
    } else {
        _deviceDesc = {};
        _isDeviceDescEmpty = true;
        _logger->debug("DeviceDesc is empty or has no device ID. Using user config value for offline compilation.");
    }

    // Register compiler configuration options
    _options->add<vpux::OV::PERFORMANCE_HINT>();
    _options->add<vpux::OV::PERFORMANCE_HINT_NUM_REQUESTS>();
    _options->add<vpux::OV::INFERENCE_PRECISION_HINT>();
    _options->add<vpux::OV::PERF_COUNT>();
    _options->add<vpux::OV::LOG_LEVEL>();
    _options->add<vpux::OV::PLATFORM>();
    _options->add<vpux::OV::COMPILER_TYPE>();
    _options->add<vpux::OV::DEVICE_ID>();
    _options->add<vpux::OV::BATCH_MODE>();
    _options->add<vpux::OV::COMPILATION_MODE>();
    _options->add<vpux::OV::COMPILATION_MODE_PARAMS>();
    _options->add<vpux::OV::BACKEND_COMPILATION_PARAMS>();
    _options->add<vpux::OV::COMPILATION_NUM_THREADS>();
    _options->add<vpux::OV::TILES>();
    _options->add<vpux::OV::STEPPING>();
    _options->add<vpux::OV::MAX_TILES>();
    _options->add<vpux::OV::DMA_ENGINES>();
    _options->add<vpux::OV::DYNAMIC_SHAPE_TO_STATIC>();
    _options->add<vpux::OV::EXECUTION_MODE_HINT>();
    _options->add<vpux::OV::COMPILER_DYNAMIC_QUANTIZATION>();
    _options->add<vpux::OV::BATCH_COMPILER_MODE_SETTINGS>();
    _options->add<vpux::OV::QDQ_OPTIMIZATION_AGGRESSIVE>();
    _options->add<vpux::OV::QDQ_OPTIMIZATION>();
    _options->add<vpux::OV::TURBO>();
    _options->add<vpux::OV::MODEL_SERIALIZER_VERSION>();
    // don't enable for NPU3720 platforms: MTL and ARL
    if (_deviceDesc.deviceID != 0x7D1D && _deviceDesc.deviceID != 0xAD1D) {
        _options->add<vpux::OV::ENABLE_STRIDES_FOR>();
    }

    _options->add<vpux::OV::ENABLE_WEIGHTLESS>();
    _options->add<vpux::OV::SEPARATE_WEIGHTS_VERSION>();
    _options->add<vpux::OV::WS_COMPILE_CALL_NUMBER>();
    _options->add<vpux::OV::CACHE_MODE>();
    _options->add<vpux::OV::COMPATIBILITY_CHECK>();
    _options->add<vpux::OV::EXTENSION_LIB_PATH>();
    _options->add<vpux::OV::CUSTOM_KERNEL_CONFIG_PATH>();

    // Update the compiler properties
    _compilerProp.id = COMPILER_VERSION;
    _compilerProp.version.major = NPU_COMPILER_VERSION_MAJOR;
    _compilerProp.version.minor = NPU_COMPILER_VERSION_MINOR;
    _compilerProp.supportedOpsets = SUPPORTED_OPSET;
}

std::pair<VPUXExecutableL0*, vcl_result_t> VPUXCompilerL0::importNetwork(const vcl_executable_desc_t& desc) {
    VPUXExecutableL0* exe = nullptr;
    try {
        // Call compiler to compile the model and create blob
        // Create executable with the result NetworkDescription, profiling option and logger
        // Note we rely on implicit move semantics thanks to compile result being an rvalue,
        // failure to move here would lead to a blob copy!
        vpux::OV::Config config(_options);
        if (!_isDeviceDescEmpty) {
            setPlatformAndDeviceIdT(_deviceDesc.deviceID, config);
        }

        // Load the compiler library on the calling thread rather than the worker thread spawned below.
        // In some sandboxed environments, worker threads only hold a restricted process token and cannot
        // call LoadLibrary(). Loading the library here ensures it happens on a thread with sufficient rights.
        _compilerLoader->ensureLoaded();

        // Isolate the MLIR thread to safely destroy MLIR thread_local objects before CiD unload.
        auto network = std::make_shared<const NetworkDescription>(runInWorkerThreadSync([&] {
            return _compilerLoader->getCompiler()->compileFromDesc(desc, _compilerProp, _compilerDesc, _deviceDesc,
                                                                   config, _isDeviceDescEmpty);
        }));

        exe = new VPUXExecutableL0(network, config, _logger);
    } catch (const InvalidIrError& error) {
        _logger->outputError(error.what());
        return {nullptr, VCL_RESULT_ERROR_INVALID_IR};
    } catch (const std::exception& error) {
        _logger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return {nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT};
    } catch (...) {
        _logger->outputError("Internal exception! Can not compile!");
        return {nullptr, VCL_RESULT_ERROR_INVALID_ARGUMENT};
    }

    return {exe, VCL_RESULT_SUCCESS};
}

NetworkDescriptionView VPUXCompilerL0::importNetwork(const vcl_executable_desc_t& desc, BlobAllocator& allocator) {
    // Load the compiler library on the calling thread rather than the worker thread spawned below.
    // In some sandboxed environments, worker threads only hold a restricted process token and cannot
    // call LoadLibrary(). Loading the library here ensures it happens on a thread with sufficient rights.
    _compilerLoader->ensureLoaded();

    // Isolate the MLIR thread to safely destroy MLIR thread_local objects before CiD unload.
    return runInWorkerThreadSync([&] {
        vpux::OV::Config config(_options);
        if (!_isDeviceDescEmpty) {
            setPlatformAndDeviceIdT(_deviceDesc.deviceID, config);
        }
        return _compilerLoader->getCompiler()->compileFromDesc(desc, _compilerProp, _compilerDesc, _deviceDesc, config,
                                                               _isDeviceDescEmpty, allocator);
    });
}

std::vector<std::shared_ptr<NetworkDescriptionView>> VPUXCompilerL0::importNetworkWSOneShot(
        const vcl_executable_desc_t& desc, BlobAllocator& allocator) {
    // Load the compiler library on the calling thread rather than the worker thread spawned below.
    // In some sandboxed environments, worker threads only hold a restricted process token and cannot
    // call LoadLibrary(). Loading the library here ensures it happens on a thread with sufficient rights.
    _compilerLoader->ensureLoaded();

    // Isolate the MLIR thread to safely destroy MLIR thread_local objects before CiD unload.
    return runInWorkerThreadSync([&] {
        vpux::OV::Config config(_options);
        if (!_isDeviceDescEmpty) {
            setPlatformAndDeviceIdT(_deviceDesc.deviceID, config);
        }
        return _compilerLoader->getCompiler()->compileFromDescWsOneShot(desc, _compilerProp, _compilerDesc, _deviceDesc,
                                                                        config, _isDeviceDescEmpty, allocator);
    });
}

vcl_result_t VPUXCompilerL0::queryNetwork(const vcl_query_desc_t& desc, VPUXQueryNetworkL0* pQueryNetwork) {
    _logger->info("Start to call query function from compiler to get supported layers!");
    ov::SupportedOpsMap queryNetworkResult;
    try {
        vpux::OV::Config config(_options);
        if (!_isDeviceDescEmpty) {
            setPlatformAndDeviceIdT(_deviceDesc.deviceID, config);
        }
        queryNetworkResult = _compilerLoader->getCompiler()->queryFromDesc(desc, _compilerDesc, _compilerProp,
                                                                           _deviceDesc, config, _isDeviceDescEmpty);
    } catch (const InvalidIrError& error) {
        _logger->outputError(error.what());
        return VCL_RESULT_ERROR_INVALID_IR;
    } catch (const std::exception& error) {
        _logger->outputError(formatv("Compiler returned msg:\n{0}", error.what()));
        return VCL_RESULT_ERROR_UNKNOWN;
    } catch (...) {
        _logger->outputError("Failed to call query from compiler!");
        return VCL_RESULT_ERROR_UNKNOWN;
    }
    _logger->info("Successfully query supported layers!");

    // Serialize the result to predefined format
    auto ret = pQueryNetwork->setQueryResult(queryNetworkResult);
    return ret;
}

vcl_result_t VPUXCompilerL0::getSupportedOptions(char* buffer, uint64_t size) const {
    // get the registered options list, excluding private options (false param)
    std::string optListStr = _options->getSupportedAsString(false);
    // check if it fits
    uint64_t stringsize = optListStr.size() + 1;
    if (stringsize > size) {
        _logger->outputError("Compiler supported options list does not fit into the provided buffer!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    // serialize
    std::memcpy(buffer, optListStr.c_str(), stringsize);

    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXCompilerL0::getSupportedOptionsSize(uint64_t* stringSize) const {
    // get the registered options list, excluding private options (false param)
    std::string optionsList = _options->getSupportedAsString(false);
    // get string size +1 for null-termination
    *stringSize = optionsList.size() + 1;
    return VCL_RESULT_SUCCESS;
}

bool VPUXCompilerL0::isOptionValueSupported(const char* option, const char* value) const {
    // From OV 25.2, the plugin can check private options
    // Return true if the option is supported by transformation in model_preprocessor
    static std::unordered_set<std::string> compatibilityConfig = {"NPU_DPU_GROUPS"};

    std::string optName(option);
    if (compatibilityConfig.count(optName) > 0) {
        _logger->debug("Option {0} is a compatibility option, returning true", optName);
        return true;
    }

    if (optName == vpux::OV::COMPATIBILITY_CHECK::key()) {
        if (!value) {
            return true;
        }
        auto config = vpux::OV::Config(_options);
        if (_isDeviceDescEmpty || _deviceDesc.tileCount == static_cast<uint32_t>(-1)) {
            _logger->outputError("Cannot validate the compatibility string: device descriptor is missing or "
                                 "MAX_TILES is invalid.");
            throw std::runtime_error("Cannot validate the compatibility string: device descriptor is missing or "
                                     "MAX_TILES is invalid.");
        }
        if (!setPlatformAndDeviceId(_deviceDesc.deviceID, config)) {
            // Update the device config failed, return false for safety.
            return false;
        }
        config.update({{vpux::OV::MAX_TILES::key().data(), std::to_string(_deviceDesc.tileCount)}});
        return checkRuntimeRequirements(std::string_view{value}, config);
    }

    std::vector<std::string> optList = _options->getSupported(true);  // include private
    if (std::find(optList.begin(), optList.end(), optName) == optList.end()) {
        return false;
    }

    return value ? _options->get(optName).isValueSupported(value) : true;
}

}  // namespace VPUXDriverCompiler
