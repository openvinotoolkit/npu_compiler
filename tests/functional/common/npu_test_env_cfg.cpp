//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/config/options.hpp"
#include "vpux/utils/ov/config.hpp"

#include <cstdlib>
#include <stdexcept>

using vpux::OV::envVarStrToBool;

namespace test_utils {
// Default set to 'NPU' Plugin, but can be overwritten via CLI argument. Example: -d IMD
const char* TARGET_DEVICE = "NPU";
}  // namespace test_utils

namespace ov::test::utils {

VpuTestEnvConfig::VpuTestEnvConfig() {
    // start reading obsolete environment variables
    if (auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_KMB_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = envVarStrToBool("IE_KMB_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = envVarStrToBool("IE_KMB_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = envVarStrToBool("IE_KMB_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = envVarStrToBool("IE_KMB_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = envVarStrToBool("IE_KMB_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = envVarStrToBool("IE_KMB_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = envVarStrToBool("IE_KMB_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_KMB_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }
    // end reading obsolete environment variables

    if (auto var = std::getenv("IE_NPU_TESTS_DEVICE_NAME")) {
        IE_NPU_TESTS_DEVICE_NAME = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_DUMP_PATH")) {
        IE_NPU_TESTS_DUMP_PATH = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_RUN_EXPORT")) {
        IE_NPU_TESTS_RUN_EXPORT = envVarStrToBool("IE_NPU_TESTS_RUN_EXPORT", var);
    }

    if (auto var = std::getenv("IE_NPU_COMPILE_ONLY")) {
        IE_NPU_COMPILE_ONLY = envVarStrToBool("IE_NPU_COMPILE_ONLY", var);
    }

    if (auto var = std::getenv("IE_NPU_FORCE_COMPILER")) {
        IE_NPU_FORCE_COMPILER = var;
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_INPUT")) {
        IE_NPU_TESTS_EXPORT_INPUT = envVarStrToBool("IE_NPU_TESTS_EXPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_OUTPUT")) {
        IE_NPU_TESTS_EXPORT_OUTPUT = envVarStrToBool("IE_NPU_TESTS_EXPORT_OUTPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_EXPORT_REF")) {
        IE_NPU_TESTS_EXPORT_REF = envVarStrToBool("IE_NPU_TESTS_EXPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_INPUT")) {
        IE_NPU_TESTS_IMPORT_INPUT = envVarStrToBool("IE_NPU_TESTS_IMPORT_INPUT", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_IMPORT_REF")) {
        IE_NPU_TESTS_IMPORT_REF = envVarStrToBool("IE_NPU_TESTS_IMPORT_REF", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_LONG_FILE_NAME")) {
        IE_NPU_TESTS_LONG_FILE_NAME = envVarStrToBool("IE_NPU_TESTS_LONG_FILE_NAME", var);
    }

    if (auto var = std::getenv("IE_NPU_TESTS_PLATFORM")) {
        IE_NPU_TESTS_PLATFORM = var;
    }

    if (auto var = std::getenv("IE_NPU_SINGLE_CLUSTER_MODE")) {
        IE_NPU_SINGLE_CLUSTER_MODE = envVarStrToBool("IE_NPU_SINGLE_CLUSTER_MODE", var);
    }

    if (auto var = std::getenv("OV_NPU_TESTS_SKIP_CONFIG_FILE")) {
        OV_NPU_TESTS_SKIP_CONFIG_FILE = var;
    }
}

const VpuTestEnvConfig& VpuTestEnvConfig::getInstance() {
    static VpuTestEnvConfig instance{};
    return instance;
}

std::string getTestsDeviceNameFromEnvironmentOr(const std::string& instead) {
    return (!VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME.empty())
                   ? VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME
                   : instead;
}

std::string getTestsPlatformFromEnvironmentOr(const std::string& instead) {
    return (!VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty())
                   ? VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM
                   : instead;
}

std::string getTestsPlatformCompilerInPlugin() {
    return getTestsPlatformFromEnvironmentOr(
            getTestsDeviceNameFromEnvironmentOr(std::string(ov::intel_npu::Platform::AUTO_DETECT)));
}

std::string getDeviceNameTestCase(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_name() + parser.get_device_id();
}

std::string getDeviceName() {
    return LayerTestsUtils::getTestsDeviceNameFromEnvironmentOr("NPU.3720");
}

std::string getDeviceNameID(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_id();
}

std::string getTestDeviceId() {
    return ov::intel_npu::Platform::standardize(VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM);
}

}  // namespace ov::test::utils
