//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_test_tool.hpp"
#include <functional_test_utils/ov_plugin_cache.hpp>
#include "vpux/utils/core/format.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace ov::test::utils {

VpuTestTool::VpuTestTool(const VpuTestEnvConfig& envCfg)
        : envConfig(envCfg),
          DEVICE_NAME(envConfig.IE_NPU_TESTS_DEVICE_NAME.empty() ? DEVICE_NPU : envConfig.IE_NPU_TESTS_DEVICE_NAME),
          _log(vpux::Logger::global().nest("VpuTestTool", 1)) {
}

void VpuTestTool::exportModel(ov::CompiledModel& compiledModel, const std::string& fsName) {
    OPENVINO_ASSERT(!envConfig.IE_NPU_TESTS_DUMP_PATH.empty());
    std::filesystem::create_directories(envConfig.IE_NPU_TESTS_DUMP_PATH);

    const auto fileName = vpux::printToString("{0}/{1}", envConfig.IE_NPU_TESTS_DUMP_PATH, fsName);
    _log.info("Exporting model into {0}, device {1}", fileName, DEVICE_NAME);

    std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
    if (!file.is_open()) {
        OPENVINO_THROW("exportModel(). Can't open file ", fileName);
    }
    compiledModel.export_model(file);
    _log.info("Exported model into file {0}", fileName);
}

void VpuTestTool::importTensor(ov::Tensor& tensor, const std::string& fsName) {
    OPENVINO_ASSERT(!envConfig.IE_NPU_TESTS_DUMP_PATH.empty());

    const auto fileName = vpux::printToString("{0}/{1}", envConfig.IE_NPU_TESTS_DUMP_PATH, fsName);
    _log.debug("Importing `ov::Tensor` blob from {0}, device {1}", fileName, DEVICE_NAME);

    std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        OPENVINO_THROW("importTensor(). Can't open file ", fileName);
    }
    file.read(static_cast<char*>(tensor.data()), static_cast<std::streamsize>(tensor.get_byte_size()));
    if (!file) {
        OPENVINO_THROW("importTensor(). Error when reading file ", fileName);
    }
}

void VpuTestTool::exportTensor(const ov::Tensor& tensor, const std::string& fsName) {
    OPENVINO_ASSERT(!envConfig.IE_NPU_TESTS_DUMP_PATH.empty());
    std::filesystem::create_directories(envConfig.IE_NPU_TESTS_DUMP_PATH);

    const auto fileName = vpux::printToString("{0}/{1}", envConfig.IE_NPU_TESTS_DUMP_PATH, fsName);
    _log.debug("Exporting `ov::Tensor` blob from {0}, device {1}", fileName, DEVICE_NAME);

    std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
    if (!file.is_open()) {
        OPENVINO_THROW("exportTensor(). Can't open file ", fileName);
    }
    file.write(static_cast<const char*>(tensor.data()), static_cast<std::streamsize>(tensor.get_byte_size()));
    if (!file) {
        OPENVINO_THROW("exportTensor(). Error when writing file ", fileName);
    }
}

std::string VpuTestTool::getDeviceMetric(std::string name) {
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core(DEVICE_NAME);

    return core->get_property(DEVICE_NAME, name).as<std::string>();
}

unsigned long int FNV_hash(const std::string& str) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(str.c_str());
    unsigned long int h = 2166136261UL;

    for (size_t i = 0; i < str.size(); i++) {
        h = (h * 16777619) ^ p[i];
    }

    return h;
}

std::string cleanName(std::string name) {
    std::replace_if(
            name.begin(), name.end(),
            [](char c) {
                return !(std::isalnum(c) || c == '.');
            },
            '_');
    return name;
}

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength) {
    const size_t maxExpectedFileNameLen = 256, maxExpectedDirLen = 100, extLen = ext.size();
    const size_t maxFileNameLen =
                         (limitAbsPathLength ? maxExpectedFileNameLen - maxExpectedDirLen : maxExpectedFileNameLen),
                 maxNoExtLen = maxFileNameLen - extLen, maxNoExtShortenedLen = maxNoExtLen - 20 - 1;
    const auto testName = vpux::printToString("{0}_{1}", testInfo->test_case_name(), testInfo->name());
    auto fnameNoExt =
            (testName.size() < maxNoExtLen)
                    ? testName
                    : vpux::printToString("{0}_{1}", testName.substr(0, maxNoExtShortenedLen), FNV_hash(testName));

    return cleanName(fnameNoExt.append(ext));
}

}  // namespace ov::test::utils
