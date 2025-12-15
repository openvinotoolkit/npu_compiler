//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#include <functional_test_utils/summary/op_summary.hpp>
#include <iostream>
#include <sstream>
#include <vpux/utils/logger/logger.hpp>
#include "gtest/gtest.h"
#include "intel_npu/npu_private_properties.hpp"
#include "vpu_test_report.hpp"
#include "vpu_test_tool.hpp"
#include "vpux/utils/IE/config.hpp"

namespace testing {
namespace internal {
extern bool g_help_flag;
}  // namespace internal
}  // namespace testing

void sigsegv_handler(int errCode) {
    auto& s = ov::test::utils::OpSummary::getInstance();
    s.saveReport();
    std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
    std::abort();
}

int main(int argc, char** argv, char** envp) {
    // register crashHandler for SIGSEGV signal
    signal(SIGSEGV, sigsegv_handler);

    std::ostringstream oss;
    oss << "Command line args (" << argc << "): ";
    for (int c = 0; c < argc; ++c) {
        oss << " " << argv[c];
    }
    oss << std::endl;

    oss << "Process id: " << getpid() << std::endl;
    std::cout << oss.str();
    oss.str("");

    oss << "Environment variables: ";
    for (char** env = envp; *env != 0; env++) {
        oss << *env << "; ";
    }

    std::string targetDevice(""), option;
    for (int i = 1; i < argc; i++) {
        option = argv[i];
        if (option == "-d") {
            if (i + 1 == argc) {
                break;
            }
            targetDevice = argv[i + 1];
            break;
        }
    }

    if (targetDevice != "NPU") {
        targetDevice = "NPU";
        std::cout << "\nTarget device was not set or it is not recognized. Using NPU by default.\n";
    }
    test_utils::TARGET_DEVICE = targetDevice.c_str();

    // GTest removes recognized arguments
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::VpuTestReportEnvironment());

    const bool dryRun = ::testing::GTEST_FLAG(list_tests) || ::testing::internal::g_help_flag;

    if (!dryRun) {
        std::vector<std::string> availableDevices;
        const auto core = ov::test::utils::PluginCache::get().core();
        if (core != nullptr) {
            availableDevices = core->get_available_devices();
            auto it = std::find(availableDevices.begin(), availableDevices.end(), "NPU");
            if (it == availableDevices.end()) {
                std::cerr << "Driver not found, exiting." << std::endl;
                return -1;
            }
        } else {
            std::cerr << "Failed to get OpenVINO Core from cache!" << std::endl;
        }

        const std::string noFetch{"<not fetched>"};
        std::string backend{noFetch}, arch{noFetch}, full{noFetch};
        try {
            LayerTestsUtils::VpuTestTool kmbTestTool(LayerTestsUtils::VpuTestEnvConfig::getInstance());
            backend = kmbTestTool.getDeviceMetric(ov::intel_npu::backend_name.name());
            arch = kmbTestTool.getDeviceMetric(ov::device::architecture.name());
            full = kmbTestTool.getDeviceMetric(ov::device::full_name.name());
        } catch (const std::exception& e) {
            std::cerr << "Exception while trying to determine device characteristics: " << e.what() << std::endl;
        }
        std::cout << "Tests run with: Backend name: '" << backend << "'; Device arch: '" << arch
                  << "'; Full device name: '" << full << "'" << std::endl;
    }

    std::string dTest = ::testing::internal::GTEST_FLAG(internal_run_death_test);
    if (dTest.empty()) {
        std::cout << oss.str() << std::endl;
    } else {
        std::cout << "gtest death test process is running" << std::endl;
    }

    auto& log = vpux::Logger::global();
    auto level = LayerTestsUtils::VpuTestEnvConfig::getInstance().IE_NPU_TESTS_LOG_LEVEL;
    vpux::LogLevel logLevel =
            level.empty() ? vpux::LogLevel::Info
                          : vpux::getLogLevel(intel_npu::OptionParser<ov::log::Level>::parse(level.c_str()));
    log.setLevel(logLevel);

    return RUN_ALL_TESTS();
}
