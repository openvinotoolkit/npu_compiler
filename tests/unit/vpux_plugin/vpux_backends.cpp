//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/config/options.hpp"
#include "vpux_backends.hpp"

#include "test_utils/npu_backends_test.hpp"

using VPUXBackendsUnitTests = ::testing::Test;

TEST_F(VPUXBackendsUnitTests, notStopSearchingIfBackendThrow) {
    const std::vector<std::string> dummyBackendRegistry = {"throw_test_backend", "npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    auto options = std::make_shared<vpux::OptionsDesc>();
    options->add<intel_npu::PERFORMANCE_HINT>();
    options->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
    options->add<intel_npu::INFERENCE_PRECISION_HINT>();
    options->add<intel_npu::PERF_COUNT>();
    options->add<intel_npu::LOG_LEVEL>();
    options->add<intel_npu::PLATFORM>();
    options->add<intel_npu::DEVICE_ID>();
    options->add<intel_npu::CACHE_DIR>();
    options->add<intel_npu::LOADED_FROM_CACHE>();
    options->add<intel_npu::BATCH_MODE>();
    options->add<intel_npu::EXCLUSIVE_ASYNC_REQUESTS>();
    options->add<intel_npu::PROFILING_TYPE>();
    options->add<intel_npu::MODEL_PRIORITY>();
    options->add<intel_npu::CREATE_EXECUTOR>();
    options->add<intel_npu::DEFER_WEIGHTS_LOAD>();
    options->add<intel_npu::NUM_STREAMS>();
    options->add<intel_npu::ENABLE_CPU_PINNING>();
    options->add<intel_npu::WORKLOAD_TYPE>();
    options->add<intel_npu::TURBO>();
    options->add<intel_npu::WEIGHTS_PATH>();
    options->add<intel_npu::BYPASS_UMD_CACHING>();
    options->add<intel_npu::RUN_INFERENCES_SEQUENTIALLY>();

    vpux::Config config(options);
    config.update({{"LOG_LEVEL", "LOG_DEBUG"}});

    backends->setup(config);

    auto device = backends->getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("3720.dummyDevice", device->getName());
}

TEST_F(VPUXBackendsUnitTests, notStopSearchingIfBackendNotExists) {
    const std::vector<std::string> dummyBackendRegistry = {"not_exists_backend", "npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    auto options = std::make_shared<vpux::OptionsDesc>();
    options->add<intel_npu::PERFORMANCE_HINT>();
    options->add<intel_npu::PERFORMANCE_HINT_NUM_REQUESTS>();
    options->add<intel_npu::INFERENCE_PRECISION_HINT>();
    options->add<intel_npu::PERF_COUNT>();
    options->add<intel_npu::LOG_LEVEL>();
    options->add<intel_npu::PLATFORM>();
    options->add<intel_npu::DEVICE_ID>();
    options->add<intel_npu::CACHE_DIR>();
    options->add<intel_npu::LOADED_FROM_CACHE>();
    options->add<intel_npu::BATCH_MODE>();
    options->add<intel_npu::EXCLUSIVE_ASYNC_REQUESTS>();
    options->add<intel_npu::PROFILING_TYPE>();
    options->add<intel_npu::MODEL_PRIORITY>();
    options->add<intel_npu::CREATE_EXECUTOR>();
    options->add<intel_npu::DEFER_WEIGHTS_LOAD>();
    options->add<intel_npu::NUM_STREAMS>();
    options->add<intel_npu::ENABLE_CPU_PINNING>();
    options->add<intel_npu::WORKLOAD_TYPE>();
    options->add<intel_npu::TURBO>();
    options->add<intel_npu::WEIGHTS_PATH>();
    options->add<intel_npu::BYPASS_UMD_CACHING>();
    options->add<intel_npu::RUN_INFERENCES_SEQUENTIALLY>();

    vpux::Config config(options);
    config.update({{"LOG_LEVEL", "LOG_DEBUG"}});

    backends->setup(config);

    auto device = backends->getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("3720.dummyDevice", device->getName());
}

TEST_F(VPUXBackendsUnitTests, canFindDeviceIfAtLeastOneBackendHasDevicesAvailable) {
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend", "npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    auto device = backends->getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("3720.dummyDevice", device->getName());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfNoBackends) {
    const std::vector<std::string> dummyBackendRegistry = {};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    ASSERT_EQ(nullptr, backends->getDevice());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfPassedBackendsNotExist) {
    const std::vector<std::string> dummyBackendRegistry = {"wrong_path", "one_more_wrong_path"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    ASSERT_EQ(nullptr, backends->getDevice());
}

TEST_F(VPUXBackendsUnitTests, findDeviceAfterName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string deviceName = "3720.dummyDevice";
    auto device = backends->getDevice(deviceName);
    ASSERT_NE(nullptr, device);
    ASSERT_EQ(deviceName, device->getName());
}

TEST_F(VPUXBackendsUnitTests, noDeviceFoundedAfterName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string deviceName = "wrong_device";
    ASSERT_EQ(nullptr, backends->getDevice(deviceName));
}

TEST_F(VPUXBackendsUnitTests, findDeviceAfterParamMap) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    ov::AnyMap paramMap = {{ov::device::id.name(), 3720}};
    auto device = backends->getDevice(paramMap);
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("3720.dummyDevice", device->getName());
}

TEST_F(VPUXBackendsUnitTests, noDeviceFoundedAfterParamMap) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    ov::AnyMap paramMap = {{ov::device::id.name(), 3720}};
    ASSERT_EQ(nullptr, backends->getDevice(paramMap));
}

TEST_F(VPUXBackendsUnitTests, getDeviceNamesSecondIsDummyName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::vector<std::string> deviceNames = backends->getAvailableDevicesNames();

    ASSERT_EQ("3720.dummyDevice", deviceNames[0]);
    ASSERT_EQ("noOtherDevice", deviceNames[1]);
}

TEST_F(VPUXBackendsUnitTests, getBackendName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string backendName = backends->getBackendName();

    ASSERT_EQ("3720.dummyDevice", backendName);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string compilationPlatform = backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "");

    ASSERT_EQ("3720", compilationPlatform);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByPlatform) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string compilationPlatform3720 = backends->getCompilationPlatform(ov::intel_npu::Platform::NPU3720, "");
    std::string compilationPlatform4000 = backends->getCompilationPlatform(ov::intel_npu::Platform::NPU4000, "");
    std::string compilationPlatform5000 = backends->getCompilationPlatform(ov::intel_npu::Platform::NPU5000, "");
    std::string compilationPlatform5010 = backends->getCompilationPlatform(ov::intel_npu::Platform::NPU5010, "");

    ASSERT_EQ("3720", compilationPlatform3720);
    ASSERT_EQ("4000", compilationPlatform4000);
    ASSERT_EQ("5000", compilationPlatform5000);
    ASSERT_EQ("5010", compilationPlatform5010);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceId) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    std::string compilationPlatform3720 =
            backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "3720");
    std::string compilationPlatform4000 =
            backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "4000");
    std::string compilationPlatform5000 =
            backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "5000");
    std::string compilationPlatform5010 =
            backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "5010");

    ASSERT_EQ("3720", compilationPlatform3720);
    ASSERT_EQ("4000", compilationPlatform4000);
    ASSERT_EQ("5000", compilationPlatform5000);
    ASSERT_EQ("5010", compilationPlatform5010);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceNameNoDevice) {
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    try {
        std::string compilationPlatform = backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "");
    } catch (const std::exception& ex) {
        std::string expectedMessage("No devices found - platform must be explicitly specified for compilation. "
                                    "Example: -d NPU.3720 instead of -d NPU.\n");
        std::string exceptionMessage(ex.what());
        // exception message contains information about path to file and line number, where the exception occurred.
        // We should ignore this part of the message on comparision step
        ASSERT_TRUE(exceptionMessage.length() >= expectedMessage.length());
        ASSERT_EQ(expectedMessage, exceptionMessage.substr(exceptionMessage.length() - expectedMessage.length()));
    } catch (...) {
        FAIL() << "UNEXPECTED RESULT";
    }
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceNameWrongNameFormat) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};

    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);

    try {
        std::string compilationPlatform = backends->getCompilationPlatform(ov::intel_npu::Platform::AUTO_DETECT, "");
    } catch (const std::exception& ex) {
        std::string expectedMessage("Unexpected device name: 3720.dummyDevice\n");
        std::string exceptionMessage(ex.what());
        // exception message contains information about path to file and line number, where the exception occurred.
        // We should ignore this part of the message on comparision step
        ASSERT_TRUE(exceptionMessage.length() >= expectedMessage.length());
        ASSERT_EQ(expectedMessage, exceptionMessage.substr(exceptionMessage.length() - expectedMessage.length()));
    } catch (...) {
        FAIL() << "UNEXPECTED RESULT";
    }
}
