//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <base/behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>

namespace {
const std::string expectedDeviceName =
        std::getenv("IE_KMB_TESTS_DEVICE_NAME") != nullptr ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "VPUX";
}

TEST(smoke_InterfaceTests, TestEngineClassGetMetric) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::Core ie;
        auto deviceIDs = ie.GetMetric(expectedDeviceName, METRIC_KEY(AVAILABLE_DEVICES)).as<std::vector<std::string>>();
        if (deviceIDs.empty())
            GTEST_SKIP() << "No devices available";

        const auto fullDeviceName = expectedDeviceName + "." + deviceIDs[0];
        const std::vector<std::string> supportedMetrics = ie.GetMetric(fullDeviceName, METRIC_KEY(SUPPORTED_METRICS));
        auto supportedConfigKeysFound = false;
        for (const auto& metricName : supportedMetrics) {
            if (metricName == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                ASSERT_FALSE(supportedConfigKeysFound);  // may be we should make more strict test for duplicates of key
                supportedConfigKeysFound = true;
            }
            ASSERT_FALSE(ie.GetMetric(fullDeviceName, metricName).empty());
        }
        ASSERT_TRUE(supportedConfigKeysFound);  // plus implicit check for !supportedMetrics.empty()

        ASSERT_THROW(ie.GetMetric(fullDeviceName, "THISMETRICNOTEXIST"), InferenceEngine::Exception);
    }
}

TEST(smoke_InterfaceTests, TestEngineClassGetConfig) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::Core ie;

        const std::vector<std::string> supportedConfigKeys =
                ie.GetMetric(expectedDeviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        ASSERT_FALSE(supportedConfigKeys.empty());
        for (const auto& configKey : supportedConfigKeys) {
            ASSERT_FALSE(ie.GetConfig(expectedDeviceName, configKey).empty());
        }

        ASSERT_THROW(ie.GetConfig(expectedDeviceName, "THISKEYNOTEXIST"), InferenceEngine::Exception);
    }
}
