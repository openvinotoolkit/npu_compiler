// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <vpux/vpux_plugin_config.hpp>
#include <vpux/vpux_compiler_config.hpp>
#include "behavior/infer_request_config.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> configs = {
    {},
    // Public options
    {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
    {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}},
    {{CONFIG_KEY(DEVICE_ID), ""}},
    {{VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), "1"}},
    {{KMB_CONFIG_KEY(THROUGHPUT_STREAMS), "1"}},
    {{VPUX_CONFIG_KEY(CSRAM_SIZE), "2097152"}},

    // Private options
    {{"VPUX_GRAPH_COLOR_FORMAT", "RGB"}},
    {{"VPUX_USE_M2I", CONFIG_VALUE(YES)}},
    {{"VPU_KMB_USE_M2I", CONFIG_VALUE(NO)}},
    {{"VPUX_USE_SHAVE_ONLY_M2I", CONFIG_VALUE(YES)}},
    {{"VPU_KMB_USE_SHAVE_ONLY_M2I", CONFIG_VALUE(NO)}},
    {{"VPUX_USE_SIPP", CONFIG_VALUE(YES)}},
    {{"VPU_KMB_USE_SIPP", CONFIG_VALUE(NO)}},
    {{"VPUX_PREPROCESSING_SHAVES", "4"}},
    {{"VPUX_PREPROCESSING_LPI", "8"}},
    {{"VPUX_VPUAL_REPACK_INPUT_LAYOUT", CONFIG_VALUE(YES)}},
    {{"VPUX_EXECUTOR_STREAMS", "2"}},
    {{"VPU_KMB_EXECUTOR_STREAMS", "1"}},
    {{"VPUX_PLATFORM", "AUTO"}},
};

const std::vector<std::map<std::string, std::string>> Inconfigs = {
    // Public options
    {{CONFIG_KEY(LOG_LEVEL), "SOME_LEVEL"}},
    {{CONFIG_KEY(PERF_COUNT), "YEP"}},
    {{CONFIG_KEY(DEVICE_ID), "SOME_DEVICE_ID"}},
    {{VPUX_CONFIG_KEY(THROUGHPUT_STREAMS), "TWENTY"}},
    {{KMB_CONFIG_KEY(THROUGHPUT_STREAMS), "TWENTY"}},
    {{VPUX_CONFIG_KEY(CSRAM_SIZE), "-3"}},

    // Private options
    {{"VPUX_GRAPH_COLOR_FORMAT", "NV12"}},
    {{"VPUX_USE_M2I", "YEP"}},
    {{"VPU_KMB_USE_M2I", "NOP"}},
    {{"VPUX_USE_SHAVE_ONLY_M2I", "YEP"}},
    {{"VPU_KMB_USE_SHAVE_ONLY_M2I", "NOP"}},
    {{"VPUX_USE_SIPP", "NOP"}},
    {{"VPU_KMB_USE_SIPP", "NOP"}},
    {{"VPUX_PREPROCESSING_SHAVES", "FOUR"}},
    {{"VPUX_PREPROCESSING_LPI", "EIGHT"}},
    {{"VPUX_VPUAL_REPACK_INPUT_LAYOUT", "YEP"}},
    {{"VPUX_EXECUTOR_STREAMS", "ONE"}},
    {{"VPU_KMB_EXECUTOR_STREAMS", "TWO"}},
    {{"VPUX_PLATFORM", "SOME_PLATFORM"}},
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        InferConfigTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigInTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(Inconfigs)),
                        InferConfigInTests::getTestCaseName);
}  // namespace