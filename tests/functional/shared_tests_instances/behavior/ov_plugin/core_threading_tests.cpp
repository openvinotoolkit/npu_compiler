// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include "behavior/ov_plugin/core_threading.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

const Params params[] = {
        std::tuple<Device, Config>{test_utils::TARGET_DEVICE, {{ov::enable_profiling(true)}}},
        std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
                                   {{ov::device::priorities(test_utils::TARGET_DEVICE, ov::test::utils::DEVICE_CPU)}}},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_NPU, CoreThreadingTestsWithCacheEnabled,
                         testing::Combine(testing::ValuesIn(params), testing::Values(20), testing::Values(10)),
                         (ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithCacheEnabled>));
