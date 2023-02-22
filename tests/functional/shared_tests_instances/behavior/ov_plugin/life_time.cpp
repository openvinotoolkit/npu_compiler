// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest, ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                         OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTestOnImportedNetwork,
                         ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                         OVHoldersTestOnImportedNetwork::getTestCaseName);

}  // namespace
