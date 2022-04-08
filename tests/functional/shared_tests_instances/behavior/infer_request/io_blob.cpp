// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"


using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
        ::testing::Combine(
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
        ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobTest::getTestCaseName);
}  // namespace
