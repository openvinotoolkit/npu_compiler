// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/remote_tensor_tests/dx12_remote_run.hpp"

#include "common/utils.hpp"

#ifdef _WIN32
#ifdef ENABLE_DX12
using namespace ov::test::behavior;

namespace {

std::shared_ptr<ov::Model> getFunction() {
    const std::vector<size_t> inputShape = {1, 10, 12};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape({inputShape}))};
    params.front()->get_output_tensor(0).set_names({"Parameter_1"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"Relu_2"});

    return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
}

}  // namespace

auto dynamicRemoteConfigs = []() {
    return std::vector<ov::AnyMap>{{{"NPU_COMPILER_TYPE", "MLIR"}, {"NPU_COMPILATION_MODE", "ReferenceSW"}}};
};

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, NPUInferRequestDynamicRemoteTensorsTests_NPU3720,
        ::testing::Combine(::testing::Values(getFunction()),
                           ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                   {{1, 10, 12}, {1, 10, 12}}, {{1, 18, 15}, {1, 18, 15}}}),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::ValuesIn(dynamicRemoteConfigs())),
        ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);
#endif
#endif
