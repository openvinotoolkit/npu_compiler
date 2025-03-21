//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/constant.hpp"

namespace ov::test {
using LargeMishTestParams = std::tuple<ov::Shape>;

class LargeMishTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMishTestParams> {
    void SetUp() override {
        auto inputShape = std::get<ov::Shape>(GetParam());
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input->set_friendly_name("input_0");
        const ov::ParameterVector params{input};

        auto mish0 = std::make_shared<ov::op::v4::Mish>(input);
        auto const0 = ov::op::v0::Constant::create(ov::element::f16, inputShape, {1.0f});
        auto mish1 = std::make_shared<ov::op::v4::Mish>(const0);
        auto minimum = std::make_shared<ov::op::v1::Minimum>(mish0, mish1);
        auto mish2 = std::make_shared<ov::op::v4::Mish>(minimum);
        auto result = std::make_shared<ov::op::v0::Result>(mish2);

        const ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, params, "LargeMishTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<LargeMishTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(LargeMishTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_LargeMishInDDR, LargeMishTest_NPU3720,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         LargeMishTest_NPU3720::getTestCaseName);

class LargeMishTest_NPU4000 : public LargeMishTest_NPU3720 {};

TEST_P(LargeMishTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_LargeMishInDDR, LargeMishTest_NPU4000,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         LargeMishTest_NPU3720::getTestCaseName);
class TwoMishTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMishTestParams> {
    void SetUp() override {
        auto inputShape = std::get<ov::Shape>(GetParam());
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input->set_friendly_name("input_0");
        const ov::ParameterVector params{input};

        auto mish0 = std::make_shared<ov::op::v4::Mish>(input);
        auto mish1 = std::make_shared<ov::op::v4::Mish>(mish0);
        auto result = std::make_shared<ov::op::v0::Result>(mish1);

        const ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, params, "TwoMishTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<LargeMishTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(TwoMishTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoMishInDDR, TwoMishTest_NPU3720,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 32, 32, 514}  // in_shape
                         }),
                         TwoMishTest_NPU3720::getTestCaseName);

class TwoMishTest_NPU4000 : public TwoMishTest_NPU3720 {};

TEST_P(TwoMishTest_NPU4000, HW) {
    setDefaultHardwareMode();
    // TODO: E129229
    configuration["NPU_BACKEND_COMPILATION_PARAMS"] = "workload-management-enable=false";
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoMishInDDR, TwoMishTest_NPU4000,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 32, 32, 514}  // in_shape
                         }),
                         TwoMishTest_NPU3720::getTestCaseName);
class TwoScatterUpdateTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMishTestParams> {
    void SetUp() override {
        auto inputShape = std::get<ov::Shape>(GetParam());
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input->set_friendly_name("input_0");
        const ov::ParameterVector params{input};

        auto scatterIndices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{32, 2}, {2});
        auto axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto updates1 = ov::op::v0::Constant::create(
                ov::element::f16, ov::Shape{inputShape[0], 32, 2, inputShape[2], inputShape[3]}, {1.0f});
        auto scatterUpdate1 = std::make_shared<ov::op::v3::ScatterUpdate>(input, scatterIndices, updates1, axis1);

        auto updates2 = ov::op::v0::Constant::create(
                ov::element::f16, ov::Shape{inputShape[0], 32, 2, inputShape[2], inputShape[3]}, {1.0f});
        auto scatterUpdate2 =
                std::make_shared<ov::op::v3::ScatterUpdate>(scatterUpdate1, scatterIndices, updates2, axis1);
        auto result = std::make_shared<ov::op::v0::Result>(scatterUpdate2);

        const ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, params, "TwoScatterUpdateTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<LargeMishTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(TwoScatterUpdateTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoScatterUpdateInDDR, TwoScatterUpdateTest_NPU3720,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         TwoScatterUpdateTest_NPU3720::getTestCaseName);

class TwoScatterUpdateTest_NPU4000 : public TwoScatterUpdateTest_NPU3720 {};

TEST_P(TwoScatterUpdateTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoScatterUpdateInDDR, TwoScatterUpdateTest_NPU4000,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         TwoScatterUpdateTest_NPU3720::getTestCaseName);
}  // namespace ov::test
