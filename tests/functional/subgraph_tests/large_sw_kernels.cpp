//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <random>

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/constant.hpp"

#include "openvino/op/minimum.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/scatter_update.hpp"

namespace ov::test {
using LargeMishTestParams = std::tuple<ov::Shape>;

class LargeMishTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMishTestParams> {
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

TEST_P(LargeMishTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LargeMishTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(LargeMishTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(LargeMishTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

INSTANTIATE_TEST_SUITE_P(smoke_LargeMishInDDR, LargeMishTestCommon,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         LargeMishTestCommon::getTestCaseName);

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
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoMishInDDR, TwoMishTest_NPU4000,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 32, 32, 514}  // in_shape
                         }),
                         TwoMishTest_NPU3720::getTestCaseName);
class TwoMishTest_NPU5010 : public TwoMishTest_NPU3720 {};

TEST_P(TwoMishTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoMishInDDR, TwoMishTest_NPU5010,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 32, 32, 514}  // in_shape
                         }),
                         TwoMishTest_NPU5010::getTestCaseName);

class TwoMishTest_NPU5020 : public TwoMishTest_NPU3720 {};

TEST_P(TwoMishTest_NPU5020, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoMishInDDR, TwoMishTest_NPU5020,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 32, 32, 514}  // in_shape
                         }),
                         TwoMishTest_NPU5020::getTestCaseName);

class TwoScatterUpdateTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMishTestParams> {
    void SetUp() override {
        auto inputShape = std::get<ov::Shape>(GetParam());
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input->set_friendly_name("input_0");
        const ov::ParameterVector params{input};

        auto axisNo = 1;
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axisNo});
        auto axisDim = inputShape.at(axisNo);

        std::mt19937 rng(123);
        std::uniform_int_distribution<int64_t> dist(0, static_cast<int64_t>(axisDim) - 1);
        const ov::Shape indicesShape{3, 5};  // 2D indices

        std::vector<int64_t> indicesData1(ov::shape_size(indicesShape));
        std::generate(indicesData1.begin(), indicesData1.end(), [&]() {
            return dist(rng);
        });
        auto scatterIndices1 = ov::op::v0::Constant::create(ov::element::i64, indicesShape, indicesData1);

        std::vector<int64_t> indicesData2(ov::shape_size(indicesShape));
        std::generate(indicesData2.begin(), indicesData2.end(), [&]() {
            return dist(rng);
        });
        auto scatterIndices2 = ov::op::v0::Constant::create(ov::element::i64, indicesShape, indicesData2);

        const size_t updates1Size = inputShape[0] * indicesData1.size() * inputShape[2] * inputShape[3];
        std::uniform_real_distribution<float> floatDist(-1.0f, 1.0f);
        std::vector<float> updates1Data(updates1Size);
        std::generate(updates1Data.begin(), updates1Data.end(), [&]() {
            return floatDist(rng);
        });
        auto updates1 = ov::op::v0::Constant::create(
                ov::element::f16,
                ov::Shape{inputShape[0], indicesShape[0], indicesShape[1], inputShape[2], inputShape[3]}, updates1Data);
        auto scatterUpdate1 = std::make_shared<ov::op::v3::ScatterUpdate>(input, scatterIndices1, updates1, axis);

        const std::vector<float> updates2Data(updates1Data.rbegin(), updates1Data.rend());
        auto updates2 = ov::op::v0::Constant::create(
                ov::element::f16,
                ov::Shape{inputShape[0], indicesShape[0], indicesShape[1], inputShape[2], inputShape[3]}, updates2Data);
        auto scatterUpdate2 =
                std::make_shared<ov::op::v3::ScatterUpdate>(scatterUpdate1, scatterIndices2, updates2, axis);
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

TEST_P(TwoScatterUpdateTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(TwoScatterUpdateTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TwoScatterUpdateTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(TwoScatterUpdateTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

INSTANTIATE_TEST_SUITE_P(smoke_TwoScatterUpdateInDDR, TwoScatterUpdateTestCommon,
                         ::testing::Values(LargeMishTestParams{
                                 {1, 64, 32, 514}  // in_shape
                         }),
                         TwoScatterUpdateTestCommon::getTestCaseName);
}  // namespace ov::test
