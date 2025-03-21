// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cmath>
#include <ov_ops/rms.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

using RMSNormDecompositionParams = std::tuple<ov::Shape,          // input shapes
                                              ov::element::Type,  // input precision
                                              bool>;              // fq stripped

class FuseRMSReduceSumTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<RMSNormDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
    void SetUp() override {
        const auto& [inputShapes, inputPrecision, fqStripped] = GetParam();
        inType = outType = inputPrecision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShapes}));
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes.front())};

        // x^2
        auto powerConst = ov::op::v0::Constant::create(inputPrecision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], powerConst);

        // ReduceSum(x^2,axes)
        auto sumAxes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(power, sumAxes, true);

        // Sqrt(ReduceSum(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(sum);

        // x/Sqrt(ReduceSum(x^2,axes))
        std::shared_ptr<ov::Node> div = std::make_shared<ov::op::v1::Divide>(params[0], sqrt);

        if (!fqStripped) {
            std::array<float, 2> range{};
            if (inputShapes.size() == 4) {
                range = {0.002, 0.04};
            } else {
                range = {0, 1};
            }

            auto lo = ov::op::v0::Constant::create(inputPrecision, {}, {range.at(0)});
            auto hi = ov::op::v0::Constant::create(inputPrecision, {}, {range.at(1)});
            div = std::make_shared<ov::op::v0::FakeQuantize>(div, lo, hi, lo, hi, 256);
        }

        // x/Sqrt(ReduceSum(x^2,axes)) * Sqrt(inputDim[axes])
        auto dim = *inputShapes.rbegin();
        auto mulValue = std::sqrt(dim);
        auto mulConst = ov::op::v0::Constant::create(inputPrecision, {1}, {mulValue});
        std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(mulConst, div);

        if (!fqStripped) {
            std::array<float, 2> range{};
            if (inputShapes.size() == 4) {
                range = {0.14, 1.75};
            } else {
                range = {0, 2.2};
            }

            auto lo = ov::op::v0::Constant::create(inputPrecision, {}, {range.at(0)});
            auto hi = ov::op::v0::Constant::create(inputPrecision, {}, {range.at(1)});
            mul = std::make_shared<ov::op::v0::FakeQuantize>(mul, lo, hi, lo, hi, 256);
        }

        auto comp = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f16);

        function = std::make_shared<ov::Model>(comp, params, "fuse_rms");
    }
};

class FuseRMSReduceSumTestUnstripped : public FuseRMSReduceSumTestCommon {};

TEST_P(FuseRMSReduceSumTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
TEST_P(FuseRMSReduceSumTestUnstripped, NPU3720_HW) {
    const float fqRange = 3, fqLevels = 256;
    abs_threshold = fqRange / fqLevels;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseRMSReduceSumTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(FuseRMSReduceSumTestUnstripped, NPU4000_HW) {
    const float fqRange = 3, fqLevels = 256;
    abs_threshold = fqRange / fqLevels;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
namespace {
const std::vector<ov::element::Type> inputPrecisions = {ov::element::f32};

const std::vector<ov::Shape> inputShapesBasic = {{{1, 2, 6}}, {{2, 2, 6}}};
const std::vector<ov::Shape> inputShapes = {{{1, 1, 512, 3072}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS_ReduceSum, FuseRMSReduceSumTestCommon,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBasic), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(true)),
                         FuseRMSReduceSumTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS_ReduceSum, FuseRMSReduceSumTestCommon,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(true)),
                         FuseRMSReduceSumTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS_ReduceSum, FuseRMSReduceSumTestUnstripped,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBasic), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(false)),
                         FuseRMSReduceSumTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS_ReduceSum, FuseRMSReduceSumTestUnstripped,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(false)),
                         FuseRMSReduceSumTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
