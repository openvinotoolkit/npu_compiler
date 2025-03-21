// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/rotary_positional_embeddings.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {
struct RoPEParams {
    ov::Shape inputShape;
    ov::Shape inputCosShape;
    ov::Shape inputSinShape;
};

class FuseRoPETestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RoPEParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RoPEParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        for (size_t i = 0; i < 3; ++i) {
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(
                    funcInputs[i].get_element_type(), targetInputStaticShapes[i], 2, -1.0f, 32768);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void SetUp() override {
        inType = outType = ov::element::f32;
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;
        const auto inputCosShape = testParams.inputCosShape;
        const auto inputSinShape = testParams.inputSinShape;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape, inputCosShape, inputSinShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputCos = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputSin = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(2));

        // Input * cos
        const auto multiply1 = std::make_shared<ov::opset1::Multiply>(input, inputCos);

        std::vector<int64_t> begin_mask{1, 1, 1, 0};
        std::vector<int64_t> end_mask{1, 1, 1, 0};
        size_t width = inputShape[3] / 2;
        const ov::Shape beginShape = {0, 0, 0, width};

        // Strided slice of input on width ( W/2 -> W )
        const auto sliceBeginConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, beginShape.data());
        const auto sliceEndConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, inputShape.data());
        const auto sliceStridesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1});

        const auto stridedSlice_negativeHalf = std::make_shared<ov::op::v1::StridedSlice>(
                input, sliceBeginConst, sliceEndConst, sliceStridesConst, begin_mask, end_mask,
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        // Negate second half
        const auto multiplyConst = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {-1.0f});
        const auto multiply2 = std::make_shared<ov::opset1::Multiply>(stridedSlice_negativeHalf, multiplyConst);

        // Strided slice of input on width ( 0 -> W/2 )
        const auto sliceBeginConst2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0});
        const ov::Shape endShape = {inputShape[0], inputShape[1], inputShape[2], width};
        const auto sliceEndConst2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, endShape);

        const auto stridedSlice_positiveHalf = std::make_shared<ov::op::v1::StridedSlice>(
                input, sliceBeginConst2, sliceEndConst2, sliceStridesConst, begin_mask, end_mask,
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        // Concat negativeHalf + positiveHalf
        const auto concat =
                std::make_shared<ov::opset1::Concat>(ov::OutputVector({multiply2, stridedSlice_positiveHalf}), -1);

        // Concat * Sin
        const auto multiply3 = std::make_shared<ov::opset1::Multiply>(concat, inputSin);

        const auto add = std::make_shared<ov::op::v1::Add>(multiply1, multiply3);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, inputCos, inputSin}, "FuseRoPETest");
    }
};

TEST_P(FuseRoPETestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
TEST_P(FuseRoPETestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(precommit_FuseRoPE, FuseRoPETestCommon,
                         ::testing::ValuesIn({RoPEParams{{1, 32, 32, 96}, {1, 1, 32, 96}, {1, 1, 32, 96}}}),
                         FuseRoPETestCommon::getTestCaseName);
}  // namespace ov::test
