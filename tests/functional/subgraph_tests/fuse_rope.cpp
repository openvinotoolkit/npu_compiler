// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/rotary_positional_embeddings.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {
struct RoPEParams {
    ov::Shape inputShape;
    ov::Shape inputCosShape;
    ov::Shape inputSinShape;
    bool isInterleaved;
};

class FuseRoPETestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RoPEParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RoPEParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;

        const auto& params = obj.param;  // Access the RoPEParams instance
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "IS={" << params.inputShape << "}" << sep;
        result << "ICosS={" << params.inputCosShape << "}" << sep;
        result << "ISin={" << params.inputSinShape << "}" << sep;
        result << "IsInterleaved=" << (params.isInterleaved ? "true" : "false");

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

    // Pattern: Multiply1(input*cos) -> Slice(input:2) -> Multiply2(Slice:1 *(-1)) -> Concat(Multiply2, Slice:0) ->
    //          Multiply3(Concat*sin) -> Add(Multiply1,Multiply3)

    std::shared_ptr<ov::Node> buildRoPE(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& inputSin,
                                        const ov::Shape& inputShape) {
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
        return multiply3;
    }

    // Pattern: Multiply1(input*cos) -> Reshape1(input) -> Slice(Reshape:2) -> Reshape2(Slice:1)
    //          -> Multiply2(Reshape2*(-1)) -> Reshape3(Multiply2) ->Concat(Reshape3, Slice:1)
    //          -> Reshape4(Concat) ->  Multiply3(Reshape4*sin) -> Add(Multiply1,Multiply3)

    std::shared_ptr<ov::Node> buildRoPEInterleaved(const ov::Output<ov::Node>& input,
                                                   const ov::Output<ov::Node>& inputSin, const ov::Shape& inputShape) {
        size_t N = inputShape[0];
        size_t C = inputShape[1];
        size_t H = inputShape[2];
        size_t W = inputShape[3];

        // Reshape to 5D for Split
        const ov::Shape targetShape1 = {N, C, H, W / 2, 2};
        const auto shapeConst1 =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{targetShape1.size()}, targetShape1);
        const auto reshape1 = std::make_shared<ov::op::v1::Reshape>(input, shapeConst1, true);

        // Split W in 2 interleaved sections
        const int64_t axisValue = 4;
        const int64_t numSplits = 2;
        const auto split = std::make_shared<ov::op::v1::Split>(
                reshape1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axisValue}), numSplits);
        // Reshape
        const ov::Shape targetShape2 = {N, C, H, W / 2};
        const auto shapeConst2 =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{targetShape2.size()}, targetShape2);
        const auto reshape2 = std::make_shared<ov::op::v1::Reshape>(split->output(1), shapeConst2, true);

        // Negate odd-row elements
        const auto constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {-1.0f});
        const auto multiply2 = std::make_shared<ov::op::v1::Multiply>(reshape2, constant);

        // Reshape
        const ov::Shape targetShape3 = {N, C, H, W / 2, 1};
        const auto shapeConst3 =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{targetShape3.size()}, targetShape3);
        const auto reshape3 = std::make_shared<ov::op::v1::Reshape>(multiply2, shapeConst3, true);

        // Concat
        const auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector({reshape3, split->output(0)}), 4);

        // Reshape
        const ov::Shape targetShape4 = {N, C, H, W};
        const auto shapeConst4 =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{targetShape4.size()}, targetShape4);
        const auto reshape4 = std::make_shared<ov::op::v1::Reshape>(concat, shapeConst4, true);

        // Concat * Sin
        const auto multiply3 = std::make_shared<ov::op::v1::Multiply>(reshape4, inputSin);
        return multiply3;
    }

    void SetUp() override {
        inType = outType = ov::element::f32;
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;
        const auto inputCosShape = testParams.inputCosShape;
        const auto inputSinShape = testParams.inputSinShape;
        const auto isInterleaved = testParams.isInterleaved;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape, inputCosShape, inputSinShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputCos = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputSin = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(2));

        // Input * cos
        const auto multiply1 = std::make_shared<ov::opset1::Multiply>(input, inputCos);

        // Concat * Sin
        const auto multiply3 = isInterleaved ? buildRoPEInterleaved(input, inputSin, inputShape)
                                             : buildRoPE(input, inputSin, inputShape);

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

const std::vector<RoPEParams> precommit_testValues = {{{1, 32, 32, 96}, {1, 1, 32, 96}, {1, 1, 32, 96}, false},
                                                      {{1, 1, 256, 80}, {1, 1, 256, 80}, {1, 1, 256, 80}, true}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRoPE, FuseRoPETestCommon, ::testing::ValuesIn(precommit_testValues),
                         FuseRoPETestCommon::getTestCaseName);

const std::vector<RoPEParams> smoke_testValues = {{{1, 512, 18, 80}, {1, 512, 1, 80}, {1, 512, 1, 80}, false},
                                                  {{2, 1, 256, 64}, {2, 1, 256, 64}, {2, 1, 256, 64}, true}};
INSTANTIATE_TEST_SUITE_P(smoke_FuseRoPE, FuseRoPETestCommon, ::testing::ValuesIn(smoke_testValues),
                         FuseRoPETestCommon::getTestCaseName);
}  // namespace ov::test
