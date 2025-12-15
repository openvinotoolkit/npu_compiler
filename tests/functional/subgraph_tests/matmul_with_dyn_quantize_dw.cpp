//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"

namespace {
struct DynQuantShapes {
    const ov::Shape _arg0;
    const ov::Shape _arg1;
    const ov::Shape _arg2;
    const ov::Shape _arg3;
    const ov::Shape _arg4;
    const ov::Shape _arg5;
    const ov::Shape _arg6;
    const ov::Shape _arg7;
    const bool _transposeB;
};
using DynQuantParams = std::tuple<DynQuantShapes>;
}  // namespace

namespace ov::test::subgraph {

class MatMulWithDynQDWTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<DynQuantParams> {
public:
    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        /* creates subgraph
            - phi-3-mini-i4-sym_prefill_repeated_fold inspired subgraph
                                    arg4  arg1  arg2
                                    |       \  /
                                Convert2    Add0
                                        \    / |
                      arg5   Constant   Power /
                       |        \       /    /
                    Convert3   ReduceMean   /
                            \   /          /
                    arg3    Add1          /
                     |       |           /
                  Convert1  Sqrt        /
                      \     /          /
              arg0    Divide          /
                 \          \        /
         arg6   Convert0     Multiply0 -> this will be converted to ScaleShift and than NCE.DepthConvolution
             \          \      /
            Convert4     Multiply1
                \          |
             Multiply2     |
                   \       |
                Convert5   |
                      \   /
                      MatMul
                        |
                      Output
        */
        const auto& [shapes] = GetParam();

        const std::vector<ov::Shape> a0inferenceShapes = {shapes._arg0};
        const ov::test::InputShape a0dataShape = {shapes._arg0, a0inferenceShapes};
        const std::vector<ov::Shape> a1inferenceShapes = {shapes._arg1};
        const ov::test::InputShape a1dataShape = {shapes._arg1, a1inferenceShapes};
        const std::vector<ov::Shape> a2inferenceShapes = {shapes._arg2};
        const ov::test::InputShape a2dataShape = {shapes._arg2, a2inferenceShapes};
        const std::vector<ov::Shape> a3inferenceShapes = {shapes._arg3};
        const ov::test::InputShape a3dataShape = {shapes._arg3, a3inferenceShapes};
        const std::vector<ov::Shape> a4inferenceShapes = {shapes._arg4};
        const ov::test::InputShape a4dataShape = {shapes._arg4, a4inferenceShapes};
        const std::vector<ov::Shape> a5inferenceShapes = {shapes._arg5};
        const ov::test::InputShape a5dataShape = {shapes._arg5, a5inferenceShapes};
        const std::vector<ov::Shape> a6inferenceShapes = {shapes._arg6};
        const ov::test::InputShape a6dataShape = {shapes._arg6, a6inferenceShapes};
        const std::vector<ov::Shape> a7inferenceShapes = {shapes._arg7};
        const ov::test::InputShape a7dataShape = {shapes._arg7, a7inferenceShapes};
        init_input_shapes({a0dataShape, a1dataShape, a2dataShape, a3dataShape, a4dataShape, a5dataShape, a6dataShape,
                           a7dataShape});

        const auto param0 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto param1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(1));
        const auto param2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(2));
        const auto param3 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(3));
        const auto param4 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(4));
        const auto param5 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(5));
        const auto param6 = std::make_shared<ov::opset1::Parameter>(ov::element::i4, inputDynamicShapes.at(6));
        const auto param7 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(7));

        const auto convert0 = std::make_shared<ov::opset1::Convert>(param0->output(0), ov::element::f32);

        const auto add0 = std::make_shared<ov::opset1::Add>(param1->output(0), param2->output(0),
                                                            ov::op::AutoBroadcastType::NUMPY);

        const auto convert1 = std::make_shared<ov::opset1::Convert>(param3->output(0), ov::element::f32);
        const auto convert2 = std::make_shared<ov::opset1::Convert>(param4->output(0), ov::element::f32);

        const auto power = std::make_shared<ov::op::v1::Power>(add0->output(0), convert2->output(0));

        const auto cst = ov::opset1::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
        const auto reducemean = std::make_shared<ov::opset1::ReduceMean>(power->output(0), cst->output(0), true);

        const auto convert3 = std::make_shared<ov::opset1::Convert>(param5->output(0), ov::element::f32);

        const auto add1 = std::make_shared<ov::opset1::Add>(reducemean->output(0), convert3->output(0),
                                                            ov::op::AutoBroadcastType::NUMPY);

        const auto sqrt = std::make_shared<ov::opset1::Sqrt>(add1->output(0));

        const auto divide = std::make_shared<ov::opset1::Divide>(convert1->output(0), sqrt->output(0),
                                                                 ov::op::AutoBroadcastType::NUMPY);

        const auto mul0 = std::make_shared<ov::opset1::Multiply>(add0->output(0), divide->output(0));

        const auto mul1 = std::make_shared<ov::opset1::Multiply>(convert0->output(0), mul0->output(0));

        const auto convert4 = std::make_shared<ov::opset1::Convert>(param6->output(0), ov::element::f16);

        const auto mul2 = std::make_shared<ov::opset1::Multiply>(convert4->output(0), param7->output(0));

        const auto convert5 = std::make_shared<ov::opset1::Convert>(mul2->output(0), ov::element::f32);

        const auto matmul =
                std::make_shared<ov::opset1::MatMul>(mul1->output(0), convert5->output(0), false, shapes._transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(
                results, ov::ParameterVector{param0, param1, param2, param3, param4, param5, param6, param7},
                "MatMulWithDynQDW");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DynQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [shapes] = obj.param;
        result << "InputShape=" << shapes._arg0;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(MatMulWithDynQDWTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulWithDynQDWTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulWithDynQDWTestCommon, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
const std::vector<DynQuantShapes> shapes = {
        /*case1=*/{
                /*_arg0=*/{1, 1, 3072},
                /*_arg1=*/{1, 1024, 3072},
                /*_arg2=*/{1, 1024, 3072},
                /*_arg3=*/{1, 1, 1},
                /*_arg4=*/{1, 1, 1},
                /*_arg5=*/{1, 1, 1},
                /*_arg6=*/{9216, 3072},
                /*_arg7=*/{9216, 1},
                /*_transposeB=*/true,
        }};

// Tracking number [E#144857]
INSTANTIATE_TEST_SUITE_P(DISABLED_MatMulWithDynQDW, MatMulWithDynQDWTestCommon,
                         ::testing::Combine(::testing::ValuesIn(shapes)), MatMulWithDynQDWTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
