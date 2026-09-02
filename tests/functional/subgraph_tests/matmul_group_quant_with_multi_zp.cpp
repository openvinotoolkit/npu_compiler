//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <vector>

namespace {
struct GroupQuantShapes {
    const ov::Shape _lhsShape;
    const ov::Shape _weightShape;
    const ov::Shape _scaleShape;
    const ov::Shape _zpShape;
    const ov::Shape _rhsShape;
    const bool _transposeB = false;
    const bool _isScaleDynamic = false;
    const bool _isGroupOutermost = false;
};
using GroupQuantParams = std::tuple<ov::element::Type, GroupQuantShapes>;
}  // namespace

namespace ov::test::subgraph {

class GroupQuantWithMultiZpTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<GroupQuantParams> {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == funcInputs.size(),
                        "Input shapes number {0} does not match with inputs number {1}", inputShapes.size(),
                        funcInputs.size());

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto elemType = funcInput.get_element_type();
            ov::Tensor tensor;
            if (elemType == ov::element::f32) {
                tensor = ov::test::utils::create_and_fill_tensor_real_distribution(elemType, inputShapes[i], -1.0f,
                                                                                   1.0f, /*seed=*/7235346);
            } else if (elemType == ov::element::u4 || elemType == ov::element::u2) {
                tensor = ov::test::utils::create_and_fill_tensor(elemType, inputShapes[i]);
            } else {
                OPENVINO_THROW("Unsupported element type: {0}", elemType);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

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
        const auto& [weightsType, shapes] = GetParam();

        if (shapes._isScaleDynamic) {
            configuration["NPU_COMPILER_DYNAMIC_QUANTIZATION"] = "YES";
        }

        const ov::test::InputShape lhsInputShape = {shapes._lhsShape, {shapes._lhsShape}};
        const ov::test::InputShape wInputShape = {shapes._weightShape, {shapes._weightShape}};
        if (shapes._isScaleDynamic) {
            const ov::test::InputShape scaleInputShape = {shapes._scaleShape, {shapes._scaleShape}};
            init_input_shapes({lhsInputShape, wInputShape, scaleInputShape});
        } else {
            init_input_shapes({lhsInputShape, wInputShape});
        }

        const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shapes._lhsShape);
        const auto weights = std::make_shared<ov::opset1::Parameter>(weightsType, shapes._weightShape);
        const auto weightsF32 = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f32);

        const uint32_t maxZpVal = (weightsType == ov::element::u2) ? 3u : 15u;
        const auto zpTensor = ov::test::utils::create_and_fill_tensor(weightsType, shapes._zpShape,
                                                                      ov::test::utils::InputGenerateData(0, maxZpVal));
        const auto zpConst = std::make_shared<ov::opset1::Constant>(weightsType, shapes._zpShape, zpTensor.data());
        const auto zpF32 = std::make_shared<ov::opset1::Convert>(zpConst->output(0), ov::element::f32);

        const auto sub = std::make_shared<ov::opset1::Subtract>(weightsF32->output(0), zpF32->output(0));

        ov::Output<ov::Node> scaleOutput;
        ov::ParameterVector funcParams{input, weights};

        if (shapes._isScaleDynamic) {
            const auto scaleParam = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shapes._scaleShape);
            funcParams.push_back(scaleParam);

            if (shapes._scaleShape.size() == 2) {
                const ov::Shape scale3d = {shapes._scaleShape[0], shapes._scaleShape[1], 1};
                const auto reshapeConst = std::make_shared<ov::opset1::Constant>(
                        ov::element::i64, ov::Shape{3}, std::vector<int64_t>(scale3d.begin(), scale3d.end()));
                scaleOutput =
                        std::make_shared<ov::opset1::Reshape>(scaleParam->output(0), reshapeConst, false)->output(0);
            } else {
                scaleOutput = scaleParam->output(0);
            }
        } else {
            const auto scaleTensor = ov::test::utils::create_and_fill_tensor_real_distribution(
                    ov::element::f32, shapes._scaleShape, -0.1f, 0.1f, /*seed=*/1);
            scaleOutput =
                    std::make_shared<ov::opset1::Constant>(ov::element::f32, shapes._scaleShape, scaleTensor.data())
                            ->output(0);
        }

        const auto mul = std::make_shared<ov::opset1::Multiply>(sub->output(0), scaleOutput);

        ov::Output<ov::Node> weightNode = mul->output(0);
        if (shapes._isGroupOutermost) {
            const auto transposeOrder = std::make_shared<ov::opset1::Constant>(ov::element::i32, ov::Shape{3},
                                                                               std::vector<int32_t>{1, 0, 2});
            weightNode = std::make_shared<ov::opset1::Transpose>(mul->output(0), transposeOrder->output(0))->output(0);
        }

        // Reshape to 2D weight matrix
        const auto reshapeConst = std::make_shared<ov::opset1::Constant>(
                ov::element::i64, ov::Shape{shapes._rhsShape.size()},
                std::vector<int64_t>(shapes._rhsShape.begin(), shapes._rhsShape.end()));
        const auto reshape = std::make_shared<ov::opset1::Reshape>(weightNode, reshapeConst, false);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(input->output(0), reshape->output(0),
                                                                 /*transposeA=*/false, shapes._transposeB);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul)},
                                               funcParams, "GroupQuantWithMultiZp");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GroupQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [weightsType, shapes] = obj.param;
        result << "WeightsType=" << weightsType << sep;
        result << "InShape=" << shapes._lhsShape << sep;
        result << "WeightShape=" << shapes._weightShape << sep;
        result << "ScaleShape=" << shapes._scaleShape << sep;
        result << "ZpShape=" << shapes._zpShape << sep;
        result << "RhsShape=" << shapes._rhsShape;
        if (shapes._isScaleDynamic) {
            result << sep << "DynScale";
        }
        if (shapes._isGroupOutermost) {
            result << sep << "WithTranspose";
        }
        return result.str();
    }
};

//
// Platform test definition
//

TEST_P(GroupQuantWithMultiZpTestCommon, NPU4000_DebugTestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(GroupQuantWithMultiZpTestCommon, NPU5010_DebugTestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// --------------------------------------------------------------------------
// Static-scale tests
// --------------------------------------------------------------------------

const std::vector<GroupQuantShapes> testShapesStaticScalePrefill = {
        /*case1=*/{/*_lhsShape=*/{1, 16, 3072},
                   /*_weightShape=*/{64, 48, 64},
                   /*_scaleShape=*/{64, 48, 1},
                   /*_zpShape=*/{64, 48, 1},
                   /*_rhsShape=*/{64, 3072},
                   /*_transposeB=*/true}};

const std::vector<GroupQuantShapes> testShapesStaticScaleDecode = {
        /*case1=*/{/*_lhsShape=*/{1, 1, 3072},
                   /*_weightShape=*/{64, 48, 64},
                   /*_scaleShape=*/{64, 48, 1},
                   /*_zpShape=*/{64, 48, 1},
                   /*_rhsShape=*/{64, 3072},
                   /*_transposeB=*/true}};

INSTANTIATE_TEST_SUITE_P(GroupQuantStaticScaleWithMultiZp_Prefill_U4, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u4),
                                            ::testing::ValuesIn(testShapesStaticScalePrefill)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantStaticScaleWithMultiZp_Prefill_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(testShapesStaticScalePrefill)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantStaticScaleWithMultiZp_Decode_U4, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u4),
                                            ::testing::ValuesIn(testShapesStaticScaleDecode)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantStaticScaleWithMultiZp_Decode_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(testShapesStaticScaleDecode)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

// --------------------------------------------------------------------------
// Dynamic-scale tests
// --------------------------------------------------------------------------

const std::vector<GroupQuantShapes> testShapesDynamicScaleDecode = {{/*_lhsShape=*/{1, 1, 1024},
                                                                     /*_weightShape=*/{16, 64, 64},
                                                                     /*_scaleShape=*/{16, 64},
                                                                     /*_zpShape=*/{16, 64, 1},
                                                                     /*_rhsShape=*/{64, 1024},
                                                                     /*_transposeB=*/true,
                                                                     /*_isScaleDynamic=*/true,
                                                                     /*_isGroupOutermost=*/true}};

const std::vector<GroupQuantShapes> testShapesDynamicScalePrefill = {{/*_lhsShape=*/{1, 16, 1024},
                                                                      /*_weightShape=*/{16, 64, 64},
                                                                      /*_scaleShape=*/{16, 64},
                                                                      /*_zpShape=*/{16, 64, 1},
                                                                      /*_rhsShape=*/{64, 1024},
                                                                      /*_transposeB=*/true,
                                                                      /*_isScaleDynamic=*/true,
                                                                      /*_isGroupOutermost=*/true}};

INSTANTIATE_TEST_SUITE_P(GroupQuantDynamicScaleWithMultiZp_Decode_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(testShapesDynamicScaleDecode)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantDynamicScaleWithMultiZp_Prefill_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::u2),
                                            ::testing::ValuesIn(testShapesDynamicScalePrefill)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
