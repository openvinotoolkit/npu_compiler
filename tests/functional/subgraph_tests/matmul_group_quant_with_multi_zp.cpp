//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <random>
#include <vector>

namespace {
struct GroupQuantShapes {
    const ov::Shape _lhsShape;
    const ov::Shape _weightShape;
    const ov::Shape _scaleShape;
    const ov::Shape _zpShape;
    const ov::Shape _rhsShape;
    const bool _transposeB;
};
using GroupQuantParams = std::tuple<ov::element::Type, ov::element::Type, ov::element::Type, GroupQuantShapes>;
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

        auto createAndFillTensor = [](ov::Shape inputStaticShape, ov::element::Type elemType) -> ov::Tensor {
            const auto totalSize =
                    std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());

            ov::Tensor inputTensor;
            if (elemType == ov::element::f32 || elemType == ov::element::f16) {
                inputTensor = ov::test::utils::create_and_fill_tensor_real_distribution(elemType, inputStaticShape,
                                                                                        -1.0f, 1.0f, 7235346);
            } else if (elemType == ov::element::u4 || elemType == ov::element::u2) {
                inputTensor = ov::test::utils::create_and_fill_tensor(elemType, inputStaticShape);
            } else {
                OPENVINO_THROW("Unsupported element type: {0}", elemType);
            }

            return inputTensor;
        };

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            const auto& inputElemType = funcInput.get_element_type();
            auto tensor = createAndFillTensor(inputShapes[i], inputElemType);
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
        /* creates subgraph

        weights(as arg)   QuantZp
                |            |
         Convert(fp16)   Convert(fp16)
                    \     /
                    Subtract QuantScale
                       |    /
                    Multiply
                       |
                    Reshape
                       |
           Input    Convert(optional)
               \    /
               Matmul
                 |
               Output
        */
        const auto& [_matmulOpType, _weightsType, _zpType, shapes] = GetParam();

        const std::vector<ov::Shape> lhsInferenceShapes = {shapes._lhsShape};
        const ov::test::InputShape lhsShape = {shapes._lhsShape, lhsInferenceShapes};
        const std::vector<ov::Shape> wInferenceShapes = {shapes._weightShape};
        const ov::test::InputShape wShape = {shapes._weightShape, wInferenceShapes};
        init_input_shapes({lhsShape, wShape});

        std::mt19937 intMersenneEngine(0);
        std::uniform_int_distribution<> uint4Dist(0, 15);
        std::uniform_int_distribution<> uint2Dist(0, 3);
        const auto& zpType = _zpType;  // Lambdas could not capture structured binding members prior to C++20.
        auto zpGen = [&]() {
            if (zpType == ov::element::u4) {
                return static_cast<int8_t>(std::round(uint4Dist(intMersenneEngine)));
            } else if (zpType == ov::element::u2) {
                return static_cast<int8_t>(std::round(uint2Dist(intMersenneEngine)));
            } else {
                OPENVINO_THROW("unsupported zp type!");
            }
        };

        std::mt19937 floatMersenneEngine(0);
        std::uniform_real_distribution<float> uniformDist(-0.1f, 0.1f);
        auto scaleGen = [&]() {
            return uniformDist(floatMersenneEngine);
        };

        const auto input = std::make_shared<ov::opset1::Parameter>(_matmulOpType, shapes._lhsShape);
        const auto weights = std::make_shared<ov::opset1::Parameter>(_weightsType, shapes._weightShape);
        const auto convert0 = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f16);

        std::vector<int8_t> zeroPoints(ov::shape_size(shapes._zpShape));
        std::generate(zeroPoints.begin(), zeroPoints.end(), zpGen);
        const auto quantZp = std::make_shared<ov::opset1::Constant>(_zpType, shapes._zpShape, zeroPoints);
        const auto convert1 = std::make_shared<ov::opset1::Convert>(quantZp->output(0), ov::element::f16);
        const auto sub = std::make_shared<ov::opset1::Subtract>(convert0->output(0), convert1->output(0));

        std::vector<float> scales(ov::shape_size(shapes._scaleShape));
        std::generate(scales.begin(), scales.end(), scaleGen);
        const auto quantScale = std::make_shared<ov::opset1::Constant>(ov::element::f16, shapes._scaleShape, scales);
        const auto mul = std::make_shared<ov::opset1::Multiply>(sub->output(0), quantScale->output(0));

        std::vector<int64_t> shapePatternValues(shapes._rhsShape.begin(), shapes._rhsShape.end());
        const auto shapePattern = std::make_shared<ov::opset1::Constant>(
                ov::element::i64, ov::Shape({shapes._rhsShape.size()}), shapePatternValues);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(mul->output(0), shapePattern, false);
        const auto convert2 = std::make_shared<ov::opset1::Convert>(reshape->output(0), _matmulOpType);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(
                input->output(0), _matmulOpType == ov::element::f16 ? reshape->output(0) : convert2->output(0), false,
                shapes._transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, weights},
                                               "MatMulGroupQuantWithMultiZp");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GroupQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [_matmulOpType, _weightsType, _zpType, shapes] = obj.param;
        result << "WeightsType=" << _weightsType << sep;
        result << "MatmulOpType=" << _matmulOpType << sep;
        result << "ZpType=" << _zpType << sep;
        result << "InShape=" << shapes._lhsShape << sep;
        result << "WeightShape=" << shapes._weightShape << sep;
        result << "ScaleShape=" << shapes._scaleShape << sep;
        result << "ZpShape=" << shapes._zpShape << sep;
        result << "RhsShape=" << shapes._rhsShape;
        return result.str();
    };
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

const std::vector<GroupQuantShapes> testShapes = {
        /*case1=*/{/*_lhsShape=*/{1, 16, 3072},
                   /*_weightShape=*/{3072, 48, 64},
                   /*_scaleShape=*/{3072, 48, 1},
                   /*_zpShape=*/{3072, 48, 1},
                   /*_rhsShape=*/{3072, 3072},
                   /*_transposeB=*/true}};

INSTANTIATE_TEST_SUITE_P(GroupQuantWithMultiZp_FP16_U4, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f16), ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::u4), ::testing::ValuesIn(testShapes)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantWithMultiZp_FP32_U4, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f32), ::testing::Values(ov::element::u4),
                                            ::testing::Values(ov::element::u4), ::testing::ValuesIn(testShapes)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantWithMultiZp_FP16_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f16), ::testing::Values(ov::element::u2),
                                            ::testing::Values(ov::element::u2), ::testing::ValuesIn(testShapes)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupQuantWithMultiZp_FP32_U2, GroupQuantWithMultiZpTestCommon,
                         ::testing::Combine(::testing::Values(ov::element::f32), ::testing::Values(ov::element::u2),
                                            ::testing::Values(ov::element::u2), ::testing::ValuesIn(testShapes)),
                         GroupQuantWithMultiZpTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
