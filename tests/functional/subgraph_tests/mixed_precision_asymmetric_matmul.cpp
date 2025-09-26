//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

struct MatMulParams {
    size_t batchSize;
    size_t numRowsA;
    size_t numColumnsA;
    size_t numRowsB;
    size_t numColsB;
    bool transposeA;
    bool transposeB;
    float zeroPoint;
    float scale;
};

class MixedPrecisionAsymmetricPerTensorMatmul :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MatMulParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "Batch=" << obj.param.batchSize << sep;
        result << "RowA=" << obj.param.numRowsA << sep;
        result << "ColA=" << obj.param.numColumnsA << sep;
        result << "RowB=" << obj.param.numRowsB << sep;
        result << "ColB=" << obj.param.numColsB << sep;
        result << (obj.param.transposeA ? "transA" : "noTransA") << sep;
        result << (obj.param.transposeB ? "transB" : "noTransB");
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::floor(10.f * std::sin(i));
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());
        ov::test::utils::compare(expected, actual);
    }

    void SetUp() override {
        const auto testParams = GetParam();
        const std::vector<ov::Shape> inferenceShapes = {{1, testParams.batchSize, testParams.numColumnsA}};
        const ov::test::InputShape dataShape = {{1, testParams.batchSize, testParams.numColumnsA}, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto weightShape = ov::Shape{testParams.numRowsB, testParams.numColsB};
        const auto weightTotalSize = ov::shape_size(weightShape);
        std::vector<ov::element_type_traits<ov::element::u8>::value_type> weightsData(weightTotalSize, 0);
        const float zeroPoint = testParams.zeroPoint;
        auto idx = 0;
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % 256;
            ++idx;
        }
        const auto weights = ov::opset1::Constant::create(ov::element::u8, weightShape, weightsData);
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f16);

        const auto scaleShiftShape = ov::Shape{1, 1};
        const auto scaleShiftTotalSize = ov::shape_size(scaleShiftShape);
        using scaleShiftValueType = ov::element_type_traits<ov::element::f16>::value_type;
        const std::vector<scaleShiftValueType> zeroPointData(1, zeroPoint);
        const auto zeroPoints = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1}, zeroPointData);
        const auto shift = std::make_shared<ov::opset1::Subtract>(convert->output(0), zeroPoints->output(0));

        std::vector<scaleShiftValueType> scaleData(scaleShiftTotalSize, 0);
        for (size_t i = 0; i < scaleData.size(); i++) {
            scaleData.at(i) = testParams.scale;
        }
        const auto scales = ov::opset1::Constant::create(ov::element::f16, scaleShiftShape, scaleData);
        const auto mul = std::make_shared<ov::opset1::Multiply>(shift->output(0), scales->output(0));

        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), mul->output(0),
                                                                 testParams.transposeA, testParams.transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MixedPrecisionAsymmetricMatmul");
    }
};

}  // namespace ov::test::subgraph

namespace {
using namespace ov::test::subgraph;

const std::vector<MatMulParams> allParams = {MatMulParams{64, 49, 128, 384, 128, false, true, 255.f, 0.001f},
                                             MatMulParams{1, 32, 256, 256, 64, false, false, 0.f, 1.f},
                                             MatMulParams{32, 64, 256, 256, 128, false, false, 30.f, 1.f},
                                             MatMulParams{1, 1, 2048, 2048, 512, false, false, 30.f, 0.01f}};

INSTANTIATE_TEST_SUITE_P(precommit_MixedAsymMatmul, MixedPrecisionAsymmetricPerTensorMatmul,
                         ::testing::ValuesIn(allParams), MixedPrecisionAsymmetricPerTensorMatmul::getTestCaseName);

}  // namespace
