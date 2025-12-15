//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

struct MatMulPerChannelParams {
    size_t batchSize;    // numConv
    size_t numRowsA;     // numRowsA
    size_t numColumnsA;  // numColsA
    size_t numRowsB;
    size_t numColsB;
    bool transposeA;
    bool transposeB;
    std::vector<float> zeroPoints;  // Per-channel zero points
    std::vector<float> scales;      // Per-channel scales
};

class MatMulWithAsymZeroPointPerChannel :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MatMulPerChannelParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulPerChannelParams> obj) {
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

        const float absThreshold = 0.01f;  // Default
        // Default is 0.001, this optimization introduces small inaccuracy due to extra calculations with fp16
        const float relThreshold = 0.03f;
        ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
    }

    void SetUp() override {
        const auto testParams = GetParam();
        const std::vector<ov::Shape> inferenceShapes = {
                {testParams.batchSize, testParams.numRowsA, testParams.numColumnsA}};
        const ov::test::InputShape dataShape = {{testParams.batchSize, testParams.numRowsA, testParams.numColumnsA},
                                                inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto weightShape = ov::Shape{testParams.numRowsB, testParams.numColsB};
        const auto weightTotalSize = ov::shape_size(weightShape);
        std::vector<uint8_t> weightsData(weightTotalSize, 0);

        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = 129 + i;
        }
        const auto weights = ov::opset1::Constant::create(ov::element::u8, weightShape, weightsData);
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f16);

        using scaleShiftValueType = ov::element_type_traits<ov::element::f16>::value_type;
        const auto zeroPointShape =
                testParams.transposeB ? ov::Shape{testParams.numRowsB, 1} : ov::Shape{1, testParams.numColsB};
        const auto zeroPoints = ov::opset1::Constant::create(ov::element::f16, zeroPointShape, testParams.zeroPoints);
        const auto shift = std::make_shared<ov::opset1::Subtract>(convert->output(0), zeroPoints->output(0));

        const auto scaleShiftShape =
                testParams.transposeB ? ov::Shape{testParams.numRowsB, 1} : ov::Shape{1, testParams.numColsB};
        const auto scales = ov::opset1::Constant::create(ov::element::f16, scaleShiftShape, testParams.scales);
        const auto mul = std::make_shared<ov::opset1::Multiply>(shift->output(0), scales->output(0));

        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), mul->output(0),
                                                                 testParams.transposeA, testParams.transposeB);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MatMulWithFQ");
    }
};

TEST_P(MatMulWithAsymZeroPointPerChannel, NPU3720_HW) {
    setDefaultHardwareMode();
    configuration[ov::intel_npu::compilation_mode_params.name()] = "matmul-mixed-precision-decomposition-ratio=0.5";
    run(Platform::NPU3720);
}

TEST_P(MatMulWithAsymZeroPointPerChannel, NPU4000_HW) {
    setDefaultHardwareMode();
    configuration[ov::intel_npu::compilation_mode_params.name()] = "matmul-mixed-precision-decomposition-ratio=0.5";
    run(Platform::NPU4000);
}

TEST_P(MatMulWithAsymZeroPointPerChannel, NPU5010_HW) {
    setDefaultHardwareMode();
    configuration[ov::intel_npu::compilation_mode_params.name()] = "matmul-mixed-precision-decomposition-ratio=0.5";
    run(Platform::NPU5010);
}

namespace {
using namespace ov::test::subgraph;

std::vector<float> generateZeroPoints(size_t size) {
    std::vector<float> zeroPoints(size);
    for (size_t i = 0; i < size; i++) {
        zeroPoints[i] = 50.0f + static_cast<float>(i % 200);
    }
    return zeroPoints;
}

std::vector<float> generateScales(size_t size) {
    std::vector<float> scales(size);
    for (size_t i = 0; i < size; i++) {
        scales[i] = 0.1f + (i % 2) * 0.1f;
    }
    return scales;
}

const std::vector<MatMulPerChannelParams> allParams = {
        // These parameters are disabled until functional test flags are fixed, as they need lower
        // matmul-mixed-precision-decomposition-ratio

        // MatMulPerChannelParams{1, 4, 320, 255, 320, false, true, generateZeroPoints(255), generateScales(255)},
        // MatMulPerChannelParams{2, 16, 32, 32, 64, false, false, generateZeroPoints(64), generateScales(64)},

        MatMulPerChannelParams{1, 1, 2048, 512, 2048, false, true, generateZeroPoints(512), generateScales(512)}

};

INSTANTIATE_TEST_SUITE_P(precommit_MatMulDecomposition, MatMulWithAsymZeroPointPerChannel,
                         ::testing::ValuesIn(allParams), MatMulWithAsymZeroPointPerChannel::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
