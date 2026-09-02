//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <random>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"
struct TestParams {
    size_t numChannels;
    size_t numGroupsPerChannel;
    size_t groupSize;
};
namespace ov::test::subgraph {

class MatMulGroupQuantLowPrecZpTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<TestParams> {
public:
    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto& expected = expectedTensors[0];
        const auto& actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], 2, -1, 1000);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        /* creates subgraph
                 cst zp  cst weights
                    |       |
                 Convert   Convert
                     \     /
      const scale    Subtract
               \     /
              Multiply    Input
                  |        |
               Reshape   Convert
                    \    /
                    Matmul
                      |
                    Output
        */
        const auto testParams = GetParam();
        const ov::Shape inputShape{1, 1, (testParams.numGroupsPerChannel * testParams.groupSize)};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));

        const ov::Shape weightsShape{testParams.numChannels, testParams.numGroupsPerChannel, testParams.groupSize};
        const ov::Shape scaleShape{testParams.numChannels, testParams.numGroupsPerChannel, 1};
        const ov::Shape zpShape{testParams.numChannels, testParams.numGroupsPerChannel, 1};

        constexpr int quantMin = 0;
        constexpr int quantMax = 3;  // U2: values 0..3
        constexpr int bitsPerElement = 2;
        constexpr int elementsPerByte = 4;
        constexpr uint8_t elementMask = 0x3;
        constexpr float minRangeWidth = 1e-6f;

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

        auto weightsTensor = ov::Tensor(ov::element::u2, weightsShape);
        auto scaleTensor = ov::Tensor(ov::element::f16, scaleShape);
        auto zpTensor = ov::Tensor(ov::element::u2, zpShape);

        auto* weightsData = static_cast<uint8_t*>(weightsTensor.data());
        auto* scaleData = static_cast<ov::float16*>(scaleTensor.data());
        auto* zpData = static_cast<uint8_t*>(zpTensor.data());

        std::memset(weightsData, 0, weightsTensor.get_byte_size());
        std::memset(zpData, 0, zpTensor.get_byte_size());

        for (size_t channelIdx = 0; channelIdx < testParams.numChannels; ++channelIdx) {
            for (size_t groupIdx = 0; groupIdx < testParams.numGroupsPerChannel; ++groupIdx) {
                // Generate random subrange [rangeLo, rangeHi] within [-1, 1]
                const float firstEndpoint = unitDist(gen) * 2.0f - 1.0f;
                const float secondEndpoint = unitDist(gen) * 2.0f - 1.0f;
                const float rangeLo = std::min(firstEndpoint, secondEndpoint);
                const float rangeHi = [&] {
                    const float hi = std::max(firstEndpoint, secondEndpoint);
                    return (hi - rangeLo < minRangeWidth) ? rangeLo + 0.01f : hi;
                }();

                // Calculate scale and zero point for U2 quantization
                // Dequantization: floatVal = (quantized - zeroPoint) * scale
                const float scale = (rangeHi - rangeLo) / static_cast<float>(quantMax - quantMin);
                const float invScale = 1.0f / scale;
                const int zeroPoint = std::clamp(static_cast<int>(std::round(-rangeLo * invScale)), quantMin, quantMax);

                const size_t flatGroupIdx = static_cast<size_t>(channelIdx) * testParams.numGroupsPerChannel + groupIdx;
                scaleData[flatGroupIdx] = ov::float16(scale);

                // Store zero point (U2 packed, one per group)
                const size_t zpByteIdx = flatGroupIdx / elementsPerByte;
                const int zpBitOffset =
                        (elementsPerByte - 1 - static_cast<int>(flatGroupIdx % elementsPerByte)) * bitsPerElement;
                zpData[zpByteIdx] = static_cast<uint8_t>((zpData[zpByteIdx] & ~(elementMask << zpBitOffset)) |
                                                         ((zeroPoint & elementMask) << zpBitOffset));

                // Generate float weights in [rangeLo, rangeHi], quantize, store as U2
                const float range = rangeHi - rangeLo;
                for (size_t elemIdx = 0; elemIdx < testParams.groupSize; ++elemIdx) {
                    const float floatWeight = rangeLo + range * unitDist(gen);
                    const int quantVal = std::clamp(static_cast<int>(std::round(floatWeight * invScale + zeroPoint)),
                                                    quantMin, quantMax);

                    const size_t flatWeightIdx = flatGroupIdx * testParams.groupSize + elemIdx;
                    const size_t weightByteIdx = flatWeightIdx / elementsPerByte;
                    const int weightBitOffset =
                            (elementsPerByte - 1 - static_cast<int>(flatWeightIdx % elementsPerByte)) * bitsPerElement;
                    weightsData[weightByteIdx] =
                            static_cast<uint8_t>((weightsData[weightByteIdx] & ~(elementMask << weightBitOffset)) |
                                                 ((quantVal & elementMask) << weightBitOffset));
                }
            }
        }

        const auto weightsConst = std::make_shared<ov::op::v0::Constant>(weightsTensor);
        const auto scalesConst = std::make_shared<ov::op::v0::Constant>(scaleTensor);
        const auto zpConst = std::make_shared<ov::op::v0::Constant>(zpTensor);

        const auto convertWeights = std::make_shared<ov::opset1::Convert>(weightsConst->output(0), ov::element::f16);
        const auto convertZp = std::make_shared<ov::opset1::Convert>(zpConst->output(0), ov::element::f16);

        auto subtract = std::make_shared<ov::op::v1::Subtract>(convertWeights, convertZp);

        const auto multiply = std::make_shared<ov::opset1::Multiply>(subtract->output(0), scalesConst->output(0));

        const auto targetShape =
                std::vector<size_t>{testParams.numChannels, (testParams.numGroupsPerChannel * testParams.groupSize)};
        const auto targetShapeConst = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{targetShape.size()}, targetShape.data());
        const auto reshapedWeights = std::make_shared<ov::op::v1::Reshape>(multiply, targetShapeConst, true);

        const auto matmul =
                std::make_shared<ov::opset1::MatMul>(input->output(0), reshapedWeights->output(0), false, true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "MatMulGroupQuantLowPrecZp");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(MatMulGroupQuantLowPrecZpTestCommon, NPU5000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(MatMulGroupQuantLowPrecZp, MatMulGroupQuantLowPrecZpTestCommon,

                         ::testing::Values(TestParams{/*numChannels*/ 3072, /*numGroupsPerChannel*/ 128,
                                                      /*groupSize*/ 64}),
                         MatMulGroupQuantLowPrecZpTestCommon::getTestCaseName);

}  // namespace ov::test::subgraph
