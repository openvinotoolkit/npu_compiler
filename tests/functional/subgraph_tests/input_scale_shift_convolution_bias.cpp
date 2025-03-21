//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

enum class ScalesMode {
    SPLAT,
    DIFFERENT,
    NO_SCALES,
};

std::string to_string(ScalesMode mode) {
    switch (mode) {
    case ScalesMode::SPLAT:
        return "SPLAT";
    case ScalesMode::DIFFERENT:
        return "DIFFERENT";
    case ScalesMode::NO_SCALES:
        return "NO_SCALES";
    }
    return "UNKNOWN";
}

// FuseInputScaleShift transformation replaces:
//       [input]        [Weights]
//          |              |
//      (Multiply)?        |
//          |              |
//        (Add)          (FQ2)
//          |              |
//        (FQ1)            |
//          |              |
//        (conv) --------- |
//          |
//        (Add) -------- [Bias]
//          |
//       [output]
// with:
//       [input]      [new Weights]
//          |              |
//       (new FQ1)      (new FQ2)
//          |              |
//        (conv) --------- |
//          |
//        (Add) -------- [new Bias]
//          |
//       [output]

class FuseInputScaleShiftCommon : public VpuOv2LayerTest, public testing::WithParamInterface<ScalesMode> {
private:
    float inCoeff = 0.2f;
    float absThreshold = 1.5f;

public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();

        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i) * inCoeff;
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

        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        configuration["NPU_COMPILER_TYPE"] = "MLIR";
        auto scalesMode = GetParam();
        // TODO: #151977 the threshold is quite big, which looks suspicious.
        // Moreover the test fails if the FuseInputScaleShift pass is disabled
        // The same was for nGraph implementation
        if (scalesMode != ScalesMode::NO_SCALES) {
            inCoeff = 2.f;
            absThreshold = 1.0f;
        }

        const ov::Shape inputShape{1, 3, 224, 224};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputShape);

        // Create weights FQ
        const ov::Shape weightsShape{16, 3, 3, 3};
        const auto weightTotalSize = ov::shape_size(weightsShape);

        auto rangeMin = -64;
        auto rangeMax = 63;
        const auto weightsI8 = ov::test::utils::make_constant(ov::element::i8, weightsShape,
                                                              ov::test::utils::InputGenerateData(rangeMin, rangeMax));
        const auto convert = std::make_shared<ov::opset1::Convert>(weightsI8->output(0), ov::element::f32);

        const ov::Shape scaleShape{16, 1, 1, 1};
        const auto scaleTotalSize = ov::shape_size(scaleShape);
        using ScaleShiftValueType = ov::element_type_traits<ov::element::f32>::value_type;

        std::vector<ScaleShiftValueType> scaleData(scaleTotalSize, 0);
        for (size_t i = 0; i < scaleData.size(); i++) {
            scaleData.at(i) = (i % 7) / 128.f;
        }
        const auto scales = ov::opset1::Constant::create(ov::element::f32, scaleShape, scaleData);
        const auto mul = std::make_shared<ov::opset1::Multiply>(convert->output(0), scales->output(0));

        // Create input ScaleShift
        const ov::Shape scaleShiftShape{1, 3, 1, 1};

        ov::Output<ov::Node> scaleOut = param->output(0);
        if (scalesMode != ScalesMode::NO_SCALES) {
            std::vector<ScaleShiftValueType> scaleShiftMultData{0.0174255371};
            if (scalesMode == ScalesMode::DIFFERENT) {
                scaleShiftMultData = {0.0174255371, 0.0175018311, 0.0170593262};
            }

            const auto scaleCst = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, scaleShiftMultData);
            scaleOut = std::make_shared<ov::opset1::Multiply>(param->output(0), scaleCst->output(0))->output(0);
        }

        std::vector<ScaleShiftValueType> scaleShiftAddData{-1.8046875, -2.03515625, -2.109375};
        const auto shiftCst = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, scaleShiftAddData);
        const auto shift = std::make_shared<ov::opset1::Add>(scaleOut, shiftCst->output(0));

        // Create input FakeQuantize
        const size_t dataLevels = 256;
        const std::vector<float> dataLow = {-2.527646541595459F};
        const std::vector<float> dataHigh = {2.507899284362793F};
        const auto inputFq = ov::test::utils::make_fake_quantize(shift->output(0), ov::element::f32, dataLevels, {},
                                                                 dataLow, dataHigh, dataLow, dataHigh);

        // Create Convolution
        const ov::Strides strides = {2, 2};
        const ov::CoordinateDiff padsBegin = {1, 1};
        const ov::CoordinateDiff padsEnd = {1, 1};
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(inputFq->output(0), mul->output(0), strides,
                                                                    padsBegin, padsEnd, dilations);

        // Create bias
        const ov::Shape biasShape{1, 16, 1, 1};
        const auto biasTotalSize = ov::shape_size(biasShape);
        std::vector<ScaleShiftValueType> biasData(biasTotalSize, 0);
        for (size_t i = 0; i < biasTotalSize; i++) {
            biasData.at(i) = i % 32;
        }

        const auto biasCst = ov::opset1::Constant::create(ov::element::f32, biasShape, biasData);
        const auto bias = std::make_shared<ov::opset1::Add>(conv->output(0), biasCst->output(0));

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(bias)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "FuseInputScaleShift");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ScalesMode>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "ScalesMode=" << to_string(obj.param) << sep;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(FuseInputScaleShiftCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseInputScaleShiftCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
const std::vector<ScalesMode> scalesModes = {ScalesMode::DIFFERENT, ScalesMode::SPLAT, ScalesMode::NO_SCALES};

INSTANTIATE_TEST_SUITE_P(FuseInputScaleShift, FuseInputScaleShiftCommon, ::testing::ValuesIn(scalesModes),
                         FuseInputScaleShiftCommon::getTestCaseName);

}  // namespace ov::test::subgraph
