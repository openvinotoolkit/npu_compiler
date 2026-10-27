//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov::test {

using ov::element::Type;
using ov::op::PadMode;
using AttrType = std::vector<int32_t>;
using DynamicPadParamSet = std::tuple<InputShape,                   // Input shape
                                      Type,                         // Input element type
                                      AttrType,                     // padsBegin
                                      AttrType,                     // padsEnd
                                      float,                        // argPadValue
                                      std::vector<InputLayerType>,  // for {begin, end, padValue}
                                      PadMode>;                     // padMode

class DynamicPadLayerTest : public testing::WithParamInterface<DynamicPadParamSet>, virtual public VpuOv2LayerTest {
public:
    AttrType padsBegin, padsEnd;
    float argPadValue;

    static std::string getTestCaseName(testing::TestParamInfo<DynamicPadParamSet> obj) {
        const auto& [shapes, model_type, padsBegin, padsEnd, argPadValue, inputLayerTypes, padMode] = obj.param;

        std::ostringstream results;
        results << "TestKind" << testKind(__FILE__) << "_";
        results << "IS=" << partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << vec2str(item) << "_";
        }
        results << "Prc=" << model_type << "_";
        results << "padsBegin=" << vec2str(padsBegin) << "_";
        results << "padsEnd=" << vec2str(padsEnd) << "_";
        if (padMode == PadMode::CONSTANT) {
            results << "Value=" << argPadValue << "_";
        }
        results << "constantInput=" << inputLayerTypes[0] << "/" << inputLayerTypes[1] << "/" << inputLayerTypes[2]
                << "_";
        results << "PadMode=" << padMode;

        return results.str();
    }

    void SetUp() override {
        InputShape inShape;
        PadMode padMode;
        std::vector<InputLayerType> inputLayerTypes;
        Type inputType;
        std::tie(inShape, inputType, padsBegin, padsEnd, argPadValue, inputLayerTypes, padMode) = this->GetParam();
        auto inputShapes = std::vector<ov::test::InputShape>();

        inputShapes.push_back(inShape);                         // input
        if (inputLayerTypes[0] == InputLayerType::PARAMETER) {  // pad_begin
            inputShapes.push_back(InputShape({static_cast<int32_t>(padsBegin.size())},
                                             std::vector<Shape>(inShape.second.size(), {padsBegin.size()})));
        }
        if (inputLayerTypes[1] == InputLayerType::PARAMETER) {  // pad_end
            inputShapes.push_back(InputShape({static_cast<int32_t>(padsEnd.size())},
                                             std::vector<Shape>(inShape.second.size(), {padsEnd.size()})));
        }

        init_input_shapes(inputShapes);

        // add empty shape for parameter input of scalar 'pad_value'
        if (inputLayerTypes[2] == InputLayerType::PARAMETER) {
            inputDynamicShapes.push_back(PartialShape({}));
            for (size_t i = 0; i < inShape.second.size(); ++i) {
                for (size_t k = 0; k < targetStaticShapes.size(); ++k) {
                    targetStaticShapes[k].push_back(Shape({}));
                }
            }
        }
        ParameterVector functionParams;
        functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes.front()));
        functionParams.front()->set_friendly_name("data");

        std::shared_ptr<Node> pads_begin, pads_end, arg_pad_value;

        // padsBegin
        if (inputLayerTypes[0] == InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{padsBegin.size()}));
            functionParams.back()->set_friendly_name("padsBegin");
            pads_begin = functionParams.back();
        } else {
            pads_begin =
                    std::make_shared<ov::op::v0::Constant>(element::i32, Shape{padsBegin.size()}, padsBegin.data());
        }

        // padsEnd
        if (inputLayerTypes[1] == InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{padsEnd.size()}));
            functionParams.back()->set_friendly_name("padsEnd");
            pads_end = functionParams.back();
        } else {
            pads_end = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{padsEnd.size()}, padsEnd.data());
        }

        // argPadValue
        if (inputLayerTypes[2] == InputLayerType::PARAMETER) {
            functionParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, PartialShape({})));
            functionParams.back()->set_friendly_name("padValue");
            arg_pad_value = functionParams.back();
        } else {
            arg_pad_value = std::make_shared<ov::op::v0::Constant>(inputType, Shape{}, &argPadValue);
        }

        auto pad = std::make_shared<ov::op::v1::Pad>(functionParams[0], pads_begin, pads_end, arg_pad_value, padMode);
        ResultVector results;
        for (size_t i = 0; i < pad->get_output_size(); ++i) {
            results.push_back(std::make_shared<ov::op::v0::Result>(pad->output(i)));
        }

        function = std::make_shared<Model>(results, functionParams, "DynamicPadLayerTest");
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); i++) {
            Tensor tensor;
            const auto& funcInput = funcInputs[i];
            const auto elementType = funcInput.get_element_type();

            if (funcInput.get_node()->get_friendly_name() == "padsBegin") {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};
                auto data = tensor.data<int>();
                for (size_t i = 0; i < padsBegin.size(); i++) {
                    data[i] = static_cast<int>(padsBegin[i]);
                }

            } else if (funcInput.get_node()->get_friendly_name() == "padsEnd") {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};
                auto data = tensor.data<int>();
                for (size_t i = 0; i < padsEnd.size(); i++) {
                    data[i] = static_cast<int>(padsEnd[i]);
                }

            } else if (funcInput.get_node()->get_friendly_name() == "padValue") {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};
                auto data = tensor.data<float>();
                data[0] = argPadValue;

            } else {
                if (elementType.is_real()) {
                    InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = create_and_fill_tensor(elementType, targetInputStaticShapes[i], in_data);
                } else {
                    tensor = create_and_fill_tensor(elementType, targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

class DynamicPadDmaLayerTest : public DynamicPadLayerTest {};

// Tracking number [E#188947]
TEST_P(DynamicPadLayerTest, DISABLED_NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicPadLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicPadLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicPadDmaLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<Type> inputPrecisions = {element::f32};

const std::vector<std::vector<InputLayerType>> attributeTypes = {
        /*pads_begin_type | pads_end_type | pads_value_type*/
        {InputLayerType::CONSTANT, InputLayerType::CONSTANT, InputLayerType::CONSTANT},
        {InputLayerType::PARAMETER, InputLayerType::PARAMETER, InputLayerType::CONSTANT},
        {InputLayerType::CONSTANT, InputLayerType::PARAMETER, InputLayerType::CONSTANT},
        {InputLayerType::PARAMETER, InputLayerType::CONSTANT, InputLayerType::CONSTANT},
        {InputLayerType::CONSTANT, InputLayerType::CONSTANT, InputLayerType::PARAMETER},
        {InputLayerType::PARAMETER, InputLayerType::PARAMETER, InputLayerType::PARAMETER},
};

//====================== Efficentdet model case ======================
const std::vector<InputShape> inputShapesDynamic1D = {
        InputShape{{Dimension(1, 100)}, std::vector<Shape>{{100}}},
};
const std::vector<float> padValue = {0.f};
// Tracking number [E#188767]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PadDynamic_EfficientDetModelCase, DynamicPadLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic1D),  // input_shape
                                            ::testing::ValuesIn(inputPrecisions),       // input_precision
                                            ::testing::Values(AttrType{0}),             // pads_begin
                                            ::testing::Values(AttrType{1}),             // pads_end
                                            ::testing::ValuesIn(padValue),              // pad_value
                                            ::testing::ValuesIn(attributeTypes),  // type combos for Pad's 'attributes'
                                            ::testing::Values(PadMode::CONSTANT)),  // pad_mode
                         DynamicPadLayerTest::getTestCaseName);

//====================== 2D cases ======================
const std::vector<InputShape> inputShapesDynamic2D = {
        InputShape{{1, Dimension(1, 100)}, std::vector<Shape>{{1, 100}}},
};
const std::vector<AttrType> padsBegin2D = {{0, 0}};
const std::vector<AttrType> padsEnd2D = {{0, 3}};

// Tracking number [E#188767]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PadDynamic_2D_cases, DynamicPadLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic2D),  // input_shape
                                            ::testing::ValuesIn(inputPrecisions),       // input_precision
                                            ::testing::ValuesIn(padsBegin2D),           // pads_begin
                                            ::testing::ValuesIn(padsEnd2D),             // pads_end
                                            ::testing::ValuesIn(padValue),              // pad_value
                                            ::testing::ValuesIn(attributeTypes),  // type combos for Pad's 'attributes'
                                            ::testing::Values(PadMode::CONSTANT)),  // pad_mode
                         DynamicPadLayerTest::getTestCaseName);

//====================== Non-constant pad modes ======================
// Tracking number [E#188767]
const std::vector<PadMode> padNonConstModes = {PadMode::EDGE, PadMode::REFLECT, PadMode::SYMMETRIC};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_PadDynamic_NonConstant, DynamicPadLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic1D),  // input_shape
                                            ::testing::ValuesIn(inputPrecisions),       // input_precision
                                            ::testing::Values(AttrType{0}),             // pads_begin
                                            ::testing::Values(AttrType{1}),             // pads_end
                                            ::testing::Values(0.f),                     // pad_value
                                            ::testing::ValuesIn(attributeTypes),  // type combos for Pad's 'attributes'
                                            ::testing::ValuesIn(padNonConstModes)),  // pad_mode
                         DynamicPadLayerTest::getTestCaseName);

const std::vector<Type> padDmaPrecisions = {element::f16};
const std::vector<InputLayerType> allConstant = {InputLayerType::CONSTANT, InputLayerType::CONSTANT,
                                                 InputLayerType::CONSTANT};
const std::vector<float> padDmaValues = {0.f, -1.f};

//====================== PadDma 1D ======================
const std::vector<InputShape> inputShapesPadDma1D = {
        InputShape{{Dimension(1, 100)}, std::vector<Shape>{{100}}},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_PadDma_Direct_1D, DynamicPadDmaLayerTest,
        ::testing::Combine(::testing::ValuesIn(inputShapesPadDma1D), ::testing::ValuesIn(padDmaPrecisions),
                           ::testing::Values(AttrType{0}),  // pads_begin
                           ::testing::Values(AttrType{1}),  // pads_end
                           ::testing::ValuesIn(padDmaValues),
                           ::testing::Values(std::vector<InputLayerType>{
                                   InputLayerType::CONSTANT, InputLayerType::CONSTANT, InputLayerType::CONSTANT}),
                           ::testing::Values(PadMode::CONSTANT)),
        DynamicPadDmaLayerTest::getTestCaseName);

//====================== PadDma 2D ======================
const std::vector<InputShape> inputShapesDma2D = {
        InputShape{{4, Dimension(1, 50)}, std::vector<Shape>{{4, 50}}},
};
const std::vector<AttrType> padsBegin2Ddma = {{1, 2}};
const std::vector<AttrType> padsEnd2Ddma = {{2, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_PadDma_2D, DynamicPadDmaLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDma2D),
                                            ::testing::ValuesIn(padDmaPrecisions), ::testing::ValuesIn(padsBegin2Ddma),
                                            ::testing::ValuesIn(padsEnd2Ddma), ::testing::ValuesIn(padDmaValues),
                                            ::testing::Values(allConstant), ::testing::Values(PadMode::CONSTANT)),
                         DynamicPadDmaLayerTest::getTestCaseName);

//====================== PadDma 3D ======================
const std::vector<InputShape> inputShapesDma3D = {
        InputShape{{3, 4, Dimension(1, 30)}, std::vector<Shape>{{3, 4, 30}}},
};
const std::vector<AttrType> padsBegin3Ddma = {{0, 1, 2}};
const std::vector<AttrType> padsEnd3Ddma = {{0, 2, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_PadDma_3D, DynamicPadDmaLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDma3D),
                                            ::testing::ValuesIn(padDmaPrecisions), ::testing::ValuesIn(padsBegin3Ddma),
                                            ::testing::ValuesIn(padsEnd3Ddma), ::testing::ValuesIn(padDmaValues),
                                            ::testing::Values(allConstant), ::testing::Values(PadMode::CONSTANT)),
                         DynamicPadDmaLayerTest::getTestCaseName);

//====================== PadDma 4D ======================
const std::vector<InputShape> inputShapesPadDmaDirect = {
        InputShape{{1, 1, 1, Dimension(1, 100)}, std::vector<Shape>{{1, 1, 1, 100}}},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_PadDma_Direct_4D, DynamicPadDmaLayerTest,
        ::testing::Combine(::testing::ValuesIn(inputShapesPadDmaDirect), ::testing::ValuesIn(padDmaPrecisions),
                           ::testing::Values(AttrType{0, 0, 0, 0}),  // pads_begin
                           ::testing::Values(AttrType{1, 0, 0, 0}),  // pads_end (pad on N)
                           ::testing::ValuesIn(padDmaValues),
                           ::testing::Values(std::vector<InputLayerType>{
                                   InputLayerType::CONSTANT, InputLayerType::CONSTANT, InputLayerType::CONSTANT}),
                           ::testing::Values(PadMode::CONSTANT)),
        DynamicPadDmaLayerTest::getTestCaseName);

const std::vector<InputShape> inputShapesDma4D = {
        InputShape{{2, 3, 4, Dimension(1, 20)}, std::vector<Shape>{{2, 3, 4, 20}}},
};
const std::vector<AttrType> padsBegin4Ddma = {{1, 2, 3, 4}};
const std::vector<AttrType> padsEnd4Ddma = {{2, 1, 4, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_PadDma_4D, DynamicPadDmaLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDma4D),
                                            ::testing::ValuesIn(padDmaPrecisions), ::testing::ValuesIn(padsBegin4Ddma),
                                            ::testing::ValuesIn(padsEnd4Ddma), ::testing::ValuesIn(padDmaValues),
                                            ::testing::Values(allConstant), ::testing::Values(PadMode::CONSTANT)),
                         DynamicPadDmaLayerTest::getTestCaseName);

}  // namespace ov::test
