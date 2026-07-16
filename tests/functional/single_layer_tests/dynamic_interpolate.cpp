//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::test::utils;
namespace ov::test {

// Common base for interpolate tests: shared input generation logic
class InterpolateScaleAsParamTestBase : virtual public VpuOv2LayerTest {
public:
    std::vector<float> scalesValues;
    std::vector<int32_t> sizesValues;

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); i++) {
            Tensor tensor;
            const auto& funcInput = funcInputs[i];
            const auto elementType = funcInput.get_element_type();

            if (funcInput.get_node()->get_friendly_name() == "scales") {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};
                if (elementType == ov::element::f16) {
                    auto data = tensor.data<ov::float16>();
                    for (size_t j = 0; j < scalesValues.size(); j++) {
                        data[j] = static_cast<ov::float16>(scalesValues[j]);
                    }
                } else {
                    auto data = tensor.data<float>();
                    for (size_t j = 0; j < scalesValues.size(); j++) {
                        data[j] = scalesValues[j];
                    }
                }
            } else if (funcInput.get_node()->get_friendly_name() == "sizes") {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};
                auto data = tensor.data<int>();
                for (size_t j = 0; j < sizesValues.size(); j++) {
                    data[j] = static_cast<int>(sizesValues[j]);
                }
            } else {
                tensor = Tensor{elementType, targetInputStaticShapes[i]};

                const std::vector<float> customInputValues = {
                        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                };

                const size_t elemCount = tensor.get_size();
                if (elementType == ov::element::f32) {
                    auto* data = tensor.data<float>();
                    for (size_t idx = 0; idx < elemCount; ++idx) {
                        data[idx] = customInputValues[idx % customInputValues.size()];
                    }
                } else if (elementType == ov::element::f16) {
                    auto* data = tensor.data<ov::float16>();
                    for (size_t idx = 0; idx < elemCount; ++idx) {
                        data[idx] = static_cast<ov::float16>(customInputValues[idx % customInputValues.size()]);
                    }
                } else {
                    tensor = create_and_fill_tensor(elementType, targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

using InterpolateSAPParamSet =
        std::tuple<ov::Shape,             // Input shape
                   ov::element::Type,     // Input element type
                   std::vector<float>,    // Scales
                   std::vector<int32_t>,  // Sizes
                   std::vector<int64_t>,  // Axes values
                   ov::op::util::InterpolateBase::ShapeCalcMode, op::v11::Interpolate::InterpolateMode,
                   op::v11::Interpolate::CoordinateTransformMode, op::v11::Interpolate::NearestMode,
                   std::vector<size_t>,  // padsBegin values
                   std::vector<size_t>,  // padsEnd values
                   bool,                 // antialias
                   ov::Layout,           // Input layout
                   float,                // cube coeff
                   ov::element::Type>;   // scales element type

class InterpolateSAPStaticInputLayerTest :
        public testing::WithParamInterface<InterpolateSAPParamSet>,
        public InterpolateScaleAsParamTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateSAPParamSet> obj) {
        using ov::test::utils::operator<<;
        const auto& [dataShape, modelType, scalesValues, sizesValues, axes, shapeCalcMode, mode,
                     coordinateTransformMode, nearestMode, padBegin, padEnd, antialias, layout, cubeCoef, scalesType] =
                obj.param;

        std::ostringstream result;
        result << "ScaleInput=PARAM_";
        result << "TestKind" << testKind(__FILE__) << "_";
        result << "DataIS=" << ov::test::utils::vec2str(dataShape) << "_";
        result << "ScalesIS=" << ov::test::utils::vec2str(scalesValues) << "_";
        result << "SizesIS=" << ov::test::utils::vec2str(sizesValues) << "_";
        result << "InterpolateMode=" << mode << "_";
        result << "ShapeCalcMode=" << shapeCalcMode << "_";
        result << "CoordinateTransformMode=" << coordinateTransformMode << "_";
        result << "NearestMode=" << nearestMode << "_";
        result << "cube_coef=" << cubeCoef << "_";
        result << "Antialias=" << std::boolalpha << antialias << "_";
        result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
        result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "Layout=" << layout.to_string() << "_";
        result << "netType=" << modelType.get_type_name() << "_";
        result << "scalesType=" << scalesType.get_type_name();
        return result.str();
    }

    void SetUp() override {
        using ov::op::util::InterpolateBase;

        ov::Shape dataShape;
        ov::element::Type inputType;
        std::vector<int64_t> axes;
        InterpolateBase::ShapeCalcMode shapeCalcMode;
        op::v11::Interpolate::InterpolateMode mode;
        op::v11::Interpolate::CoordinateTransformMode coordinateTransformMode;
        op::v11::Interpolate::NearestMode nearestMode;
        std::vector<size_t> padsBegin;
        std::vector<size_t> padsEnd;
        bool antialias;
        ov::Layout layout;
        float cubeCoef;
        ov::element::Type scalesType;

        std::tie(dataShape, inputType, scalesValues, sizesValues, axes, shapeCalcMode, mode, coordinateTransformMode,
                 nearestMode, padsBegin, padsEnd, antialias, layout, cubeCoef, scalesType) = this->GetParam();

        // Only scales mode is supported in current implementation
        OPENVINO_ASSERT(shapeCalcMode == InterpolateBase::ShapeCalcMode::SCALES);

        auto dataParam = std::make_shared<ov::op::v0::Parameter>(inputType, dataShape);
        dataParam->set_friendly_name("interp_input");

        InterpolateBase::InterpolateAttrs interpolateAttrs{
                mode, shapeCalcMode, padsBegin, padsEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};

        auto scalesShape = ov::Shape{scalesValues.size()};
        std::vector<ov::Shape> param_shapes{dataShape, scalesShape};
        init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

        ov::ParameterVector params{dataParam};
        std::shared_ptr<ov::Node> shapeInput;
        if (shapeCalcMode == InterpolateBase::ShapeCalcMode::SCALES) {
            auto scalesParam = std::make_shared<ov::op::v0::Parameter>(scalesType, scalesShape);
            scalesParam->set_friendly_name("scales");
            params.push_back(scalesParam);
            shapeInput = scalesParam;
        } else {
            auto sizesParam = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, Shape{sizesValues.size()});
            sizesParam->set_friendly_name("sizes");
            params.push_back(sizesParam);
            shapeInput = sizesParam;
        }

        std::shared_ptr<ov::op::v11::Interpolate> interpolate;
        if (axes.empty()) {
            interpolate = std::make_shared<ov::op::v11::Interpolate>(dataParam, shapeInput, interpolateAttrs);
        } else {
            auto axesConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
            interpolate =
                    std::make_shared<ov::op::v11::Interpolate>(dataParam, shapeInput, axesConst, interpolateAttrs);
        }

        auto result = std::make_shared<ov::op::v0::Result>(interpolate);

        if (inputType == ov::element::f32) {
            abs_threshold = 1e-6;
        }

        function = std::make_shared<ov::Model>(ResultVector{result}, params, "InterpolateSAPStaticInputLayerTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input(0).tensor().set_layout(layout);
        preProc.input(0).model().set_layout(layout);
        preProc.output().tensor().set_layout(layout);
        preProc.output().model().set_layout(layout);
        function = preProc.build();
    }
};

TEST_P(InterpolateSAPStaticInputLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    enableTurbo();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// [Tracking number E#196823]
TEST_P(InterpolateSAPStaticInputLayerTest, DISABLED_TMP_NPU4000_HC) {
    abs_threshold = 0.0f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(InterpolateSAPStaticInputLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// [Tracking number E#196823]
TEST_P(InterpolateSAPStaticInputLayerTest, DISABLED_TMP_NPU5010_HC) {
    abs_threshold = 0.0f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

// Subset of platforms for large-shape tests
class InterpolateSAPStaticLargeShapeLayerTest : public InterpolateSAPStaticInputLayerTest {};

TEST_P(InterpolateSAPStaticLargeShapeLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    enableTurbo();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(InterpolateSAPStaticLargeShapeLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// ===== Test parameters for InterpolateSAPStaticInputLayerTest =====

const std::vector<ov::Shape> interpShapes = {
        ov::Shape{1, 3, 4, 6},
};

const std::vector<ov::element::Type> interpInputFullPrecisions = {
        ov::element::f16,
        ov::element::f32,
};

const std::vector<ov::element::Type> interpInputF16Precisions = {
        ov::element::f16,
};

// Scale values has 8.0f as upper bound for now
const std::vector<std::vector<float>> interpScalesList = {
        {1.0f, 1.0f},
        {4.0f, 4.0f},
        {1.5f, 2.0f},
        {8.0f, 8.0f},
};

const std::vector<std::vector<float>> interpTypicalScalesList = {
        {4.0f, 4.0f},
};

// no size mode yet
const std::vector<std::vector<int32_t>> interpSizesList = {
        {},
};

// linear interpolation only supports spatial axes, so axes values are {2, 3} in this test
const std::vector<std::vector<int64_t>> interpAxesList = {
        {2, 3},
        {1, 2},
};

const std::vector<std::vector<size_t>> interpPadsBeginList = {
        {0, 0, 0, 0},
};

const std::vector<std::vector<size_t>> interpPadsEndList = {
        {0, 0, 0, 0},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_ScalesAsParam, InterpolateSAPStaticInputLayerTest,
        ::testing::Combine(::testing::ValuesIn(interpShapes),              // dataShape
                           ::testing::ValuesIn(interpInputF16Precisions),  // inputType
                           ::testing::ValuesIn(interpScalesList),          // scalesValues
                           ::testing::ValuesIn(interpSizesList),           // sizesValues (unused)
                           ::testing::ValuesIn(interpAxesList),            // axes
                           ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
                           ::testing::Values(op::v11::Interpolate::InterpolateMode::LINEAR),
                           ::testing::Values(op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                           ::testing::Values(op::v11::Interpolate::NearestMode::FLOOR),
                           ::testing::ValuesIn(interpPadsBeginList), ::testing::ValuesIn(interpPadsEndList),
                           ::testing::Values(false),                   // antialias
                           ::testing::ValuesIn({ov::Layout("NCHW")}),  // layout
                           ::testing::Values(-0.75f),                  // cubeCoef
                           ::testing::Values(ov::element::f32)),       // scalesType
        InterpolateSAPStaticInputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_ScalesAsParam_LargeShape_1, InterpolateSAPStaticLargeShapeLayerTest,
        ::testing::Combine(::testing::Values(ov::Shape{1, 3, 368, 432}),                   // dataShape
                           ::testing::ValuesIn(interpInputFullPrecisions),                 // inputType
                           ::testing::Values(std::vector<float>{1.0f, 1.0f, 4.0f, 4.0f}),  // scalesValues
                           ::testing::ValuesIn(interpSizesList),                           // sizesValues (unused)
                           ::testing::Values(std::vector<int64_t>{}),                      // axes
                           ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
                           ::testing::Values(op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
                           ::testing::Values(op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                           ::testing::Values(op::v11::Interpolate::NearestMode::FLOOR),
                           ::testing::ValuesIn(interpPadsBeginList), ::testing::ValuesIn(interpPadsEndList),
                           ::testing::Values(false),                   // antialias
                           ::testing::ValuesIn({ov::Layout("NCHW")}),  // layout
                           ::testing::Values(-0.75f),                  // cubeCoef
                           ::testing::Values(ov::element::f32)),       // scalesType
        InterpolateSAPStaticInputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_ScalesAsParam_LargeShape_2, InterpolateSAPStaticLargeShapeLayerTest,
        ::testing::Combine(::testing::Values(ov::Shape{1, 368, 432, 3}),       // dataShape
                           ::testing::ValuesIn(interpInputF16Precisions),      // inputType
                           ::testing::Values(std::vector<float>{4.0f, 4.0f}),  // scalesValues
                           ::testing::ValuesIn(interpSizesList),               // sizesValues (unused)
                           ::testing::Values(std::vector<int64_t>{1, 2}),      // axes
                           ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
                           ::testing::Values(op::v11::Interpolate::InterpolateMode::LINEAR),
                           ::testing::Values(op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                           ::testing::Values(op::v11::Interpolate::NearestMode::FLOOR),
                           ::testing::ValuesIn(interpPadsBeginList), ::testing::ValuesIn(interpPadsEndList),
                           ::testing::Values(false),                   // antialias
                           ::testing::ValuesIn({ov::Layout("NCHW")}),  // layout
                           ::testing::Values(-0.75f),                  // cubeCoef
                           ::testing::Values(ov::element::f32)),       // scalesType
        InterpolateSAPStaticInputLayerTest::getTestCaseName);

// ===== Dynamic input shape (with bounds) + dynamic scales test =====
// Contrast to InterpolateSAPStaticInputLayerTest which uses static input shapes.
// Here the data input has bounded dynamic spatial dimensions while scales
// remain as runtime parameters.

using InterpolateSAPDynParamSet =
        std::tuple<InputShape,                                     // Input shape (dynamic with bounds)
                   ov::element::Type,                              // Input element type
                   std::vector<float>,                             // Scales
                   std::vector<int64_t>,                           // Axes values
                   op::v11::Interpolate::InterpolateMode,          // Interpolation mode
                   op::v11::Interpolate::CoordinateTransformMode,  // Coordinate transform mode
                   ov::Layout>;                                    // Input layout

class InterpolateSAPDynInputLayerTest :
        public testing::WithParamInterface<InterpolateSAPDynParamSet>,
        public InterpolateScaleAsParamTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateSAPDynParamSet> obj) {
        using ov::test::utils::operator<<;
        const auto& [dataShape, modelType, scalesValues, axes, mode, coordinateTransformMode, layout] = obj.param;

        std::ostringstream result;
        result << "DynInput=PARAM_";
        result << "TestKind" << testKind(__FILE__) << "_";
        result << "IS=" << partialShape2str({dataShape.first}) << "_";
        result << "TS=";
        for (const auto& item : dataShape.second) {
            result << vec2str(item) << "_";
        }
        result << "ScalesIS=" << ov::test::utils::vec2str(scalesValues) << "_";
        result << "InterpolateMode=" << mode << "_";
        result << "CoordinateTransformMode=" << coordinateTransformMode << "_";
        result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
        result << "Layout=" << layout.to_string() << "_";
        result << "netType=" << modelType.get_type_name();
        return result.str();
    }

    void SetUp() override {
        using ov::op::util::InterpolateBase;

        InputShape dataInputShape;
        ov::element::Type inputType;
        std::vector<int64_t> axes;
        op::v11::Interpolate::InterpolateMode mode;
        op::v11::Interpolate::CoordinateTransformMode coordinateTransformMode;
        ov::Layout layout;

        std::tie(dataInputShape, inputType, scalesValues, axes, mode, coordinateTransformMode, layout) =
                this->GetParam();

        auto scalesShape = ov::Shape{scalesValues.size()};

        // Build input shapes: data (dynamic with bounds) + scales (static)
        std::vector<InputShape> inputShapes;
        inputShapes.push_back(dataInputShape);
        inputShapes.push_back(InputShape{ov::PartialShape{static_cast<int64_t>(scalesValues.size())},
                                         std::vector<Shape>(dataInputShape.second.size(), scalesShape)});
        init_input_shapes(inputShapes);

        auto dataParam = std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[0]);
        dataParam->set_friendly_name("interp_input");

        auto scalesParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scalesShape);
        scalesParam->set_friendly_name("scales");

        ov::ParameterVector params{dataParam, scalesParam};

        InterpolateBase::InterpolateAttrs interpolateAttrs{mode,
                                                           InterpolateBase::ShapeCalcMode::SCALES,
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           coordinateTransformMode,
                                                           op::v11::Interpolate::NearestMode::FLOOR,
                                                           false,
                                                           -0.75f};

        std::shared_ptr<ov::op::v11::Interpolate> interpolate;
        if (axes.empty()) {
            interpolate = std::make_shared<ov::op::v11::Interpolate>(dataParam, scalesParam, interpolateAttrs);
        } else {
            auto axesConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
            interpolate =
                    std::make_shared<ov::op::v11::Interpolate>(dataParam, scalesParam, axesConst, interpolateAttrs);
        }

        auto result = std::make_shared<ov::op::v0::Result>(interpolate);

        if (inputType == ov::element::f32) {
            abs_threshold = 1e-6;
        }

        function = std::make_shared<ov::Model>(ResultVector{result}, params, "InterpolateSAPDynInputLayerTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input(0).tensor().set_layout(layout);
        preProc.input(0).model().set_layout(layout);
        preProc.output().tensor().set_layout(layout);
        preProc.output().model().set_layout(layout);
        function = preProc.build();
    }
};

TEST_P(InterpolateSAPDynInputLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    enableTurbo();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// [Tracking number E#196823]
TEST_P(InterpolateSAPDynInputLayerTest, DISABLED_TMP_NPU4000_HC) {
    abs_threshold = 0.0f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(InterpolateSAPDynInputLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// [Tracking number E#196823]
TEST_P(InterpolateSAPDynInputLayerTest, DISABLED_TMP_NPU5010_HC) {
    abs_threshold = 0.0f;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

// Input data has bounded dynamic spatial dims; scales remain runtime parameters
const std::vector<InputShape> interpDynamicInputShapes = {
        generateTestShape(1, 3, 50_Dyn, 50_Dyn),
};

const std::vector<InputShape> interpDynamicInputLargeShapes = {
        generateTestShape(1, 3, 368_Dyn, 432_Dyn),
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_DynamicInput_ScalesAsParam, InterpolateSAPDynInputLayerTest,
        ::testing::Combine(::testing::ValuesIn(interpDynamicInputShapes),  // dataShape (dynamic)
                           ::testing::ValuesIn(interpInputF16Precisions),  // inputType
                           ::testing::ValuesIn(interpScalesList),          // scalesValues
                           ::testing::ValuesIn(interpAxesList),            // axes
                           ::testing::Values(op::v11::Interpolate::InterpolateMode::LINEAR),
                           ::testing::Values(op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                           ::testing::ValuesIn({ov::Layout("NCHW")})),  // layout
        InterpolateSAPDynInputLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_DynamicInput_ScalesAsParam_LargeShape, InterpolateSAPDynInputLayerTest,
        ::testing::Combine(::testing::ValuesIn(interpDynamicInputLargeShapes),  // dataShape (dynamic)
                           ::testing::ValuesIn(interpInputF16Precisions),       // inputType
                           ::testing::ValuesIn(interpTypicalScalesList),        // scalesValues
                           ::testing::ValuesIn(interpAxesList),                 // axes
                           ::testing::Values(op::v11::Interpolate::InterpolateMode::LINEAR),
                           ::testing::Values(op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL),
                           ::testing::ValuesIn({ov::Layout("NCHW")})),  // layout
        InterpolateSAPDynInputLayerTest::getTestCaseName);

// =====================================================================
// Realistic cases [Tracking number E#217769]
// =====================================================================

namespace {
struct RealisticCase {
    size_t h;
    size_t w;
    float scale;
};

const std::vector<RealisticCase> realisticCases = {
        {240, 424, 1.1f}, {180, 320, 0.75f},  {360, 640, 0.75f},
        {360, 640, 5.0f}, {1080, 1920, 1.1f}, {1080, 1920, 2.0f},
};

std::vector<InterpolateSAPParamSet> buildRealisticSAPParams() {
    std::vector<InterpolateSAPParamSet> out;
    out.reserve(realisticCases.size());
    for (const auto& c : realisticCases) {
        out.emplace_back(ov::Shape{1, 3, c.h, c.w},             // dataShape
                         ov::element::f16,                      // inputType
                         std::vector<float>{c.scale, c.scale},  // scalesValues
                         std::vector<int32_t>{},                // sizesValues (unused)
                         std::vector<int64_t>{2, 3},            // axes
                         ov::op::util::InterpolateBase::ShapeCalcMode::SCALES,
                         op::v11::Interpolate::InterpolateMode::LINEAR,
                         op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                         op::v11::Interpolate::NearestMode::FLOOR, std::vector<size_t>{0, 0, 0, 0},  // padsBegin
                         std::vector<size_t>{0, 0, 0, 0},                                            // padsEnd
                         false,                                                                      // antialias
                         ov::Layout("NCHW"),                                                         // layout
                         -0.75f,                                                                     // cubeCoef
                         // certain scales only supported in fp16 Tracking number E#217769
                         ov::element::f16);  // scalesType
    }
    return out;
}
}  // namespace

// Dedicated subclass so the realistic-case suite can use a relaxed tolerance
// without affecting the strict (0.0) thresholds of the other Static suites.
class InterpolateSAPStaticRealisticLayerTest : public InterpolateSAPStaticInputLayerTest {};

TEST_P(InterpolateSAPStaticRealisticLayerTest, NPU4000_HW) {
    abs_threshold = 0.02f;
    enableTurbo();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(InterpolateSAPStaticRealisticLayerTest, NPU5010_HW) {
    abs_threshold = 0.02f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_ScalesAsParam_Realistic, InterpolateSAPStaticRealisticLayerTest,
                         ::testing::ValuesIn(buildRealisticSAPParams()),
                         InterpolateSAPStaticInputLayerTest::getTestCaseName);

}  // namespace ov::test
