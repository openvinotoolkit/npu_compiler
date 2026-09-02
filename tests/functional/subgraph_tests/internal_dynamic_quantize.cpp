//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/dynamic_quantize.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"

#include <limits>
#include <numeric>

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test::subgraph {
namespace {

constexpr uint64_t WHOLE_DIM = std::numeric_limits<uint64_t>::max();

using InternalDQParams =
        std::tuple<ov::Shape, ov::element::Type, ov::element::Type, ov::element::Type, std::vector<uint64_t>>;

struct NegativeInternalDQParams {
    std::string testName;
    ov::Shape inputShape;
    ov::element::Type inputType;
    ov::op::internal::DynamicQuantize::Attributes attrs;
    std::string expectedError;
};

ov::op::internal::DynamicQuantize::Attributes makeSupportedAttrs(const ov::element::Type& quantType,
                                                                 const ov::element::Type& scaleType,
                                                                 const std::vector<uint64_t>& groupSizes) {
    ov::op::internal::DynamicQuantize::Attributes attrs;
    attrs.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
    attrs.quantization_dt = quantType;
    attrs.scale_dt = scaleType;
    attrs.zp_dt = quantType;
    attrs.group_sizes = groupSizes;
    attrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
    attrs.scales_zp_output_order.resize(groupSizes.size());
    std::iota(attrs.scales_zp_output_order.begin(), attrs.scales_zp_output_order.end(), 0);
    return attrs;
}

std::vector<int64_t> getReductionAxes(const ov::Shape& inputShape, const std::vector<uint64_t>& groupSizes) {
    OPENVINO_ASSERT(inputShape.size() == groupSizes.size());

    std::vector<int64_t> axes;
    for (size_t ind = 0; ind < groupSizes.size(); ++ind) {
        const auto groupSize = groupSizes[ind];
        OPENVINO_ASSERT(groupSize > 0);

        if (groupSize == 1) {
            continue;
        }

        OPENVINO_ASSERT(groupSize == WHOLE_DIM || groupSize == inputShape[ind]);
        axes.push_back(static_cast<int64_t>(ind));
    }

    OPENVINO_ASSERT(!axes.empty());
    return axes;
}

std::shared_ptr<ov::Model> buildInternalDQModel(const ov::Shape& inputShape, const ov::element::Type& inputType,
                                                const ov::op::internal::DynamicQuantize::Attributes& attrs,
                                                const std::string& modelName) {
    auto input = std::make_shared<ov::op::v0::Parameter>(inputType, inputShape);
    auto dq = std::make_shared<ov::op::internal::DynamicQuantize>(input, attrs);

    ov::ResultVector results;
    for (const auto& output : dq->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }

    return std::make_shared<ov::Model>(results, ov::ParameterVector{input}, modelName);
}

std::shared_ptr<ov::Model> buildDecomposedReferenceDQModel(const ov::Shape& inputShape,
                                                           const ov::element::Type& inputType,
                                                           const ov::element::Type& quantType,
                                                           const ov::element::Type& scaleType,
                                                           const std::vector<uint64_t>& groupSizes) {
    auto input = std::make_shared<ov::op::v0::Parameter>(inputType, inputShape);
    std::shared_ptr<ov::Node> dqInput = input;
    if (inputType == ov::element::f16) {
        dqInput = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }

    const auto reduceAxes = getReductionAxes(inputShape, groupSizes);
    auto axesNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{reduceAxes.size()}, reduceAxes);
    constexpr auto keepDims = true;

    auto reducedMin = std::make_shared<ov::op::v1::ReduceMin>(dqInput, axesNode, keepDims);
    auto reducedMax = std::make_shared<ov::op::v1::ReduceMax>(dqInput, axesNode, keepDims);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto xMin = std::make_shared<ov::op::v1::Minimum>(zero, reducedMin);
    auto xMax = std::make_shared<ov::op::v1::Maximum>(zero, reducedMax);
    auto xSpan = std::make_shared<ov::op::v1::Subtract>(xMax, xMin);

    const bool isSigned = quantType.is_signed();
    const float quantRangeMinValue = isSigned ? -127.0f : 0.0f;
    const float quantRangeMaxValue = isSigned ? 127.0f : 255.0f;
    auto quantRangeMin = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {quantRangeMinValue});
    auto quantRangeMax = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {quantRangeMaxValue});
    auto quantRangeSpan = std::make_shared<ov::op::v1::Subtract>(quantRangeMax, quantRangeMin);

    auto yScale = std::make_shared<ov::op::v1::Divide>(xSpan, quantRangeSpan);
    auto xMinShifted = std::make_shared<ov::op::v1::Subtract>(quantRangeMin, xMin);
    auto intermediateZeroPoint = std::make_shared<ov::op::v5::Round>(
            std::make_shared<ov::op::v1::Divide>(xMinShifted, yScale), ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto yZeroPoint = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<ov::op::v0::Clamp>(intermediateZeroPoint, quantRangeMinValue, quantRangeMaxValue),
            quantType);

    auto xScaled = std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Multiply>(dqInput, quantRangeSpan),
                                                        xSpan);
    auto xRounded = std::make_shared<ov::op::v5::Round>(xScaled, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto yZeroPointF32 = std::make_shared<ov::op::v0::Convert>(yZeroPoint, ov::element::f32);
    auto resultShifted = std::make_shared<ov::op::v1::Add>(xRounded, yZeroPointF32);
    auto resultClamped = std::make_shared<ov::op::v0::Clamp>(resultShifted, quantRangeMinValue, quantRangeMaxValue);
    auto y = std::make_shared<ov::op::v0::Convert>(resultClamped, quantType);

    std::shared_ptr<ov::Node> scaleOutput = yScale;
    if (scaleType == ov::element::f16) {
        scaleOutput = std::make_shared<ov::op::v0::Convert>(yScale, ov::element::f16);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{y, scaleOutput, yZeroPoint}, ov::ParameterVector{input},
                                       "DecomposedReferenceDynamicQuantize");
}

class InternalDynamicQuantizeTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<InternalDQParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InternalDQParams>& obj) {
        const auto& [inputShape, inputType, quantType, scaleType, groupSizes] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "InputShape=" << inputShape << sep;
        result << "InputType=" << inputType << sep;
        result << "QuantType=" << quantType << sep;
        result << "ScaleType=" << scaleType << sep;
        result << "GroupSizes=" << ov::test::utils::vec2str(groupSizes);
        return result.str();
    }

    void SetUp() override {
        ov::Shape inputShape;
        ov::element::Type inputType;
        ov::element::Type quantType;
        ov::element::Type scaleType;
        std::vector<uint64_t> groupSizes;

        std::tie(inputShape, inputType, quantType, scaleType, groupSizes) = GetParam();
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto attrs = makeSupportedAttrs(quantType, scaleType, groupSizes);
        // NPU frontend decomposes internal::DynamicQuantize into the existing compiler DynamicQuantize subgraph.
        function = buildInternalDQModel(inputShape, inputType, attrs, "InternalDynamicQuantize");
        // CPU executes a decomposed reference subgraph so the test validates DynamicQuantize semantics rather than
        // relying on CPU support for the internal op.
        _decomposedReferenceFunction =
                buildDecomposedReferenceDQModel(inputShape, inputType, quantType, scaleType, groupSizes);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        auto tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 3);
        ASSERT_EQ(expectedTensors.size(), 3);

        ov::test::utils::compare(expectedTensors[0], actualTensors[0], 1.0f);
        ov::test::utils::compare(expectedTensors[1], actualTensors[1], 0.01f);
        ov::test::utils::compare(expectedTensors[2], actualTensors[2], 1.0f);
    }

    void validate() override {
        std::vector<ov::Tensor> actualOutputs;
        if (envConfig.IE_NPU_TESTS_RUN_INFER) {
            actualOutputs = get_plugin_outputs();
        }

        // Compare NPU outputs from the dedicated internal op against CPU outputs from the decomposed reference graph.
        auto referenceCompiledModel = core->compile_model(_decomposedReferenceFunction, "CPU");
        auto referenceInferRequest = referenceCompiledModel.create_infer_request();
        referenceInferRequest.set_input_tensor(0, inputs.at(function->inputs().front().get_node_shared_ptr()));
        referenceInferRequest.infer();

        std::vector<ov::Tensor> expectedOutputs;
        for (const auto& output : referenceCompiledModel.outputs()) {
            expectedOutputs.push_back(referenceInferRequest.get_tensor(output));
        }

        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
                << "CPU reference outputs count mismatch for internal DynamicQuantize test";
        compare(expectedOutputs, actualOutputs);
    }

private:
    std::shared_ptr<ov::Model> _decomposedReferenceFunction;
};

class UnsupportedInternalDynamicQuantizeTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<NegativeInternalDQParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NegativeInternalDQParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << obj.param.testName;
        return result.str();
    }

    void SetUp() override {
        const auto& params = GetParam();
        init_input_shapes(ov::test::static_shapes_to_test_representation({params.inputShape}));
        function = buildInternalDQModel(params.inputShape, params.inputType, params.attrs, params.testName);
    }

    void runNegativeTest(const std::string_view platform) {
        configuration[ov::intel_npu::platform.name()] = std::string(platform);
        targetDevice = test_utils::TARGET_DEVICE;
        OV_EXPECT_THROW_HAS_SUBSTRING(compile_model(), std::runtime_error, GetParam().expectedError);
    }
};

TEST_P(InternalDynamicQuantizeTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    setPluginCompilerType();
    run(Platform::NPU3720);
}

TEST_P(InternalDynamicQuantizeTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(InternalDynamicQuantizeTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(InternalDynamicQuantizeTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    setPluginCompilerType();
    run(Platform::NPU5020);
}

TEST_P(UnsupportedInternalDynamicQuantizeTest, NPU4000) {
    setPluginCompilerType();
    runNegativeTest(Platform::NPU4000);
}

const std::vector<InternalDQParams> positiveCases = {
        {{1, 4, 8}, ov::element::f32, ov::element::u8, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM}},
        {{1, 4, 8}, ov::element::f16, ov::element::i8, ov::element::f16, {1, 1, 8}},
};

const std::vector<NegativeInternalDQParams> negativeCases = [] {
    std::vector<NegativeInternalDQParams> cases;

    auto symmetricAttrs = makeSupportedAttrs(ov::element::u8, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM});
    symmetricAttrs.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
    cases.push_back(
            {"RejectSymmetric", {1, 4, 8}, ov::element::f32, symmetricAttrs, "supports only asymmetric quantization"});

    auto interleavedAttrs = makeSupportedAttrs(ov::element::u8, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM});
    interleavedAttrs.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;
    interleavedAttrs.scale_dt = ov::element::u8;
    interleavedAttrs.zp_dt = ov::element::u8;
    cases.push_back({"RejectInterleaved",
                     {1, 4, 8},
                     ov::element::f32,
                     interleavedAttrs,
                     "supports only planar output storage"});

    auto precomputedAttrs = makeSupportedAttrs(ov::element::u8, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM});
    precomputedAttrs.precomputed_reduction = true;
    precomputedAttrs.precomputed_reduction_dt = ov::element::f32;
    cases.push_back({"RejectPrecomputedReduction",
                     {1, 4, 8},
                     ov::element::f32,
                     precomputedAttrs,
                     "does not support precomputed reduction"});

    auto non8BitAttrs = makeSupportedAttrs(ov::element::u4, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM});
    cases.push_back({"RejectNon8BitQuantType",
                     {1, 4, 8},
                     ov::element::f32,
                     non8BitAttrs,
                     "supports only 8-bit integer quantization"});

    auto nonIdentityOrderAttrs =
            makeSupportedAttrs(ov::element::u8, ov::element::f32, {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM});
    nonIdentityOrderAttrs.scales_zp_output_order = {0, 2, 1};
    cases.push_back({"RejectNonIdentityOrder",
                     {1, 4, 8},
                     ov::element::f32,
                     nonIdentityOrderAttrs,
                     "supports only identity scales_zp_output_order"});

    auto partialGroupAttrs = makeSupportedAttrs(ov::element::u8, ov::element::f32, {1, 1, 4});
    cases.push_back({"RejectPartialGroupSize",
                     {1, 4, 8},
                     ov::element::f32,
                     partialGroupAttrs,
                     "Only 1 or full-dimension group sizes are supported"});

    return cases;
}();

// [E#-220941] Temporarily disabled: these tests exercise ov::op::internal::DynamicQuantize through the PLUGIN
// (in-plugin VCL) compiler path, which serializes the model to IR and deserializes it inside the compiler. The op
// attributes (group_sizes, scales_zp_output_order, quantization_type, per-tensor data types, output_storage_type)
// only survive that round-trip once ov::op::internal::DynamicQuantize::visit_attributes() exists in OpenVINO.
// The OpenVINO commit currently pinned in validation/openvino_config.json (655b96740d) predates that support, so the
// deserialized op is rebuilt with empty group_sizes and shape_infer aborts with
// "Scale_shape and group_size are supposed to have same rank: N / 0" before any NPU-side logic runs. This breaks both
// the positive round-trip cases and the negative rejection cases (the "N / 0" abort masks the expected NPU error).
//
// The required support already exists upstream: PR openvinotoolkit/openvino#35283
// (commit c508c7595e307571828f7d15e979471614cdcbe2) added DynamicQuantize::visit_attributes(), and it is present on
// current OpenVINO master. Remove the DISABLED_ prefixes once validation/openvino_config.json is bumped to an
// OpenVINO commit that includes c508c7595e; the tests then pass unchanged with no NPU-side or OpenVINO-side fix.
INSTANTIATE_TEST_SUITE_P(DISABLED_precommit_InternalDynamicQuantize, InternalDynamicQuantizeTestCommon,
                         ::testing::ValuesIn(positiveCases), InternalDynamicQuantizeTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_InternalDynamicQuantize_Negative, UnsupportedInternalDynamicQuantizeTest,
                         ::testing::ValuesIn(negativeCases), UnsupportedInternalDynamicQuantizeTest::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
