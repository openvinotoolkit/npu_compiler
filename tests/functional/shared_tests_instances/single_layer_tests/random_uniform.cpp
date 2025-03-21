//
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/random_uniform.hpp"
#include "common_test_utils/test_constants.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class RandomLayerTestCommon : public RandomUniformLayerTest, virtual public VpuOv2LayerTest {
    // TODO: E#92001 Resolve the dependency of dummy parameter for layer with all constant inputs
    // OpenVino 'SetUp' builds the test-ngraph without any Parameter (since inputs are Constants) => Exception:
    // "/vpux-plugin/src/vpux_imd_backend/src/infer_request.cpp:73 No information about network's output/input"
    // So cloning locally 'SetUp' and providing a 'dummy' Parameter

    template <ov::element::Type_t e>
    std::shared_ptr<ov::op::v0::Constant> createRangeConst(const fundamental_type_for<e>& value) {
        return std::make_shared<ov::op::v0::Constant>(e, ov::Shape{}, std::vector<fundamental_type_for<e>>{value});
    }

    std::shared_ptr<ov::op::v0::Constant> createConstant(ov::element::Type e, double value) {
        switch (e) {
        case ov::element::f32:
            return createRangeConst<ov::element::f32>(
                    static_cast<fundamental_type_for<ov::element::Type_t::f32>>(value));
        case ov::element::f16:
            return createRangeConst<ov::element::f16>(
                    static_cast<fundamental_type_for<ov::element::Type_t::f16>>(value));
        default:
            return createRangeConst<ov::element::i32>(
                    static_cast<fundamental_type_for<ov::element::Type_t::i32>>(value));
        }
    }

    void SetUp() override {
        RandomUniformTypeSpecificParams randomUniformParams;
        ov::Shape inputShape;
        int64_t globalSeed;
        int64_t opSeed;
        std::tie(inputShape, randomUniformParams, globalSeed, opSeed, std::ignore /*PhiloxAlignment*/, std::ignore) =
                this->GetParam();
        auto model_type = randomUniformParams.model_type;

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto input = std::make_shared<ov::op::v0::Parameter>(model_type, inputShape);
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input);

        std::shared_ptr<ov::op::v0::Constant> minValue, maxValue;
        minValue = createConstant(model_type, randomUniformParams.min_value);
        maxValue = createConstant(model_type, randomUniformParams.max_value);
        auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(shape_of, minValue, maxValue, model_type,
                                                                          globalSeed, opSeed);

        VpuOv2LayerTest::function =
                std::make_shared<ov::Model>(random_uniform->outputs(), ov::ParameterVector{input}, "random_uniform");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

class RandomLayerTest_F32 : public RandomLayerTestCommon {
    void configure_model() override {
        VpuOv2LayerTest::configuration[ov::intel_npu::compilation_mode_params.name()] =
                "convert-precision-to-fp16=false";
    }
};

TEST_P(RandomLayerTestCommon, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(RandomLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

TEST_P(RandomLayerTest_F32, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<RandomUniformTypeSpecificParams> randomUniformSpecificParams = {
        {ov::element::f16, 0.0f, 1.0f}, {ov::element::f16, -10.0, 10.0}, {ov::element::i32, -20, 90}};

const std::vector<RandomUniformTypeSpecificParams> randomUniformSpecificParamsF32 = {{ov::element::f32, 0.0f, 1.0f},
                                                                                     {ov::element::f32, -10.0, 10.0}};

const std::vector<int64_t> globalSeeds = {0, 3456};
const std::vector<int64_t> opSeeds = {11, 876};

const std::vector<ov::Shape> outputShapes = {{1, 200}, {1, 4, 64, 64}};

// FIXME(E#151501): this param is ignored by the tests
const std::vector<ov::op::PhiloxAlignment> philoxAlignments = {ov::op::PhiloxAlignment::TENSORFLOW,
                                                               ov::op::PhiloxAlignment::PYTORCH};

const auto randParams =
        ::testing::Combine(::testing::ValuesIn(outputShapes), ::testing::ValuesIn(randomUniformSpecificParams),
                           ::testing::ValuesIn(globalSeeds), ::testing::ValuesIn(opSeeds),
                           ::testing::ValuesIn(philoxAlignments), ::testing::Values(DEVICE_NPU));

const auto randParamsF32 =
        ::testing::Combine(::testing::Values(outputShapes[1]), ::testing::ValuesIn(randomUniformSpecificParamsF32),
                           ::testing::Values(globalSeeds[0]), ::testing::Values(opSeeds[1]),
                           ::testing::ValuesIn(philoxAlignments), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RandomUniform, RandomLayerTestCommon, randParams,
                         RandomLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RandomUniform, RandomLayerTest_F32, randParamsF32, RandomLayerTest_F32::getTestCaseName);

}  // namespace
