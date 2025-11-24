//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "single_op_tests/eltwise.hpp"
#include "vpu_ov2_layer_test.hpp"

using ov::test::utils::EltwiseTypes;
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;

namespace ov {
namespace test {

class EltwiseLayerTestCommon : public EltwiseLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override;
    // Helper for filling base tensor for POWER operation.
    // Force only positive base values for stability in POWER tests.
    template <typename T>
    void fill_power_base_tensor(ov::Tensor& tensor) {
        auto* data = tensor.data<T>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            float val = static_cast<float>(data[i]);
            if (val == 0.0f) {
                val = 1.0f;
            }
            val = std::abs(val);
            data[i] = static_cast<T>(val);
        }
    }

    // Helper for filling exponent tensor for POWER operation.
    // Scalars default to 0.5 to stress the fractional exponent path, while
    // vector tensors still cycle through a mix of integer and fractional values.
    template <typename T>
    void fill_power_exp_tensor(ov::Tensor& tensor) {
        auto* data = tensor.data<T>();
        constexpr float exponents[] = {2.0f, 3.0f, 0.5f, -1.0f, -0.5f, 0.0f};
        if (tensor.get_size() == 1) {
            data[0] = static_cast<T>(0.5f);
        } else {
            constexpr size_t exponents_size = sizeof(exponents) / sizeof(exponents[0]);
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<T>(exponents[i % exponents_size]);
            }
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto secondInputType = std::get<2>(GetParam());
        const auto netPrecision = std::get<4>(GetParam());

        // This block generates test inputs for POWER with positive bases and fractional/integer exponents
        // expTensor: covers integer and fractional exponents
        // Create and fill base tensor with values in positive range
        // f16 is more prone to precision issues on large magnitudes; keep values smaller
        if (eltwiseType == EltwiseTypes::POWER &&
            (netPrecision == ov::element::f16 || netPrecision == ov::element::f32) &&
            targetInputStaticShapes.size() == 2) {
            ov::Tensor baseTensor = [&]() {
                if (netPrecision == ov::element::f16) {
                    return ov::test::utils::create_and_fill_tensor(netPrecision, targetInputStaticShapes[0], 5, 1, 1);
                } else {
                    return ov::test::utils::create_and_fill_tensor(netPrecision, targetInputStaticShapes[0], 11, 1, 1);
                }
            }();
            ov::Tensor expTensor(netPrecision, targetInputStaticShapes[1]);
            if (netPrecision == ov::element::f16) {
                fill_power_base_tensor<ov::float16>(baseTensor);
                fill_power_exp_tensor<ov::float16>(expTensor);
            } else if (netPrecision == ov::element::f32) {
                fill_power_base_tensor<float>(baseTensor);
                fill_power_exp_tensor<float>(expTensor);
            }
            inputs[function->get_parameters()[0]] = baseTensor;
            if (secondInputType == InputLayerType::PARAMETER) {
                inputs[function->get_parameters()[1]] = expTensor;
            }
        } else {
            EltwiseLayerTest::generate_inputs(targetInputStaticShapes);
        }
    }
};

void EltwiseLayerTestCommon::SetUp() {
    EltwiseLayerTest::SetUp();

    const auto params = GetParam();
    const auto eltwiseType = std::get<1>(params);
    const auto secondInputType = std::get<2>(params);
    const auto netPrecision = std::get<4>(params);

    if (eltwiseType != EltwiseTypes::POWER || secondInputType != InputLayerType::CONSTANT ||
        !(netPrecision == ov::element::f16 || netPrecision == ov::element::f32)) {
        return;
    }

    auto orderedOps = function->get_ordered_ops();
    auto constantIt = std::find_if(orderedOps.begin(), orderedOps.end(), [](const std::shared_ptr<ov::Node>& node) {
        return std::dynamic_pointer_cast<ov::op::v0::Constant>(node) != nullptr &&
               node->get_friendly_name() == "param1";
    });

    if (constantIt == orderedOps.end()) {
        return;
    }

    auto constantNode = std::dynamic_pointer_cast<ov::op::v0::Constant>(*constantIt);
    ov::Tensor expTensor(netPrecision, constantNode->get_shape());

    if (netPrecision == ov::element::f16) {
        fill_power_exp_tensor<ov::float16>(expTensor);
    } else {
        fill_power_exp_tensor<float>(expTensor);
    }

    auto newConstant = std::make_shared<ov::op::v0::Constant>(expTensor);
    newConstant->set_friendly_name(constantNode->get_friendly_name());
    ov::replace_node(constantNode, newConstant);
    function->validate_nodes_and_infer_types();
}

class EltwiseLayerTestF32Common : public EltwiseLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class EltwiseLayerTestDynamic : public EltwiseLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> shapes;
        ov::test::utils::InputLayerType secondInputType;
        ov::test::Config testConfig;
        ov::element::Type modelType;
        ov::test::utils::EltwiseTypes eltwiseOpType;

        std::tie(shapes, eltwiseOpType, secondInputType,
                 std::ignore,  // OpType
                 modelType,    // precision
                 std::ignore,  // Type1
                 std::ignore,  // Type2
                 std::ignore,  // TARGET_DEVICE
                 testConfig) = this->GetParam();
        configuration.insert(testConfig.begin(), testConfig.end());

        init_input_shapes(shapes);

        ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> secondInput;
        if (secondInputType == InputLayerType::PARAMETER) {
            secondInput = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes[1]);
            inputs.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondInput));
        } else {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(modelType, targetStaticShapes.front()[1]);
            secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        }

        auto eltwiseNode = ov::test::utils::make_eltwise(inputs[0], secondInput, eltwiseOpType);

        auto convertedEltwiseNode = std::make_shared<ov::op::v0::Convert>(eltwiseNode, modelType);
        function = std::make_shared<ov::Model>(convertedEltwiseNode, inputs, "DynamicEltwise");
    }
};

class EltwiseEmptyShapeInputLayerTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};
class EltwiseIntegerLayerTest : public EltwiseLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(EltwiseLayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.6;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestCommon, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        // [Tracking number: E#82236]
        if (netPrecisions == ov::element::i32) {
            skip << "Type is not supported";
        }
    });

    abs_threshold = 0.6;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.6;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseLayerTestF32Common, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseLayerTestF32Common, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestF32Common, NPU3720_HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(GetParam());
        const auto netPrecisions = std::get<4>(GetParam());
        if (netPrecisions == ov::element::f32) {
            skip << "FP32 operations will be converted to IE.scaleshift in AdjustScaleShiftForDWConv in HW Mode. "
                    "IE.scaleshift is a NCE task which do not support FP32";
        }
    });

    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
TEST_P(EltwiseEmptyShapeInputLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseEmptyShapeInputLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
void setCommonSkipCompilationCallback(EltwiseIntegerLayerTest* test) {
    test->setSkipCompilationCallback([test](std::stringstream& skip) {
        const auto eltwiseType = std::get<1>(test->GetParam());
        const auto netPrecisions = std::get<4>(test->GetParam());

        // Define sets of unsupported types for specific precisions
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForU16 = {
                EltwiseTypes::SUBTRACT, EltwiseTypes::FLOOR_MOD, EltwiseTypes::MULTIPLY, EltwiseTypes::DIVIDE,
                EltwiseTypes::POWER};
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForU8 = {
                EltwiseTypes::MULTIPLY, EltwiseTypes::DIVIDE, EltwiseTypes::POWER};
        static const std::unordered_set<EltwiseTypes> unsupportedTypesForI16 = {EltwiseTypes::FLOOR_MOD};

        // Check if the current combination of precision and eltwiseType is unsupported
        bool isUnsupported = (netPrecisions == ov::element::u16 && unsupportedTypesForU16.count(eltwiseType)) ||
                             (netPrecisions == ov::element::u8 && unsupportedTypesForU8.count(eltwiseType)) ||
                             (netPrecisions == ov::element::i16 && unsupportedTypesForI16.count(eltwiseType));

        if (isUnsupported) {
            skip << eltwiseType << " SingleLayerTest is not enabled with precision: " << netPrecisions;
        }
    });
}

TEST_P(EltwiseIntegerLayerTest, NPU3720_SW) {
    abs_threshold = 0.6;
    setCommonSkipCompilationCallback(this);
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseIntegerLayerTest, NPU4000_SW) {
    abs_threshold = 0.6;
    setCommonSkipCompilationCallback(this);
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseLayerTestDynamic, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseLayerTestDynamic, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

namespace {

using namespace ov::test;

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f16,
        ov::element::i32,
};

std::vector<ov::test::ElementType> netPrecisionsF16 = {
        ov::element::f16,
};

std::vector<ov::test::ElementType> netPrecisionsF32 = {
        ov::element::f32,
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::VECTOR,
        ov::test::utils::OpType::SCALAR,
};

//
// Test supported Eltwise types + Tiling
//

std::set<EltwiseTypes> eltwiseTypes = {EltwiseTypes::ADD,       EltwiseTypes::MULTIPLY,     EltwiseTypes::SUBTRACT,
                                       EltwiseTypes::DIVIDE,    EltwiseTypes::SQUARED_DIFF, EltwiseTypes::POWER,
                                       EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD};

std::set<EltwiseTypes> eltwiseTypesF32 = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY, EltwiseTypes::POWER,
                                          EltwiseTypes::DIVIDE, EltwiseTypes::SUBTRACT};

std::vector<std::vector<ov::Shape>> bigShape = {{{1, 10, 256, 256}, {1, 10, 256, 256}}};

const auto typesParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypes, EltwiseLayerTestCommon, typesParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto typesParamsF32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bigShape)),
        ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes), ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisionsF32), ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_EltwiseTypesF32, EltwiseLayerTestF32Common, typesParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
// Dynamic shape tests
//

std::set<EltwiseTypes> DynamicEltwiseOpTypes = {EltwiseTypes::SUBTRACT};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic = {
        {{{1, 1, 1, ov::Dimension(1, 10)}, {{1, 1, 1, 10}, {1, 1, 1, 5}}},
         {{1, 1, 1, ov::Dimension(1, 10)}, {{1, 1, 1, 1}}}},
};

std::vector<ov::test::ElementType> precision = {
        ov::element::f32,
        ov::element::f16,
};

std::vector<InputLayerType> secondInputType = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

const auto eltwise_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(in_shapes_dynamic), ::testing::ValuesIn(DynamicEltwiseOpTypes),
        ::testing::ValuesIn(secondInputType), ::testing::ValuesIn(opTypes), ::testing::ValuesIn(precision),
        ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

//  Dynamic shapes cases
INSTANTIATE_TEST_SUITE_P(smoke_EltwiseDynamic, EltwiseLayerTestDynamic, eltwise_params_dynamic,
                         EltwiseLayerTestDynamic::getTestCaseName);

//
// Test Eltwise input broadcast
//

std::set<EltwiseTypes> broadcastTestEltwiseTypes = {EltwiseTypes::ADD};

std::vector<std::vector<ov::Shape>> broadcastTestInputShape = {{{1, 320, 128, 128}, {1, 320, 1, 1}}};

const auto broadcastTestParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(broadcastTestInputShape)),
                           ::testing::ValuesIn(broadcastTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_InputBroadcastEltwise, EltwiseLayerTestCommon, broadcastTestParams,
                         EltwiseLayerTestCommon::getTestCaseName);

std::set<EltwiseTypes> scalarInput2broadcastTestEltwiseTypes = {EltwiseTypes::DIVIDE};
std::vector<std::vector<ov::Shape>> scalarInput2broadcastTestInputShape = {
        {{1, 1, 1, 512}, {1, 1, 1, 1}},
        {{1, 1, 512, 1}, {1, 1, 1, 1}},
        {{1, 512, 1, 1}, {1, 1, 1, 1}},
};
const auto scalarInput2BroadcastTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(scalarInput2broadcastTestInputShape)),
        ::testing::ValuesIn(scalarInput2broadcastTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16), ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::element::dynamic), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(ov::test::Config{}));
INSTANTIATE_TEST_SUITE_P(precommit_scalarInput2BroadcastEltwise, EltwiseLayerTestCommon,
                         scalarInput2BroadcastTestParams, EltwiseLayerTestCommon::getTestCaseName);

//
// Test Eltwise batch input
//

std::set<EltwiseTypes> batchInputTestEltwiseTypes = {EltwiseTypes::ADD};

std::vector<std::vector<ov::Shape>> batchInputTestInputShape = {{{361, 4, 48, 48}, {361, 4, 48, 48}}};

const auto batchInputTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(batchInputTestInputShape)),
        ::testing::ValuesIn(batchInputTestEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisionsF16), ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::element::dynamic), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_BatchInputEltwise, EltwiseLayerTestCommon, batchInputTestParams,
                         EltwiseLayerTestCommon::getTestCaseName);

//
// Scalar mode
//

std::vector<std::vector<ov::Shape>> inShapesScalar = {
        {{10}},              // 1D
        {{1, 9}},            // NC
        {{1, 128, 32}},      // CHW
        {{1, 3, 224, 224}},  // NCHW
};

const auto scalarParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesND, EltwiseLayerTestCommon, scalarParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto scalarParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesNDF32, EltwiseLayerTestF32Common, scalarParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
// Vector mode
//

std::vector<std::vector<ov::Shape>> inShapesVector = {
        {{24}, {24}},                          // 1D
        {{1, 9}, {1, 1}},                      // NC + scalar
        {{1, 128, 32}, {1, 128, 32}},          // CHW, eltwise
        {{1, 128, 32}, {1, 128, 1}},           // CHW, input1 != input2, broadcast over W
        {{1, 128, 32}, {1, 1, 32}},            // CHW, input1 != input2, broadcast over H
        {{1, 128, 32}, {1, 1, 1}},             // CHW + scalar
        {{1, 3, 224, 224}, {1, 3, 224, 224}},  // NCHW, eltwise
        {{1, 3, 224, 224}, {1, 1, 1, 1}},      // NCHW + scalar
        {{1, 3, 224, 224}, {1, 3, 1, 1}},      // NCHW, broadcast over HW
        {{2, 3, 224, 224}, {1, 1, 1, 224}},    // NCHW, N != 1, broadcast over NCH
};

const auto vectorParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesND, EltwiseLayerTestCommon, vectorParams,
                         EltwiseLayerTestCommon::getTestCaseName);

const auto vectorParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesVector)),
                           ::testing::ValuesIn(eltwiseTypesF32), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesNDF32, EltwiseLayerTestF32Common, vectorParamsF32,
                         EltwiseLayerTestF32Common::getTestCaseName);

//
//  This case to test the support for empty shape input for Add and Multiply ops
//
std::set<EltwiseTypes> eltwise0DInputOps = {EltwiseTypes::ADD, EltwiseTypes::MULTIPLY};

std::vector<std::vector<ov::Shape>> eltwise0DInputShape = {
        {{}},  // 0D
};

const auto vectorParamsEmptyShapeInput =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(eltwise0DInputShape)),
                           ::testing::ValuesIn(eltwise0DInputOps), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_0DInputTest, EltwiseEmptyShapeInputLayerTest, vectorParamsEmptyShapeInput,
                         EltwiseEmptyShapeInputLayerTest::getTestCaseName);

//
// Bitwise
//

std::vector<std::vector<ov::Shape>> bitwiseInput = {{{1, 1, 256, 56}, {1, 1, 256, 56}},
                                                    {{1, 1, 256, 56}, {1, 1, 256, 1}}};

std::vector<ov::test::ElementType> bitwiseNetPrecisions = {ov::element::i32};

std::set<EltwiseTypes> bitwiseTypes = {EltwiseTypes::BITWISE_AND, EltwiseTypes::BITWISE_OR, EltwiseTypes::BITWISE_XOR};

const auto bitwiseParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwiseInput)),
                           ::testing::ValuesIn(bitwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(bitwiseNetPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_Bitwise, EltwiseLayerTestCommon, bitwiseParams,
                         EltwiseLayerTestCommon::getTestCaseName);

std::vector<std::vector<ov::Shape>> bitwiseInputi8 = {{{1, 1, 256, 56}, {1, 1, 256, 1}}};

std::vector<ov::test::ElementType> bitwiseNetPrecisionsi8 = {ov::element::i8};

std::set<EltwiseTypes> bitwiseTypesi8 = {EltwiseTypes::BITWISE_OR};

const auto bitwiseParamsi8 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwiseInputi8)),
                           ::testing::ValuesIn(bitwiseTypesi8), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(bitwiseNetPrecisionsi8),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_Bitwisei8, EltwiseLayerTestCommon, bitwiseParamsi8,
                         EltwiseLayerTestCommon::getTestCaseName);

std::vector<std::vector<ov::Shape>> bitwiseNotInput = {{{1, 1, 256, 56}, {}}};

const auto bitwiseNotParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bitwiseNotInput)),
                           ::testing::Values(EltwiseTypes::BITWISE_NOT), ::testing::Values(InputLayerType::CONSTANT),
                           ::testing::ValuesIn(opTypes), ::testing::ValuesIn(bitwiseNetPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(precommit_BitwiseNot, EltwiseLayerTestCommon, bitwiseNotParams,
                         EltwiseLayerTestCommon::getTestCaseName);

//
// Test Unsigned Integer data types
//

std::vector<std::vector<ov::Shape>> inShape = {{{1, 5, 16, 32}, {1, 5, 16, 32}}};

std::vector<ov::test::ElementType> netPrecisionsUnsigned = {ov::element::u8, ov::element::u16, ov::element::u32,
                                                            ov::element::u64};

std::set<EltwiseTypes> eltwiseTypesUnsigned = {EltwiseTypes::ADD,    EltwiseTypes::SUBTRACT, EltwiseTypes::MULTIPLY,
                                               EltwiseTypes::DIVIDE, EltwiseTypes::POWER,    EltwiseTypes::FLOOR_MOD,
                                               EltwiseTypes::MOD};

const auto typesParamsUnsigned = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(eltwiseTypesUnsigned), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsUnsigned),
        ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Unsigned, EltwiseIntegerLayerTest, typesParamsUnsigned,
                         EltwiseIntegerLayerTest::getTestCaseName);

//
// Test Integer data types
//

std::vector<ov::test::ElementType> netPrecisionsInteger = {ov::element::i8, ov::element::i16, ov::element::i32};

std::set<EltwiseTypes> eltwiseTypesInteger = {EltwiseTypes::FLOOR_MOD, EltwiseTypes::MOD};

const auto typesParamsInteger = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(eltwiseTypesInteger), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsInteger),
        ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Signed, EltwiseIntegerLayerTest, typesParamsInteger,
                         EltwiseIntegerLayerTest::getTestCaseName);

}  // namespace

// ShaveCodeGen tests

namespace ov {
namespace test {

class ShaveCodeGenEltwiseLayerTestCommon : public EltwiseLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

class ShaveCodeGenEltwiseLayerTestF32Common : public ShaveCodeGenEltwiseLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-shave-code-gen=true convert-precision-to-fp16=false";
    }
};

class ShaveCodeGenEltwiseIntegerLayerTest : public EltwiseIntegerLayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-shave-code-gen=true";
    }
};

TEST_P(ShaveCodeGenEltwiseLayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.6;
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenEltwiseLayerTestF32Common, NPU4000_SW) {
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ShaveCodeGenEltwiseIntegerLayerTest, NPU4000_SW) {
    abs_threshold = 0.6;
    setCommonSkipCompilationCallback(this);
    setReferenceSoftwareMode();
    setMLIRCompilerType();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

namespace {

using namespace ov::test;

//
// Test supported Eltwise types
//

// Tests for the ADD operator are not enabled due to E#163155
std::set<EltwiseTypes> scgEltwiseTypes = {EltwiseTypes::DIVIDE, EltwiseTypes::SQUARED_DIFF, EltwiseTypes::SUBTRACT,
                                          EltwiseTypes::MULTIPLY};

//
// Scalar mode
//

const auto scgScalarParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(scgEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesND, ShaveCodeGenEltwiseLayerTestCommon, scgScalarParams,
                         ShaveCodeGenEltwiseLayerTestCommon::getTestCaseName);

const auto scgScalarParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesScalar)),
                           ::testing::ValuesIn(scgEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::SCALAR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_ScalarShapesNDF32, ShaveCodeGenEltwiseLayerTestF32Common, scgScalarParamsF32,
                         ShaveCodeGenEltwiseLayerTestF32Common::getTestCaseName);

//
// Vector mode
//

std::vector<std::vector<ov::Shape>> scgInShapesVector = {
        {{24}, {24}},                      // 1D
        {{1, 9}, {1, 1}},                  // NC + scalar
        {{1, 128, 32}, {1, 128, 32}},      // CHW, eltwise
        {{1, 128, 32}, {1, 128, 1}},       // CHW, input1 != input2, broadcast over W
        {{1, 128, 32}, {1, 1, 32}},        // CHW, input1 != input2, broadcast over H
        {{1, 128, 32}, {1, 1, 1}},         // CHW + scalar
        {{1, 3, 16, 16}, {1, 3, 16, 16}},  // NCHW, eltwise
        {{1, 3, 16, 16}, {1, 1, 1, 1}},    // NCHW + scalar
        {{1, 3, 16, 16}, {1, 3, 1, 1}},    // NCHW, broadcast over HW
                                           // E-152367: ShaveCodeGen Tiling support
                                           // Fails in unroll_cluster_tiling:
                                           // {{2, 1, 8, 16}, {1, 1, 1, 16}},        // NCHW, N != 1, broadcast over NCH
};

const auto scgVectorParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(scgInShapesVector)),
                           ::testing::ValuesIn(scgEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesND, ShaveCodeGenEltwiseLayerTestCommon, scgVectorParams,
                         ShaveCodeGenEltwiseLayerTestCommon::getTestCaseName);

const auto scgVectorParamsF32 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(scgInShapesVector)),
                           ::testing::ValuesIn(scgEltwiseTypes), ::testing::ValuesIn(secondaryInputTypes),
                           ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsF32),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::f32),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_VectorShapesNDF32, ShaveCodeGenEltwiseLayerTestF32Common, scgVectorParamsF32,
                         ShaveCodeGenEltwiseLayerTestF32Common::getTestCaseName);

//
// Test Unsigned Integer data types
//

// Tests for the ADD operator are not enabled due to E#163155
std::set<EltwiseTypes> scgEltwiseTypesUnsigned = {EltwiseTypes::DIVIDE, EltwiseTypes::SUBTRACT, EltwiseTypes::MULTIPLY};

const auto scgTypesParamsUnsigned = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(scgEltwiseTypesUnsigned), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(netPrecisionsUnsigned),
        ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Unsigned, ShaveCodeGenEltwiseIntegerLayerTest, scgTypesParamsUnsigned,
                         ShaveCodeGenEltwiseIntegerLayerTest::getTestCaseName);

//
// Test Integer data types
//

std::vector<ov::test::ElementType> scgNetPrecisionsInteger = {ov::element::i32, ov::element::i64};
std::set<EltwiseTypes> scgEltwiseTypesInteger = {EltwiseTypes::DIVIDE};

const auto scgTypesParamsInteger = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
        ::testing::ValuesIn(scgEltwiseTypesInteger), ::testing::Values(InputLayerType::PARAMETER),
        ::testing::Values(ov::test::utils::OpType::VECTOR), ::testing::ValuesIn(scgNetPrecisionsInteger),
        ::testing::Values(ov::element::dynamic), ::testing::Values(ov::element::dynamic),
        ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_Eltwise_Signed, ShaveCodeGenEltwiseIntegerLayerTest, scgTypesParamsInteger,
                         ShaveCodeGenEltwiseIntegerLayerTest::getTestCaseName);

//
// Bitwise
//

INSTANTIATE_TEST_SUITE_P(precommit_Bitwise, ShaveCodeGenEltwiseLayerTestCommon, bitwiseParams,
                         ShaveCodeGenEltwiseLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_Bitwisei8, ShaveCodeGenEltwiseLayerTestCommon, bitwiseParamsi8,
                         ShaveCodeGenEltwiseLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(precommit_BitwiseNot, ShaveCodeGenEltwiseLayerTestCommon, bitwiseNotParams,
                         ShaveCodeGenEltwiseLayerTestCommon::getTestCaseName);
}  // namespace
