// Copyright (C) 2021-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/mat_mul.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
class MatMulLayerTestCommon : public MatMulLayerTest, virtual public VpuOv2LayerTest {};

class MatMulLayerTest_HW_NPU3720 : public MatMulLayerTestCommon {};
class MatMulLayerTest_SW_NPU3720 : public MatMulLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-grouped-matmul=false";
    }
};

class MatMulLayerTest_HW_NPU3720_ppe_fp16_clamp : public MatMulLayerTestCommon {
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == funcInputs.size(),
                        "Input shapes number does not match with inputs number");

        auto createAndFillTensor = [](ov::Shape inputStaticShape) -> ov::Tensor {
            auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
            const auto totalSize =
                    std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
            for (size_t i = 0; i < totalSize; i++) {
                inputData[i] = static_cast<float>(300.0f);
            }
            return inputTensor;
        };

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = createAndFillTensor(inputShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

class MatMulLayerTest_HW_NPU4000 : public MatMulLayerTestCommon {};
class MatMulLayerTest_SW_NPU4000 : public MatMulLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-grouped-matmul=false";
    }
};
TEST_P(MatMulLayerTest_HW_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulLayerTest_SW_NPU3720, SW) {
    rel_threshold = 0.001;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulLayerTest_SW_NPU4000, SW) {
    rel_threshold = 0.001;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(MatMulLayerTest_HW_NPU4000, HW) {
    rel_threshold = 0.001;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(MatMulLayerTest_HW_NPU3720_ppe_fp16_clamp, HW) {
    rel_threshold = 1;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f32, ov::element::f16};

const std::vector<std::pair<bool, bool>> transposeInputs = {
        std::make_pair(false, false),
        std::make_pair(false, true),
        std::make_pair(true, false),
        std::make_pair(true, true),
};

const std::vector<std::vector<ov::Shape>> shapeRelatedParamsPrecommit = {
        {{1, 4, 5, 6}, {1, 4, 6, 4}}, {{4, 5, 6}, {6, 3}}, {{9, 9, 9}, {9, 9}}};

const std::vector<std::vector<ov::Shape>> shapeRelatedParamsNoTrans = {{{1, 2, 5, 16}, {1, 2, 16, 4}},
                                                                       {{1, 16, 16, 2}, {1, 2, 2}},
                                                                       {{8, 1, 1500}, {8, 1500, 64}},
                                                                       {{1, 1, 8, 1, 64}, {8, 64, 64}},
                                                                       {{64}, {64, 32}},
                                                                       {{64, 32}, {32}}};

const std::vector<std::vector<ov::Shape>> shapeRelatedParamsFirstTrans = {{{8, 64, 76}, {64, 4}}};

const std::vector<std::vector<ov::Shape>> shapeRelatedParamsSecondTrans = {{{1, 8, 76, 64}, {1, 8, 4, 64}},
                                                                           {{8, 76, 64}, {4, 64}},
                                                                           {{1, 1, 1, 3}, {12, 3}},
                                                                           {{16, 4, 49, 32}, {16, 4, 49, 32}},
                                                                           {{1, 1, 1, 1, 64}, {1, 32, 64}}};

const std::vector<std::vector<ov::Shape>> shapeRelatedParamsBothTrans = {
        {{2, 16, 5}, {16, 16}},
};

const std::vector<std::vector<ov::Shape>> fullyConnectedShapeParamsSecondTrans = {
        {{1, 16}, {64, 16}},
        {{2, 16}, {64, 16}},
        {{2, 1, 512}, {2, 40, 512}},
        {{1, 1, 256}, {1, 16, 256}},
};

const std::vector<std::vector<ov::Shape>> fullyConnectedShapeParamsNoTrans = {
        {{1, 16}, {16, 64}},
        {{1, 8, 4, 64}, {1, 8, 64, 76}},
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

const auto fullyConnectedCaseSecondTrans = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(fullyConnectedShapeParamsSecondTrans)),
        ::testing::Values(transposeInputs[1]), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

const auto fullyConnectedCaseNoTrans =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(fullyConnectedShapeParamsNoTrans)),
                           ::testing::Values(transposeInputs[0]), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto matMulParamsPrecommit =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParamsPrecommit)),
                           ::testing::Values(transposeInputs[0]), ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto matMulParamsNoTrans =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParamsNoTrans)),
                           ::testing::Values(transposeInputs[0]), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto matMulParamsFirstTrans =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParamsFirstTrans)),
                           ::testing::Values(transposeInputs[2]), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto matMulParamsSecondTrans =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParamsSecondTrans)),
                           ::testing::Values(transposeInputs[1]), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

const auto matMulParamsBothTrans =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParamsBothTrans)),
                           ::testing::Values(transposeInputs[3]), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(secondaryInputTypes), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

/* ============= NPU3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_MatMul, MatMulLayerTest_HW_NPU3720, matMulParamsPrecommit,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_to_FC_caseSecondTrans, MatMulLayerTest_SW_NPU3720, fullyConnectedCaseSecondTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_to_FC_caseNoTrans, MatMulLayerTest_SW_NPU3720, fullyConnectedCaseNoTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulNoTrans, MatMulLayerTest_SW_NPU3720, matMulParamsNoTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulFirstTrans, MatMulLayerTest_SW_NPU3720, matMulParamsFirstTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSecondTrans, MatMulLayerTest_SW_NPU3720, matMulParamsSecondTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulBothTrans, MatMulLayerTest_SW_NPU3720, matMulParamsBothTrans,
                         MatMulLayerTest_SW_NPU3720::getTestCaseName);

const std::vector<std::vector<ov::Shape>> shape1dBothTransClamp = {
        {{10, 10}, {10, 10}},
};

const auto matMul1dBothTransClampParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape1dBothTransClamp)),
                           ::testing::Values(transposeInputs.at(3)), ::testing::Values(ov::element::f32),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMul1dBothTransClamp, MatMulLayerTest_HW_NPU3720_ppe_fp16_clamp,
                         matMul1dBothTransClampParams, MatMulLayerTest_HW_NPU3720_ppe_fp16_clamp::getTestCaseName);

/* ============= NPU4000 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_MatMul, MatMulLayerTest_SW_NPU4000, matMulParamsPrecommit,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_to_FC_caseSecondTrans, MatMulLayerTest_SW_NPU4000, fullyConnectedCaseSecondTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_to_FC_caseNoTrans, MatMulLayerTest_SW_NPU4000, fullyConnectedCaseNoTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulNoTrans, MatMulLayerTest_SW_NPU4000, matMulParamsNoTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulFirstTrans, MatMulLayerTest_SW_NPU4000, matMulParamsFirstTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSecondTrans, MatMulLayerTest_SW_NPU4000, matMulParamsSecondTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMulBothTrans, MatMulLayerTest_SW_NPU4000, matMulParamsBothTrans,
                         MatMulLayerTest_SW_NPU4000::getTestCaseName);

const std::vector<std::vector<ov::Shape>> shapeRelatedParams = {{{49, 4, 49, 32}, {49, 4, 32, 49}}};

const auto matMulParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParams)),
                           ::testing::Values(std::make_pair(false, false)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMagicConfig, MatMulLayerTest_HW_NPU4000, matMulParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);

const std::vector<std::vector<ov::Shape>> shapeRelatedParams1 = {{{2, 2, 49, 49}, {2, 2, 49, 32}}};

const auto matMulParams1 =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeRelatedParams1)),
                           ::testing::Values(std::make_pair(false, false)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMagicConfig1, MatMulLayerTest_HW_NPU4000, matMulParams1,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);

const std::vector<std::vector<ov::Shape>> shape1dNoTrans = {
        {{32}, {32}},
        {{32}, {32, 16}},
        {{16, 32}, {32}},
};

const auto matMul1dNoTransParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape1dNoTrans)),
                           ::testing::Values(transposeInputs.at(0)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMul1dNoTrans, MatMulLayerTest_HW_NPU3720, matMul1dNoTransParams,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul1dNoTrans, MatMulLayerTest_HW_NPU4000, matMul1dNoTransParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);
const std::vector<std::vector<ov::Shape>> shape1dFirstTrans = {
        {{32}, {32}},
        {{32}, {32, 16}},
        {{64, 32}, {64}},
};

const auto matMul1dFirstTransParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape1dFirstTrans)),
                           ::testing::Values(transposeInputs.at(2)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMul1dFirstTrans, MatMulLayerTest_HW_NPU3720, matMul1dFirstTransParams,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul1dFirstTrans, MatMulLayerTest_HW_NPU4000, matMul1dFirstTransParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);
const std::vector<std::vector<ov::Shape>> shape1dSecondTrans = {
        {{32}, {32}},
        {{32}, {16, 32}},
        {{16, 32}, {32}},
};

const auto matMul1dSecondTransParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape1dSecondTrans)),
                           ::testing::Values(transposeInputs.at(1)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMul1dSecondTrans, MatMulLayerTest_HW_NPU3720, matMul1dSecondTransParams,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul1dSecondTrans, MatMulLayerTest_HW_NPU4000, matMul1dSecondTransParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);
const std::vector<std::vector<ov::Shape>> shapeWithBatch = {
        {{4, 16, 32}, {32}},
        {{4, 8, 16}, {4, 16, 64}},
        {{2, 3, 4, 16}, {16, 32}},
        {{2, 3, 4, 16}, {2, 3, 16, 32}},
};

const auto matMulWithBatchParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shapeWithBatch)),
                           ::testing::Values(transposeInputs.at(0)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMulWithBatch, MatMulLayerTest_HW_NPU3720, matMulWithBatchParams,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMulWithBatch, MatMulLayerTest_HW_NPU4000, matMulWithBatchParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);
const std::vector<std::vector<ov::Shape>> shape1dBothTrans = {
        {{32}, {32}},
        {{32}, {16, 32}},
        {{16, 32}, {16}},
};

const auto matMul1dBothTransParams =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape1dBothTrans)),
                           ::testing::Values(transposeInputs.at(3)), ::testing::Values(ov::element::f16),
                           ::testing::Values(InputLayerType::PARAMETER), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(MatMul1dBothTrans, MatMulLayerTest_HW_NPU3720, matMul1dBothTransParams,
                         MatMulLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul1dBothTrans, MatMulLayerTest_HW_NPU4000, matMul1dBothTransParams,
                         MatMulLayerTest_HW_NPU4000::getTestCaseName);
}  // namespace
