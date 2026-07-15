//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/cum_sum.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CumSumLayerTest);

class CumSumLayerTestCommon : public CumSumLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(CumSumLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CumSumLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(CumSumLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
TEST_P(CumSumLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<std::vector<ov::Shape>> shapes = {{{5, 14, 5, 7}},
                                                    // Values from real neural networks
                                                    {{1, 1}},
                                                    {{1, 1024}},
                                                    {{1, 128}},
                                                    {{1, 25, 36}},
                                                    {{1, 384}},
                                                    {{1, 5}},
                                                    {{1, 9}},
                                                    {{8, 128}},
                                                    {{8, 384}}};

const std::vector<ov::element::Type> inputPrecision = {ov::element::f16, ov::element::f32};
const std::vector<ov::element::Type> inputLowPrecision = {ov::element::f16, ov::element::i32};

const std::vector<int64_t> axes = {0, 1};
const std::vector<int64_t> negativeAxes = {-2, -1};

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse = {true, false};

const auto testCaseAxis_0 =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}),  // Input shapes
                         testing::Values(inputPrecision[0]),                                      // Model type
                         testing::Values(axes[0]),                                                // Axis
                         testing::Values(exclusive[0]),                                           // Exclusive
                         testing::Values(reverse[1]),                                             // Reverse
                         testing::Values(test_utils::TARGET_DEVICE));                             // Device name

const auto testCasesNegativeAxis = testing::Combine(
        testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}), testing::Values(inputPrecision[1]),
        testing::ValuesIn(negativeAxes), testing::Values(exclusive[1]), testing::Values(reverse[0]),
        testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::Shape>> iShape(shapes.begin() + 1, shapes.end());
const auto testCasesRealNet =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShape)),
                         testing::Values(inputPrecision[0]), testing::Values(axes[1]), testing::Values(exclusive[1]),
                         testing::Values(reverse[1]), testing::Values(test_utils::TARGET_DEVICE));

const auto testCasePrecommit = testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[4]})}),
                                                testing::ValuesIn(inputLowPrecision), testing::ValuesIn(negativeAxes),
                                                testing::ValuesIn(exclusive), testing::ValuesIn(reverse),
                                                testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::test::InputShape>> cumSumDynamicShapes = {
        {{{{1, 1, 1, ov::Dimension(1, 10)}, {{1, 1, 1, 10}, {1, 1, 1, 5}}}},
         {{{1, ov::Dimension(1, 32)}, {{1, 16}, {1, 32}}}},
         {{{ov::Dimension(1, 8), 128}, {{4, 128}, {8, 128}}}}}};

const auto cumSumParamsDynamic = ::testing::Combine(
        ::testing::ValuesIn(cumSumDynamicShapes), ::testing::Values(inputPrecision[0]), ::testing::Values(axes[1]),
        ::testing::Values(exclusive[1]), ::testing::Values(reverse[1]), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_CumSumDynamic, CumSumLayerTestCommon, cumSumParamsDynamic,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_axis_0, CumSumLayerTestCommon, testCaseAxis_0,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_negative_axis, CumSumLayerTestCommon, testCasesNegativeAxis,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_real_net, CumSumLayerTestCommon, testCasesRealNet,
                         CumSumLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_CumSum, CumSumLayerTestCommon, testCasePrecommit,
                         CumSumLayerTestCommon::getTestCaseName);

// Tests for CumSumToMatMulRewriter (in FuseOpsToMatMulPass).
// Large tensors are rewritten to IE.MatMul with a triangular constant; later in the pipeline this MatMul is typically
// converted to an NCE-backed 1x1 IE::Convolution (see ConvertMatMulToConvPass), which accumulates in fp16 on HW.
class CumSumMatMulLayerTest : public CumSumLayerTest, virtual public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData inData;
            inData.start_from = 0;
            inData.range = 1;
            inData.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], inData);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-cumsum-to-matmul=true";
    }

    void SetUp() override {
        CumSumLayerTest::SetUp();
        abs_threshold = 1.5f;
        rel_threshold = 0.001f;
    }
};

TEST_P(CumSumMatMulLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CumSumMatMulLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(CumSumMatMulLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(CumSumMatMulLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<std::vector<ov::Shape>> matmulShapes = {
        // Primary use case: [1,64,4,256,256] axis=3 (axis=-2), batchSize=256, L=256, trailing=256
        {{1, 64, 4, 256, 256}},
        // Smaller 5D variant for fast smoke test
        {{1, 4, 2, 64, 64}},
        // 2D: batch=1, L=256, trailing=128
        {{256, 128}},
        // 3D
        {{8, 128, 64}},
};

const auto testCaseMatMulPrecommit =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({matmulShapes[1]})}),
                         testing::Values(inputPrecision[0]), testing::Values(int64_t{3}), testing::ValuesIn(exclusive),
                         testing::ValuesIn(reverse), testing::Values(test_utils::TARGET_DEVICE));

const auto testCaseMatMulPrimary =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({matmulShapes[0]})}),
                         testing::Values(inputPrecision[0]), testing::Values(int64_t{3}), testing::Values(false),
                         testing::Values(false), testing::Values(test_utils::TARGET_DEVICE));

const auto testCaseMatMul2D =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({matmulShapes[2]})}),
                         testing::Values(inputPrecision[0]), testing::Values(int64_t{0}), testing::Values(false),
                         testing::Values(false), testing::Values(test_utils::TARGET_DEVICE));

const auto testCaseMatMul3D =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({matmulShapes[3]})}),
                         testing::Values(inputPrecision[0]), testing::Values(int64_t{1}), testing::Values(false),
                         testing::Values(false), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_CumSumMatMul, CumSumMatMulLayerTest, testCaseMatMulPrecommit,
                         CumSumMatMulLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumMatMul_primary, CumSumMatMulLayerTest, testCaseMatMulPrimary,
                         CumSumMatMulLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumMatMul_2D, CumSumMatMulLayerTest, testCaseMatMul2D,
                         CumSumMatMulLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumMatMul_3D, CumSumMatMulLayerTest, testCaseMatMul3D,
                         CumSumMatMulLayerTest::getTestCaseName);

}  // namespace
