//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/normalize_l2.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
class NormalizeL2LayerTestCommon : public NormalizeL2LayerTest, virtual public VpuOv2LayerTest {};
class NormalizeL2LayerTest_6DPU : public NormalizeL2LayerTestCommon {};
class NormalizeL2LayerTest_2DPU : public NormalizeL2LayerTestCommon {};

TEST_P(NormalizeL2LayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NormalizeL2LayerTestCommon, NPU4000_HW) {
    abs_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(NormalizeL2LayerTest_6DPU, NPU4000_HW) {
    abs_threshold = 0.02;
    setDefaultHardwareMode();
    configuration["NPU_TILES"] = "6";
    run(Platform::NPU4000);
}

TEST_P(NormalizeL2LayerTest_2DPU, NPU4000_HW) {
    abs_threshold = 0.02;
    setDefaultHardwareMode();
    configuration["NPU_TILES"] = "2";
    run(Platform::NPU4000);
}
TEST_P(NormalizeL2LayerTestCommon, NPU5010_HW) {
    abs_threshold = 0.04;
    rel_threshold = 0.04;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<ov::op::EpsMode> epsMode = {
        ov::op::EpsMode::ADD,
        ov::op::EpsMode::MAX,
};

const std::vector<float> eps = {
        9.9999999392252903e-09,
        1.000000013351432e-10,
        9.999999960041972e-13,
        9.9999998245167004e-14,
};

const std::vector<std::vector<int64_t>> axes2D = {{1}};
const std::vector<std::vector<int64_t>> axes3D = {{1}, {1, 2}, {0, 1, 2}};
const std::vector<std::vector<int64_t>> axes4D = {{1}, {1, 2}, {0, 1, 2}, {0, 1, 2, 3}};
const std::vector<std::vector<int64_t>> axesMini4D = {{1}, {1, 2}};
const std::vector<std::vector<int64_t>> axesTiling4D = {{1}, {2}, {3}, {1, 2}};

std::vector<std::vector<ov::Shape>> shapes2D = {
        {{1, 128}},
        {{1, 256}},
        {{1, 512}},
};

std::vector<std::vector<ov::Shape>> shapes3D = {{{1, 5, 3}}, {{1, 20, 200}}};

std::vector<std::vector<ov::Shape>> shapes4D = {
        {{1, 8, 24, 64}},   {{1, 1024, 1, 1}},  {{1, 128, 50, 85}}, {{1, 512, 64, 64}},
        {{1, 512, 40, 40}}, {{1, 512, 20, 20}}, {{1, 512, 38, 38}}, {{1, 128, 25, 43}},
};

std::vector<std::vector<ov::Shape>> shapesTiling4D = {{{1, 512, 36, 36}}, {{1, 512, 37, 37}}, {{1, 512, 44, 43}}};

const auto params2D = testing::Combine(testing::ValuesIn(axes2D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(static_shapes_to_test_representation(shapes2D)),
                                       testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto params3D = testing::Combine(testing::ValuesIn(axes3D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(static_shapes_to_test_representation(shapes3D)),
                                       testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto params4D = testing::Combine(testing::ValuesIn(axes4D), testing::ValuesIn(eps), testing::ValuesIn(epsMode),
                                       testing::ValuesIn(static_shapes_to_test_representation(shapes4D)),
                                       testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsMini4D =
        testing::Combine(testing::ValuesIn(axesMini4D), testing::Values(eps[0]), testing::Values(epsMode[0]),
                         testing::ValuesIn(static_shapes_to_test_representation(shapes4D)),
                         testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_2D, NormalizeL2LayerTestCommon, params2D,
                         NormalizeL2LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_3D, NormalizeL2LayerTestCommon, params3D,
                         NormalizeL2LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_4D, NormalizeL2LayerTestCommon, params4D,
                         NormalizeL2LayerTestCommon::getTestCaseName);

/* ============= Tiling ============= */

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_1, NormalizeL2LayerTest_2DPU,
                         testing::Combine(testing::ValuesIn(axesTiling4D),
                                          testing::ValuesIn(std::vector<float>{3.0815954528967052E-41}),
                                          testing::Values(epsMode[0]),
                                          testing::ValuesIn(static_shapes_to_test_representation(shapesTiling4D)),
                                          testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE)),
                         NormalizeL2LayerTest_2DPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_1, NormalizeL2LayerTestCommon,
                         testing::Combine(testing::ValuesIn(axesTiling4D),
                                          testing::ValuesIn(std::vector<float>{3.0815954528967052E-41}),
                                          testing::Values(epsMode[0]),
                                          testing::ValuesIn(static_shapes_to_test_representation(shapesTiling4D)),
                                          testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE)),
                         NormalizeL2LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_real_net_tiling_2, NormalizeL2LayerTestCommon,
                         testing::Combine(testing::Values(std::vector<int64_t>({3})),
                                          testing::ValuesIn(std::vector<float>{9.9999997473787516e-05}),
                                          testing::Values(epsMode[1]),
                                          testing::Values(static_shapes_to_test_representation(
                                                  std::vector<ov::Shape>({{3, 3, 64, 2304}}))),
                                          testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE)),
                         NormalizeL2LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2_4D, NormalizeL2LayerTest_6DPU, paramsMini4D,
                         NormalizeL2LayerTest_6DPU::getTestCaseName);

}  // namespace
