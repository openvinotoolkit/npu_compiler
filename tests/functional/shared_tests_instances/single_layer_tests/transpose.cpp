//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/transpose.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class TransposeLayerTestCommon : public TransposeLayerTest, virtual public VpuOv2LayerTest {};
class TransposeLayerTest_NPU3720 : public TransposeLayerTestCommon {};
class TransposeLayerTest_NPU4000 : public TransposeLayerTestCommon {};

TEST_P(TransposeLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TransposeLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using ov::test::TransposeLayerTest_NPU3720;
using ov::test::TransposeLayerTest_NPU4000;

namespace {

const std::vector<ov::element::Type> modelTypes = {
        ov::element::f16,
};

// MLIR 2D instantiation
const std::vector<std::vector<ov::Shape>> inputShapes2D = {
        std::vector<ov::Shape>{{50, 100}},
};

const std::vector<std::vector<size_t>> inputOrder2D = {
        std::vector<size_t>{},
};

// MLIR 4D instantiation
const std::vector<std::vector<ov::Shape>> inputShapes4D = {
        std::vector<ov::Shape>{{1, 3, 100, 100}},
};

// Tracking number [E#85137]
const std::vector<std::vector<size_t>> inputOrder4D = {
        std::vector<size_t>{0, 3, 2, 1},
};

const std::vector<std::vector<ov::Shape>> inputShapesMemPerm = {
        std::vector<ov::Shape>{{1, 3, 100, 100}},
};

const std::vector<std::vector<ov::Shape>> inputShapesMemPermchannel16 = {
        std::vector<ov::Shape>{{1, 16, 48, 289}},
};

const std::vector<std::vector<size_t>> inputOrderMemPerm = {
        std::vector<size_t>{0, 2, 3, 1},
};

const std::vector<std::vector<size_t>> inputOrderMemPermNWCH = {
        std::vector<size_t>{0, 3, 1, 2},
};

const auto paramsMemPermNWCHtoNHWC =
        testing::Combine(testing::ValuesIn(inputOrderMemPermNWCH), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesMemPermchannel16)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

/* ============= NPU3720  ============= */

const auto paramsNPU3720 =
        testing::Combine(testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesMemPerm)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Transpose, TransposeLayerTest_NPU3720, paramsNPU3720,
                         TransposeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TransposeMemPermNWCHtoNHWC, TransposeLayerTest_NPU3720, paramsMemPermNWCHtoNHWC,
                         TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 ND  ============= */

const std::vector<std::vector<ov::Shape>> shape_5D = {std::vector<ov::Shape>{{1, 10, 10, 4, 6}},
                                                      std::vector<ov::Shape>{{1, 10, 4, 6, 1}}};
const std::vector<std::vector<size_t>> reorder_5D = {std::vector<size_t>{4, 1, 2, 3, 0},
                                                     std::vector<size_t>{4, 0, 2, 3, 1}};

const auto params_5D = testing::Combine(testing::ValuesIn(reorder_5D), testing::ValuesIn(modelTypes),
                                        testing::ValuesIn(ov::test::static_shapes_to_test_representation(shape_5D)),
                                        testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_5D, TransposeLayerTest_NPU3720, params_5D,
                         TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test complex tensor optimization  ============= */

const std::vector<std::vector<size_t>> orderCNo4d = {std::vector<size_t>{0, 1, 3, 2, 4},
                                                     std::vector<size_t>{0, 3, 2, 1, 4}};

const std::vector<std::vector<ov::Shape>> inputShapesCNo4d = {
        std::vector<ov::Shape>{{1, 3, 8, 8, 2}}, std::vector<ov::Shape>{{1, 3, 4, 5, 2}},
        std::vector<ov::Shape>{{1, 3, 2, 5, 2}}, std::vector<ov::Shape>{{1, 3, 5, 2, 2}},
        std::vector<ov::Shape>{{1, 3, 9, 7, 2}}, std::vector<ov::Shape>{{1, 2, 33, 33, 2}}};

const auto paramsCNo4d =
        testing::Combine(testing::ValuesIn(orderCNo4d), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesCNo4d)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TransposeCNo4d, TransposeLayerTest_NPU3720, paramsCNo4d,
                         TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test optimization with merged input shape 4D ============= */

const std::vector<std::vector<size_t>> orderMerged4d = {
        std::vector<size_t>{0, 2, 1, 3}, std::vector<size_t>{2, 1, 0, 3}, std::vector<size_t>{0, 3, 2, 1}};

const std::vector<std::vector<ov::Shape>> inputShapesMerged4d = {std::vector<ov::Shape>{{6, 4, 8, 512}},
                                                                 std::vector<ov::Shape>{{12, 7, 12, 4}}};

const auto paramsMerged4d =
        testing::Combine(testing::ValuesIn(orderMerged4d), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesMerged4d)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TransposeMerged4d, TransposeLayerTest_NPU3720, paramsMerged4d,
                         TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test permutation decomposition  ============= */

const std::vector<std::vector<size_t>> complex5DReorder = {std::vector<size_t>{0, 2, 4, 1, 3}};
const std::vector<std::vector<ov::Shape>> inputShapeBatched5D = {std::vector<ov::Shape>{{3, 128, 4, 128, 4}}};

const auto paramsPermuteDecomposition =
        testing::Combine(testing::ValuesIn(complex5DReorder), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapeBatched5D)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Transpose_Permutation_Decomposition, TransposeLayerTest_NPU3720,
                         paramsPermuteDecomposition, TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU3720 Test transpose to DMA  ============= */

const std::vector<std::vector<size_t>> specificReorder = {std::vector<size_t>{2, 0, 3, 1}};
const std::vector<std::vector<ov::Shape>> specificInShape = {std::vector<ov::Shape>{{1, 15, 2, 128}}};

const auto paramsTransposeToDMA =
        testing::Combine(testing::ValuesIn(specificReorder), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(specificInShape)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Transpose_To_DMA, TransposeLayerTest_NPU3720, paramsTransposeToDMA,
                         TransposeLayerTest_NPU3720::getTestCaseName);

/* ============= NPU4000  ============= */

const std::vector<std::vector<ov::Shape>> inShapesMemPerm = {std::vector<ov::Shape>{{1, 8, 80, 960}},
                                                             std::vector<ov::Shape>{{1, 3, 12, 16}}};

const auto paramsMemPerm =
        testing::Combine(testing::ValuesIn(inputOrderMemPerm), testing::ValuesIn(modelTypes),
                         testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesMemPerm)),
                         testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Transpose, TransposeLayerTest_NPU4000, paramsMemPerm,
                         TransposeLayerTest_NPU4000::getTestCaseName);
}  // namespace
