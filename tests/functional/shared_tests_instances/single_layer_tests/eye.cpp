//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/eye.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/eye.hpp"
#include "openvino/op/select.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Layer setup with:
// - rows -> Constant
// - cols -> Constant
// - diag_shift -> Parameter
// - batch_shape -> Constant
class EyeLayerTestCommon : public EyeLayerTest, virtual public VpuOv2LayerTest {
    std::vector<ov::Shape> inputShapes;
    std::vector<int> outBatchShape;
    std::vector<int> eyeParams;
    ov::element::Type modelType;

    int32_t rowNum, colNum, shift;

    void SetUp() override {
        std::tie(inputShapes, outBatchShape, eyeParams, modelType, std::ignore) = GetParam();

        inType = outType = modelType;
        rowNum = eyeParams[0];
        colNum = eyeParams[1];
        shift = eyeParams[2];

        init_input_shapes(static_shapes_to_test_representation({inputShapes}));

        const auto rowsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, inputShapes[0], &rowNum);
        rowsConst->set_friendly_name("rows_const");
        const auto colsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, inputShapes[1], &colNum);
        colsConst->set_friendly_name("cols_const");
        const auto diagShiftPar = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputShapes[2]);

        std::shared_ptr<ov::op::v9::Eye> eyeOp;
        if (outBatchShape.empty()) {
            eyeOp = std::make_shared<ov::op::v9::Eye>(rowsConst, colsConst, diagShiftPar, modelType);
        } else {
            const auto batchShapeConst = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i32, ov::Shape{outBatchShape.size()}, outBatchShape.data());
            eyeOp = std::make_shared<ov::op::v9::Eye>(rowsConst, colsConst, diagShiftPar, batchShapeConst, modelType);
        }
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eyeOp)};

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{diagShiftPar}, "eye");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 1, shift);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

// Layer setup with:
// - rows -> Constant
// - cols -> Constant
// - diag_shift -> Constant
// - batch_shape -> Constant
// With OV constant folding (enabled by default), this layer will be calculated by CPU and replaced to Constant operator
class EyeLayerTestWithConstantFoldingCommon : public EyeLayerTest, virtual public VpuOv2LayerTest {
    std::vector<ov::Shape> inputShapes;
    std::vector<int> outBatchShape;
    std::vector<int> eyeParams;
    ov::element::Type modelType;

    int32_t rowNum, colNum, shift;

    void SetUp() override {
        std::tie(inputShapes, outBatchShape, eyeParams, modelType, std::ignore) = GetParam();

        rowNum = eyeParams[0];
        colNum = eyeParams[1];
        shift = eyeParams[2];

        const auto rowsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, inputShapes[0], &rowNum);
        rowsConst->set_friendly_name("rows_const");
        const auto colsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, inputShapes[1], &colNum);
        colsConst->set_friendly_name("cols_const");
        const auto diagShiftConst = std::make_shared<ov::op::v0::Constant>(ov::element::i32, inputShapes[2], &shift);
        diagShiftConst->set_friendly_name("diag_shift_const");

        std::shared_ptr<ov::op::v9::Eye> eyeOp;
        if (outBatchShape.empty()) {
            eyeOp = std::make_shared<ov::op::v9::Eye>(rowsConst, colsConst, diagShiftConst, modelType);
        } else {
            const auto batchShapeConst = std::make_shared<ov::op::v0::Constant>(
                    ov::element::i32, ov::Shape{outBatchShape.size()}, outBatchShape.data());
            eyeOp = std::make_shared<ov::op::v9::Eye>(rowsConst, colsConst, diagShiftConst, batchShapeConst, modelType);
        }
        // `Parameter` op was also needed to be added to the result of `Eye` and `Select` op was used as an interface
        // between `Parameter` and `Eye` ops.
        auto condition =
                std::make_shared<ov::op::v0::Constant>(ov::element::boolean, eyeOp->output(0).get_shape(), false);
        std::vector<ov::test::InputShape> inputShape =
                ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{eyeOp->output(0).get_shape()});

        init_input_shapes(inputShape);

        auto params = ov::ParameterVector{
                std::make_shared<ov::op::v0::Parameter>(modelType, targetStaticShapes.front().at(0)),
        };

        auto select = std::make_shared<ov::op::v1::Select>(condition, params[0], eyeOp);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        function = std::make_shared<ov::Model>(results, params, "eye");
    }
};

TEST_P(EyeLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EyeLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(EyeLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(EyeLayerTestWithConstantFoldingCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EyeLayerTestWithConstantFoldingCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(EyeLayerTestWithConstantFoldingCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

// Shape for 'rows', 'cols', and 'diag_shift'
const std::vector<ov::Shape> eyeShape = {{1}, {1}, {1}};

const std::vector<std::vector<int>> batchShapes = {
        {},        // No 'batch_shape' -> output shape = 2D
        {2},       // 1D 'batch_shape' -> output shape = 3D
        {3, 2},    // 2D 'batch_shape' -> output shape = 4D
        {4, 3, 2}  // 3D 'batch_shape' -> output shape = 5D
};

const std::vector<std::vector<int>> eyePars = {
        // rows, cols, diag_shift
        {8, 2, 1},
        {9, 4, 6},
        {5, 7, -3}};

const std::vector<ov::element::Type> modelTypes = {ov::element::f32, ov::element::f16, ov::element::i32,
                                                   ov::element::i8, ov::element::u8};

const auto noBatchShapeParams =
        testing::Combine(testing::Values(eyeShape), testing::Values(batchShapes[0]), testing::ValuesIn(eyePars),
                         testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto withBatchShapeParams = testing::Combine(
        testing::Values(eyeShape),
        testing::ValuesIn(std::vector<std::vector<int>>(batchShapes.begin() + 1, batchShapes.end())),
        testing::Values(eyePars[0]), testing::Values(modelTypes[0]), testing::Values(test_utils::TARGET_DEVICE));

const auto realNetParams = testing::Combine(testing::Values(eyeShape), testing::Values(batchShapes[0]),
                                            testing::Values(std::vector<int>{128, 128, 0}),
                                            testing::Values(modelTypes[0]), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Eye, EyeLayerTestCommon, noBatchShapeParams, EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_with_batch_shape, EyeLayerTestCommon, withBatchShapeParams,
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_real_net, EyeLayerTestCommon, realNetParams, EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_const_fold_real_net, EyeLayerTestWithConstantFoldingCommon, realNetParams,
                         EyeLayerTest::getTestCaseName);

}  // namespace
