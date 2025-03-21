//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reduce_ops.hpp"

#include <vector>

#include <common/functions.h>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/reduce.hpp"
#include "common_test_utils/test_constants.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
class ReduceLayerTestCommon : public ReduceOpsLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
    void SetUp() override {
        ov::element::Type modelType;
        OpType opType;
        ReductionType reductionType;
        ov::NodeVector convertedInputs;
        ov::OutputVector paramOuts;
        std::shared_ptr<ov::Node> reduceNode;
        std::vector<size_t> inputShape, shapeAxes;
        std::vector<int> axes;
        bool keepDims;

        std::tie(axes, opType, keepDims, reductionType, modelType, inputShape, std::ignore) = GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector inputs{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front())};
        switch (opType) {
        case OpType::SCALAR: {
            if (axes.size() > 1)
                FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
            break;
        }
        case OpType::VECTOR: {
            shapeAxes.push_back(axes.size());
            break;
        }
        default:
            FAIL() << "Reduce op doesn't support operation type: " << opType;
        }
        auto reductionAxesNode = std::dynamic_pointer_cast<ov::Node>(
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(shapeAxes), axes));

        const auto reduce = make_reduce(inputs[0], reductionAxesNode, keepDims, reductionType);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(reduce->outputs(), inputs, "Reduce");
    }

    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};  // namespace test

// FP16/FP32
class ReduceLayerTest_HW_FP16 : public ReduceLayerTestCommon {};
class ReduceLayerTest_SW_FP16 : public ReduceLayerTestCommon {};
class ReduceLayerTest_FP32 : public ReduceLayerTestCommon {
    void configure_model() override {
        VpuOv2LayerTest::configuration[ov::intel_npu::compilation_mode_params.name()] =
                "convert-precision-to-fp16=false";
    }
};

// ChannelAxis Reduce Test for DPU support
class ReduceLayerTest_ChannelAxis_HW_FP16 : public ReduceLayerTestCommon {
    void configure_model() override {
        VpuOv2LayerTest::configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-is-reduce-supported=false enable-auto-padding-odu=false";
    }
};
/// FP16 SW/HW
TEST_P(ReduceLayerTest_HW_FP16, NPU3720) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReduceLayerTest_SW_FP16, NPU3720) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReduceLayerTest_SW_FP16, NPU4000) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
/// FP32 HW
TEST_P(ReduceLayerTest_FP32, NPU3720_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReduceLayerTest_FP32, NPU4000_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
/// FP32 SW
TEST_P(ReduceLayerTest_FP32, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReduceLayerTest_FP32, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
class ReduceOpsLayerWithSpecificInputTestCommon :
        public ReduceOpsLayerWithSpecificInputTest,
        virtual public VpuOv2LayerTest {
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<bool> keepDims = {
        true,
        false,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
        std::vector<size_t>{3, 5, 7, 9},
};

const std::vector<std::vector<int>> axes = {{1}, {2}, {1, 3}, {2, 3}, {1, -1}};

const std::vector<ReductionType> reduceOperations = {
        ReductionType::Mean, ReductionType::Max, ReductionType::Min, ReductionType::Sum,
        // By documentation, operations LogicalOr and LogicalAnd return boolean type. This type is not supported yet in
        // NPU and OV is using this rule. Enable these operations when this feature is enabled.
        // [Tracking number E#107046]
        // ReductionType::LogicalOr,
        // ReductionType::LogicalAnd,
        ReductionType::L1, ReductionType::L2, ReductionType::Prod};

//
// FP16 SW
const auto paramsSWFP16 =
        testing::Combine(testing::ValuesIn(axes), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
                         testing::ValuesIn(reduceOperations), testing::ValuesIn(modelTypes),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}), testing::Values(DEVICE_NPU));

const auto paramsTiling = testing::Combine(
        testing::ValuesIn(decltype(axes){{2}, {1, -1}}), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::Values(ReductionType::Sum), testing::ValuesIn(modelTypes),
        testing::Values(std::vector<size_t>{1, 20, 175, 512}), testing::Values(DEVICE_NPU));

// ReduceMax config for U8 data type resnet-50-pytorch
const auto paramsResnet =
        testing::Combine(testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(OpType::VECTOR),
                         testing::Values(true), testing::Values(ReductionType::Max), testing::Values(ov::element::u8),
                         testing::Values(std::vector<size_t>{1, 2048, 7, 7}), testing::Values(DEVICE_NPU));

auto paramsReduceAllAxis = testing::Combine(
        testing::ValuesIn(decltype(axes){{0, 1, 2, 3}}), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::ValuesIn(reduceOperations), testing::Values(ov::element::f16),
        testing::Values(std::vector<size_t>{1, 4, 2, 38}), testing::Values(DEVICE_NPU));

//
// FP16 HW
const auto paramsHWFP16 = testing::Combine(
        testing::ValuesIn(decltype(axes){{1}, {2}}), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::Values(ReductionType::Sum, ReductionType::Mean, ReductionType::Min, ReductionType::Max),
        testing::ValuesIn(modelTypes),
        testing::Values(std::vector<size_t>{1, 9, 32, 32}, std::vector<size_t>{1, 1, 2},
                        std::vector<size_t>{1, 4, 32, 32}, std::vector<size_t>{1, 16, 32, 32}),
        testing::Values(DEVICE_NPU));

//
// FP16 HW
const auto channelAxisHWFP16 = testing::Combine(
        testing::ValuesIn(decltype(axes){{1}}), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::Values(ReductionType::Mean, ReductionType::Sum), testing::ValuesIn(modelTypes),
        testing::Values(std::vector<size_t>{1, 16, 32, 32}, std::vector<size_t>{1, 4, 32, 32}),
        testing::Values(DEVICE_NPU));

//
// FP32
const auto paramsFP32 = testing::Combine(
        testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::Values(ReductionType::Mean, ReductionType::Sum, ReductionType::L2), testing::Values(ov::element::f32),
        testing::Values(std::vector<size_t>{1, 1024, 7, 7}), testing::Values(DEVICE_NPU));

//
// FP16 HW
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reduce, ReduceLayerTest_HW_FP16, paramsHWFP16,
                         ReduceLayerTest_HW_FP16::getTestCaseName);

//
// FP16 SW

// Passing on master branch. Please reenable when backmerge
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Reduce, ReduceLayerTest_SW_FP16, paramsSWFP16,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_tiling, ReduceLayerTest_SW_FP16, paramsTiling,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Resnet, ReduceLayerTest_SW_FP16, paramsResnet,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

// All axes reduced tests
INSTANTIATE_TEST_SUITE_P(smoke_ReduceAllAxis, ReduceLayerTest_SW_FP16, paramsReduceAllAxis,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

// FP32 HW and SW
INSTANTIATE_TEST_SUITE_P(smoke_Reduce_FP32, ReduceLayerTest_FP32, paramsFP32, ReduceLayerTest_FP32::getTestCaseName);

// ChannelAxis Reduce test for DPU support
INSTANTIATE_TEST_SUITE_P(smoke_reduce_axis, ReduceLayerTest_ChannelAxis_HW_FP16, channelAxisHWFP16,
                         ReduceLayerTest_ChannelAxis_HW_FP16::getTestCaseName);

}  // namespace
