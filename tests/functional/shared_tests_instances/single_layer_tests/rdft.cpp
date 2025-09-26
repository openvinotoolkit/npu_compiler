//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/rdft.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/rdft.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class RdftLayerTestCommon : public RDFTLayerTest, virtual public VpuOv2LayerTest {
    // C#125993
    // Reduce resolution of ov::float16 data generation to prevent NaN values
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
    void SetUp() override {
        std::vector<size_t> inputShape;
        ov::element::Type modelType;
        std::vector<int64_t> axes;
        std::vector<int64_t> signalSize;
        ov::test::utils::DFTOpType opType;
        std::tie(inputShape, modelType, axes, signalSize, opType, std::ignore) = this->GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front());
        auto rdft = ov::test::utils::make_rdft(param, axes, signalSize, opType);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(rdft->outputs(), ov::ParameterVector{param}, "RDFT");

        if (modelType == ov::element::f16) {
            VpuOv2LayerTest::rel_threshold = 0.15f * axes.size();
        }
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(RdftLayerTestCommon, NPU3720) {
    VpuOv2LayerTest::abs_threshold = 1.0;
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(RdftLayerTestCommon, NPU4000) {
    VpuOv2LayerTest::abs_threshold = 1.0;
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<DFTOpType> opTypes = {
        DFTOpType::FORWARD,
        DFTOpType::INVERSE,
};

const std::vector<ov::element::Type> modelTypes = {
        // disable FP32  tests as default compiler pipelines pass createConvertPrecisionToFP16Pass will convert anyway
        // to fp16 the operation, so test precision will be precision for fp16
        // ov::element::f32,
        ov::element::f16,

};

const auto combine = [](const std::vector<std::vector<size_t>>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(inputShapes), testing::ValuesIn(modelTypes), testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes), testing::ValuesIn(opTypes),
                            testing::Values(test_utils::TARGET_DEVICE));
};

// RDFT can support 1d
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_1d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(std::vector<size_t>{10}), testing::ValuesIn(modelTypes),
                                          testing::Values(std::vector<int64_t>{0}),
                                          testing::Values(std::vector<int64_t>{}), testing::Values(DFTOpType::FORWARD),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(std::vector<size_t>{10, 2}), testing::ValuesIn(modelTypes),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{0}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3}, {12}}),
                                          testing::Values(DFTOpType::FORWARD),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2dx, RdftLayerTestCommon,
                         combine({{10, 2}},         // input shapes
                                 {{0}},             // axes
                                 {{}, {3}, {11}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_3d, RdftLayerTestCommon,
                         combine({{10, 4, 2}},    // input shapes
                                 {{0, 1}},        // axes
                                 {{}, {3, 10}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{0, 1, 2}},        // axes
                                 {{}, {3, 10, 8}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_4d_negative_reversed_axes, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{-1, -2, -3}},     // axes
                                 {{}, {8, 10, 3}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_single_axis, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},        // input shapes
                                 {{0}, {1}, {2}},        // axes
                                 {{}, {1}, {5}, {20}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

// IRDFT can support 5d
INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_5d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(std::vector<size_t>{10, 4, 8, 2, 2}),
                                          testing::ValuesIn(modelTypes),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3, 10, 8, 6}}),
                                          testing::Values(DFTOpType::INVERSE),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         RDFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_tile_FORWARD, RdftLayerTestCommon,
                         testing::Combine(testing::Values(std::vector<size_t>{1, 80, 64, 64}),
                                          testing::ValuesIn(modelTypes), testing::Values(std::vector<int64_t>{2, 3}),
                                          testing::Values(std::vector<int64_t>{}), testing::Values(DFTOpType::FORWARD),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         RDFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_tile_INVERSE, RdftLayerTestCommon,
                         testing::Combine(testing::Values(std::vector<size_t>{1, 120, 64, 33, 2}),
                                          testing::ValuesIn(modelTypes), testing::Values(std::vector<int64_t>{2, 3}),
                                          testing::Values(std::vector<int64_t>{}), testing::Values(DFTOpType::INVERSE),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         RDFTLayerTest::getTestCaseName);

}  // namespace
