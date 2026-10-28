//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include "common/print_test_case_name.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"

#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/gather_nd.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>

namespace ov::test {

PRETTY_PARAM(DataShape, ov::Shape);
PRETTY_PARAM(IndicesShape, ov::test::InputShape);
PRETTY_PARAM(DataType, ov::element::Type);
PRETTY_PARAM(IndicesType, ov::element::Type);
PRETTY_PARAM(BatchDims, int32_t);

using DynamicGatherNDTestParams = std::tuple<DataShape, IndicesShape, DataType, IndicesType, BatchDims>;

class DynamicGatherNDLayerTest : public testing::WithParamInterface<DynamicGatherNDTestParams>, public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VPUX_THROW_WHEN(targetInputStaticShapes.size() != 1, "Expected 1 target input shape, but got {0}",
                        targetInputStaticShapes.size());
        inputs.clear();

        auto dataInput = function->input(0);
        const auto& dataPartialShape = dataInput.get_partial_shape();
        auto dataShape = dataPartialShape.is_dynamic() ? targetInputStaticShapes.front() : dataPartialShape.to_shape();
        ov::Tensor dataTensor = ov::test::utils::create_and_fill_tensor(dataInput.get_element_type(), dataShape);

        auto indicesInput = function->input(1);
        const auto& indicesShape = targetInputStaticShapes.front();

        auto batchDims = std::get<4>(this->GetParam());
        auto range = static_cast<uint32_t>(dataShape[batchDims.value()]);
        ov::Tensor indicesTensor =
                ov::test::utils::create_and_fill_tensor(indicesInput.get_element_type(), indicesShape, {0, range});

        inputs.insert({dataInput.get_node_shared_ptr(), dataTensor});
        inputs.insert({indicesInput.get_node_shared_ptr(), indicesTensor});
    }

protected:
    void SetUp() override {
        const auto& [dataShape, indicesInputShape, dataType, indicesType, batchDims] = this->GetParam();
        init_input_shapes({indicesInputShape.value()});

        auto dataParam = std::make_shared<ov::op::v0::Parameter>(dataType.value(), dataShape.value());
        auto indicesParam = std::make_shared<ov::op::v0::Parameter>(indicesType.value(), inputDynamicShapes[0]);
        auto gatherND = std::make_shared<ov::op::v8::GatherND>(dataParam, indicesParam, batchDims.value());
        auto result = std::make_shared<ov::op::v0::Result>(gatherND);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{dataParam, indicesParam}, "DynamicGatherND");
    }
};

TEST_P(DynamicGatherNDLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicGatherNDLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicGatherNDLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicGatherNDLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DataShape> dataShapes1D = {ov::Shape{14}};
const std::vector<DataShape> dataShapesSlice = {ov::Shape{2, 3, 4}};
const std::vector<DataType> dataPrecision = {ov::element::i64};
const std::vector<IndicesType> indicesPrecision = {ov::element::i64};
const std::vector<BatchDims> batchDims = {0};

const std::vector<IndicesShape> indicesShapes1D = {generateTestShape(14_Dyn, 1), generateTestShape(128_Dyn, 1)};

const std::vector<IndicesShape> indicesShapesSlice = {
        generateTestShape(std::vector<BoundedDim>{16_Dyn, 2},
                          [](const BoundedDim&) {
                              return std::vector<int>{8, 16};
                          }),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherND_1D, DynamicGatherNDLayerTest,
                         ::testing::Combine(::testing::ValuesIn(dataShapes1D), ::testing::ValuesIn(indicesShapes1D),
                                            ::testing::ValuesIn(dataPrecision), ::testing::ValuesIn(indicesPrecision),
                                            ::testing::ValuesIn(batchDims)),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherND_SliceOutput, DynamicGatherNDLayerTest,
                         ::testing::Combine(::testing::ValuesIn(dataShapesSlice),
                                            ::testing::ValuesIn(indicesShapesSlice), ::testing::ValuesIn(dataPrecision),
                                            ::testing::ValuesIn(indicesPrecision), ::testing::ValuesIn(batchDims)),
                         PrintTestCaseName());

}  // namespace ov::test
