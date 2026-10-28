//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include "common/print_test_case_name.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"

#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>

namespace ov::test {

PRETTY_PARAM(InputType, ov::element::Type);
PRETTY_PARAM(IndicesType, ov::element::Type);
PRETTY_PARAM(AxisType, int32_t);

using GatherInputType = std::tuple<ov::test::InputShape, ov::test::InputShape>;
using DynamicGatherTestParams = std::tuple<GatherInputType, InputType, IndicesType, AxisType>;
//
// DynamicGatherLayerTest
//

class DynamicGatherLayerTest : public testing::WithParamInterface<DynamicGatherTestParams>, public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VPUX_THROW_WHEN(targetInputStaticShapes.size() != 1 && targetInputStaticShapes.size() != 2,
                        "Target input shapes expected to have 1 or 2 shapes, but got {0}",
                        targetInputStaticShapes.size());
        inputs.clear();
        auto dataInput = function->input(0);
        const auto& dataPartialShape = dataInput.get_partial_shape();
        auto dataShape = dataPartialShape.is_dynamic() ? targetInputStaticShapes.front() : dataPartialShape.to_shape();

        auto indicesInput = function->input(1);
        const auto& indicesPartialShape = indicesInput.get_partial_shape();
        auto indicesShape =
                indicesPartialShape.is_dynamic() ? targetInputStaticShapes.back() : indicesPartialShape.to_shape();
        ov::Tensor dataTensor = ov::test::utils::create_and_fill_tensor(dataInput.get_element_type(), dataShape);

        auto axis = std::get<3>(this->GetParam());
        auto range = static_cast<uint32_t>(dataShape[axis.value()]);
        ov::Tensor indicesTensor =
                ov::test::utils::create_and_fill_tensor(indicesInput.get_element_type(), indicesShape, {0, range});

        inputs.insert({dataInput.get_node_shared_ptr(), dataTensor});
        inputs.insert({indicesInput.get_node_shared_ptr(), indicesTensor});
    }

protected:
    std::tuple<PartialShape, PartialShape> getParamShapes(const GatherInputType& shapes) {
        auto [dataShape, indicesShape] = shapes;
        auto isDataDynamic = dataShape.first.is_dynamic();
        auto isIndicesDynamic = indicesShape.first.is_dynamic();
        if (isDataDynamic && isIndicesDynamic) {
            init_input_shapes({dataShape, indicesShape});
            return {inputDynamicShapes[0], inputDynamicShapes[1]};
        }
        if (isDataDynamic && !isIndicesDynamic) {
            init_input_shapes({dataShape});
            return {inputDynamicShapes[0], indicesShape.first};
        }
        if (!isDataDynamic && isIndicesDynamic) {
            init_input_shapes({indicesShape});
            return {dataShape.first, inputDynamicShapes[0]};
        }
        return {dataShape.first, indicesShape.first};
    }

    void SetUp() override {
        const auto& [inputShapes, dataType, indicesType, axis] = this->GetParam();
        auto [dataParamShape, indicesParamShape] = getParamShapes(inputShapes);
        auto dataParam = std::make_shared<ov::opset1::Parameter>(dataType.value(), dataParamShape);
        auto indicesParam = std::make_shared<ov::opset1::Parameter>(indicesType.value(), indicesParamShape);
        auto axisParam = ov::op::v0::Constant::create(ov::element::i32, Shape{}, std::vector<int32_t>{axis.value()});
        auto gather = std::make_shared<ov::op::v8::Gather>(dataParam, indicesParam, axisParam);
        function = std::make_shared<ov::Model>(gather, ov::ParameterVector{dataParam, indicesParam}, "DynamicGather");
    }
};

TEST_P(DynamicGatherLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicGatherLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicGatherLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicGatherLayerTest, NPU5020_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<InputType> inputPrecision = {ov::element::f16, ov::element::i64};
const std::vector<IndicesType> indicesPrecision = {ov::element::u64};
const std::vector<AxisType> inputAxis = {0, 1};
const std::vector<GatherInputType> inShapes = {{generateTestShape(2, 128), generateTestShape(2, 128_Dyn)},
                                               {generateTestShape(2, 128_Dyn), generateTestShape(2, 128)},
                                               {generateTestShape(2, 128_Dyn), generateTestShape(2, 128_Dyn)},
                                               {generateTestShape(1, 128_Dyn, 16), generateTestShape(128_Dyn)}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGather, DynamicGatherLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision),
                                            ::testing::ValuesIn(indicesPrecision), ::testing::ValuesIn(inputAxis)),
                         PrintTestCaseName());
}  // namespace ov::test
