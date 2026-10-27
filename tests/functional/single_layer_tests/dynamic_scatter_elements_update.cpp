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
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/scatter_elements_update.hpp>

namespace ov::test {
namespace {

PRETTY_PARAM(DataShape, ov::Shape);
PRETTY_PARAM(IndicesShape, ov::test::InputShape);
PRETTY_PARAM(UpdatesShape, ov::test::InputShape);
PRETTY_PARAM(DataType, ov::element::Type);
PRETTY_PARAM(IndicesType, ov::element::Type);
PRETTY_PARAM(Axis, int64_t);
PRETTY_PARAM(Reduction, ov::op::v12::ScatterElementsUpdate::Reduction);
PRETTY_PARAM(UseInitVal, bool);

using DynamicScatterElementsUpdateTestParams =
        std::tuple<DataShape, IndicesShape, UpdatesShape, DataType, IndicesType, Axis, Reduction, UseInitVal>;

class DynamicScatterElementsUpdateLayerTest :
        public testing::WithParamInterface<DynamicScatterElementsUpdateTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicScatterElementsUpdateTestParams>& obj) {
        return PrintTestCaseName()(obj);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VPUX_THROW_WHEN(targetInputStaticShapes.size() != 2,
                        "Expected 2 target input shapes (indices and updates), but got {0}",
                        targetInputStaticShapes.size());
        inputs.clear();

        auto dataInput = function->input(0);
        const auto& dataPartialShape = dataInput.get_partial_shape();
        auto dataShape = dataPartialShape.is_dynamic() ? targetInputStaticShapes.front() : dataPartialShape.to_shape();
        ov::Tensor dataTensor = ov::test::utils::create_and_fill_tensor(dataInput.get_element_type(), dataShape);

        auto indicesInput = function->input(1);
        auto updatesInput = function->input(2);
        const auto& indicesShape = targetInputStaticShapes[0];
        const auto& updatesShape = targetInputStaticShapes[1];
        auto axis = std::get<5>(this->GetParam());
        auto range = static_cast<uint32_t>(dataShape[axis.value()]);
        ov::Tensor indicesTensor =
                ov::test::utils::create_and_fill_tensor(indicesInput.get_element_type(), indicesShape, {0, range});
        ov::Tensor updatesTensor =
                ov::test::utils::create_and_fill_tensor(updatesInput.get_element_type(), updatesShape);

        inputs.insert({dataInput.get_node_shared_ptr(), dataTensor});
        inputs.insert({indicesInput.get_node_shared_ptr(), indicesTensor});
        inputs.insert({updatesInput.get_node_shared_ptr(), updatesTensor});
    }

protected:
    void SetUp() override {
        const auto& [dataShape, indicesShape, updatesShape, dataType, indicesType, axis, reduction, useInitVal] =
                this->GetParam();
        init_input_shapes({indicesShape.value(), updatesShape.value()});

        auto dataParam = std::make_shared<ov::op::v0::Parameter>(dataType.value(), dataShape.value());
        auto indicesParam = std::make_shared<ov::op::v0::Parameter>(indicesType.value(), inputDynamicShapes[0]);
        auto updatesParam = std::make_shared<ov::op::v0::Parameter>(dataType.value(), inputDynamicShapes[1]);
        auto axisConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{axis.value()});

        auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
                dataParam, indicesParam, updatesParam, axisConst, reduction.value(), useInitVal.value());
        auto result = std::make_shared<ov::op::v0::Result>(scatter);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{dataParam, indicesParam, updatesParam},
                                               "DynamicScatterElementsUpdate");
    }
};

TEST_P(DynamicScatterElementsUpdateLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicScatterElementsUpdateLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DynamicScatterElementsUpdateLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<DataShape> dataShapes = {ov::Shape{1, 1}};
const std::vector<DataType> dataPrecision = {ov::element::f16};
const std::vector<IndicesType> indicesPrecision = {ov::element::i32};
const std::vector<Axis> axes = {0};
const std::vector<Reduction> reductions = {ov::op::v12::ScatterElementsUpdate::Reduction::NONE,
                                           ov::op::v12::ScatterElementsUpdate::Reduction::MAX,
                                           ov::op::v12::ScatterElementsUpdate::Reduction::MEAN};
const std::vector<UseInitVal> useInitVals = {false, true};

const std::vector<IndicesShape> indicesShapes = {generateTestShape(14_Dyn, 1)};
const std::vector<UpdatesShape> updatesShapes = {generateTestShape(14_Dyn, 1)};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicScatterElementsUpdate, DynamicScatterElementsUpdateLayerTest,
                         ::testing::Combine(::testing::ValuesIn(dataShapes), ::testing::ValuesIn(indicesShapes),
                                            ::testing::ValuesIn(updatesShapes), ::testing::ValuesIn(dataPrecision),
                                            ::testing::ValuesIn(indicesPrecision), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(reductions), ::testing::ValuesIn(useInitVals)),
                         DynamicScatterElementsUpdateLayerTest::getTestCaseName);

}  // namespace
}  // namespace ov::test
