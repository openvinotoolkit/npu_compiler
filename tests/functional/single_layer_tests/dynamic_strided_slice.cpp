// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

#include <common/print_test_case_name.hpp>
#include <openvino/core/type/element_type.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include <random>
#include <vector>

#include "openvino/op/strided_slice.hpp"

namespace ov::test {

PRETTY_PARAM(InputType, ov::element::Type);

using BeginAndInputShape = std::pair<ov::test::InputShape, std::vector<int32_t>>;
using DynamicStridedSliceTestParams = std::tuple<BeginAndInputShape, InputType, int64_t>;

std::vector<int64_t> generateConst(const ov::Shape& shape) {
    size_t totalElements = 1;
    for (size_t dim : shape) {
        totalElements *= dim;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dis(0, 100);

    std::vector<int64_t> randomNumbers(totalElements);
    for (size_t i = 0; i < totalElements; ++i) {
        randomNumbers[i] = dis(gen);
    }

    return randomNumbers;
}

//
// DynamicStridedSliceLayerTest
//

class DynamicStridedSliceLayerTest :
        public testing::WithParamInterface<DynamicStridedSliceTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicStridedSliceTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 3;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [Inputs, type, sliceSize] = this->GetParam();
        const auto& inputShape = Inputs.first;
        const auto& constSize = Inputs.second;
        ov::Shape inputConstShape(constSize.begin(), constSize.end());

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(type.value(), shape));
        }

        std::vector<int64_t> randomInput = generateConst(inputConstShape);
        auto inputConst = ov::op::v0::Constant::create(ov::element::i64, inputConstShape, randomInput);

        const auto paramsShape = std::get<0>(inputShape).to_shape();
        const auto inputShapeRank = inputConstShape.size();
        const std::vector<int64_t> strides(inputShapeRank, 1);
        std::vector<int64_t> ends(constSize.begin(), constSize.end());
        // Slice the tensor by width
        ends.back() = sliceSize;
        auto endParam = ov::op::v0::Constant::create(ov::element::i64, paramsShape, ends);
        auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, paramsShape, strides);
        auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                inputConst, inputParams[0], endParam, stridesParam, std::vector<std::int64_t>{},
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        inputParams[0]->set_friendly_name("input");
        function = std::make_shared<ov::Model>(stridedSlice, inputParams, "DynamicStridedSlice");
    }
};

TEST_P(DynamicStridedSliceLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicStridedSliceLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicStridedSliceLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<InputType> inputPrecision = {ov::element::i32};
const std::vector<int64_t> sliceSize = {150};
const std::vector<BeginAndInputShape> inShapes = {
        {generateTestShape(1), {12}},
        {generateTestShape(1), {300}},
        {generateTestShape(3), {4, 8, 320}},
        {generateTestShape(4), {4, 6, 8, 320}},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicStridedSlice, DynamicStridedSliceLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision),
                                            ::testing::ValuesIn(sliceSize)),
                         DynamicStridedSliceLayerTest::getTestCaseName);

//
// DynamicStridedSliceDynamicEndsLayerTest
//

PRETTY_PARAM(Input, ov::test::InputShape);
PRETTY_PARAM(EndsValues, std::vector<int64_t>);
using DynamicStridedSliceDynamicEndsParams = std::tuple<Input, EndsValues, InputType>;

class DynamicStridedSliceDynamicEndsLayerTest :
        public testing::WithParamInterface<DynamicStridedSliceDynamicEndsParams>,
        public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& staticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        auto type = std::get<InputType>(GetParam());
        auto& dataStaticShape = staticShapes[0];
        auto dataTensor = utils::create_and_fill_tensor(type.value(), dataStaticShape);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), dataTensor});

        auto endsValues = std::get<EndsValues>(GetParam()).value();
        // Clamp ends values by data static shape
        for (auto i = 0; i < static_cast<int64_t>(endsValues.size()); i++) {
            endsValues[i] = std::min(endsValues[i], static_cast<int64_t>(dataStaticShape[i]));
        }

        auto endsSize = endsValues.size();
        auto endsTensor = ov::Tensor(ov::element::i64, ov::Shape{endsSize});
        std::copy_n(endsValues.begin(), endsSize, endsTensor.data<int64_t>());

        inputs.insert({funcInputs[1].get_node_shared_ptr(), endsTensor});
    }

protected:
    void SetUp() override {
        const auto& [dataTestShape, endsValues, type] = this->GetParam();

        auto endsShape = ov::Shape{endsValues.value().size()};
        init_input_shapes({dataTestShape.value(), generateTestShape(endsShape)});

        VPUX_THROW_UNLESS(inputDynamicShapes.size() == 2, "Expected to have 2 input shapes, got {0}",
                          inputDynamicShapes.size());

        auto inputParams = ov::ParameterVector{
                std::make_shared<ov::op::v0::Parameter>(type.value(), inputDynamicShapes.at(0)),
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes.at(1))};

        inputParams[0]->set_friendly_name("data");
        inputParams[1]->set_friendly_name("ends");

        const auto dataShape = dataTestShape.value().first.get_max_shape();
        const auto dataRank = dataShape.size();
        VPUX_THROW_UNLESS(dataRank == endsValues.value().size(),
                          "Input shape rank '{0}' and the size of 'ends' input '{1}' must be equal", dataRank,
                          endsValues.value().size());

        const auto begins = std::vector<int64_t>(dataRank, 0);
        const auto strides = std::vector<int64_t>(dataRank, 1);
        const auto attrShape = ov::Shape{dataRank};

        auto beginsParam = ov::op::v0::Constant::create(ov::element::i64, attrShape, begins);
        auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, attrShape, strides);

        auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                inputParams[0], beginsParam, inputParams[1], stridesParam, std::vector<std::int64_t>{},
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        function = std::make_shared<ov::Model>(stridedSlice, inputParams, "DynamicStridedSlice");
    }
};

TEST_P(DynamicStridedSliceDynamicEndsLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicStridedSliceDynamicEndsLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DynamicStridedSliceDynamicEndsLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// dynamic shape inputs will cause the test to fail because a strided slice layer
// works with strides of the input buffer. But the input data is packed and does
// not respect the strides of an upper-bounded buffer.
// Need to have a full support of strided data for dynamic tensors by the NPU plugin.
auto in = std::vector<Input>{generateTestShape(1, 2, 35, 512)};
auto ends = std::vector<EndsValues>{{1, 2, 35, 512}, {1, 2, 10, 512}, {1, 2, 1, 512}, {1, 1, 10, 512}};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicStridedSlice_DynamicEnds, DynamicStridedSliceDynamicEndsLayerTest,
                         ::testing::Combine(::testing::ValuesIn(in), ::testing::ValuesIn(ends),
                                            ::testing::Values(ov::element::i32)),
                         PrintTestCaseName());

//
// StridedSliceWithDynamicInputLayerTest
//

class StridedSliceWithDynamicInputLayerTest : public VpuOv2LayerTest {
public:
    void SetUp() override {
        const ov::Shape staticShape{1, 3, 16, 32};
        const std::vector<ov::Shape> inferenceShapes = {staticShape};
        const ov::PartialShape lhsDynamicShape{1, 3, 16, ov::Dimension(1, 32)};
        const ov::test::InputShape dataShape = {lhsDynamicShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset3::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const std::vector<int64_t> strides{1, 1, 1, 1};
        const std::vector<int64_t> begins{0, 0, 0, 1};
        const std::vector<int64_t> ends{1, 3, 16, 30};
        const auto beginConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, begins);
        const auto endConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, ends);
        const auto stridesConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, strides);
        auto stridedSlice = std::make_shared<ov::opset3::StridedSlice>(
                param, beginConst, endConst, stridesConst, std::vector<std::int64_t>{}, std::vector<std::int64_t>{},
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        const auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(stridedSlice->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicSlice");
    }
};

TEST_F(StridedSliceWithDynamicInputLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(StridedSliceWithDynamicInputLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(StridedSliceWithDynamicInputLayerTest, NPU5010_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
