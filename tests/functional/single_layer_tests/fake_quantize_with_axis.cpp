//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/fake_quantize.hpp"

namespace ov::test {

using AxisWithLayout = std::tuple<size_t, ov::Layout>;

class FqWithArbitraryAxisLayerTestCommon : public testing::WithParamInterface<AxisWithLayout>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AxisWithLayout>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (int64_t i = 0; i < totalSize; i++) {
            inputData[i] = (i % 255) - 127;
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    ov::Shape transposeShape(const ov::Shape& inputShape, const ov::Layout& order) const {
        ov::Shape transposedShape = inputShape;
        std::string neutralDims = "NCHW";
        std::vector<int64_t> indices;
        const auto dimToIdx = [&order](const char dim) -> int64_t {
            const std::string dimStr(1, dim);
            return order.get_index_by_name(dimStr);
        };
        std::transform(neutralDims.cbegin(), neutralDims.cend(), std::back_inserter(indices), dimToIdx);
        for (size_t srcIdx = 0; srcIdx < inputShape.size(); srcIdx++) {
            const auto dstIdx = indices.at(srcIdx);
            transposedShape.at(dstIdx) = inputShape.at(srcIdx);
        }
        return transposedShape;
    }

    void SetUp() override {
        const ov::Shape inputShape = {1, 4, 8, 16};
        const auto& params = GetParam();
        const auto& order = std::get<1>(params);
        const auto transposedShape = transposeShape(inputShape, order);
        const std::vector<ov::Shape> inferenceShapes = {transposedShape};
        const ov::test::InputShape dataShape = {inputShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto fqInLow = ov::opset1::Constant::create(ov::element::f32, {1, 1, 1, 1}, std::vector<float>{-128.f});
        const auto fqInHigh = ov::opset1::Constant::create(ov::element::f32, {1, 1, 1, 1}, std::vector<float>{127.f});

        ov::Shape fqShape = {1, 1, 1, 1};
        const auto& axis = std::get<0>(params);
        fqShape.at(axis) = inputShape.at(axis);
        using fqValueType = ov::element_type_traits<ov::element::f32>::value_type;
        const auto fqTotalSize = ov::shape_size(fqShape);
        std::vector<fqValueType> fqOutLowData(fqTotalSize, 0.f);
        for (size_t idx = 0; idx < fqOutLowData.size(); idx++) {
            fqOutLowData.at(idx) = -32.0 * (1 + idx % 3);
        }
        const auto fqOutLow = ov::opset1::Constant::create(ov::element::f32, fqShape, fqOutLowData);
        std::vector<fqValueType> fqOutHighData(fqTotalSize, 0.f);
        for (size_t idx = 0; idx < fqOutHighData.size(); idx++) {
            fqOutHighData.at(idx) = -fqOutLowData.at(idx) - 1;
        }
        const auto fqOutHigh = ov::opset1::Constant::create(ov::element::f32, fqShape, fqOutHighData);
        const size_t fqLevels = 256;
        const auto fq = std::make_shared<ov::opset1::FakeQuantize>(
                param->output(0), fqInLow->output(0), fqInHigh->output(0), fqOutLow->output(0), fqOutHigh->output(0),
                fqLevels, ov::op::AutoBroadcastType::NUMPY);
        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(fq->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "FQWithAxis");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout("NCHW");
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout("NCHW");
        function = preProc.build();
    }
};

//
// Platform test definition
//

TEST_P(FqWithArbitraryAxisLayerTestCommon, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FqWithArbitraryAxisLayerTestCommon, NPU4000) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<size_t> axes = {0, 1, 2, 3};
const std::vector<ov::Layout> layouts = {
        ov::Layout("NHWC"),
        ov::Layout("NCHW"),
};

INSTANTIATE_TEST_SUITE_P(FakeQuantize, FqWithArbitraryAxisLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(axes), ::testing::ValuesIn(layouts)),
                         FqWithArbitraryAxisLayerTestCommon::getTestCaseName);
}  // namespace ov::test
