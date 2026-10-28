//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::test::utils;

namespace ov::test {

class PropagateFQSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<std::vector<int64_t>>> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 32, 32};
        const auto transposeOrder = std::get<0>(GetParam());

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};

        const size_t dataLevels = 256;
        const std::vector<float> inDataLow = {0.0f};
        const std::vector<float> inDataHigh = {100.0f};

        std::vector<size_t> newShape;
        newShape.push_back(inputShape[0]);
        newShape.push_back(inputShape[3]);
        newShape.push_back(inputShape[2]);
        newShape.push_back(inputShape[1]);
        const auto reshape = buildReshape(params[0], newShape);

        const auto lhsTranspose = buildTranspose(reshape, transposeOrder);

        const auto dataFq = ov::test::utils::make_fake_quantize(lhsTranspose, ov::element::f32, dataLevels, {},
                                                                inDataLow, inDataHigh, inDataLow, inDataHigh);
        const ov::Strides strides = {2, 2};
        const std::vector<size_t> pads_begin = {0, 0};
        const std::vector<size_t> pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const std::vector<size_t> kernelSize = {2, 2};
        const ov::op::PadType padType = ov::op::PadType::AUTO;
        const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

        const auto pooling = std::make_shared<ov::op::v1::MaxPool>(dataFq, strides, pads_begin, pads_end, kernelSize,
                                                                   roundingType, padType);

        // Quantized 1x1 Convolution + output FQ so the FakeQuantize fuses into the Convolution PPE
        // instead of being stripped as a boundary Quantize/Dequantize
        const auto poolChannels = pooling->get_output_shape(0).at(1);
        const auto convWShape = ov::Shape{poolChannels, poolChannels, 1, 1};
        const auto convWeights = ov::op::v0::Constant::create(ov::element::f32, convWShape,
                                                              std::vector<float>(ov::shape_size(convWShape), 0.0625f));
        const auto convWeightsFq = ov::test::utils::make_fake_quantize(convWeights, ov::element::f32, dataLevels, {},
                                                                       {0.0f}, {0.0625f}, {0.0f}, {0.0625f});
        const auto conv = std::make_shared<ov::op::v1::Convolution>(pooling, convWeightsFq, ov::Strides{1, 1},
                                                                    ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0},
                                                                    ov::Strides{1, 1});
        const auto relu = std::make_shared<ov::op::v0::Relu>(conv);
        const auto outFq = ov::test::utils::make_fake_quantize(relu, ov::element::f32, dataLevels, {}, {0.0f}, {100.0f},
                                                               {0.0f}, {100.0f});

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq),
                                       std::make_shared<ov::op::v0::Result>(lhsTranspose)};
        function = std::make_shared<ov::Model>(results, params, "PropagateFQSubGraph");
        rel_threshold = 0.1f;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                std::make_shared<ov::op::v1::Reshape>(param, constNode, false));
        return reshape;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::vector<int64_t>>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};  // namespace ov::test

using PropagateFQUpwardTestParams = std::tuple<std::vector<float>,  // fqRanges1
                                               std::vector<float>   // fqRanges2
                                               >;

class PropagateFQUpwardSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<PropagateFQUpwardTestParams> {
    void SetUp() override {
        const ov::Shape inputShape{1, 288, 20, 20};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};

        const auto sigmoidOp = std::make_shared<ov::op::v0::Sigmoid>(params[0]);

        const auto sliceBeginConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 16, 0, 0});
        const auto sliceEndConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 32, 0, 0});
        const auto sliceStridesConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
        std::vector<int64_t> begin_mask{1, 0, 1, 1};
        std::vector<int64_t> end_mask{1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};
        const auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                sigmoidOp, sliceBeginConst, sliceEndConst, sliceStridesConst, begin_mask, end_mask, new_axis_mask,
                shrink_axis_mask, ellipsis_mask);

        const size_t dataLevels = 256;
        std::vector<float> firstFQRanges;
        std::vector<float> secondFQRanges;
        std::tie(firstFQRanges, secondFQRanges) = this->GetParam();
        const auto firstFq = ov::test::utils::make_fake_quantize(stridedSlice, ov::element::f32, dataLevels, {},
                                                                 {firstFQRanges.at(0)}, {firstFQRanges.at(1)},
                                                                 {firstFQRanges.at(2)}, {firstFQRanges.at(3)});
        const auto secondFq = ov::test::utils::make_fake_quantize(stridedSlice, ov::element::f32, dataLevels, {},
                                                                  {secondFQRanges.at(0)}, {secondFQRanges.at(1)},
                                                                  {secondFQRanges.at(2)}, {secondFQRanges.at(3)});

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(firstFq),
                                       std::make_shared<ov::op::v0::Result>(secondFq)};

        function = std::make_shared<ov::Model>(results, params, "PropagateFQUpwardSubGraph");
        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<PropagateFQUpwardTestParams> obj) {
        std::vector<float> firstFQRanges;
        std::vector<float> secondFQRanges;
        std::tie(firstFQRanges, secondFQRanges) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "FQ1={" << firstFQRanges.at(0) << ", " << firstFQRanges.at(1) << ", " << firstFQRanges.at(2) << ", "
               << firstFQRanges.at(3) << "}" << sep;
        result << "FQ2={" << secondFQRanges.at(0) << ", " << secondFQRanges.at(1) << ", " << secondFQRanges.at(2)
               << ", " << secondFQRanges.at(3) << "}" << sep;
        return result.str();
    }
};

// PropagateFQSubGraphTest_NPU3720

TEST_P(PropagateFQSubGraphTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(PropagateFQSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 3, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_PropagateFQSubGraph, PropagateFQSubGraphTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(transposes)),
                         PropagateFQSubGraphTest_NPU3720::getTestCaseName);

// PropagateFQUpwardSubGraphTest_NPU3720

TEST_P(PropagateFQUpwardSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(PropagateFQUpwardSubGraphTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

std::vector<std::vector<float>> fqRanges1 = {{0.0f, 8.0f, 0.0f, 8.0f}};
std::vector<std::vector<float>> fqRanges2 = {{0.0f, 8.0f, 0.0f, 8.0f}, {0.0f, 4.0f, 0.0f, 4.0f}};

INSTANTIATE_TEST_SUITE_P(smoke_PropagateUpFQSubGraph, PropagateFQUpwardSubGraphTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(fqRanges1), ::testing::ValuesIn(fqRanges2)),
                         PropagateFQUpwardSubGraphTest_NPU3720::getTestCaseName);

}  // namespace ov::test
