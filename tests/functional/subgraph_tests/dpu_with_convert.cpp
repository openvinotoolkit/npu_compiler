// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "vpux/utils/core/error.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov::test {

class DpuWithF16ToF32ConvertTestBase :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, ov::Layout, ov::Layout, size_t>> {
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() <= 2, "Can have at most 2 inputs supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() <= 2, "Can have at most 2 inputs supported");

        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i);
        }
        inputs = {{funcInputs[0].get_node_shared_ptr(), inputTensor}};

        if (funcInputs.size() > 1) {
            inputs.emplace(funcInputs[1].get_node_shared_ptr(), inputTensor);
        }
    }

    void SetUp() override {
        const auto& [lhsInputShape, inLayout, outLayout, outputChannels] = GetParam();

        const ov::ParameterVector params = getParams(lhsInputShape);

        const auto conv = buildDpuOp(params, outputChannels);
        const auto convert = buildConvert(conv->output(0), lhsInputShape);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convert)};

        function = std::make_shared<ov::Model>(results, params, "DpuWithF16ToF32ConvertTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input(0).tensor().set_layout(inLayout);
        preProc.input(0).model().set_layout(inLayout);

        if (params.size() == 2) {
            preProc.input(1).tensor().set_layout(inLayout);
            preProc.input(1).model().set_layout(inLayout);
        }

        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout(outLayout);
        function = preProc.build();

        // With the exception of the Conv tests, all other are accurate within the default threshold
        // The Conv tests with a large number of input channels have a slightly higher difference compared to CPU. It is
        // likely due to the way the DPU does the FP16 accumulation internally. Highest observed diff on random data was
        // 0.033, so setting threshold 0.04.
        abs_threshold = 0.04f;
    }

    std::shared_ptr<ov::Node> buildConvert(const ov::Output<ov::Node>& param, const ov::Shape& shape) {
        auto like = std::make_shared<op::v0::Constant>(ov::element::f32, shape);
        return std::make_shared<ov::op::v1::ConvertLike>(param, like);
    }

protected:
    virtual std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector&, const size_t) = 0;
    virtual ov::ParameterVector getParams(const ov::Shape& lhsInputShape) {
        init_input_shapes(static_shapes_to_test_representation({lhsInputShape}));

        return ov::ParameterVector{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<ov::Shape, ov::Layout, ov::Layout, size_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class ConvWithF16ToF32ConvertTest : public DpuWithF16ToF32ConvertTestBase {
    std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector& params, const size_t outCh) override {
        const ov::Shape& inputShape = params.at(0)->output(0).get_shape();
        const auto weightsSize = inputShape.at(1) * outCh * 1 * 1;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ov::Shape{outCh, inputShape.at(1), 1, 1};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);
        auto conv2d = std::make_shared<ov::op::v1::Convolution>(
                params.at(0), weights->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        return conv2d;
    }
};

class GroupConvWithF16ToF32ConvertTest : public DpuWithF16ToF32ConvertTestBase {
    std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector& params, const size_t /*outCh*/) override {
        const ov::Shape& inputShape = params.at(0)->output(0).get_shape();
        const auto weightsSize = inputShape.at(1) * 1 * 1;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ov::Shape{inputShape.at(1), 1, 1, 1, 1};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);

        const auto groupConv = std::make_shared<ov::op::v1::GroupConvolution>(
                params.at(0), weights, ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        return groupConv;
    }
};

class AvgPoolWithF16ToF32ConvertTest : public DpuWithF16ToF32ConvertTestBase {
    std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector& params, const size_t /*outCh*/) override {
        const auto avgpooling = std::make_shared<ov::op::v1::AvgPool>(
                params.at(0), ov::Strides(std::vector<size_t>{1, 1}), ov::Shape({0, 0}), ov::Shape({0, 0}),
                ov::Strides(std::vector<size_t>{3, 3}), false, ov::op::RoundingType::FLOOR, ov::op::PadType::AUTO);

        return avgpooling;
    }
};

class MaxPoolWithF16ToF32ConvertTest : public DpuWithF16ToF32ConvertTestBase {
    std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector& params, const size_t /*outCh*/) override {
        auto maxpool = std::make_shared<ov::op::v1::MaxPool>(
                params.at(0), ov::Strides(std::vector<size_t>{1, 1}), ov::Shape({0, 0}), ov::Shape({0, 0}),
                ov::Strides(std::vector<size_t>{3, 3}), ov::op::RoundingType::FLOOR, ov::op::PadType::AUTO);

        return maxpool;
    }
};

class EltwiseWithF16ToF32ConvertTest : public DpuWithF16ToF32ConvertTestBase {
    std::shared_ptr<ov::Node> buildDpuOp(const ov::ParameterVector& params, const size_t /*outCh*/) override {
        auto add = std::make_shared<ov::op::v1::Add>(params.at(0), params.at(1));
        return add;
    }

    ov::ParameterVector getParams(const ov::Shape& lhsInputShape) override {
        init_input_shapes(static_shapes_to_test_representation({lhsInputShape}));

        return ov::ParameterVector{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front()),
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};
    }
};

TEST_P(ConvWithF16ToF32ConvertTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupConvWithF16ToF32ConvertTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(AvgPoolWithF16ToF32ConvertTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(EltwiseWithF16ToF32ConvertTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvWithF16ToF32ConvertTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(GroupConvWithF16ToF32ConvertTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(AvgPoolWithF16ToF32ConvertTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(EltwiseWithF16ToF32ConvertTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvWithF16ToF32ConvertTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(GroupConvWithF16ToF32ConvertTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(AvgPoolWithF16ToF32ConvertTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(MaxPoolWithF16ToF32ConvertTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(EltwiseWithF16ToF32ConvertTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<ov::Shape> inputShapes = {{1, 16, 48, 32}, {1, 1024, 4, 8}, {1, 16, 128, 32}};
const std::vector<size_t> outputChannels = {8, 16};

const std::vector<ov::Layout> outLayout = {ov::Layout("NCHW")};
const std::vector<ov::Layout> inLayout = {ov::Layout("NHWC")};

INSTANTIATE_TEST_SUITE_P(smoke_conv_with_convertf16_to_f32, ConvWithF16ToF32ConvertTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inLayout),
                                            ::testing::ValuesIn(outLayout), ::testing::ValuesIn(outputChannels)),
                         ConvWithF16ToF32ConvertTest::getTestCaseName);

const std::vector<size_t> emptyOutputChannels = {0};

INSTANTIATE_TEST_SUITE_P(smoke_group_conv_with_convertf16_to_f32, GroupConvWithF16ToF32ConvertTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inLayout),
                                            ::testing::ValuesIn(outLayout), ::testing::ValuesIn(emptyOutputChannels)),
                         GroupConvWithF16ToF32ConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_avg_pool_with_convertf16_to_f32, AvgPoolWithF16ToF32ConvertTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inLayout),
                                            ::testing::ValuesIn(outLayout), ::testing::ValuesIn(emptyOutputChannels)),
                         AvgPoolWithF16ToF32ConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_max_pool_with_convertf16_to_f32, MaxPoolWithF16ToF32ConvertTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inLayout),
                                            ::testing::ValuesIn(outLayout), ::testing::ValuesIn(emptyOutputChannels)),
                         MaxPoolWithF16ToF32ConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_eltwise_with_convertf16_to_f32, EltwiseWithF16ToF32ConvertTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inLayout),
                                            ::testing::ValuesIn(outLayout), ::testing::ValuesIn(emptyOutputChannels)),
                         EltwiseWithF16ToF32ConvertTest::getTestCaseName);

}  // namespace ov::test
