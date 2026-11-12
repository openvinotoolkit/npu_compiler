//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>
#include <vpu_ov2_layer_test.hpp>
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test {

class ConvWithMultiplyTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<
                std::tuple<ov::Shape, std::vector<size_t>, std::vector<ptrdiff_t>, std::vector<ptrdiff_t>>> {
    void SetUp() override {
        const auto& [inShape, kernelSize, padBegin, padEnd] = GetParam();

        init_input_shapes(static_shapes_to_test_representation({{inShape}, {1, 128, 1, 1}}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]),
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[1])};

        const auto conv = buildConv(params.at(0), kernelSize, padBegin, padEnd);

        const auto multiply = std::make_shared<ov::op::v1::Multiply>(conv->output(0), params.at(1)->output(0));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(multiply)};

        function = std::make_shared<ov::Model>(results, params, "ConvWithMultiplyTest");
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& data, const std::vector<size_t>& kernelSize,
                                        const std::vector<ptrdiff_t>& padBegin, const std::vector<ptrdiff_t>& padEnd) {
        const ov::Shape& inputShape = data.get_shape();
        const auto weightsShape = ov::Shape{128, inputShape.at(1), kernelSize[0], kernelSize[1]};
        const auto weightsSize = shape_size(weightsShape);
        std::vector<float> values(weightsSize, 0.01f);
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);
        return std::make_shared<ov::op::v1::Convolution>(
                data, weights->output(0), ov::Strides(std::vector<size_t>{1, 1}), ov::CoordinateDiff(padBegin),
                ov::CoordinateDiff(padEnd), ov::Strides(std::vector<size_t>{1, 1}));
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        // Input 0: main data tensor
        {
            const auto& shape = targetInputStaticShapes[0];
            ov::Tensor tensor(ov::element::f16, shape);
            auto* data = tensor.data<ov::float16>();
            const size_t size = ov::shape_size(shape);

            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

            for (size_t i = 0; i < size; ++i) {
                data[i] = ov::float16(dist(gen));
            }
            inputs.insert({function->get_parameters()[0], tensor});
        }

        // Input 1: scale tensor
        {
            const auto& shape = targetInputStaticShapes[1];
            ov::Tensor tensor(ov::element::f16, shape);
            auto* data = tensor.data<ov::float16>();
            const size_t size = ov::shape_size(shape);

            std::mt19937 gen_scale(123);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

            for (size_t i = 0; i < size; ++i) {
                data[i] = ov::float16(dist(gen_scale));
            }
            inputs.insert({function->get_parameters()[1], tensor});
        }
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<
                    std::tuple<ov::Shape, std::vector<size_t>, std::vector<ptrdiff_t>, std::vector<ptrdiff_t>>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(ConvWithMultiplyTest, NPU6010_HW) {
    abs_threshold = 0.5;
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

const std::vector<ov::Shape> inputShapes = {
        {1, 16, 16, 32},
        {1, 16, 32, 64},
        {1, 8320, 32, 64},
};

const std::vector<std::vector<size_t>> kernelSizes = {
        {1, 1},
};

const std::vector<std::vector<ptrdiff_t>> padsBegin = {{0, 0}, {1, 1}};

const std::vector<std::vector<ptrdiff_t>> padsEnd = {{0, 0}, {1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvMultiply, ConvWithMultiplyTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(kernelSizes),
                                            ::testing::ValuesIn(padsBegin), ::testing::ValuesIn(padsEnd)),
                         ConvWithMultiplyTest::getTestCaseName);

}  // namespace ov::test
