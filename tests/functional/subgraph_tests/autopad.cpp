//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest-param-test.h>
#include <openvino/op/add.hpp>
#include <openvino/op/avg_pool.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/max_pool.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/softmax.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "common/quantization_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace ov::test {

enum class OpType {
    CONV = 0,
    GROUP_CONV = 1,
    MAXPOOL = 2,
    AVGPOOL = 3,
    REDUCE_MEAN = 4,
    REDUCE_SUM = 5,
    SOFTMAX = 6,
};

// clang-format off
static std::ostream& operator<<(std::ostream& os, const OpType opType) {
    switch (opType) {
    case OpType::CONV: { return os << "CONV"; }
    case OpType::GROUP_CONV: { return os << "GROUP_CONV"; }
    case OpType::MAXPOOL: { return os << "MAXPOOL"; }
    case OpType::AVGPOOL: { return os << "AVGPOOL"; }
    case OpType::REDUCE_MEAN: { return os << "REDUCE_MEAN"; }
    case OpType::REDUCE_SUM: { return os << "REDUCE_SUM"; }
    case OpType::SOFTMAX: { return os << "SOFTMAX"; }
    default: { return os << "NONE"; }
    }
}
// clang-format on

struct Params {
    ov::Shape inputShape;
    OpType firstOp;
    OpType secondOp;
    bool quantized;
};

class AutoPaddingTest : public VpuOv2LayerTest, public testing::WithParamInterface<Params> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-auto-padding-idu=true enable-auto-padding-odu=true enable-is-reduce-supported=true";
    }

    void SetUp() override {
        const auto createOp = [&](const Output<Node>& input, OpType opType, bool quantized) -> Output<Node> {
            if (opType == OpType::CONV) {
                return buildConv(input, quantized);
            } else if (opType == OpType::GROUP_CONV) {
                return buildGroupConv(input, quantized);
            } else if (opType == OpType::MAXPOOL) {
                return buildMaxPool(input);
            } else if (opType == OpType::AVGPOOL) {
                return buildAvgPool(input);
            } else if (opType == OpType::REDUCE_MEAN) {
                return buildReduceMean(input);
            } else if (opType == OpType::REDUCE_SUM) {
                return buildReduceSum(input);
            } else if (opType == OpType::SOFTMAX) {
                return buildSoftmax(input);
            }
            VPUX_THROW("Unknown op type for first op");
        };

        const auto parameters = GetParam();

        init_input_shapes(static_shapes_to_test_representation({parameters.inputShape}));
        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        auto result = params.at(0)->get_default_output();
        if (parameters.quantized) {
            result = utils::makeFakeQuantize(result, ov::element::f16, 256,
                                             FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                             ->get_default_output();
        }

        result = createOp(result, parameters.firstOp, parameters.quantized);

        if (parameters.quantized) {
            result = utils::makeFakeQuantize(result, ov::element::f16, 256,
                                             FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                             ->get_default_output();
        }

        result = createOp(result, parameters.secondOp, parameters.quantized);

        if (parameters.quantized) {
            result = utils::makeFakeQuantize(result, ov::element::f16, 256,
                                             FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                             ->get_default_output();
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(result)};
        function = std::make_shared<ov::Model>(results, params, "AutoPaddingTest");
    }

    ov::Output<ov::Node> buildConv(const ov::Output<ov::Node>& input, bool quantized) const {
        const auto inputChannels = input.get_shape().at(1);
        const auto weightsShape = ov::Shape{3, inputChannels, 1, 1};
        const auto weightsSize = shape_size(weightsShape);
        auto weights =
                ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>(weightsSize, 0.5f))
                        ->get_default_output();
        if (quantized) {
            weights = utils::makeFakeQuantize(weights, ov::element::f16, 256,
                                              FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                              ->get_default_output();
        }

        return std::make_shared<ov::op::v1::Convolution>(input, weights, /*strides=*/ov::Strides({1, 1}),
                                                         /*pads_begin=*/ov::CoordinateDiff({0, 0}),
                                                         /*pads_end=*/ov::CoordinateDiff({0, 0}),
                                                         /*dilations=*/ov::Strides({1, 1}))
                ->get_default_output();
    }

    ov::Output<ov::Node> buildGroupConv(const ov::Output<ov::Node>& input, bool quantized) const {
        const auto inputChannels = input.get_shape().at(1);
        const auto weightsShape = ov::Shape{/*groups=*/inputChannels, 1, 1, 1, 1};
        const auto weightsSize = shape_size(weightsShape);
        auto weights =
                ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>(weightsSize, 0.5f))
                        ->get_default_output();
        if (quantized) {
            weights = utils::makeFakeQuantize(weights, ov::element::f16, 256,
                                              FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                              ->get_default_output();
        }

        return std::make_shared<ov::op::v1::GroupConvolution>(input, weights, /*strides=*/ov::Strides({1, 1}),
                                                              /*pads_begin=*/ov::CoordinateDiff({0, 0}),
                                                              /*pads_end=*/ov::CoordinateDiff({0, 0}),
                                                              /*dilations=*/ov::Strides({1, 1}))
                ->get_default_output();
    }

    ov::Output<ov::Node> buildMaxPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::MaxPool>(input, /*strides=*/ov::Strides{1, 1},
                                                     /*pads_begin=*/ov::Shape{0, 0}, /*pads_end=*/ov::Shape{0, 0},
                                                     /*kernel=*/ov::Shape{1, 1})
                ->get_default_output();
    }

    ov::Output<ov::Node> buildAvgPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::AvgPool>(input, /*strides=*/ov::Strides{1, 1},
                                                     /*pads_begin=*/ov::Shape{0, 0}, /*pads_end=*/ov::Shape{0, 0},
                                                     /*kernel=*/ov::Shape{1, 1}, /*exclude_pad=*/false)
                ->get_default_output();
    }

    ov::Output<ov::Node> buildReduceMean(const ov::Output<ov::Node>& input) const {
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})->get_default_output();
        return std::make_shared<ov::op::v1::ReduceMean>(input, axes, /*keep_dims=*/true)->get_default_output();
    }

    ov::Output<ov::Node> buildReduceSum(const ov::Output<ov::Node>& input) const {
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})->get_default_output();
        return std::make_shared<ov::op::v1::ReduceSum>(input, axes, /*keep_dims=*/true)->get_default_output();
    }

    ov::Output<ov::Node> buildSoftmax(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::Softmax>(input, /*axis=*/2)->get_default_output();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<Params>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << utils::testKind(__FILE__);
        result << sep << "firstOp=" << obj.param.firstOp;
        result << sep << "secondOp=" << obj.param.secondOp;
        result << sep << "quantized=" << std::boolalpha << obj.param.quantized;
        return result.str();
    };
};

TEST_P(AutoPaddingTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(IDU, AutoPaddingTest,
                         testing::ValuesIn({
                                 Params{ov::Shape{1, 3, 16, 16}, OpType::SOFTMAX, OpType::CONV, /*quantized=*/false},
                         }),
                         AutoPaddingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        ODU, AutoPaddingTest,
        testing::ValuesIn({
                Params{ov::Shape{1, 3, 16, 16}, OpType::CONV, OpType::SOFTMAX, /*quantized=*/false},
                Params{ov::Shape{1, 3, 16, 16}, OpType::CONV, OpType::SOFTMAX, /*quantized=*/true},
                Params{ov::Shape{1, 3, 16, 16}, OpType::GROUP_CONV, OpType::SOFTMAX, /*quantized=*/false},
                Params{ov::Shape{1, 3, 16, 16}, OpType::MAXPOOL, OpType::SOFTMAX, /*quantized=*/false},
                Params{ov::Shape{1, 3, 16, 16}, OpType::AVGPOOL, OpType::SOFTMAX, /*quantized=*/false},
                Params{ov::Shape{1, 3, 16, 16}, OpType::REDUCE_SUM, OpType::SOFTMAX, /*quantized=*/false},
                Params{ov::Shape{1, 3, 16, 16}, OpType::REDUCE_MEAN, OpType::SOFTMAX, /*quantized=*/false},
        }),
        AutoPaddingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(IDU_ODU, AutoPaddingTest,
                         testing::ValuesIn({
                                 Params{ov::Shape{1, 3, 16, 16}, OpType::CONV, OpType::CONV, /*quantized=*/false},
                         }),
                         AutoPaddingTest::getTestCaseName);

}  // namespace ov::test
