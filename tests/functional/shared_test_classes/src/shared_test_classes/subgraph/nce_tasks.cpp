//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/nce_tasks.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/reshape.hpp"

namespace {

std::shared_ptr<ov::Node> buildConv2d(const ov::Output<ov::Node>& param) {
    const std::vector<size_t> inputShape = param.get_shape();
    const size_t FILT_IN = inputShape.at(1);
    const std::vector<size_t> kenelShape = {16, FILT_IN, 3, 3};
    const size_t KERNEL_W = kenelShape.at(3);
    const size_t KERNEL_H = kenelShape.at(2);
    const size_t FILT_OUT = kenelShape.at(0);

    const auto quantInputRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantInput = NCETasksHelpers::quantize(param, quantInputRange);

    // Create non-trivial weights distribution to check accuracy.
    std::vector<ov::float16> weights(FILT_IN * FILT_OUT * KERNEL_W * KERNEL_H, 0.f);
    for (size_t oc = 0; oc < FILT_OUT; oc++) {
        size_t ic = oc;
        for (size_t ky = 0; ky < KERNEL_H; ky++) {
            for (size_t kx = 0; kx < KERNEL_W; kx++) {
                size_t i = kx + ky * KERNEL_W + ic * KERNEL_W * KERNEL_H + oc * KERNEL_W * KERNEL_H * FILT_IN;
                weights.at(i) = 1.f * (i % 2);
            }
        }
    }
    const auto weightsLayer = std::make_shared<ov::op::v0::Constant>(
            ov::element::f16, ov::Shape{FILT_OUT, FILT_IN, KERNEL_H, KERNEL_W}, weights.data());
    const auto quantWeightsRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantWeights = NCETasksHelpers::quantize(weightsLayer->output(0), quantWeightsRange);

    const auto convLayer = std::make_shared<ov::op::v1::Convolution>(
            quantInput, quantWeights, ov::Strides(std::vector<size_t>{1, 1}),
            ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
            ov::Strides(std::vector<size_t>{1, 1}));

    std::vector<ov::float16> biases(FILT_OUT, 1.0);
    for (size_t i = 0; i < biases.size(); i++) {
        biases.at(i) = 0.125f * (i % 2);
    }
    auto bias_weights_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, FILT_OUT, 1, 1}, biases.data());

    return std::make_shared<ov::op::v1::Add>(convLayer->output(0), bias_weights_node->output(0));
}

std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& param) {
    const auto quantInputRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantInput = NCETasksHelpers::quantize(param, quantInputRange);

    const std::vector<size_t> inputShape = param.get_shape();
    const auto totalSize = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<size_t>());
    // Create non-trivial weights distribution to check accuracy.
    std::vector<ov::float16> weights(totalSize, 0.f);
    for (size_t i = 0; i < weights.size(); i++) {
        weights.at(i) = 1.f * (i % 20);
    }
    const auto weightsLayer = std::make_shared<ov::op::v0::Constant>(ov::element::f16, inputShape, weights.data());
    const auto quantWeightsRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantWeights = NCETasksHelpers::quantize(weightsLayer->output(0), quantWeightsRange);
    return std::make_shared<ov::op::v1::Add>(quantInput, quantWeights);
}

std::shared_ptr<ov::Node> buildGroupConv2d(const ov::Output<ov::Node>& param) {
    const std::vector<size_t> inputShape = param.get_shape();
    const size_t FILT_IN_OUT = inputShape.at(1);
    const std::vector<size_t> kenelShape = {16, FILT_IN_OUT, 3, 3};
    const size_t KERNEL_W = kenelShape.at(3);
    const size_t KERNEL_H = kenelShape.at(2);

    const auto quantInputRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantInput = NCETasksHelpers::quantize(param, quantInputRange);

    std::vector<ov::float16> weights(FILT_IN_OUT * KERNEL_W * KERNEL_H, 0.f);
    const auto quantWeightRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    // Create non-trivial weights distribution to check accuracy.
    for (size_t i = 0; i < weights.size(); i++) {
        weights.at(i) = 1.f * (i % 2);
    }
    const auto weightsLayer = std::make_shared<ov::op::v0::Constant>(
            ov::element::f16, ov::Shape{FILT_IN_OUT, 1, KERNEL_H, KERNEL_W}, weights.data());
    const auto quantWeights = NCETasksHelpers::quantize(weightsLayer->output(0), quantWeightRange);
    const auto targetShape = std::vector<int64_t>{static_cast<int64_t>(FILT_IN_OUT), 1, 1,
                                                  static_cast<int64_t>(KERNEL_H), static_cast<int64_t>(KERNEL_W)};
    const auto targetShapeConst =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape.size()}, targetShape.data());
    const auto reshapedWeights = std::make_shared<ov::op::v1::Reshape>(quantWeights, targetShapeConst, false);

    auto groupConvLayer = std::make_shared<ov::op::v1::GroupConvolution>(
            quantInput, reshapedWeights->output(0), ov::Strides(std::vector<size_t>{1, 1}),
            ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
            ov::Strides(std::vector<size_t>{1, 1}));

    std::vector<ov::float16> biases(FILT_IN_OUT, 1.0);
    for (size_t i = 0; i < biases.size(); i++) {
        biases.at(i) = 0.125f * (i % 2);
    }
    auto bias_weights_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, FILT_IN_OUT, 1, 1}, biases.data());

    return std::make_shared<ov::op::v1::Add>(groupConvLayer->output(0), bias_weights_node->output(0));
}

std::shared_ptr<ov::Node> buildAvgPool(const ov::Output<ov::Node>& param) {
    const auto quantRange = std::array<float, 4>{0.f, 32.f, 0.f, 32.f};
    const auto quantInput = NCETasksHelpers::quantize(param, quantRange);

    const ov::Strides poolStrides = {1, 1};
    const ov::Shape padsBegin = {0, 0};
    const ov::Shape padsEnd = {0, 0};
    const ov::Shape poolKernel = {3, 3};
    return std::make_shared<ov::op::v1::AvgPool>(quantInput, poolStrides, padsBegin, padsEnd, poolKernel, true);
}

std::shared_ptr<ov::Node> buildMaxPool(const ov::Output<ov::Node>& param) {
    const auto quantRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantInput = NCETasksHelpers::quantize(param, quantRange);

    std::vector<uint64_t> poolStridesVec = {1, 1};
    std::vector<uint64_t> poolKernelVec = {1, 1};
    const ov::Strides poolStrides = {1, 1};
    const ov::Shape padsBegin = {0, 0};
    const ov::Shape padsEnd = {0, 0};
    const ov::Shape poolKernel = {3, 3};
    return std::make_shared<ov::op::v1::MaxPool>(quantInput, poolStrides, padsBegin, padsEnd, poolKernel);
}

}  // namespace

std::shared_ptr<ov::Node> NCETasksHelpers::buildNCETask(const ov::Output<ov::Node>& param, const NCEOpType& opType) {
    switch (opType) {
    case NCETasksHelpers::NCEOpType::AveragePooling:
        return buildAvgPool(param);
    case NCETasksHelpers::NCEOpType::Conv2d:
        return buildConv2d(param);
    case NCETasksHelpers::NCEOpType::EltwiseAdd:
        return buildAdd(param);
    case NCETasksHelpers::NCEOpType::GroupConv2d:
        return buildGroupConv2d(param);
    case NCETasksHelpers::NCEOpType::MaxPooling:
        return buildMaxPool(param);
    default:
        OPENVINO_THROW("buildNCETask: unsupported operation type: " + opType);
    }
}

std::shared_ptr<ov::Node> NCETasksHelpers::quantize(const ov::Output<ov::Node>& producer,
                                                    const std::array<float, 4>& fqRange, const size_t dataFqLvl) {
    const ov::Shape fqShape = {1};

    const std::vector<ov::float16> dataFqInLoVec = {fqRange.at(0)};
    const auto dataFqInLo = std::make_shared<ov::op::v0::Constant>(ov::element::f16, fqShape, dataFqInLoVec.data());

    const std::vector<ov::float16> dataFqInHiVec = {fqRange.at(1)};
    const auto dataFqInHi = std::make_shared<ov::op::v0::Constant>(ov::element::f16, fqShape, dataFqInHiVec.data());

    const std::vector<ov::float16> dataFqOutLoVec = {fqRange.at(2)};
    const auto dataFqOutLo = std::make_shared<ov::op::v0::Constant>(ov::element::f16, fqShape, dataFqOutLoVec.data());

    const std::vector<ov::float16> dataFqOutHiVec = {fqRange.at(3)};
    const auto dataFqOutHi = std::make_shared<ov::op::v0::Constant>(ov::element::f16, fqShape, dataFqOutHiVec.data());

    const auto dataFq = std::make_shared<ov::op::v0::FakeQuantize>(
            producer, dataFqInLo, dataFqInHi, dataFqOutLo, dataFqOutHi, dataFqLvl, ov::op::AutoBroadcastType::NUMPY);

    return dataFq;
}

std::string NCETasksHelpers::NCEOpTypeToString(const NCEOpType& opType) {
    switch (opType) {
    case NCETasksHelpers::NCEOpType::AveragePooling:
        return "AveragePooling";
    case NCETasksHelpers::NCEOpType::Conv2d:
        return "Conv2d";
    case NCETasksHelpers::NCEOpType::EltwiseAdd:
        return "EltwiseAdd";
    case NCETasksHelpers::NCEOpType::GroupConv2d:
        return "GroupConv2d";
    case NCETasksHelpers::NCEOpType::MaxPooling:
        return "MaxPooling";
    default:
        OPENVINO_THROW("buildNCETask: unsupported operation type: " + opType);
    }
}
