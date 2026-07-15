//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <intel_npu/npu_private_properties.hpp>
#include <vpux/utils/core/error.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset15_decl.hpp"

using namespace ov::test;
using DynamicStridesTestParams = std::tuple<ov::Shape, std::vector<std::vector<size_t>>>;

class DynamicStridesTestBase : public ::testing::TestWithParam<DynamicStridesTestParams> {
protected:
    void SetUp() override {
        if (utils::getTestDeviceId() == "3720") {
            GTEST_SKIP() << "Dynamic strides are unsupported on NPU3720";
        }
    }

    virtual std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) = 0;

    void initializeInputTensors(ov::Tensor& cpuInput, ov::Tensor& npuInput) {
        auto cpuData = cpuInput.data<float>();
        auto npuData = npuInput.data<float>();
        for (size_t idx = 0; idx < cpuInput.get_size(); ++idx) {
            cpuData[idx] = static_cast<float>(idx % 100);
            npuData[idx] = static_cast<float>(idx % 100);
        }
    }

    void runInferenceOnAllSlices(ov::CompiledModel& cpuModel, ov::CompiledModel& npuModel, ov::Tensor& cpuFullTensor,
                                 ov::Tensor& cpuFullOutputTensor, ov::Tensor& npuFullTensor,
                                 ov::Tensor& npuFullOutputTensor, const ov::Shape& sliceShape,
                                 const std::vector<size_t>& slices) {
        ov::InferRequest npuRequest = npuModel.create_infer_request();
        ov::InferRequest cpuRequest = cpuModel.create_infer_request();

        auto runInferenceOnSlice = [&](ov::Shape start, ov::Shape end) {
            ov::Tensor cpuRoi(cpuFullTensor, start, end);
            ov::Tensor cpuRoiOut(cpuFullOutputTensor, start, end);
            ov::Tensor npuRoi(npuFullTensor, start, end);
            ov::Tensor npuRoiOut(npuFullOutputTensor, start, end);
            npuRequest.set_input_tensor(0, npuRoi);
            npuRequest.set_input_tensor(1, npuRoi);
            npuRequest.set_output_tensor(0, npuRoiOut);
            npuRequest.infer();
            cpuRequest.set_input_tensor(0, cpuRoi);
            cpuRequest.set_input_tensor(1, cpuRoi);
            cpuRequest.set_output_tensor(0, cpuRoiOut);
            cpuRequest.infer();
        };

        runOnAllSlices(sliceShape, slices, std::move(runInferenceOnSlice));
    }

    ov::Shape getSliceShape(const ov::Shape& fullTensor, const std::vector<size_t>& slices) {
        std::vector<size_t> sliceDims;
        for (size_t shapeIdx = 0; shapeIdx < slices.size(); shapeIdx++) {
            VPUX_THROW_WHEN(fullTensor[shapeIdx] % slices[shapeIdx], "Dimension must be divisible by number of slices");
            sliceDims.push_back(fullTensor[shapeIdx] / slices[shapeIdx]);
        }
        return ov::Shape(sliceDims);
    }

    void runOnAllSlices(const ov::Shape& sliceShape, const std::vector<size_t>& slices,
                        std::function<void(ov::Shape start, ov::Shape end)> function) {
        ov::Shape start(sliceShape.size(), 0);
        ov::Shape end(sliceShape);

        auto runOnAllSlicesRecursive = [&](const auto& self, ov::Shape start, ov::Shape end, size_t dimIndex) -> void {
            for (size_t sliceIdx = 0; sliceIdx < slices[dimIndex]; sliceIdx++) {
                if (dimIndex == slices.size() - 1) {
                    function(start, end);
                    start[dimIndex] += sliceShape[dimIndex];
                    end[dimIndex] += sliceShape[dimIndex];
                } else {
                    self(self, start, end, dimIndex + 1);
                    start[dimIndex] += sliceShape[dimIndex];
                    end[dimIndex] += sliceShape[dimIndex];
                }
            }
        };

        runOnAllSlicesRecursive(runOnAllSlicesRecursive, start, end, 0);
    }

    void run() {
        _npuCompilationParams[ov::intel_npu::platform.name()] = utils::getTestDeviceId();
        auto fullTensor = std::get<0>(GetParam());
        auto slicings = std::get<1>(GetParam());
        size_t sliceIdx = 0;
        for (const auto& slicing : slicings) {
            auto sliceShape = getSliceShape(fullTensor, slicing);

            ov::Core core;
            auto model = getTestModel(sliceShape);
            auto inputs = model->inputs();
            auto outputs = model->outputs();

            std::stringstream ss;
            for (auto input : inputs) {
                ss << input.get_node()->get_name() << ",";
            }
            for (auto output : outputs) {
                ss << output.get_node()->get_name() << ",";
            }

            _npuCompilationParams[ov::intel_npu::enable_strides_for.name()] = ss.str();
            auto cpuModel = core.compile_model(model, "CPU");
            ov::CompiledModel npuModel;
            OV_ASSERT_NO_THROW(npuModel = core.compile_model(model, test_utils::TARGET_DEVICE, _npuCompilationParams));

            auto npuContext = core.get_default_context(test_utils::TARGET_DEVICE);
            ov::Tensor cpuFullTensor(ov::element::f32, fullTensor);
            ov::Tensor cpuFullOutputTensor(ov::element::f32, fullTensor);
            auto npuFullTensor = npuContext.create_host_tensor(ov::element::f32, fullTensor);
            auto npuFullOutputTensor = npuContext.create_host_tensor(ov::element::f32, fullTensor);

            initializeInputTensors(cpuFullTensor, npuFullTensor);
            runInferenceOnAllSlices(cpuModel, npuModel, cpuFullTensor, cpuFullOutputTensor, npuFullTensor,
                                    npuFullOutputTensor, sliceShape, slicing);

            auto resultCpuData = cpuFullOutputTensor.data<float>();
            auto resultNpuData = npuFullOutputTensor.data<float>();
            for (size_t idx = 0; idx < cpuFullOutputTensor.get_size(); ++idx) {
                auto diff = std::abs(resultNpuData[idx] - resultCpuData[idx]);
                ASSERT_TRUE(diff < 0.1f) << "Value mismatch at element idx " << idx << " for slice idx " << sliceIdx;
            }
            sliceIdx++;
        }
    }

    ov::AnyMap _npuCompilationParams;
};

// Simple model without any internal model slicing. Which compute layer is used doesn't matter for dynamic strides
// so a simple add operation is used.
class DynamicStridesBehaviorTest : public DynamicStridesTestBase {
protected:
    std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) override {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto add = std::make_shared<ov::op::v1::Add>(param1, param2);
        auto result = std::make_shared<ov::opset15::Result>(add->output(0));
        const auto results = ov::ResultVector{result};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                           "SimpleDynamicStridesTestModel");
    }
};

TEST_P(DynamicStridesBehaviorTest, DynamicStridesTest) {
    run();
}

DynamicStridesTestParams smallTensorAll4DTilings{{8, 8, 8, 8},
                                                 {{1, 1, 1, 1},
                                                  {2, 1, 1, 1},
                                                  {1, 2, 1, 1},
                                                  {1, 1, 2, 1},
                                                  {1, 1, 1, 2},
                                                  {1, 1, 2, 2},
                                                  {1, 2, 1, 2},
                                                  {2, 1, 1, 2},
                                                  {1, 2, 2, 1},
                                                  {2, 1, 2, 1},
                                                  {2, 2, 1, 1},
                                                  {1, 2, 2, 2},
                                                  {2, 2, 2, 1},
                                                  {2, 2, 1, 2},
                                                  {2, 1, 2, 2},
                                                  {2, 2, 2, 2}}};

INSTANTIATE_TEST_SUITE_P(All4DTilingPermutations, DynamicStridesBehaviorTest,
                         ::testing::Values(smallTensorAll4DTilings),
                         (utils::appendPlatformTypeTestName<DynamicStridesBehaviorTest>));

DynamicStridesTestParams bigTensorTiling{{1, 16, 1280, 1280}, {{1, 4, 20, 4}}};

INSTANTIATE_TEST_SUITE_P(BigTensorTiling, DynamicStridesBehaviorTest, ::testing::Values(bigTensorTiling),
                         (utils::appendPlatformTypeTestName<DynamicStridesBehaviorTest>));

// More complex model with internal slices, used to exercise logic for calculating tile offsets at runtime.
class DynamicStridesWithSlicesBehaviorTest : public DynamicStridesTestBase {
protected:
    std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) override {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);

        std::vector<size_t> modelSlices{2, 2, 2, 2};
        auto modelSliceShape = getSliceShape(inputShape, modelSlices);
        ov::OutputVector addResults;
        ov::Shape onesShape = {1, 1, 1, 1};
        auto ones = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, onesShape);
        auto insertOpOnSlice = [&](ov::Shape start, ov::Shape end) {
            auto startConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, start);
            auto endConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, end);
            auto slice1 = std::make_shared<ov::op::v8::Slice>(param1, startConst, endConst, ones);
            auto slice2 = std::make_shared<ov::op::v8::Slice>(param2, startConst, endConst, ones);
            auto addSlice = std::make_shared<ov::op::v1::Add>(slice1, slice2);
            addResults.push_back(addSlice->output(0));
        };
        runOnAllSlices(modelSliceShape, modelSlices, std::move(insertOpOnSlice));

        ov::OutputVector previousLayerResults = addResults;
        for (size_t dimIdx = 0; dimIdx < inputShape.size(); dimIdx++) {
            ov::OutputVector currentLayerResults;
            for (size_t sliceIdx = 0; sliceIdx < previousLayerResults.size(); sliceIdx += 2) {
                auto concat = std::make_shared<ov::op::v0::Concat>(
                        ov::OutputVector{previousLayerResults[sliceIdx], previousLayerResults[sliceIdx + 1]}, dimIdx);
                currentLayerResults.push_back(concat->output(0));
            }
            previousLayerResults = currentLayerResults;
        }

        auto result = std::make_shared<ov::opset15::Result>(previousLayerResults[0]);
        const auto results = ov::ResultVector{result};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                           "DynamicStridesTestModelWithSlices");
    }
};

TEST_P(DynamicStridesWithSlicesBehaviorTest, DynamicStridesTest) {
    run();
}

INSTANTIATE_TEST_SUITE_P(All4DTilingPermutations, DynamicStridesWithSlicesBehaviorTest,
                         ::testing::Values(smallTensorAll4DTilings),
                         (utils::appendPlatformTypeTestName<DynamicStridesWithSlicesBehaviorTest>));

// Simple model with permutation added. Tests if dynamic strides relocation is permuted correctly.
class DynamicStridesWithTransposeBehaviorTest : public DynamicStridesTestBase {
protected:
    std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) override {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto transposeOrder = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4},
                                                                     std::vector<int32_t>{2, 3, 0, 1});
        auto transposedParam1 = std::make_shared<ov::op::v1::Transpose>(param1, transposeOrder);
        auto transposedParam2 = std::make_shared<ov::op::v1::Transpose>(param2, transposeOrder);
        auto add = std::make_shared<ov::op::v1::Add>(transposedParam1, transposedParam2);
        auto transposeResult = std::make_shared<ov::op::v1::Transpose>(add, transposeOrder);
        auto result = std::make_shared<ov::opset15::Result>(transposeResult->output(0));
        const auto results = ov::ResultVector{result};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                           "DynamicStridesWithTransposeLayerTest");
    }
};

TEST_P(DynamicStridesWithTransposeBehaviorTest, DynamicStridesTest) {
    run();
}

INSTANTIATE_TEST_SUITE_P(All4DTilingPermutations, DynamicStridesWithTransposeBehaviorTest,
                         ::testing::Values(smallTensorAll4DTilings),
                         (utils::appendPlatformTypeTestName<DynamicStridesWithTransposeBehaviorTest>));

class DynamicStridesWithStridedSlicesBehaviorTest : public DynamicStridesTestBase {
protected:
    std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) override {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);

        std::vector<size_t> modelSlices{2, 1, 1, 1};
        auto modelSliceShape = getSliceShape(inputShape, modelSlices);

        auto steps = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, ov::Shape{2, 2, 2, 2});
        auto startConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, ov::Shape{0, 0, 0, 0});
        auto endConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, inputShape);
        auto slice1 = std::make_shared<ov::op::v8::Slice>(param1, startConst, endConst, steps);
        auto slice2 = std::make_shared<ov::op::v8::Slice>(param2, startConst, endConst, steps);
        auto addSlice = std::make_shared<ov::op::v1::Add>(slice1, slice2);

        ov::OutputVector previousLayerResults{addSlice->output(0), addSlice->output(0)};
        for (size_t dimIdx = 0; dimIdx < inputShape.size(); dimIdx++) {
            ov::OutputVector currentLayerResults;
            auto concat = std::make_shared<ov::op::v0::Concat>(
                    ov::OutputVector{previousLayerResults[0], previousLayerResults[1]}, dimIdx);
            currentLayerResults.push_back(concat->output(0));
            currentLayerResults.push_back(concat->output(0));
            previousLayerResults = currentLayerResults;
        }

        auto result = std::make_shared<ov::opset15::Result>(previousLayerResults[0]);
        const auto results = ov::ResultVector{result};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                           "DynamicStridesWithStridedSlicesBehaviorTest");
    }
};

TEST_P(DynamicStridesWithStridedSlicesBehaviorTest, DynamicStridesTest) {
    run();
}

INSTANTIATE_TEST_SUITE_P(All4DTilingPermutations, DynamicStridesWithStridedSlicesBehaviorTest,
                         ::testing::Values(smallTensorAll4DTilings),
                         (utils::appendPlatformTypeTestName<DynamicStridesWithStridedSlicesBehaviorTest>));

// Simple model with reshaped inputs to shapes incompatible with input arguments.
class DynamicStridesWithIncompatibleReshapeBehaviorTest : public DynamicStridesTestBase {
protected:
    std::shared_ptr<ov::Model> getTestModel(ov::Shape inputShape) override {
        unsigned int elements = 1;
        for (size_t shapeIdx = 0; shapeIdx < inputShape.size(); shapeIdx++) {
            elements *= inputShape[shapeIdx];
        }
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, ov::Shape{elements});
        auto reshaped1 = std::make_shared<ov::op::v1::Reshape>(param1, constant, false);
        auto reshaped2 = std::make_shared<ov::op::v1::Reshape>(param2, constant, false);
        auto add = std::make_shared<ov::op::v1::Add>(reshaped1, reshaped2);
        auto constantOutReshape =
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{inputShape.size()}, inputShape);
        auto reshapedOut = std::make_shared<ov::op::v1::Reshape>(add, constantOutReshape, false);
        auto result = std::make_shared<ov::opset15::Result>(reshapedOut->output(0));
        const auto results = ov::ResultVector{result};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{std::move(param1), std::move(param2)},
                                           "DynamicStridesWithIncompatibleReshapeBehaviorTest");
    }
};

TEST_P(DynamicStridesWithIncompatibleReshapeBehaviorTest, DynamicStridesTest) {
    run();
}

INSTANTIATE_TEST_SUITE_P(All4DTilingPermutations, DynamicStridesWithIncompatibleReshapeBehaviorTest,
                         ::testing::Values(smallTensorAll4DTilings),
                         (utils::appendPlatformTypeTestName<DynamicStridesWithIncompatibleReshapeBehaviorTest>));
