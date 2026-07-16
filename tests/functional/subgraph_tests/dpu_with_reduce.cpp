// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::test {

struct DpuReduceParams {
    ov::Shape inputShape0;
    ov::element::Type inputType0;
    std::vector<int64_t> axes;
    std::string testCategory;
};

// Base class providing common input generation, reduce builders, and test-case naming.
class DPUWithReduceTestBase : public VpuOv2LayerTest, public testing::WithParamInterface<DpuReduceParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0.0;
        in_data.range = 1.0;
        in_data.resolution = 32768;

        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], in_data);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

protected:
    ov::Output<ov::Node> buildReduceMax(const ov::Output<ov::Node>& input, const std::vector<int64_t>& axes) const {
        auto axesConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes)->get_default_output();
        return std::make_shared<ov::op::v1::ReduceMax>(input, axesConst, /*keep_dims=*/true)->get_default_output();
    }

    ov::Output<ov::Node> buildReduceMin(const ov::Output<ov::Node>& input, const std::vector<int64_t>& axes) const {
        auto axesConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes)->get_default_output();
        return std::make_shared<ov::op::v1::ReduceMin>(input, axesConst, /*keep_dims=*/true)->get_default_output();
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<DpuReduceParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "Category=" << obj.param.testCategory << sep;
        result << "Shape0=" << ov::test::utils::vec2str(obj.param.inputShape0) << sep;
        result << "Type0=" << obj.param.inputType0 << sep;
        result << "Axes=" << ov::test::utils::vec2str(obj.param.axes) << sep;
        return result.str();
    }
};

class MaxPoolWithReduceMaxTest : public DPUWithReduceTestBase {
    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape0;
        const auto axes = testParams.axes;
        inType = outType = testParams.inputType0;

        const ov::Shape nhwcShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));

        auto maxPool = buildMaxPool(input);
        auto reduceMax = buildReduceMax(maxPool, axes);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(maxPool),
                                       std::make_shared<ov::op::v0::Result>(reduceMax)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "MaxPoolWithReduceMaxTest");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        preProc.output(1).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(1).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildMaxPool(const ov::Output<ov::Node>& input) const {
        return std::make_shared<ov::op::v1::MaxPool>(input, /*strides=*/ov::Strides{1, 1},
                                                     /*pads_begin=*/ov::Shape{0, 0}, /*pads_end=*/ov::Shape{0, 0},
                                                     /*kernel=*/ov::Shape{2, 2})
                ->get_default_output();
    }
};

class MatMulWithReduceMaxTest : public DPUWithReduceTestBase {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0.0;
        in_data.range = 1.0;
        in_data.resolution = 32768;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape0;
        const auto axes = testParams.axes;
        inType = outType = testParams.inputType0;

        // Build weights with matching batch dimensions to preserve the 4D batched MatMul.
        // A 2D MatMul (rank==2) would be converted to IE::FullyConnected by the UseFullyConnected
        // canonicalization pattern, bypassing the DPU+Reduce fusion path entirely.
        // The hasChannelAxisReduceConsumer check in MatMulOpConverter skips decomposition,
        // allowing downstream DPU+Reduce fusion via NCEMatMulOp.
        const auto inputRank = inputShape.size();
        const auto features = inputShape.back();
        ov::Shape weightsShape(inputShape.begin(), inputShape.begin() + inputRank - 2);
        weightsShape.push_back(features);
        weightsShape.push_back(features);

        const ov::Shape nhwcInputShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        const ov::Shape nhwcWeightsShape = {weightsShape[0], weightsShape[2], weightsShape[3], weightsShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcInputShape, nhwcWeightsShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));
        const auto weights = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(weightsShape));

        auto matMul = buildMatMul(input, weights);
        auto reduceMax = buildReduceMax(matMul, axes);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matMul),
                                       std::make_shared<ov::op::v0::Result>(reduceMax)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input, weights}, "MatMulWithReduceMaxTest");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.input(0).model().set_layout(ov::Layout("NCHW"));
        preProc.input(1).tensor().set_layout(ov::Layout("NHWC"));
        preProc.input(1).model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        preProc.output(1).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(1).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();
    }

    ov::Output<ov::Node> buildMatMul(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weights) {
        return std::make_shared<ov::op::v0::MatMul>(input, weights, /*transpose_a=*/false, /*transpose_b=*/true)
                ->get_default_output();
    }
};

// Shared convolution builder for Conv+Reduce subgraphs.
class ConvWithReduceTestBase : public DPUWithReduceTestBase {
protected:
    // Builds a 1x1 identity Convolution: output channels equal input channels, weight[i,i,0,0]=1 and 0 elsewhere.
    ov::Output<ov::Node> buildConv(const ov::Output<ov::Node>& input, const ov::Shape& inputShape) {
        const size_t inChannels = inputShape.at(1);
        const size_t outChannels = inChannels;
        const auto weightsShape = ov::Shape{outChannels, inChannels, 1, 1};
        std::vector<float> weightsData(outChannels * inChannels, 0.0f);
        for (size_t i = 0; i < inChannels; ++i) {
            weightsData[i * inChannels + i] = 1.0f;
        }
        const auto weights = ov::op::v0::Constant::create(inType, weightsShape, weightsData)->get_default_output();
        return std::make_shared<ov::op::v1::Convolution>(input, weights,
                                                         /*strides=*/ov::Strides{1, 1},
                                                         /*pads_begin=*/ov::CoordinateDiff{0, 0},
                                                         /*pads_end=*/ov::CoordinateDiff{0, 0},
                                                         /*dilations=*/ov::Strides{1, 1})
                ->get_default_output();
    }
};

class ConvWithReduceMaxTest : public ConvWithReduceTestBase {
    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape0;
        const auto axes = testParams.axes;
        inType = outType = testParams.inputType0;

        // The model is built with NCHW Parameter internally, but the external port exposed
        // after PrePostProcessor (NHWC tensor / NCHW model) is the NHWC-permuted shape.
        // Register the NHWC shape so the test framework generates tensors of the correct size.
        const ov::Shape nhwcShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));

        auto conv = buildConv(input, inputShape);
        auto reduceMax = buildReduceMax(conv, axes);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv),
                                       std::make_shared<ov::op::v0::Result>(reduceMax)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ConvWithReduceMaxTest");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        preProc.output(1).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(1).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();
    }
};

// Convolution feeding both ReduceMax and ReduceMin, exercising two reduce consumers of a single DPU op.
class ConvWithReduceMinAndMaxTest : public ConvWithReduceTestBase {
    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape0;
        const auto axes = testParams.axes;
        inType = outType = testParams.inputType0;

        const ov::Shape nhwcShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));

        auto conv = buildConv(input, inputShape);
        auto reduceMax = buildReduceMax(conv, axes);
        auto reduceMin = buildReduceMin(conv, axes);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv),
                                       std::make_shared<ov::op::v0::Result>(reduceMax),
                                       std::make_shared<ov::op::v0::Result>(reduceMin)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ConvWithReduceMinAndMaxTest");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        preProc.output(1).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(1).model().set_layout(ov::Layout("NCHW"));
        preProc.output(2).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(2).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();
    }
};

// Exercises the DecomposeSoftmaxInSdpa path by building
//   Conv1 (expand channels) -> SoftMax (axis=1) -> Conv2 (compress channels)
// Channel dimensions mirror the SDPA pattern from the LIT test (C_in=48, C_expanded=4096).
// W=4 prevents the compiler from wrapping SoftMax in ShapeCast ops, keeping the
// direct NCEConv->SoftMax->NCEConv chain visible to the pass.
class ConvWithSoftmaxTest : public DPUWithReduceTestBase {
    // Builds a 1x1 convolution mapping inChannels to outChannels.
    // Output channel i connects to input channel (i % inChannels).
    ov::Output<ov::Node> buildConvLayer(const ov::Output<ov::Node>& input, size_t inChannels,
                                        size_t outChannels) const {
        const auto weightsShape = ov::Shape{outChannels, inChannels, 1, 1};
        std::vector<float> weightsData(outChannels * inChannels, 0.0f);
        for (size_t i = 0; i < outChannels; ++i) {
            weightsData[i * inChannels + (i % inChannels)] = 1.0f;
        }
        const auto weights = ov::op::v0::Constant::create(inType, weightsShape, weightsData)->get_default_output();
        return std::make_shared<ov::op::v1::Convolution>(input, weights,
                                                         /*strides=*/ov::Strides{1, 1},
                                                         /*pads_begin=*/ov::CoordinateDiff{0, 0},
                                                         /*pads_end=*/ov::CoordinateDiff{0, 0},
                                                         /*dilations=*/ov::Strides{1, 1})
                ->get_default_output();
    }

    void SetUp() override {
        const auto testParams = GetParam();
        // inputShape0: NCHW shape of the input/output (C_in=48, W=4).
        // axes[0]: SoftMax channel axis (must be 1).
        // axes[1]: expanded channel count for the intermediate tensor (4096).
        //   expandedChannels / C_in = 4096 / 48 ≈ 85 > 64, satisfying the pass
        //   bottleneck criterion. W=4 matches the DPU tile width so the compiler
        //   does not insert ShapeCast ops around SoftMax.
        const auto inputShape = testParams.inputShape0;
        const auto axis = static_cast<int64_t>(testParams.axes.at(0));
        const auto expandedChannels = static_cast<size_t>(testParams.axes.at(1));
        inType = outType = testParams.inputType0;

        const ov::Shape nhwcShape = {inputShape[0], inputShape[2], inputShape[3], inputShape[1]};
        init_input_shapes(ov::test::static_shapes_to_test_representation({nhwcShape}));
        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, ov::PartialShape(inputShape));

        // Conv1: expand to a wide channel dimension.
        auto conv1 = buildConvLayer(input, inputShape[1], expandedChannels);
        // SoftMax on the channel axis; Conv2 is its only consumer.
        auto softmax = std::make_shared<ov::op::v8::Softmax>(conv1, axis)->get_default_output();
        // Conv2: compress back to C_in.
        auto conv2 = buildConvLayer(softmax, expandedChannels, inputShape[1]);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv2)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "ConvWithSoftmaxTest");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NHWC"));
        preProc.input().model().set_layout(ov::Layout("NCHW"));
        preProc.output(0).tensor().set_layout(ov::Layout("NHWC"));
        preProc.output(0).model().set_layout(ov::Layout("NCHW"));
        function = preProc.build();
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_FuseReduceMaxInMaxPool, MaxPoolWithReduceMaxTest,
                         ::testing::ValuesIn({
                                 DpuReduceParams{{1, 16, 2, 16}, ov::element::f16, {1}, "channel axis reduction"},
                         }),
                         DPUWithReduceTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseReduceMaxInMatMul, MatMulWithReduceMaxTest,
                         ::testing::ValuesIn({
                                 DpuReduceParams{{1, 4, 8, 16}, ov::element::f16, {-1}, "channel axis reduction"},
                         }),
                         DPUWithReduceTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseReduceMaxInConv, ConvWithReduceMaxTest,
                         ::testing::ValuesIn({
                                 DpuReduceParams{{1, 16, 2, 16}, ov::element::f16, {1}, "channel axis reduction"},
                         }),
                         DPUWithReduceTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseReduceMinMaxInConv, ConvWithReduceMinAndMaxTest,
                         ::testing::ValuesIn({
                                 DpuReduceParams{{1, 16, 2, 16}, ov::element::f16, {1}, "channel axis reduction"},
                         }),
                         DPUWithReduceTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_FuseReduceMaxFromSoftmaxInConv, ConvWithSoftmaxTest,
                         ::testing::ValuesIn({
                                 // Mirrors the SDPA LIT test: C_in=48, C_expanded=4096 (ratio ≈ 85 > 64),
                                 // W=4 prevents ShapeCast insertion around SoftMax.
                                 // axes = {softmax_axis=1, expanded_channels=4096}.
                                 DpuReduceParams{{1, 48, 1024, 4}, ov::element::f16, {1, 4096}, "channel axis softmax"},
                         }),
                         DPUWithReduceTestBase::getTestCaseName);

}  // namespace ov::test
