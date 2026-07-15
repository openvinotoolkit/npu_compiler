//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Minimal reproducer for the H-axis SCF loop unroll bug (loop-unroll-factor=1,1,2,1).
//
// Bug: second H-copy output is corrupt when H-axis unroll factor > 1.
// Observed with ESPCN_x2_gh (Conv 5x5 + Conv 3x3 + Conv 3x3 + DepthToSpace).
// Confirmed to also reproduce with all-3x3 variant (ESPCN_x2_gh_3x3).
//
// Minimal topology that reproduces the bug (mirrors ESPCN structure):
//
//   Input [1, H, W, 1]  (dynamic, NHWC, fp16)
//         |
//         v
//   Transpose [0,3,1,2]   -- NHWC → NCHW: [1, 1, H, W]
//         |
//         v
//   Conv (3x3, pad=1, stride=1, 1→4 channels)   -- preserves H, W; outputs 4ch for DepthToSpace
//         |
//         v
//   DepthToSpace (block=2, BLOCKS_FIRST)          -- [1, 4, H, W] → [1, 1, 2H, 2W]
//         |
//         v
//   Output [1, 1, 2H, 2W]
//
// Hypothesis: NHWC input + Transpose may interact with unrolling logic
// and cause the second H-copy to receive wrong spatial offset.
// With loop-unroll-factor=1,1,2,1 the H loop is unrolled by factor 2.

#include <openvino/opsets/opset1.hpp>
#include "openvino/opsets/opset6_decl.hpp"

#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using ESPCNLikeSubgraphParams = std::tuple<ov::test::InputShape,  // Input shape
                                           ov::element::Type>;    // Input precision

class ESPCNLikeSubgraph : public VpuOv2LayerTest, public testing::WithParamInterface<ESPCNLikeSubgraphParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ESPCNLikeSubgraphParams>& obj) {
        const auto& [inShape, inType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape={" << inShape.first.to_string() << "}" << sep;
        result << "InputType=" << inType;
        return result.str();
    }

    void SetUp() override {
        const auto& [inShape, inType] = GetParam();

        init_input_shapes({inShape});

        // Input is NHWC: [1, H, W, 1] — mirrors the original ESPCN model layout.
        auto input = std::make_shared<ov::opset1::Parameter>(inType, inputDynamicShapes[0]);

        // Transpose NHWC → NCHW: [1, H, W, 1] → [1, 1, H, W]
        auto transposeOrder = ov::opset1::Constant::create(ov::element::i64, {4}, {0, 3, 1, 2});
        auto transpose = std::make_shared<ov::opset1::Transpose>(input, transposeOrder);

        // Conv: 1→4 channels, 3x3 kernel, padding=1 on all sides, stride=1.
        // Produces exactly 4 output channels needed by DepthToSpace(block=2).
        // Spatial dimensions H and W are preserved.
        const int64_t inChannels = 1;
        const int64_t outChannels = 4;  // block^2 = 2^2, required by DepthToSpace
        std::vector<float> weightValues(outChannels * inChannels * 3 * 3, 1.f / 9.f);
        auto weights = ov::opset1::Constant::create(inType, {outChannels, inChannels, 3, 3}, weightValues);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads = {1, 1};
        const ov::Strides dilations = {1, 1};
        auto conv = std::make_shared<ov::opset6::Convolution>(transpose, weights, strides, pads, pads, dilations);

        // DepthToSpace: [1, 4, H, W] → [1, 1, 2H, 2W]
        auto dts = std::make_shared<ov::opset1::DepthToSpace>(conv,
                                                              ov::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                              /*block_size=*/2);

        const auto result = std::make_shared<ov::opset1::Result>(dts);
        function =
                std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "ESPCNLikeSubgraph");

        // Apply layout preprocessing matching ESPCN's compile_tool flags:
        //   input_layout=NCHW, model_input_layout=NHWC
        // PPP inserts a NCHW→NHWC Transpose at the input boundary. The compiler
        // folds it with the model's existing NHWC→NCHW Transpose into a
        // VPU.PermuteCast (memory-layout-only op). The two IE.Transpose ops cancel,
        // so @main's output-shape chain contains only tensor.dim + arith ops,
        // which OutlineDimOperations can handle. External interface becomes
        // NCHW [1, 1, H, W] — identical to the compiled ESPCN model.
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(ov::Layout("NCHW"));
        preProc.input().model().set_layout(ov::Layout("NHWC"));
        function = preProc.build();

        // After PPP, the external input interface is NCHW [1, 1, H, W].
        // targetStaticShapes was populated from NHWC inShape {N, H, W, C};
        // permute each entry to NCHW {N, C, H, W} so the test harness feeds
        // correctly-shaped tensors.
        for (auto& perInferenceShapes : targetStaticShapes) {
            for (auto& shape : perInferenceShapes) {
                OPENVINO_ASSERT(shape.size() == 4, "Expected 4D input shape");
                shape = {shape[0], shape[3], shape[1], shape[2]};  // {N,H,W,C} → {N,C,H,W}
            }
        }
    }
};

// H-axis unroll factor=2 — the configuration that triggers the bug.
class ESPCNLikeSubgraphHUnroll : public ESPCNLikeSubgraph {
public:
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "loop-unroll-factor=1,1,2,1";
    }
};

// No unrolling — baseline reference: output must be correct.
class ESPCNLikeSubgraphNoUnroll : public ESPCNLikeSubgraph {
public:
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "loop-unroll-factor=1,1,1,1";
    }
};

// W-axis unroll factor=2 — control: unrolling a different axis should not corrupt H.
class ESPCNLikeSubgraphWUnroll : public ESPCNLikeSubgraph {
public:
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "loop-unroll-factor=1,1,1,2";
    }
};

// H- and W-axis unroll factor=2 — combined 2D unroll: both axes unrolled simultaneously.
class ESPCNLikeSubgraphHWUnroll : public ESPCNLikeSubgraph {
public:
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "loop-unroll-factor=1,1,2,2";
    }
};

// NPU4000

TEST_P(ESPCNLikeSubgraphNoUnroll, NPU4000_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ESPCNLikeSubgraphHUnroll, NPU4000_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ESPCNLikeSubgraphWUnroll, NPU4000_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(ESPCNLikeSubgraphHWUnroll, NPU4000_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU4000);
}

// NPU5010

TEST_P(ESPCNLikeSubgraphNoUnroll, NPU5010_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ESPCNLikeSubgraphHUnroll, NPU5010_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ESPCNLikeSubgraphWUnroll, NPU5010_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(ESPCNLikeSubgraphHWUnroll, NPU5010_HC) {
    setHostCompileMode("HostCompile_Interpreter");
    setPluginCompilerType();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

const ov::test::InputShape kESPCNTestShape{ov::PartialShape{1, ov::Dimension(1, 2160), ov::Dimension(1, 3840), 1},
                                           std::vector<ov::Shape>{{1, 1024, 1024, 1}}};

INSTANTIATE_TEST_SUITE_P(smoke_ESPCNLike_HAxisUnroll, ESPCNLikeSubgraphHUnroll,
                         ::testing::Values(ESPCNLikeSubgraphParams{kESPCNTestShape, ov::element::f16}),
                         ESPCNLikeSubgraphHUnroll::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ESPCNLike_NoUnroll, ESPCNLikeSubgraphNoUnroll,
                         ::testing::Values(ESPCNLikeSubgraphParams{kESPCNTestShape, ov::element::f16}),
                         ESPCNLikeSubgraphNoUnroll::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ESPCNLike_WAxisUnroll, ESPCNLikeSubgraphWUnroll,
                         ::testing::Values(ESPCNLikeSubgraphParams{kESPCNTestShape, ov::element::f16}),
                         ESPCNLikeSubgraphWUnroll::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ESPCNLike_HWAxesUnroll, ESPCNLikeSubgraphHWUnroll,
                         ::testing::Values(ESPCNLikeSubgraphParams{kESPCNTestShape, ov::element::f16}),
                         ESPCNLikeSubgraphHWUnroll::getTestCaseName);

}  // namespace
