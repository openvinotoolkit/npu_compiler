//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/dimension.hpp>
#include <openvino/opsets/opset14_decl.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "private_properties.hpp"
#include "vpux/utils/core/env.hpp"

namespace ov::test {

PRETTY_PARAM(Batches, std::vector<BoundedDim>);  // N, ...
PRETTY_PARAM(SourceSeqLen, BoundedDim);          // S
PRETTY_PARAM(TargetSeqLen, BoundedDim);          // L
PRETTY_PARAM(QKEmbeddingSize, BoundedDim);       // E
PRETTY_PARAM(VEmbeddingSize, BoundedDim);        // Ev
PRETTY_PARAM(IsCausal, bool);
PRETTY_PARAM(HasAttentionMask, bool);
PRETTY_PARAM(HasScale, bool);

PRETTY_PARAM(InputType, ov::element::Type);

namespace {

std::vector<InputShape> generateInputShapes(const Batches& batches, SourceSeqLen sourceSeqLen,
                                            TargetSeqLen targetSeqLen, QKEmbeddingSize qkEmbeddingSize,
                                            VEmbeddingSize vEmbeddingSize, HasAttentionMask hasAttentionMask) {
    auto qDims = combine<BoundedDim>(batches, targetSeqLen, qkEmbeddingSize);
    auto qShape = generateTestShape(qDims);

    auto kDims = combine<BoundedDim>(batches, sourceSeqLen, qkEmbeddingSize);
    auto kShape = generateTestShape(kDims);

    auto vDims = combine<BoundedDim>(batches, sourceSeqLen, vEmbeddingSize);
    auto vShape = generateTestShape(vDims);

    auto inputShapes = std::vector<InputShape>{qShape, kShape, vShape};

    if (hasAttentionMask) {
        auto attensionMaskDims = combine<BoundedDim>(batches, targetSeqLen, sourceSeqLen);
        auto attentionMaskShape = generateTestShape(attensionMaskDims);
        inputShapes.push_back(attentionMaskShape);
    }

    return inputShapes;
}

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes, InputType inputType,
                                        HasAttentionMask hasAttentionMask, HasScale hasScale) {
    ov::ParameterVector inputParams;

    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[2]));

    inputParams[0]->set_friendly_name("q");
    inputParams[1]->set_friendly_name("k");
    inputParams[2]->set_friendly_name("v");

    if (hasScale) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(
                inputType, hasAttentionMask ? inputDynamicShapes[3] : ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else if (hasAttentionMask) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[3]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

}  // namespace

class SdpAttentionLayerTestCommon : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        auto shapes = std::vector<ov::Shape>{targetInputStaticShapes[0], targetInputStaticShapes[1],
                                             targetInputStaticShapes[2]};
        if (hasScale) {
            shapes.push_back(hasAttentionMask ? targetInputStaticShapes[3] : ov::Shape{});
            shapes.push_back(ov::Shape{1});  // NOLINT emplace_back leads to incorrect results
        } else if (hasAttentionMask) {
            shapes.push_back(targetInputStaticShapes[3]);
        }

        SubgraphBaseTest::generate_inputs(shapes);
    }

public:
    bool hasAttentionMask = false;
    bool hasScale = false;
};

//
// Cross Attention test class
//

using SdpAttentionLayerTestParams = std::tuple<Batches, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize,
                                               IsCausal, HasAttentionMask, HasScale, InputType>;

class SdpAttentionLayerTest :
        public testing::WithParamInterface<SdpAttentionLayerTestParams>,
        public SdpAttentionLayerTestCommon {
protected:
    void SetUp() override {
        SdpAttentionLayerTestCommon::SetUp();

        const auto& [batches, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, isCausal, hasAttentionMask,
                     hasScale, inputType] = GetParam();

        this->hasAttentionMask = hasAttentionMask;
        this->hasScale = hasScale;

        const auto inputShapes = generateInputShapes(batches, sourceSeqLen, targetSeqLen, qkEmbeddingSize,
                                                     vEmbeddingSize, hasAttentionMask);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, inputType, hasAttentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
        sdp->set_friendly_name("sdp");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset14::Result>(sdp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDP");

        // Interpreter backend doesn't implement evaluate method for OP ScaledDotProductAttention
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }

    void TearDown() override {
        SdpAttentionLayerTestCommon::TearDown();
    }
};

//
// Self Attention test class
//

PRETTY_PARAM(SequenceLength, BoundedDim);  // S == L
PRETTY_PARAM(EmbeddingSize, BoundedDim);   // E == Ev

using SelfAttentionTestParams =
        std::tuple<Batches, SequenceLength, EmbeddingSize, IsCausal, HasAttentionMask, HasScale, InputType>;

class SelfAttentionLayerTest :
        public testing::WithParamInterface<SelfAttentionTestParams>,
        public SdpAttentionLayerTestCommon {
protected:
    void SetUp() override {
        SdpAttentionLayerTestCommon::SetUp();

        const auto& [batches, sequenceLength, embeddingSize, isCausal, hasAttentionMask, hasScale, inputType] =
                GetParam();

        this->hasAttentionMask = hasAttentionMask;
        this->hasScale = hasScale;

        auto sourceSeqLen = SourceSeqLen{sequenceLength};
        auto targetSeqLen = TargetSeqLen{sequenceLength};
        auto qkEmbeddingSize = QKEmbeddingSize{embeddingSize};
        auto vEmbeddingSize = VEmbeddingSize{embeddingSize};

        const auto inputShapes = generateInputShapes(batches, sourceSeqLen, targetSeqLen, qkEmbeddingSize,
                                                     vEmbeddingSize, hasAttentionMask);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, inputType, hasAttentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
        sdp->set_friendly_name("sdp");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset14::Result>(sdp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDP");

        // Interpreter backend doesn't implement evaluate method for OP ScaledDotProductAttention
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }

    void TearDown() override {
        SdpAttentionLayerTestCommon::TearDown();
    }
};

//
// Online Self Attention reference subgraph implementation
//

class OnlineSelfAttentionDecomposedLayerTest : public SelfAttentionLayerTest {
public:
    void SetUp() override {
        SelfAttentionLayerTest::SetUp();

        // Disable OpenVINO ScaledDotProductAttentionDecomposition pass
        vpux::env::setEnvVar("NPU_DECOMPOSE_SDPA", "0");
    }

    void TearDown() override {
        SelfAttentionLayerTest::TearDown();
        vpux::env::unsetEnvVar("NPU_DECOMPOSE_SDPA");
    }

    void configure_model() override {
        // Enable passes that perform SDPA decomposition
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-online-sdpa-conversion=true";
    }
};

//
// Online Cross Attention Software kernel
//

class OnlineCrossAttentionSoftwareKernelLayerTest : public SdpAttentionLayerTest {
public:
    void SetUp() override {
        SdpAttentionLayerTest::SetUp();

        // Disable OpenVINO ScaledDotProductAttentionDecomposition pass
        vpux::env::setEnvVar("NPU_DECOMPOSE_SDPA", "0");

        // Disable tiling for now
        vpux::env::setEnvVar("NPU_Q_NUM_BLOCKS", "1");
        vpux::env::setEnvVar("NPU_KV_NUM_BLOCKS", "1");
    }

    void TearDown() override {
        SdpAttentionLayerTest::TearDown();

        vpux::env::unsetEnvVar("NPU_DECOMPOSE_SDPA");

        vpux::env::unsetEnvVar("NPU_Q_NUM_BLOCKS");
        vpux::env::unsetEnvVar("NPU_KV_NUM_BLOCKS");
    }

    void configure_model() override {
        // Enable passes that perform SDPA decomposition, disable IncrementalSDPA decomposition
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-online-sdpa-conversion=true disable-incremental-sdpa-decomposition=true";
    }
};

TEST_P(SdpAttentionLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SelfAttentionLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(OnlineSelfAttentionDecomposedLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(OnlineCrossAttentionSoftwareKernelLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SdpAttentionLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SelfAttentionLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(OnlineSelfAttentionDecomposedLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(OnlineCrossAttentionSoftwareKernelLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<InputType> inputPrecision = {ov::element::f16};

//
// SelfAttentionTest
//

INSTANTIATE_TEST_SUITE_P(smoke, SelfAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Batches{8}),                            // 12, 16
                                            ::testing::Values(SequenceLength{512}),                   // 1k, 2k, 4k, 8k
                                            ::testing::Values(EmbeddingSize{64}),                     // 128, 256
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),          //
                                            ::testing::ValuesIn(inputPrecision)                               //
                                            ),
                         PrintTestCaseName());
//
// OnlineSelfAttentionSubgraphLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, OnlineSelfAttentionDecomposedLayerTest,
                         ::testing::Combine(::testing::Values(Batches{8}),                             //
                                            ::testing::Values(SequenceLength{512}),                    //
                                            ::testing::Values(EmbeddingSize{64}),                      //
                                            ::testing::ValuesIn(std::vector<IsCausal>{false}),         //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{false}),         //
                                            ::testing::ValuesIn(inputPrecision)                        //
                                            ),
                         PrintTestCaseName());

//
// CrossAttentionTest
//

INSTANTIATE_TEST_SUITE_P(smoke, SdpAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Batches{8}),                                    //
                                            ::testing::Values(SourceSeqLen{512}),                             //
                                            ::testing::Values(TargetSeqLen{256}),                             //
                                            ::testing::Values(QKEmbeddingSize{64}),                           //
                                            ::testing::Values(VEmbeddingSize{32}),                            //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),          //
                                            ::testing::ValuesIn(inputPrecision)                               //
                                            ),
                         PrintTestCaseName());

//
// OnlineCrossAttentionSoftwareKernelLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, OnlineCrossAttentionSoftwareKernelLayerTest,
                         ::testing::Combine(::testing::Values(Batches{8}),                                    //
                                            ::testing::Values(SourceSeqLen{32}),                              //
                                            ::testing::Values(TargetSeqLen{64}),                              //
                                            ::testing::Values(QKEmbeddingSize{64}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                           //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),          //
                                            ::testing::ValuesIn(inputPrecision)                               //
                                            ),
                         PrintTestCaseName());

//
// Dynamic SelfAttentionTest
//

// [Tracking number: E#160081]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Dynamic, SelfAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Batches{16_Dyn, 12}),                           //
                                            ::testing::Values(SequenceLength{512_Dyn}),                       //
                                            ::testing::Values(EmbeddingSize{64}),                             //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),          //
                                            ::testing::ValuesIn(inputPrecision)                               //
                                            ),
                         PrintTestCaseName());

}  // namespace ov::test
