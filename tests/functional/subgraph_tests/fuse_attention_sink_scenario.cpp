//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset14.hpp>
#include <openvino/opsets/opset3.hpp>
#include <pretty_test_arguments.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

// Attention-with-sink scenario covering both self-attention and Grouped-Query-Attention (GQA).
// FuseAttention distributes the query heads across N (kv-group) and C (head-in-group), so a
// per-head sink packs its logits along the C axis (numQHeads entries). For GQA (numQHeads >
// numKVHeads) the reference applies repeat_kv to broadcast K/V from numKVHeads to numQHeads.
// Tiling over N then requires each kv-group tile to read its own sink slice; a distinct-per-head
// sink diverges from the decompose reference if any tile reads the wrong slice.
struct AttentionSinkPatternParams {
    size_t numQHeads;
    size_t numKVHeads;
    size_t seqLen;
    size_t headDim;
    bool broadcastSink;
};

class FuseAttentionSinkPatternTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AttentionSinkPatternParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        const size_t sinkIndex = funcInputs.size() - 1;
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = (i == sinkIndex) ? 4 : 1;
            in_data.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    void SetUp() override {
        auto elementType = ov::element::f16;
        inType = outType = elementType;

        const auto testParams = GetParam();
        const auto numQHeads = static_cast<int64_t>(testParams.numQHeads);
        const auto numKVHeads = static_cast<int64_t>(testParams.numKVHeads);
        const auto seqLen = static_cast<int64_t>(testParams.seqLen);
        const auto headDim = static_cast<int64_t>(testParams.headDim);
        const auto groupSize = numQHeads / numKVHeads;

        const ov::Shape qShape{1, testParams.numQHeads, testParams.seqLen, testParams.headDim};
        const ov::Shape kvShape{1, testParams.numKVHeads, testParams.seqLen, testParams.headDim};
        const ov::Shape maskShape{1, 1, testParams.seqLen, testParams.seqLen};
        const ov::Shape sinkShape = testParams.broadcastSink ? ov::Shape{1, testParams.numQHeads, 1, 1}
                                                             : ov::Shape{1, testParams.numQHeads, testParams.seqLen, 1};

        init_input_shapes(
                ov::test::static_shapes_to_test_representation({qShape, kvShape, kvShape, maskShape, sinkShape}));

        const auto inputQ = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputK = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputV = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(2));
        const auto inputMask = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(3));
        const auto inputSink = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(4));

        // repeat_kv broadcasts K/V from numKVHeads to numQHeads for GQA; self-attention keeps K/V as-is:
        // [1, numKVHeads, S, D] -> [1, numKVHeads, 1, S, D] -> broadcast [1, numKVHeads, groupSize, S, D]
        // -> [1, numQHeads, S, D].
        const auto repeatKV = [&](const std::shared_ptr<ov::Node>& kv) -> std::shared_ptr<ov::Node> {
            const auto expandShape = ov::opset3::Constant::create(
                    ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, numKVHeads, 1, seqLen, headDim});
            const auto expanded = std::make_shared<ov::opset3::Reshape>(kv, expandShape, false);
            const auto broadcastShape = ov::opset3::Constant::create(
                    ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, numKVHeads, groupSize, seqLen, headDim});
            const auto broadcasted = std::make_shared<ov::opset14::Broadcast>(expanded, broadcastShape,
                                                                              ov::op::BroadcastType::BIDIRECTIONAL);
            const auto mergeShape = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4},
                                                                 std::vector<int64_t>{1, numQHeads, seqLen, headDim});
            return std::make_shared<ov::opset3::Reshape>(broadcasted, mergeShape, false);
        };

        const std::shared_ptr<ov::Node> keyFull = numQHeads > numKVHeads ? repeatKV(inputK) : inputK;
        const std::shared_ptr<ov::Node> valueFull = numQHeads > numKVHeads ? repeatKV(inputV) : inputV;

        const auto scores = std::make_shared<ov::opset14::MatMul>(inputQ, keyFull, false, true);
        const auto maskedScores =
                std::make_shared<ov::opset14::Add>(scores, inputMask, ov::op::AutoBroadcastType::NUMPY);

        // A per-head sink packs one logit per query head along C; broadcast it across S before concat.
        std::shared_ptr<ov::op::v0::Concat> scoresWithSink;
        if (testParams.broadcastSink) {
            const ov::Shape sinkBroadcastShape{1, testParams.numQHeads, testParams.seqLen, 1};
            const auto sinkTargetShape =
                    std::make_shared<ov::opset3::Constant>(ov::element::i32, ov::Shape{4}, sinkBroadcastShape);
            const auto broadcastedSink = std::make_shared<ov::opset14::Broadcast>(inputSink, sinkTargetShape);
            broadcastedSink->set_friendly_name("sdp_sink_pattern");
            scoresWithSink = std::make_shared<ov::opset14::Concat>(ov::OutputVector{maskedScores, broadcastedSink}, 3);
        } else {
            scoresWithSink = std::make_shared<ov::opset14::Concat>(ov::OutputVector{maskedScores, inputSink}, 3);
        }

        const auto normalizedScores = std::make_shared<ov::opset14::Softmax>(scoresWithSink, 3);
        const auto beginConst =
                ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, 0, 0});
        const auto endConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4},
                                                           std::vector<int64_t>{1, numQHeads, seqLen, seqLen});
        const auto stridesConst =
                ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
        const auto slicedScores = std::make_shared<ov::opset3::StridedSlice>(
                normalizedScores, beginConst, endConst, stridesConst, std::vector<int64_t>{0, 0, 0, 0},
                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0},
                std::vector<int64_t>{0, 0, 0, 0});

        const auto output = std::make_shared<ov::opset14::MatMul>(slicedScores, valueFull, false, false);
        output->set_friendly_name("sdp_sink_pattern");

        ov::ParameterVector inputParams{inputQ, inputK, inputV, inputMask, inputSink};
        auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(output)};

        function = std::make_shared<ov::Model>(results, inputParams, "SDPSinkPattern");
        functionRefs = function->clone();
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<AttentionSinkPatternParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& p = obj.param;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "QHeads=" << p.numQHeads << sep;
        result << "KVHeads=" << p.numKVHeads << sep;
        result << "SeqLen=" << p.seqLen << sep;
        result << "HeadDim=" << p.headDim << sep;
        result << "BroadcastSink=" << p.broadcastSink;
        return result.str();
    };
};

TEST_P(FuseAttentionSinkPatternTestCommon, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_AttentionWithSink_model_scenario, FuseAttentionSinkPatternTestCommon,
                         ::testing::ValuesIn({
                                 AttentionSinkPatternParams{64, 64, 1024, 64, /*broadcastSink=*/false},
                                 AttentionSinkPatternParams{64, 64, 1024, 64, /*broadcastSink=*/true},
                         }),
                         FuseAttentionSinkPatternTestCommon::getTestCaseName);

// Grouped-Query-Attention scenario (numQHeads > numKVHeads) reuses the unified test class; the
// KV broadcasting is applied through repeat_kv in SetUp.
INSTANTIATE_TEST_SUITE_P(smoke_AttentionGQAWithSink_kvgroup_tiling, FuseAttentionSinkPatternTestCommon,
                         ::testing::ValuesIn({
                                 AttentionSinkPatternParams{32, 8, 1024, 64, /*broadcastSink=*/true},
                         }),
                         FuseAttentionSinkPatternTestCommon::getTestCaseName);
}  // namespace ov::test::subgraph
