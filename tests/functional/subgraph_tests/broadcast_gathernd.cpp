//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset3_decl.hpp>
#include <openvino/opsets/opset8_decl.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/gather_nd.hpp"

namespace ov::test {

struct BroadcastGatherNDTestParams {
    ov::Shape inputShape;
    ov::Shape broadcastShape;
    std::vector<std::vector<int32_t>> indices;
    ov::Shape indicesShape;  // shape of the indices constant tensor
};

class BroadcastGatherNDSubgraphTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<BroadcastGatherNDTestParams> {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const auto& p = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({p.inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        const auto targetShape = std::make_shared<ov::opset3::Constant>(
                ov::element::i64, ov::Shape{p.broadcastShape.size()},
                std::vector<int64_t>(p.broadcastShape.begin(), p.broadcastShape.end()));
        const auto broadcast = std::make_shared<ov::opset3::Broadcast>(input, targetShape);

        // Flatten indices for the constant constructor.
        std::vector<int32_t> flatIndices;
        for (const auto& idx : p.indices) {
            flatIndices.insert(flatIndices.end(), idx.begin(), idx.end());
        }
        const auto indicesConst = std::make_shared<ov::opset3::Constant>(ov::element::i32, p.indicesShape, flatIndices);

        const auto gatherND = std::make_shared<ov::opset8::GatherND>(broadcast, indicesConst, 0);

        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(gatherND)},
                                               ov::ParameterVector{input}, "BroadcastGatherND");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastGatherNDTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_";
        result << "Input=" << ov::Shape(p.inputShape) << "_";
        result << "Broadcast=" << ov::Shape(p.broadcastShape) << "_";
        result << "IndicesShape=" << ov::Shape(p.indicesShape);
        return result.str();
    }
};

TEST_P(BroadcastGatherNDSubgraphTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastGatherNDSubgraphTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(BroadcastGatherNDSubgraphTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(BroadcastGatherNDSubgraphTest, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

// Basic case: 1x1x16x1x8 broadcast to 1x4x16x1x8, indices shape [4,2].
// Matches the original motivating pattern from the ticket.
const std::vector<BroadcastGatherNDTestParams> testCases = {
        // Basic case: 1x1x16x1x8 broadcast to 1x4x16x1x8, indices shape [4,2].
        {
                /*inputShape=*/{1, 1, 16, 1, 8},
                /*broadcastShape=*/{1, 4, 16, 1, 8},
                /*indices=*/{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                /*indicesShape=*/{4, 2},
        },
        // Broadcasting axis outside lastDim: 1x1x1x1x8 -> 1x4x16x1x8, indices shape [4,2].
        {
                /*inputShape=*/{1, 1, 1, 1, 8},
                /*broadcastShape=*/{1, 4, 16, 1, 8},
                /*indices=*/{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                /*indicesShape=*/{4, 2},
        },
        // Partial slices: numSlices=3 < broadcast N=4. Only 3 of 4 identical slices selected.
        {
                /*inputShape=*/{1, 1, 16, 1, 8},
                /*broadcastShape=*/{1, 4, 16, 1, 8},
                /*indices=*/{{0, 0}, {0, 1}, {0, 2}},
                /*indicesShape=*/{3, 2},
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_BroadcastGatherND, BroadcastGatherNDSubgraphTest, ::testing::ValuesIn(testCases),
                         BroadcastGatherNDSubgraphTest::getTestCaseName);

}  // namespace ov::test
