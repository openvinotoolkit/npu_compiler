//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"

namespace ov::test {

// Functional accuracy test for Gather with repeat_interleave indices pattern
// (indices[i] = i / repeatFactor). Validates end-to-end numerical correctness
// on NPU hardware where the compiler may apply various optimizations.
class GatherRepeatInterleaveTest : public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;

        const ov::Shape inputShape{1, 1, 16, 8};
        const int64_t axis = 3;
        const int64_t repeatFactor = 2;
        const int64_t dimSize = static_cast<int64_t>(inputShape[axis]);
        const int64_t numIndices = dimSize * repeatFactor;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // Build repeat_interleave indices: [0,0,1,1,2,2,...,7,7]
        std::vector<int64_t> indicesData(numIndices);
        for (int64_t i = 0; i < numIndices; ++i) {
            indicesData[i] = i / repeatFactor;
        }
        const auto indices =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(numIndices)}, indicesData);
        const auto axisConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});

        const auto gather = std::make_shared<ov::op::v8::Gather>(input, indices, axisConst);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "GatherRepeatInterleave");
    }
};

TEST_F(GatherRepeatInterleaveTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(GatherRepeatInterleaveTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(GatherRepeatInterleaveTest, NPU5010_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
