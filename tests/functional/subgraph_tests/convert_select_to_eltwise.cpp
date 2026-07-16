//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/select.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {

struct ConvertSelectToEltwiseParams {
    ov::Shape dataShape;       // shape of the data input to Broadcast
    ov::Shape maskShape;       // shape of the bool mask constant (input to BitwiseNot)
    ov::Shape broadcastShape;  // target shape after Broadcast
};

class ConvertSelectToEltwiseTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvertSelectToEltwiseParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvertSelectToEltwiseParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "DataShape=" << ov::test::utils::vec2str(obj.param.dataShape) << sep;
        result << "MaskShape=" << ov::test::utils::vec2str(obj.param.maskShape) << sep;
        result << "BroadcastShape=" << ov::test::utils::vec2str(obj.param.broadcastShape) << sep;
        result << "TestIdx=" << obj.index;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, -5, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        inType = outType = ov::element::f16;
        const auto testParams = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({testParams.dataShape}));

        const auto data = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // Build a bool mask constant: alternating 0/1 pattern over the mask shape.
        const auto maskSize = ov::shape_size(testParams.maskShape);
        std::vector<uint8_t> maskData(maskSize);
        for (size_t i = 0; i < maskSize; ++i) {
            maskData[i] = static_cast<uint8_t>(i % 2);
        }

        const auto maskConst = ov::op::v0::Constant::create(ov::element::boolean, testParams.maskShape, maskData);
        const auto bitwiseNot = std::make_shared<ov::op::v13::BitwiseNot>(maskConst);

        // Broadcast(data, target_shape)
        const auto targetShapeConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams.broadcastShape.size()},
                std::vector<int64_t>(testParams.broadcastShape.begin(), testParams.broadcastShape.end()));
        const auto broadcast =
                std::make_shared<ov::op::v3::Broadcast>(data, targetShapeConst, ov::op::BroadcastType::BIDIRECTIONAL);

        // Zero scalar for the true-branch of Select
        const auto zeroConst = ov::op::v0::Constant::create(inType, ov::Shape{}, {0.0f});

        // Select(BitwiseNot(mask), zero, Broadcast(data))
        const auto select = std::make_shared<ov::op::v1::Select>(bitwiseNot, zeroConst, broadcast);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "ConvertSelectToEltwiseTest");
    }
};

TEST_P(ConvertSelectToEltwiseTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvertSelectToEltwiseTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvertSelectToEltwiseTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ConvertSelectToEltwiseTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<ConvertSelectToEltwiseParams> testValues = {
        {{1, 64, 5, 1}, {1, 1, 5, 5}, {1, 64, 5, 5}},
        {{64, 4, 256, 1}, {1, 1, 256, 256}, {64, 4, 256, 256}},
};

INSTANTIATE_TEST_SUITE_P(precommit_ConvertSelectToEltwise, ConvertSelectToEltwiseTestCommon,
                         ::testing::ValuesIn(testValues), ConvertSelectToEltwiseTestCommon::getTestCaseName);

}  // namespace ov::test
