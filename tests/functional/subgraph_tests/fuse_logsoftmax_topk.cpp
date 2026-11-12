//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/topk.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

struct LogSoftmaxTopKParams {
    ov::Shape inputShape;
    int64_t axis;
    int64_t k;
};

class FuseLogSoftmaxTopKTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<LogSoftmaxTopKParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LogSoftmaxTopKParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 5;
        in_data.resolution = 256;
        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], in_data);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        inType = ov::element::f32;
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;
        const auto axis = testParams.axis;
        const auto k = testParams.k;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        const auto logSoftmax = std::make_shared<ov::op::v5::LogSoftmax>(input, axis);

        const auto fqInLow = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {-39.9375f});
        const auto fqInHigh = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {0.231641f});
        const auto fqOutLow = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {-39.9375f});
        const auto fqOutHigh = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {0.231641f});
        const auto fakeQuantize =
                std::make_shared<ov::op::v0::FakeQuantize>(logSoftmax, fqInLow, fqInHigh, fqOutLow, fqOutHigh, 65536);

        const auto kConst = ov::op::v0::Constant::create(ov::element::i64, {}, {k});
        const auto topK = std::make_shared<ov::op::v11::TopK>(input, kConst, axis, ov::op::v11::TopK::Mode::MAX,
                                                              ov::op::v11::TopK::SortType::NONE, ov::element::i32);

        const auto convert = std::make_shared<ov::op::v0::Convert>(topK->output(1), ov::element::i64);

        ov::Shape outputShape = {inputShape[0]};
        const auto reshapeConst = ov::op::v0::Constant::create(ov::element::i64, {outputShape.size()}, outputShape);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(convert, reshapeConst, false);

        const auto resultIndices = std::make_shared<ov::op::v0::Result>(reshape);
        const auto resultLogSoftmax = std::make_shared<ov::op::v0::Result>(fakeQuantize);

        const ov::ResultVector results{resultIndices, resultLogSoftmax};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "FuseLogSoftmaxTopKTest");
    }
};

TEST_P(FuseLogSoftmaxTopKTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseLogSoftmaxTopKTestCommon, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

// Only inner size axis and K = 1 is supported
std::vector<LogSoftmaxTopKParams> logSoftmaxTopKTestParams = {
        {{151, 7049}, 1, 1},
        {{130, 7040}, 1, 1},
        {{142, 7070}, 1, 1},
};

INSTANTIATE_TEST_SUITE_P(precommit_FuseLogSoftmaxTopK, FuseLogSoftmaxTopKTestCommon,
                         ::testing::ValuesIn(logSoftmaxTopKTestParams), FuseLogSoftmaxTopKTestCommon::getTestCaseName);

}  // namespace ov::test
