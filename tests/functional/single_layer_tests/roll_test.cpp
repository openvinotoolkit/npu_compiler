//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/roll.hpp"

namespace ov::test {

class SEPRollLayerTestTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, std::vector<int64_t>, std::vector<int64_t>,
                                                      ov::element::Type, ov::element::Type, ov::Layout, ov::Layout>> {
    void SetUp() override {
        ov::Shape inputShape;
        std::vector<int64_t> shiftContent, aexsContent;
        ov::Layout inLayout, outLayout;
        std::tie(inputShape, shiftContent, aexsContent, inType, outType, inLayout, outLayout) = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};
        const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                 ov::Shape{aexsContent.size()}, aexsContent.data());
        const auto shift = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                  ov::Shape{shiftContent.size()}, shiftContent.data());

        auto param = params.at(0);
        const auto rollOp = std::make_shared<ov::op::v7::Roll>(param, shift, axes);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rollOp)};
        function = std::make_shared<ov::Model>(results, params, "SEPRoll");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(inLayout);
        preProc.input().model().set_layout(inLayout);
        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout(outLayout);
        function = preProc.build();
        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<ov::Shape, std::vector<int64_t>, std::vector<int64_t>,
                                                    ov::element::Type, ov::element::Type, ov::Layout, ov::Layout>>&
                    obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class SEPRollLayerTest_NPU3720 : public SEPRollLayerTestTestCommon {};
class SEPRollLayerTest_NPU4000 : public SEPRollLayerTestTestCommon {};
class SEPRollLayerTest_NPU5010 : public SEPRollLayerTestTestCommon {};

TEST_P(SEPRollLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";
    run(Platform::NPU3720);
}

TEST_P(SEPRollLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";
    run(Platform::NPU4000);
}
TEST_P(SEPRollLayerTest_NPU5010, HW) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";
    run(Platform::NPU5010);
}

const std::vector<ov::Shape> inputShapesHW = {{1, 16, 5, 5}, {1, 128, 96, 96}};
const std::vector<std::vector<int64_t>> shiftsHW = {{-2, 1}, {1, -4}};
const std::vector<std::vector<int64_t>> axesHW = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightAndWidth, SEPRollLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHW), ::testing::ValuesIn(shiftsHW),
                                            ::testing::ValuesIn(axesHW), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightAndWidth, SEPRollLayerTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHW), ::testing::ValuesIn(shiftsHW),
                                            ::testing::ValuesIn(axesHW), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightAndWidth, SEPRollLayerTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHW), ::testing::ValuesIn(shiftsHW),
                                            ::testing::ValuesIn(axesHW), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU5010::getTestCaseName);

const std::vector<ov::Shape> inputShapesHeightOrWidth = {{1, 16, 5, 5}, {1, 128, 96, 96}};
const std::vector<std::vector<int64_t>> shiftsHeightOrWidth = {{4}, {-2}};
const std::vector<std::vector<int64_t>> axesHeightOrWidth = {{2}, {3}};

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightOrWidth, SEPRollLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHeightOrWidth),
                                            ::testing::ValuesIn(shiftsHeightOrWidth),
                                            ::testing::ValuesIn(axesHeightOrWidth), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightOrWidth, SEPRollLayerTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHeightOrWidth),
                                            ::testing::ValuesIn(shiftsHeightOrWidth),
                                            ::testing::ValuesIn(axesHeightOrWidth), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_HeightOrWidth, SEPRollLayerTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapesHeightOrWidth),
                                            ::testing::ValuesIn(shiftsHeightOrWidth),
                                            ::testing::ValuesIn(axesHeightOrWidth), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU5010::getTestCaseName);

const std::vector<ov::Shape> inputShapesChannelAndHeight = {{1, 5, 7, 32}, {1, 96, 96, 128}};
const std::vector<std::vector<int64_t>> shiftsChannelAndHeight = {{-1, 3}, {1, -3}};
const std::vector<std::vector<int64_t>> axesChannelAndHeight = {{1, 2}};

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_ChannelAndHeight, SEPRollLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannelAndHeight),
                                            ::testing::ValuesIn(shiftsChannelAndHeight),
                                            ::testing::ValuesIn(axesChannelAndHeight),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_ChannelAndHeight, SEPRollLayerTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannelAndHeight),
                                            ::testing::ValuesIn(shiftsChannelAndHeight),
                                            ::testing::ValuesIn(axesChannelAndHeight),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_ChannelAndHeight, SEPRollLayerTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannelAndHeight),
                                            ::testing::ValuesIn(shiftsChannelAndHeight),
                                            ::testing::ValuesIn(axesChannelAndHeight),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU5010::getTestCaseName);

const std::vector<ov::Shape> inputShapesChannel = {{1, 5, 7, 10}, {1, 96, 96, 128}};
const std::vector<std::vector<int64_t>> shiftsChannel = {{-1}};
const std::vector<std::vector<int64_t>> axesChannel = {{1}};

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_Channel, SEPRollLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannel), ::testing::ValuesIn(shiftsChannel),
                                            ::testing::ValuesIn(axesChannel), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_Channel, SEPRollLayerTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannel), ::testing::ValuesIn(shiftsChannel),
                                            ::testing::ValuesIn(axesChannel), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_SEPRollTest_Channel, SEPRollLayerTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapesChannel), ::testing::ValuesIn(shiftsChannel),
                                            ::testing::ValuesIn(axesChannel), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::Values(ov::Layout("NCHW"))),
                         SEPRollLayerTest_NPU5010::getTestCaseName);
}  // namespace ov::test
