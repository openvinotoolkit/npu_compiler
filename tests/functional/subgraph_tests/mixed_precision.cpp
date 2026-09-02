//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/subgraph/nce_tasks.hpp"

using namespace ov::test::utils;
namespace ov::test {

ov::Output<ov::Node> quantizeOutput(const ov::Output<ov::Node>& producer, const bool needQuant, const bool isMaxPool) {
    if (!needQuant) {
        // Bypass the quantization
        return producer;
    }

    const auto quantNCEOutRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    // MaxPool output range must match its input range
    const auto quantMaxPoolOutRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
    const auto quantRange = isMaxPool ? quantMaxPoolOutRange : quantNCEOutRange;
    const auto quantOp = NCETasksHelpers::quantize(producer, quantRange);

    return quantOp->output(0);
}

enum MixedMode {
    FP16toU8,
    U8toFP16,
};

typedef std::tuple<MixedMode, NCETasksHelpers::NCEOpType, ov::element::Type, ov::Layout> MixedPrecisionParams;

class NCEMixedPrecisionTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<MixedPrecisionParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 50, 0, 1, 1);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        const auto mixedMode = std::get<0>(GetParam());
        const bool isOutputQuantized = (mixedMode == FP16toU8);
        const auto nceOpType = std::get<1>(GetParam());
        const ov::Shape inputShape = {1, 16, 32, 64};
        inType = outType = std::get<ov::element::Type>(GetParam());
        ov::Layout order = std::get<ov::Layout>(GetParam());

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};
        auto nce_task = NCETasksHelpers::buildNCETask(params.at(0), nceOpType);

        const bool isMaxPool = (NCETasksHelpers::NCEOpType::MaxPooling == nceOpType);
        const auto maybeQuantOutput = quantizeOutput(nce_task->output(0), isOutputQuantized, isMaxPool);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(maybeQuantOutput)};

        function = std::make_shared<ov::Model>(results, params, "NCEMixedPrecisionTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MixedPrecisionParams>& paramInfo) {
        const auto mixedMode = std::get<0>(paramInfo.param);
        const auto isInputQuantized = (mixedMode == U8toFP16);
        const auto isOutputQuantized = (mixedMode == FP16toU8);
        const auto nceOpType = std::get<1>(paramInfo.param);

        const std::string inPrecision = isInputQuantized ? "inPRC=U8" : "inPRC=FP16";
        const std::string outPrecision = isOutputQuantized ? "outPRC=U8" : "outPRC=FP16";
        const auto opType = NCETasksHelpers::NCEOpTypeToString(nceOpType);

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << inPrecision << sep << outPrecision << sep << opType;
        return result.str();
    }
};

TEST_P(NCEMixedPrecisionTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<MixedMode> mixedMode = {FP16toU8, U8toFP16};
const std::vector<NCETasksHelpers::NCEOpType> nceOpType = {
        NCETasksHelpers::NCEOpType::AveragePooling, NCETasksHelpers::NCEOpType::Conv2d,
        NCETasksHelpers::NCEOpType::EltwiseAdd, NCETasksHelpers::NCEOpType::GroupConv2d,
        NCETasksHelpers::NCEOpType::MaxPooling};

INSTANTIATE_TEST_SUITE_P(smoke_conv2d_with_act, NCEMixedPrecisionTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(mixedMode), ::testing::ValuesIn(nceOpType),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC"))),
                         NCEMixedPrecisionTest_NPU3720::getTestCaseName);

}  // namespace ov::test
