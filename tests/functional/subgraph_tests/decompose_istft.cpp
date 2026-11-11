//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/istft.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

struct ISTFTParams {
    ov::Shape signalShape;
    ov::Shape windowShape;
    int64_t frameSize;
    int64_t frameStep;
    std::optional<int64_t> signalLength;
    bool center;
    bool normalized;
};

using ISTFTCombinedParams = std::tuple<ISTFTParams, ov::element::Type>;

class DecomposeISTFTTest : public VpuOv2LayerTest, public testing::WithParamInterface<ISTFTCombinedParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ISTFTCombinedParams> obj) {
        const auto& [params, dataType] = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_";
        result << "Signal" << ov::test::utils::vec2str(params.signalShape) << "_";
        result << "Window" << ov::test::utils::vec2str(params.windowShape) << "_";
        result << "FrameSize" << params.frameSize << "_";
        result << "FrameStep" << params.frameStep << "_";
        result << "SignalLen"
               << (params.signalLength.has_value() ? std::to_string(params.signalLength.value()) : "None") << "_";
        result << "Center" << (params.center ? "True" : "False") << "_";
        result << "Normalized" << (params.normalized ? "True" : "False") << "_";
        result << "Type" << dataType.get_type_name();
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 3, 0, 1000);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        const auto& [baseParams, dataType] = GetParam();

        inType = outType = dataType;

        init_input_shapes(ov::test::static_shapes_to_test_representation({baseParams.signalShape}));

        const auto signalParam = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        signalParam->set_friendly_name("signal");

        std::vector<float> windowData(baseParams.frameSize);
        for (int64_t i = 0; i < baseParams.frameSize; ++i) {
            windowData[i] = 1.0f;
        }
        const auto windowConst = std::make_shared<ov::op::v0::Constant>(inType, baseParams.windowShape, windowData);
        windowConst->set_friendly_name("window");

        const auto frameSizeConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, baseParams.frameSize);
        frameSizeConst->set_friendly_name("frame_size");

        const auto frameStepConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, baseParams.frameStep);
        frameStepConst->set_friendly_name("frame_step");

        std::shared_ptr<ov::op::v16::ISTFT> istftOp;

        if (!baseParams.signalLength.has_value()) {
            istftOp = std::make_shared<ov::op::v16::ISTFT>(signalParam, windowConst, frameSizeConst, frameStepConst,
                                                           baseParams.center, baseParams.normalized);
        } else {
            const auto signalLengthConst = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{},
                                                                                  baseParams.signalLength.value());
            signalLengthConst->set_friendly_name("signal_length");

            istftOp = std::make_shared<ov::op::v16::ISTFT>(signalParam, windowConst, frameSizeConst, frameStepConst,
                                                           signalLengthConst, baseParams.center, baseParams.normalized);
        }

        istftOp->set_friendly_name("istft");

        const auto result = std::make_shared<ov::op::v0::Result>(istftOp);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{signalParam},
                                               "DecomposeISTFTTest");
    }
};

TEST_P(DecomposeISTFTTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DecomposeISTFTTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DecomposeISTFTTest, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

// {{signal}, {window}, frame_size, frame_step, signal_length, center, normalized}
const std::vector<ISTFTParams> baseParams = {
        {{2, 3, 2}, {2}, 2, 1, 4, false, false},
        {{2, 2, 2}, {2}, 2, 1, 3, false, false},
        {{33, 33, 2}, {64}, 64, 16, 576, true, false},
        {{2, 3, 3, 2}, {4}, 4, 2, 8, false, true},
        // signal_length < default
        {{2, 2, 4, 2}, {2}, 2, 1, 4, false, false},
        // signal_length > default
        {{3, 3, 2}, {4}, 4, 2, 9, false, false},
        // signal_length omitted
        {{3, 4, 2}, {4}, 4, 2, std::nullopt, false, true},
        {{1, 11, 12001, 2}, {20}, 20, 5, std::nullopt, true, false},
};

// VPU.NCE.Convolution operations only support FP16/BF16, not FP32
// Without automatic FP16 conversion, compilation fails with precision mismatch errors
const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(precommit_DecomposeISTFT, DecomposeISTFTTest,
                         ::testing::Combine(::testing::ValuesIn(baseParams), ::testing::ValuesIn(netPrecisions)),
                         DecomposeISTFTTest::getTestCaseName);

}  // namespace ov::test
