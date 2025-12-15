//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/stft.hpp"

#define pi 3.14159265358979f

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

struct STFTParams {
    ov::Shape signalShape;       // Input signal shape
    ov::Shape windowShape;       // Window shape
    int64_t frameSize;           // Size of each frame
    int64_t frameStep;           // Step between frames
    bool transposeFrames;        // Whether to transpose frames in output
    ov::element::Type dataType;  // Data type for computation
};

class DecomposeSTFTTest : public VpuOv2LayerTest, public testing::WithParamInterface<STFTParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<STFTParams> obj) {
        const auto& params = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_";
        result << "Signal" << ov::test::utils::vec2str(params.signalShape) << "_";
        result << "Window" << ov::test::utils::vec2str(params.windowShape) << "_";
        result << "FrameSize" << params.frameSize << "_";
        result << "FrameStep" << params.frameStep << "_";
        result << "Transpose" << (params.transposeFrames ? "True" : "False") << "_";
        result << "Type" << params.dataType.get_type_name();
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        // Only generate input for the signal parameter (others are constants)
        ASSERT_EQ(funcInputs.size(), 1) << "Expected exactly 1 input parameter for STFT test";

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1.0;
        in_data.range = 2.0;  // Range [-1.0, 1.0] for signal data
        in_data.resolution = 32768;

        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], in_data);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        const auto& params = GetParam();

        inType = outType = params.dataType;

        init_input_shapes(ov::test::static_shapes_to_test_representation({params.signalShape}));

        const auto signalParam = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        signalParam->set_friendly_name("signal");

        std::vector<float> windowData(params.frameSize);
        for (int64_t i = 0; i < params.frameSize; ++i) {
            windowData[i] = 0.5f * (1.0f - std::cos(2.0f * pi * i / (params.frameSize - 1)));
        }
        const auto windowConst = std::make_shared<ov::op::v0::Constant>(inType, params.windowShape, windowData);
        windowConst->set_friendly_name("window");

        const auto frameSizeConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, params.frameSize);
        frameSizeConst->set_friendly_name("frame_size");

        const auto frameStepConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, params.frameStep);
        frameStepConst->set_friendly_name("frame_step");

        const auto stftOp = std::make_shared<ov::op::v15::STFT>(signalParam, windowConst, frameSizeConst,
                                                                frameStepConst, params.transposeFrames);
        stftOp->set_friendly_name("stft");

        const auto result = std::make_shared<ov::op::v0::Result>(stftOp);
        result->set_friendly_name("result");

        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{signalParam},
                                               "DecomposeSTFTTest");
    }
};

TEST_P(DecomposeSTFTTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DecomposeSTFTTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<STFTParams> stftTestParams = {{
                                                        {1024},           // signal: [1024] samples
                                                        {512},            // window: [512]
                                                        512,              // frame_size: 512
                                                        256,              // frame_step: 256 (50% overlap)
                                                        false,            // transpose_frames: false
                                                        ov::element::f32  // data type
                                                },
                                                {
                                                        {2048},           // signal: [2048] samples
                                                        {512},            // window: [512]
                                                        512,              // frame_size: 512
                                                        128,              // frame_step: 128 (75% overlap)
                                                        true,             // transpose_frames: true
                                                        ov::element::f32  // data type
                                                },
                                                {
                                                        {2, 1024},        // signal: [batch=2, length=1024]
                                                        {512},            // window: [512]
                                                        512,              // frame_size: 512
                                                        256,              // frame_step: 256
                                                        false,            // transpose_frames: false
                                                        ov::element::f32  // data type
                                                },
                                                {
                                                        {4, 2048},        // signal: [channels=4, length=2048]
                                                        {512},            // window: [512]
                                                        512,              // frame_size: 512
                                                        256,              // frame_step: 256
                                                        true,             // transpose_frames: true
                                                        ov::element::f32  // data type
                                                },
                                                {
                                                        {512},            // signal: [512] samples
                                                        {128},            // window: [128]
                                                        128,              // frame_size: 128
                                                        64,               // frame_step: 64
                                                        false,            // transpose_frames: false
                                                        ov::element::f32  // data type
                                                },
                                                {
                                                        {1536},           // signal: [1536] samples
                                                        {256},            // window: [256]
                                                        256,              // frame_size: 256
                                                        128,              // frame_step: 128
                                                        false,            // transpose_frames: false
                                                        ov::element::f32  // data type
                                                }};

INSTANTIATE_TEST_SUITE_P(precommit_DecomposeSTFT, DecomposeSTFTTest, ::testing::ValuesIn(stftTestParams),
                         DecomposeSTFTTest::getTestCaseName);

}  // namespace ov::test
