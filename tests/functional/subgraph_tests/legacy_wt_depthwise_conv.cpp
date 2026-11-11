//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6_decl.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace LegacyWTDepthwiseConvDefinition {

struct FQAsSubMul {
    std::vector<int8_t> zp_per_channel;
    std::vector<float> scale_per_channel;
};

using LegacyWTDepthwiseConvTestParams = std::tuple<FQAsSubMul, ov::element::Type, std::string>;

namespace {
FQAsSubMul createPerChannelQuant(const std::vector<int8_t>& zp_values, const std::vector<float>& scale_values) {
    FQAsSubMul result;
    result.scale_per_channel = scale_values;
    result.zp_per_channel = zp_values;

    return result;
}
}  // namespace

class LegacyWTDepthwiseConvTest :
        public testing::WithParamInterface<LegacyWTDepthwiseConvTestParams>,
        public VpuOv2LayerTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto& test_case = std::get<0>(params);
        const auto& float_element_type = std::get<1>(params);
        targetDevice = std::get<2>(params);

        const ov::Shape weightsShape{16, 1, 1, 2, 2};
        const size_t totalWeights =
                weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3] * weightsShape[4];
        std::vector<int8_t> weights(totalWeights, 10);

        auto i_weights = std::make_shared<ov::op::v0::Constant>(ov::element::i8, weightsShape, weights);
        auto f_weights = std::make_shared<ov::opset6::Convert>(i_weights, float_element_type);

        const size_t num_channels = 16;

        // Convert i8 zero points to f16, create per-channel zero point tensor and subtract it from weights
        auto i_zp = std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{num_channels, 1, 1, 1, 1},
                                                           test_case.zp_per_channel);
        auto f_zp = std::make_shared<ov::opset6::Convert>(i_zp, float_element_type);
        std::shared_ptr<ov::opset6::Subtract> zero_points_subtracted =
                std::make_shared<ov::opset6::Subtract>(f_weights, f_zp);

        // Create per-channel scale tensor and multiply zero_points_subtracted by it
        auto scale = std::make_shared<ov::op::v0::Constant>(float_element_type, ov::Shape{num_channels, 1, 1, 1, 1},
                                                            test_case.scale_per_channel);
        std::shared_ptr<ov::opset6::Multiply> multiplied =
                std::make_shared<ov::opset6::Multiply>(zero_points_subtracted, scale);

        const ov::Shape inputShape{1, 16, 4, 4};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector conv_params{std::make_shared<ov::op::v0::Parameter>(float_element_type, inputShape)};

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};

        const auto conv = std::make_shared<ov::opset6::GroupConvolution>(conv_params[0], multiplied->output(0), strides,
                                                                         pads_begin, pads_end, dilations);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, conv_params, "LegacyWT_DepthwiseConv");
    }

    static std::string getTestCaseName(testing::TestParamInfo<LegacyWTDepthwiseConvTestParams> obj) {
        auto params = obj.param;
        FQAsSubMul test_case = std::get<0>(params);
        ov::element::Type precision = std::get<1>(params);
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "ZP=" << static_cast<float>(test_case.zp_per_channel[0]) << sep;
        result << "scale=" << test_case.scale_per_channel[0] << sep;
        result << "precision=" << precision.get_type_name() << sep;
        return result.str();
    }
};

//
// Platform test definition
//

TEST_P(LegacyWTDepthwiseConvTest, NPU6010_HW) {
    rel_threshold = 0.1;
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-asymmetric-per-channel-zp=true";
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

// clang-format off
// Per-channel case -> zero points will be put on 4 MS bits of DATA_PTR. ZP must be in range 0 - 15 (uint4). Otherwise dequantization will happen in SplitFakeQuantPass pass.
const auto basicCasesM = ::testing::Combine(
        ::testing::ValuesIn(
            std::vector<FQAsSubMul>{
                // Per-channel case, zero points are in uint4 range, so they will be stored in DATA_PTR on 4 MS bits
                createPerChannelQuant(
                    {1, 3, 2, 3, 1, 1, 3, 2, 1, 1, 2, 2, 1, 3, 3, 3},  // 16 ZP values positive
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f,1.0f, 1.0f}), // 16 scale values = 1
                // Per-tensor case -> mpe_wtbias register will contain zero point
                createPerChannelQuant(
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 16 ZP values = 0
                    {3.0f, 2.1f, 1.9f, 2.2f, 1.8f, 2.3f, 1.7f, 2.4f, 1.6f, 2.5f, 1.5f, 2.6f, 1.4f, 2.7f, 1.3f, 2.8f}), // 16 scale values positive
                createPerChannelQuant(
                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // 16 ZP values = 1
                    {3.0f, 2.1f, 1.9f, 2.2f, 1.8f, 2.3f, 1.7f, 2.4f, 1.6f, 2.5f, 1.5f, 2.6f, 1.4f, 2.7f, 1.3f, 2.8f}), // 16 scale values positive
                createPerChannelQuant(
                    {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20},  // 16 ZP values = 20
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f,1.0f, 1.0f}), // 16 scale values = 1
                // Per-channel case and zero points are out of uint4 range. Dequantization will happen in SplitFakeQuantPass pass
                createPerChannelQuant(
                    {1, -2, 3, 0, -1, 2, -3, 1, 0, -1, 2, -2, 3, -1, 1, 0},  // 16 ZP values mixed
                    {2.0f, 5.0f, 10.0f, 12.0f, 4.0f, 1.0f, 6.0f, 3.0f,2.0f, 5.0f, 7.0f, 20.0f, 6.0f,10.0f,9.0f, 1.0f}), // 16 scale values positive
                createPerChannelQuant(
                    {22, 19, 18, 17, 1, 2, 3, 20, 20, 20, 20, 20, 5, 20, 6, 20}, // 16 ZP values positive
                    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f, 1.0f, 1.0f, 1.0f, 1.0f,1.0f,1.0f, 1.0f}) // 16 scale values positive

            }),
        ::testing::ValuesIn(std::vector<ov::element::Type>{ov::element::f16, ov::element::f32}),
        ::testing::Values(test_utils::TARGET_DEVICE));
// clang-format on

INSTANTIATE_TEST_SUITE_P(precommit_LegacyWTDepthwiseConv, LegacyWTDepthwiseConvTest, basicCasesM,
                         LegacyWTDepthwiseConvTest::getTestCaseName);
}  // namespace LegacyWTDepthwiseConvDefinition
