//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1.hpp>
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"

#include "common/quantization_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/depth_to_space.hpp"

// Subgraph:
//
//  [input]
//     |
//   {FQ}
//     |
//   (D2S)
//   /   \  ... -> (Expand fused into D2S before DPU.Add)
//  ( Add )
//     |
// [output]

namespace ov::test::subgraph {

using FuseD2sExpandParams = std::tuple<ov::Shape,          // Input shapes
                                       std::size_t,        // Block size
                                       ov::element::Type,  // Input precision
                                       bool>;              // Enable FQ

class FuseD2sExpandCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseD2sExpandParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseD2sExpandParams> obj) {
        const auto& [inShape, bs, prec, enableFQ] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inShape.size() << "}" << sep;
        result << "BlockSize=" << bs << sep;
        result << "InputPrec=" << prec << sep;
        result << "EnableFQ=" << enableFQ << sep;
        return result.str();
    }
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-d2s-to-transposed-conv-conversion=false enable-ops-as-dma=false enable-fuse-d2s-expand=true";
        configuration[ov::intel_npu::tiles.name()] = 1;  // E#184822
    }
    void SetUp() override {
        const auto& [inShape, blockSize, prec, enableFQ] = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({inShape}));

        ov::ParameterVector params;
        auto param = std::make_shared<ov::op::v0::Parameter>(prec, inShape);
        params.push_back(param);

        ov::Output<ov::Node> d2sInput = param->output(0);
        if (enableFQ) {
            d2sInput = utils::makeFakeQuantize(param, ov::element::f16, 256,
                                               FakeQuantizeParams({-1.0f}, {1.0f}, {-1.0f}, {1.0f}))
                               ->get_default_output();
        }
        const auto mode = op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
        auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(d2sInput, mode, blockSize);

        const auto add = std::make_shared<ov::opset6::Add>(d2s->output(0), d2s->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(add->output(0))};

        function = std::make_shared<ov::Model>(results, params, "D2sExpand");
    }
};

TEST_P(FuseD2sExpandCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseD2sExpandCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

// Important: use odd W/H dims to avoid 'inputShape1' reshaped into {1, 16, H/2, W}
//            and have the Expand C=8 to C=16 instead
ov::Shape inputShape1 = {1, 8, 11, 11};

ov::Shape inputShape2 = {1, 12, 3, 123};

const auto params1 = ::testing::Combine(::testing::Values(inputShape1), ::testing::Values(2),
                                        ::testing::Values(ov::element::f16), ::testing::Values(false));

const auto params2 = ::testing::Combine(::testing::Values(inputShape2), ::testing::Values(2),
                                        ::testing::Values(ov::element::f16), ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_f16, FuseD2sExpandCommon, params1, FuseD2sExpandCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand_u8q, FuseD2sExpandCommon, params2, FuseD2sExpandCommon::getTestCaseName);

}  // namespace
