//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/depth_to_space.hpp"

// Subgraph:
//
//     [input]
//    /       \
// (D2S)     (D2S)
//    \       /  ... -> (Expand fused into D2S before DPU.Add)
//     ( Add )
//        |
//    [output]

namespace ov::test::subgraph {

using FuseD2sExpandParams = std::tuple<ov::Shape,          // Input shapes
                                       std::size_t,        // Block size
                                       ov::element::Type,  // Input precision
                                       std::size_t>;       // Num clusters

class FuseD2sExpandCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseD2sExpandParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseD2sExpandParams> obj) {
        const auto& [inShape, bs, prec, nClusters] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inShape.size() << "}" << sep;
        result << "BlockSize=" << bs << sep;
        result << "InputPrec=" << prec << sep;
        result << "NumClusters=" << nClusters << sep;
        return result.str();
    }
    void configure_model() override {
        const auto& [inShape, blockSize, prec, nClusters] = GetParam();
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-d2s-to-transposed-conv-conversion=false enable-ops-as-dma=false enable-fuse-d2s-expand=true";
        configuration[ov::intel_npu::tiles.name()] = nClusters;
    }
    void SetUp() override {
        const auto& [inShape, blockSize, prec, nClusters] = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({inShape}));

        ov::ParameterVector params;
        auto param = std::make_shared<ov::op::v0::Parameter>(prec, inShape);
        params.push_back(param);

        const auto modeA = op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
        auto d2sA = std::make_shared<ov::op::v0::DepthToSpace>(params[0], modeA, blockSize);

        const auto modeB = op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
        auto d2sB = std::make_shared<ov::op::v0::DepthToSpace>(params[0], modeB, blockSize);

        const auto add = std::make_shared<ov::opset6::Add>(d2sA->output(0), d2sB->output(0));
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(add->output(0))};

        function = std::make_shared<ov::Model>(results, params, "D2sExpand");
    }
};

TEST_P(FuseD2sExpandCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

// Important: use odd W/H dims to avoid Reshape into {1, 16, W/2, H}
//            and have the Expand C=8 to C=16 instead
ov::Shape inputShape = {1, 8, 11, 11};

const auto params = ::testing::Combine(::testing::Values(inputShape), ::testing::Values(2),
                                       ::testing::Values(ov::element::f16), ::testing::Values(1));

INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sExpand, FuseD2sExpandCommon, params, FuseD2sExpandCommon::getTestCaseName);

}  // namespace
