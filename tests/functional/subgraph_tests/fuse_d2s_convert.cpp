//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6_decl.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using FuseD2sConvertParams = std::tuple<ov::Shape,                                   // Input shape
                                        std::size_t,                                 // D2S block-size
                                        ov::op::v0::DepthToSpace::DepthToSpaceMode,  // D2S mode
                                        ov::element::Type,                           // Input precision
                                        ov::element::Type>;                          // Output precision

class FuseD2sConvertCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseD2sConvertParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseD2sConvertParams> obj) {
        const auto& [inShape, blockSize, mode, iPrec, oPrec] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inShape.size() << "}" << sep;
        result << "BlockSize=" << blockSize << sep;
        const auto strMode =
                (mode == ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) ? "BLOCKS_FIRST" : "DEPTH_FIRST";
        result << "Mode=" << strMode << sep;
        result << "InputPrec=" << iPrec << sep;
        result << "OutputPrec=" << oPrec << sep;
        return result.str();
    }

    void SetUp() override {
        const auto& [inShape, blockSize, mode, iPrec, oPrec] = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({inShape}));

        ov::ParameterVector params;
        auto param = std::make_shared<ov::op::v0::Parameter>(iPrec, inShape);
        params.push_back(param);

        auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(param->output(0), mode, blockSize);

        std::vector<int64_t> dimsOrder = {0, 2, 3, 1};  // NHWC
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(d2s->output(0), order);

        auto cvt = std::make_shared<ov::opset6::Convert>(transpose->output(0), oPrec);
        const auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(cvt->output(0))};

        function = std::make_shared<ov::Model>(results, params, "D2sConvert");
    }
};

TEST_P(FuseD2sConvertCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseD2sConvertCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseD2sConvertCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::Shape inputShape1 = {1, 16, 20, 40};
const auto params1 = ::testing::Combine(::testing::Values(inputShape1), ::testing::Values(2),
                                        ::testing::Values(ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST),
                                        ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f32));

INSTANTIATE_TEST_SUITE_P(smoke_FuseD2sCvt_C16_B1st_BS2_f16_f32, FuseD2sConvertCommon, params1,
                         FuseD2sConvertCommon::getTestCaseName);

}  // namespace
