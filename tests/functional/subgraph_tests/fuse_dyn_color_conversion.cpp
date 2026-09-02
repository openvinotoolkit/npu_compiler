//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/dyn_color_conversion_pattern.hpp"
#include "common/print_test_case_name.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov::test;

namespace ov::test::subgraph {

namespace {

PRETTY_PARAM(Height, BoundedDim);
PRETTY_PARAM(Width, BoundedDim);

using FuseDynColorConversionParams = std::tuple<Height, Width>;

class FuseDynColorConversionTest :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<FuseDynColorConversionParams> {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        for (size_t i = 0; i < targetInputStaticShapes.size(); i++) {
            const auto& inputStaticShape = targetInputStaticShapes[i];
            const auto& funcInput = funcInputs[i];
            auto inputTensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), inputStaticShape);
            VpuOv2LayerTest::inputs.insert({funcInput.get_node_shared_ptr(), inputTensor});
        }
    }

    void SetUp() override {
        const auto& [height, width] = GetParam();

        const auto heightDim = height.value();
        const auto widthDim = width.value();

        // UV plane is half resolution
        const auto halfHeight =
                (heightDim.dim == -1) ? BoundedDim{-1, heightDim.bound / 2} : BoundedDim(heightDim.bound / 2);
        const auto halfWidth =
                (widthDim.dim == -1) ? BoundedDim{-1, widthDim.bound / 2} : BoundedDim(widthDim.bound / 2);

        auto fullBoundOnly = [](const BoundedDim& bd) -> std::vector<int> {
            return {bd.bound};
        };

        // Inputs are physically NHWC: Y=[1,H,W,1], UV=[1,H/2,W/2,2]
        auto yShape = generateTestShape({BoundedDim(1), heightDim, widthDim, BoundedDim(1)}, fullBoundOnly);
        auto uvShape = generateTestShape({BoundedDim(1), halfHeight, halfWidth, BoundedDim(2)}, fullBoundOnly);

        init_input_shapes({yShape, uvShape});

        const auto yInput = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        yInput->set_friendly_name("Y");

        const auto uvInput = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.at(1));
        uvInput->set_friendly_name("UV");

        auto result = buildDynYuvToRgbPattern(yInput, uvInput);

        // Output is NHWC: [1,H,W,3] in f16
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(result)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{yInput, uvInput},
                                               "DynFuseColorConversionTest");
    }
};

TEST_P(FuseDynColorConversionTest, NPU4000_HC_TestKindSubgraph) {
    abs_threshold = 0.01;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

TEST_P(FuseDynColorConversionTest, NPU5010_HC_TestKindSubgraph) {
    abs_threshold = 0.01;
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(precommit, FuseDynColorConversionTest,
                         ::testing::Combine(::testing::Values(Height{1440_Dyn}), ::testing::Values(Width{2560_Dyn})),
                         PrintTestCaseName());

}  // namespace
}  // namespace ov::test::subgraph
