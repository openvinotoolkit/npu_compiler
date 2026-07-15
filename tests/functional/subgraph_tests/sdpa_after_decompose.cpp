//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::test {

struct SDPAAfterDecomposeTestParams {
    ov::Shape queryShape;
    ov::Shape keyShape;
    ov::Shape maskShape;
    ov::Shape valueShape;
    ov::Shape scaleShape;
};

class SDPAAfterDecomposeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SDPAAfterDecomposeTestParams> {
    void SetUp() override {
        const auto testParams = GetParam();
        const auto queryShape = testParams.queryShape;
        const auto keyShape = testParams.keyShape;
        const auto maskShape = testParams.maskShape;
        const auto valueShape = testParams.valueShape;
        const auto scaleShape = testParams.scaleShape;
        init_input_shapes(ov::test::static_shapes_to_test_representation(
                {queryShape, keyShape, maskShape, valueShape, scaleShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        const auto q_scaled = std::make_shared<ov::op::v1::Multiply>(params[0], params[4]);
        const auto scaled_atten = std::make_shared<ov::op::v0::MatMul>(q_scaled, params[1], false, true);
        const auto atten_mask = std::make_shared<ov::op::v1::Add>(scaled_atten, params[2]);
        const auto softmax = std::make_shared<ov::op::v8::Softmax>(atten_mask, -1);
        const auto result = std::make_shared<ov::op::v0::MatMul>(softmax, params[3]);

        const ov::ResultVector outputs{std::make_shared<ov::op::v0::Result>(result)};
        function = std::make_shared<ov::Model>(outputs, params, "SDPAAfterDecomposeTest");
        abs_threshold = 0.5;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<SDPAAfterDecomposeTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(SDPAAfterDecomposeTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SDPAAfterDecomposeTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SDPAAfterDecomposeTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPAAfterDecompose, SDPAAfterDecomposeTestCommon,
                         ::testing::Values(SDPAAfterDecomposeTestParams{
                                 {1, 3, 32, 32}, {1, 3, 32, 32}, {1, 3, 32, 32}, {1, 3, 32, 32}, {1}}),
                         SDPAAfterDecomposeTestCommon::getTestCaseName);

}  // namespace ov::test
