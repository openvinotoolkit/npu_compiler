// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/opsets/opset14_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include "openvino/op/avg_pool.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test {

class AvgPoolV14LayerTestCommon : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 30, 30};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::opset3::Parameter>(ov::element::f32, inputDynamicShapes[0])};

        const ov::Strides strides = {2, 2};
        const std::vector<size_t> padBegin = {0, 0};
        const std::vector<size_t> padEnd = {0, 0};
        const std::vector<size_t> kernel = {3, 3};
        bool exclude_pad = false;
        const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;
        const ov::op::PadType padType = ov::op::PadType::AUTO;

        const auto avgpooling = std::make_shared<ov::opset14::AvgPool>(params[0], strides, padBegin, padEnd, kernel,
                                                                       exclude_pad, roundingType, padType);

        const ov::ResultVector results{std::make_shared<ov::opset3::Result>(avgpooling)};
        function = std::make_shared<ov::Model>(results, params, "AvgPoolV14Test");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_F(AvgPoolV14LayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_F(AvgPoolV14LayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_F(AvgPoolV14LayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace ov::test
