//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/avg_pool.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::test {

class AvgPoolWithStaticScaleTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, size_t>> {
    void SetUp() override {
        inType = ov::element::f32;
        outType = ov::element::f32;

        const auto inputShape = std::get<0>(GetParam());
        const auto kernelSize = std::get<1>(GetParam());

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        // AvgPool with KxK kernel, strides 8x8, and pads 8x8
        auto avgPool = std::make_shared<ov::op::v1::AvgPool>(params.at(0), ov::Strides{8, 8}, ov::Shape{8, 8},
                                                             ov::Shape{8, 8}, ov::Shape{kernelSize, kernelSize}, false);

        const ov::Shape scalesShape{1, 1, 1, 1};
        const auto scales = ov::op::v0::Constant::create(ov::element::f32, scalesShape, {1.0f});
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(avgPool, scales);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(multiply)};
        function = std::make_shared<ov::Model>(results, params, "AvgPoolWithStaticScaleTest");

        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<ov::Shape, size_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(AvgPoolWithStaticScaleTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(AvgPoolWithStaticScaleTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(AvgPoolWithStaticScaleTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(AvgPoolWithStaticScaleTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

const std::vector<ov::Shape> testShapes = {
        {1, 512, 8, 10},

};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolWithBigKernelAndStaticScale, AvgPoolWithStaticScaleTestCommon,
                         ::testing::Combine(::testing::ValuesIn(testShapes), ::testing::Values(17)),
                         AvgPoolWithStaticScaleTestCommon::getTestCaseName);

}  // namespace ov::test
