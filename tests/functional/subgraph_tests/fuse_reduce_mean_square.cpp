// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {

struct ReduceMeanSquareParams {
    ov::Shape inputShape;
};

class FuseReduceMeanSquareTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ReduceMeanSquareParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceMeanSquareParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        inType = outType = ov::element::f32;
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // x^2 (Power operation with exponent 2.0)
        const auto powerConst = ov::op::v0::Constant::create(ov::element::f32, {}, {2.0f});
        const auto power = std::make_shared<ov::op::v1::Power>(input, powerConst);

        // ReduceMean(x^2, axes, keep_dims)
        auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        const auto reduceMean = std::make_shared<ov::op::v1::ReduceMean>(power, axesConst, true);

        // Sqrt(ReduceMean(x^2, axes, keep_dims))
        const auto sqrt = std::make_shared<ov::op::v0::Sqrt>(reduceMean);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sqrt)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "FuseReduceMeanSquareTest");
    }
};

TEST_P(FuseReduceMeanSquareTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<ReduceMeanSquareParams> testValues = {{{1, 32, 32, 96}}, {{1, 512, 18, 80}}};

INSTANTIATE_TEST_SUITE_P(precommit_FuseReduceMeanSquare, FuseReduceMeanSquareTestCommon,
                         ::testing::ValuesIn(testValues), FuseReduceMeanSquareTestCommon::getTestCaseName);

}  // namespace ov::test
