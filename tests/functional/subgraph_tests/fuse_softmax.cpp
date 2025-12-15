// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ov::test {

struct SoftmaxParams {
    ov::Shape inputShape;
    ov::element::Type inputType;
    std::vector<int64_t> axes;
    std::string testCategory;
};

class FuseSoftmaxTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<SoftmaxParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SoftmaxParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "Category=" << obj.param.testCategory << sep;
        result << "Shape=" << ov::test::utils::vec2str(obj.param.inputShape) << sep;
        result << "Type=" << obj.param.inputType << sep;
        result << "Axes=" << ov::test::utils::vec2str(obj.param.axes) << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0.0;
        in_data.range = 10.0;
        in_data.resolution = 32768;

        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], in_data);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        const auto testParams = GetParam();
        const auto inputShape = testParams.inputShape;
        const auto axes = testParams.axes;
        inType = outType = testParams.inputType;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // Build decomposed softmax pattern: ReduceMax -> Subtract -> Exp -> ReduceSum -> Divide
        const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        const auto reduceMax = std::make_shared<ov::op::v1::ReduceMax>(input, axesConst, true);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(input, reduceMax);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);
        const auto reduceSum = std::make_shared<ov::op::v1::ReduceSum>(exp, axesConst, true);
        const auto divide = std::make_shared<ov::op::v1::Divide>(exp, reduceSum);

        const auto result = std::make_shared<ov::op::v0::Result>(divide);
        const ov::ResultVector results{result};

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "FuseSoftmaxTest");
    }
};

TEST_P(FuseSoftmaxTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseSoftmaxTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(
        precommit_FuseSoftmax, FuseSoftmaxTestCommon,
        ::testing::ValuesIn({
                // ========== CONSECUTIVE AXES TESTS (Reshape Only) ==========
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {2, 3}, "ConsecutiveAxes4D"},     // Last 2 dims
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {1, 2}, "ConsecutiveAxes4D"},     // Middle 2 dims
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 1}, "ConsecutiveAxes4D"},     // First 2 dims
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {1, 2, 3}, "ConsecutiveAxes4D"},  // Last 3 dims
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 1, 2}, "ConsecutiveAxes4D"},  // First 3 dims

                // ========== NON-CONSECUTIVE AXES TESTS (Transpose + Reshape) ==========
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 2}, "NonConsecutiveAxes4D"},     // Skip dim 1
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 3}, "NonConsecutiveAxes4D"},     // Skip dims 1,2
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {1, 3}, "NonConsecutiveAxes4D"},     // Skip dim 2
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 1, 3}, "NonConsecutiveAxes4D"},  // Skip dim 2
                SoftmaxParams{{2, 4, 8, 16}, ov::element::f16, {0, 2, 3}, "NonConsecutiveAxes4D"},  // Skip dim 1
        }),
        FuseSoftmaxTestCommon::getTestCaseName);

}  // namespace ov::test
