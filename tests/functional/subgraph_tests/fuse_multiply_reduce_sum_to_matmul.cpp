//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"

using namespace ov::test::utils;
namespace ov::test {

struct BroadcastMultiplyReduceSumParams {
    ov::Shape input1Shape;
    ov::Shape input2Shape;
    std::vector<int64_t> reduceAxes;
    bool keepDims;
};

class BroadcastMultiplyReduceSumTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<BroadcastMultiplyReduceSumParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BroadcastMultiplyReduceSumParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "IS1=" << ov::test::utils::partialShape2str({obj.param.input1Shape}) << sep;
        result << "IS2=" << ov::test::utils::partialShape2str({obj.param.input2Shape}) << sep;
        result << "Axes=";
        for (size_t i = 0; i < obj.param.reduceAxes.size(); i++) {
            if (i > 0) {
                result << ".";
            }
            result << obj.param.reduceAxes[i];
        }
        result << sep;
        result << "KeepDims=" << (obj.param.keepDims ? "true" : "false") << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void SetUp() override {
        inType = outType = ov::element::f16;
        const auto testParams = GetParam();

        init_input_shapes(
                ov::test::static_shapes_to_test_representation({testParams.input1Shape, testParams.input2Shape}));

        const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));

        const auto multiply = std::make_shared<ov::op::v1::Multiply>(input1, input2);

        const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{testParams.reduceAxes.size()},
                                                            testParams.reduceAxes);
        const auto reduceSum = std::make_shared<ov::op::v1::ReduceSum>(multiply, axesConst, testParams.keepDims);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reduceSum)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2},
                                               "BroadcastMultiplyReduceSumTest");
    }
};

TEST_P(BroadcastMultiplyReduceSumTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastMultiplyReduceSumTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(BroadcastMultiplyReduceSumTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(BroadcastMultiplyReduceSumTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(BroadcastMultiplyReduceSumTestCommon, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

const std::vector<BroadcastMultiplyReduceSumParams> testValues = {{{6, 8, 1, 64}, {6, 1, 8, 64}, {3}, true},
                                                                  {{6, 8, 1, 64}, {6, 1, 8, 64}, {3}, false},
                                                                  {{6, 8, 1, 4, 64}, {6, 1, 8, 4, 64}, {4}, false},
                                                                  {{6, 8, 32, 16, 1}, {6, 8, 32, 1, 8}, {2}, true}};

INSTANTIATE_TEST_SUITE_P(precommit_BroadcastMultiplyReduceSum, BroadcastMultiplyReduceSumTestCommon,
                         ::testing::ValuesIn(testValues), BroadcastMultiplyReduceSumTestCommon::getTestCaseName);

}  // namespace ov::test
