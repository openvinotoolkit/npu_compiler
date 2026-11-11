//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {

enum class ReduceType { Mean, Sum };

struct ReduceSquareParams {
    ov::Shape inputShape;
    ReduceType reduceType;
    bool hasEpsilon;
};

class FuseReduceSquareTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<ReduceSquareParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReduceSquareParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "ReduceType=" << (obj.param.reduceType == ReduceType::Mean ? "Mean" : "Sum") << sep;
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
        const auto reduceType = testParams.reduceType;
        const auto hasEpsilon = testParams.hasEpsilon;
        const auto inputShape = testParams.inputShape;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const auto input = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));

        // x^2 (Power operation with exponent 2.0)
        const auto powerConst = ov::op::v0::Constant::create(ov::element::f32, {}, {2.0f});
        const auto power = std::make_shared<ov::op::v1::Power>(input, powerConst);

        // Reduce(x^2, axes, keep_dims) - either ReduceMean or ReduceSum
        auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        std::shared_ptr<ov::Node> reduceOp;
        if (reduceType == ReduceType::Mean) {
            reduceOp = std::make_shared<ov::op::v1::ReduceMean>(power, axesConst, true);
        } else {
            reduceOp = std::make_shared<ov::op::v1::ReduceSum>(power, axesConst, true);
        }

        // Sqrt(Reduce(x^2, axes, keep_dims))
        std::shared_ptr<ov::op::v0::Sqrt> sqrt;
        // Epsilon (Add) is only supported for ReduceMean pattern, not for ReduceSum
        if (hasEpsilon && reduceType == ReduceType::Mean) {
            // Reduce(x^2, axes, keep_dims) + eps
            auto eps = ov::op::v0::Constant::create(inType, {}, {3.5});
            auto addEps = std::make_shared<ov::op::v1::Add>(reduceOp, eps);

            // Sqrt(Reduce(x^2, axes, keep_dims) + eps)
            sqrt = std::make_shared<ov::op::v0::Sqrt>(addEps);
        } else {
            sqrt = std::make_shared<ov::op::v0::Sqrt>(reduceOp);
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sqrt)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "FuseReduceSquareTest");
    }
};

TEST_P(FuseReduceSquareTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseReduceSquareTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseReduceSquareTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(FuseReduceSquareTestCommon, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

const std::vector<ReduceSquareParams> testValues = {
        // ReduceMean pattern (Power + ReduceMean + Sqrt)
        {{1, 32, 32, 96}, ReduceType::Mean, false},  // without epsilon
        {{1, 32, 32, 96}, ReduceType::Mean, true},   // with epsilon
        {{1, 512, 18, 80}, ReduceType::Mean, false},
        {{1, 512, 18, 80}, ReduceType::Mean, true},
        // ReduceSum pattern (Power + ReduceSum + Sqrt)
        {{1, 32, 32, 96}, ReduceType::Sum, false},
        {{1, 512, 18, 80}, ReduceType::Sum, false},
};

INSTANTIATE_TEST_SUITE_P(precommit_FuseReduceSquare, FuseReduceSquareTestCommon, ::testing::ValuesIn(testValues),
                         FuseReduceSquareTestCommon::getTestCaseName);

}  // namespace ov::test
