//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"

namespace ov::test::subgraph {

using FuseMVNTestParams = std::tuple<std::vector<size_t>,  // input shape
                                     std::vector<size_t>,  // target_shape
                                     std::vector<size_t>,  // ReduceMean axis
                                     bool                  // eps inside or outside
                                     >;

class FuseMVNTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseMVNTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMVNTestParams> obj) {
        ov::Shape inputShape;
        ov::Shape targetShape;
        std::vector<size_t> axis;
        bool isEpsInside;
        std::tie(inputShape, targetShape, axis, isEpsInside) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inputShape.size() << "}" << sep;
        result << "targetShapeSize={" << targetShape.size() << "}" << sep;
        result << "axisSize={" << axis.size() << "}" << sep;
        result << "isEpsInside={" << isEpsInside << "}" << sep;
        return result.str();
    }

    void SetUp() override {
        ov::Shape input_shape;
        ov::Shape target_shape;
        std::vector<size_t> axis;
        bool isEpsInside;

        std::tie(input_shape, target_shape, axis, isEpsInside) = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shape}));
        ov::ParameterVector params{std::make_shared<ov::opset6::Parameter>(ov::element::f32, ov::Shape(input_shape))};

        auto reshape1_const = ov::opset6::Constant::create(ov::element::i32, {target_shape.size()}, target_shape);
        auto reshape1 = std::make_shared<ov::opset6::Reshape>(params[0], reshape1_const, false);

        auto mean1_axes = ov::opset6::Constant::create(ov::element::i32, {axis.size()}, axis);
        auto mean1 = std::make_shared<ov::opset6::ReduceMean>(reshape1, mean1_axes, true);

        auto sub1 = std::make_shared<ov::opset6::Subtract>(reshape1, mean1);

        auto x_square = std::make_shared<ov::opset6::Multiply>(reshape1, reshape1);
        auto x_square_mean_axes = ov::opset6::Constant::create(ov::element::i32, {axis.size()}, axis);
        auto x_square_mean = std::make_shared<ov::opset6::ReduceMean>(x_square, mean1_axes, true);
        auto mean1_square = std::make_shared<ov::opset6::Multiply>(mean1, mean1);

        auto sub2 = std::make_shared<ov::opset6::Subtract>(x_square_mean, mean1_square);
        auto eps = ov::opset6::Constant::create(ov::element::f32, {1}, {0.000001});

        if (isEpsInside) {
            auto eps_inside_add = std::make_shared<ov::opset6::Add>(sub2, eps);
            auto eps_inside_sqrt = std::make_shared<ov::opset6::Sqrt>(eps_inside_add);
            auto divide = std::make_shared<ov::opset6::Divide>(sub1, eps_inside_sqrt);
            auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(divide->output(0))};
            function = std::make_shared<ov::Model>(results, params, "FuseMVNInsideEPS");
        } else {
            auto eps_outside_sqrt = std::make_shared<ov::opset6::Sqrt>(sub2);
            auto eps_outside_add = std::make_shared<ov::opset6::Add>(eps_outside_sqrt, eps);
            auto divide = std::make_shared<ov::opset6::Divide>(sub1, eps_outside_add);
            auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(divide->output(0))};
            function = std::make_shared<ov::Model>(results, params, "FuseMVNOutsideEPS");
        }
    }
};

TEST_P(FuseMVNTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseMVNTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseMVNTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseMVNTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

//
// FuseMVNWithSquaredDiff
//
// Builds the TFLite-style LayerNorm decomposition:
//   out = Multiply(x, Reshape(rsqrt)) + Reshape(Subtract(0, Multiply(mean, rsqrt)))
//       = (x - mean) / sqrt(var + eps)    [= MVN]
// where rsqrt = Power(ReduceMean(SquaredDifference(x, Reshape(mean)), axes) + eps, -0.5).
//
// The compiler fuses this graph into a single IE::MVNOp. For axes that cannot
// be handled by a pure reshape (e.g. [0,1] on a CxHxW tensor), getMVN1Mapping
// inserts a pair of Transpose ops around the MVN.
//

using FuseMVNWithSquaredDiffTestParams = std::tuple<std::vector<size_t>,  // input shape
                                                    std::vector<size_t>,  // ReduceMean axes
                                                    std::vector<size_t>   // broadcast shape for mean / rsqrt
                                                    >;

class FuseMVNWithSquaredDiffTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<FuseMVNWithSquaredDiffTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMVNWithSquaredDiffTestParams> obj) {
        std::vector<size_t> inputShape, axes, bcastShape;
        std::tie(inputShape, axes, bcastShape) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape={";
        for (size_t i = 0; i < inputShape.size(); ++i) {
            result << (i ? "x" : "") << inputShape[i];
        }
        result << "}" << sep;
        result << "Axes={";
        for (size_t i = 0; i < axes.size(); ++i) {
            result << (i ? "," : "") << axes[i];
        }
        result << "}";
        return result.str();
    }

    void SetUp() override {
        std::vector<size_t> inputShape, axes, bcastShape;
        std::tie(inputShape, axes, bcastShape) = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({ov::Shape(inputShape)}));
        ov::ParameterVector params{std::make_shared<ov::opset6::Parameter>(ov::element::f32, ov::Shape(inputShape))};

        // mean = ReduceMean(x, axes, keep_dims=false)
        auto meanAxesConst = ov::opset6::Constant::create(ov::element::i32, {axes.size()}, axes);
        auto mean = std::make_shared<ov::opset6::ReduceMean>(params[0], meanAxesConst, false);

        // Reshape mean / rsqrt / neg to broadcast shape (shared constant)
        auto bcastConst = ov::opset6::Constant::create(ov::element::i32, {bcastShape.size()}, bcastShape);
        auto meanR = std::make_shared<ov::opset6::Reshape>(mean, bcastConst, false);

        // sq_diff = SquaredDifference(x, mean_r)
        auto sqDiff = std::make_shared<ov::op::v0::SquaredDifference>(params[0], meanR);

        // var = ReduceMean(sq_diff, axes, keep_dims=false)
        auto varAxesConst = ov::opset6::Constant::create(ov::element::i32, {axes.size()}, axes);
        auto var = std::make_shared<ov::opset6::ReduceMean>(sqDiff, varAxesConst, false);

        // rsqrt = Power(var + eps, -0.5)
        auto eps = ov::opset6::Constant::create(ov::element::f32, {}, {1e-6f});
        auto varEps = std::make_shared<ov::opset6::Add>(var, eps);
        auto negHalf = ov::opset6::Constant::create(ov::element::f32, {}, {-0.5f});
        auto rsqrt = std::make_shared<ov::op::v1::Power>(varEps, negHalf);

        auto rsqrtR = std::make_shared<ov::opset6::Reshape>(rsqrt, bcastConst, false);

        // x_mul = Multiply(x, rsqrt_r)
        auto xMul = std::make_shared<ov::opset6::Multiply>(params[0], rsqrtR);

        // neg = Subtract(0, Multiply(mean, rsqrt))
        auto negMul = std::make_shared<ov::opset6::Multiply>(mean, rsqrt);
        auto zero = ov::opset6::Constant::create(ov::element::f32, {}, {0.0f});
        auto neg = std::make_shared<ov::opset6::Subtract>(zero, negMul);
        auto negR = std::make_shared<ov::opset6::Reshape>(neg, bcastConst, false);

        // result = Add(x_mul, neg_r)  = (x - mean) / sqrt(var + eps)
        auto output = std::make_shared<ov::opset6::Add>(xMul, negR);

        auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(output->output(0))};
        function = std::make_shared<ov::Model>(results, params, "FuseMVNWithSquaredDiff");
    }
};

TEST_P(FuseMVNWithSquaredDiffTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseMVNWithSquaredDiffTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FuseMVNWithSquaredDiffTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(FuseMVNWithSquaredDiffTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::Shape inputShape = {1, 1500, 512};
ov::Shape targetShape = {1500, 512};
std::vector<size_t> axis = {1};
std::vector<bool> isEpsInside = {true, false};

const auto epsCase = ::testing::Combine(::testing::Values(inputShape), ::testing::Values(targetShape),
                                        ::testing::Values(axis), ::testing::ValuesIn(isEpsInside));

INSTANTIATE_TEST_SUITE_P(precommit_FuseMVN, FuseMVNTestCommon, epsCase, FuseMVNTestCommon::getTestCaseName);

// Basic SquaredDiff-based LayerNorm: 1x8x64, axes=[2], no transpose needed.
const auto sqDiffBasicCase =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 8, 64}), ::testing::Values(std::vector<size_t>{2}),
                           ::testing::Values(std::vector<size_t>{1, 8, 1}));

INSTANTIATE_TEST_SUITE_P(precommit_FuseMVNWithSquaredDiff, FuseMVNWithSquaredDiffTestCommon, sqDiffBasicCase,
                         FuseMVNWithSquaredDiffTestCommon::getTestCaseName);

// Transpose path: 8x64x49, axes=[0,1]. getMVN1Mapping inserts Transpose [2,0,1] / [1,2,0].
const auto sqDiffTransposeCase = ::testing::Combine(::testing::Values(std::vector<size_t>{8, 64, 49}),
                                                    ::testing::Values(std::vector<size_t>{0, 1}),
                                                    ::testing::Values(std::vector<size_t>{1, 1, 49}));

INSTANTIATE_TEST_SUITE_P(precommit_FuseMVNWithSquaredDiffTranspose, FuseMVNWithSquaredDiffTestCommon,
                         sqDiffTransposeCase, FuseMVNWithSquaredDiffTestCommon::getTestCaseName);

}  // namespace
