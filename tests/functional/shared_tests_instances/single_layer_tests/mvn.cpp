//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_op_tests/mvn.hpp>
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/mvn.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// -------------- MVN1 test classes

typedef std::tuple<std::vector<InputShape>,  // Input shapes
                   ov::element::Type,        // Input precision
                   ov::Layout,               // Input layout
                   ov::AxisSet,              // Reduction axes
                   bool,                     // Across channels
                   bool,                     // Normalize variance
                   double                    // Epsilon
                   >
        mvn2Params;  // MVN1 with layout params

/**
 * For testing particular kernel with variable order currently there is no way to generalize that in
 LayerTestCommon
 */
class Mvn1LayerTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<mvn2Params> {
    void SetUp() override {
        std::vector<InputShape> shapes;
        ov::element::Type modelType;
        ov::AxisSet axes;
        ov::Layout order;
        bool acrossChannels, normalizeVariance;
        double eps;
        std::tie(shapes, modelType, order, axes, acrossChannels, normalizeVariance, eps) = this->GetParam();
        init_input_shapes(shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());

        std::shared_ptr<ov::op::v0::MVN> mvn;

        if (axes.empty()) {
            mvn = std::make_shared<ov::op::v0::MVN>(param, acrossChannels, normalizeVariance, eps);

            // OpenVINO MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid
            // behavior
            ov::AxisSet axes;
            const size_t startAxis = acrossChannels ? 1 : 2;
            const size_t numOfDims = param->output(0).get_partial_shape().size();
            for (size_t i = startAxis; i < numOfDims; i++) {
                axes.insert(i);
            }
            mvn->set_reduction_axes(axes);
        } else {
            mvn = std::make_shared<ov::op::v0::MVN>(param, axes, normalizeVariance, eps);
        }

        auto result = std::make_shared<ov::op::v0::Result>(mvn);
        function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "MVN1");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvn2Params>& obj) {
        mvn1Params mvn1;
        ov::Layout hwl;

        std::tie(std::get<0>(mvn1), std::get<1>(mvn1), hwl, std::get<2>(mvn1), std::get<3>(mvn1), std::get<4>(mvn1),
                 std::get<5>(mvn1)) = obj.param;

        std::ostringstream resultName;
        resultName << Mvn1LayerTest::getTestCaseName(testing::TestParamInfo<mvn1Params>(mvn1, 0)) << "_";
        resultName << "Layout=" << hwl.to_string();
        return resultName.str();
    }
};

/**
 * MVN1 Input=0 special case E#96869
 */
class Mvn1ZeroInputLayerTestCommon : public Mvn1LayerTest, virtual public VpuOv2LayerTest {};

class Mvn1ZeroInputLayerTest_NPU3720_HW : public Mvn1ZeroInputLayerTestCommon {};
class Mvn1ZeroInputLayerTest_NPU4000_HW : public Mvn1ZeroInputLayerTestCommon {};
class Mvn1LayerTest_NPU4000_HW : public Mvn1ZeroInputLayerTestCommon {};
class Mvn1LayerTest_SW : public Mvn1ZeroInputLayerTestCommon {};

TEST_P(Mvn1LayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Mvn1LayerTest_NPU4000_HW, HW) {
    abs_threshold = 0.03;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(Mvn1LayerTest_SW, NPU4000_SW) {
    abs_threshold = 0.03;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(Mvn1ZeroInputLayerTest_NPU3720_HW, HW) {
    abs_threshold = 0.003;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Mvn1ZeroInputLayerTest_NPU4000_HW, HW) {
    abs_threshold = 0.003;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// -------------- MVN6 test classes

class Mvn6LayerTestCommon : public Mvn6LayerTest, virtual public VpuOv2LayerTest {};

class Mvn6LayerTestCommonFP32 : public Mvn6LayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(Mvn6LayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(Mvn6LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
// -------------- MVN6 F32 tests

TEST_P(Mvn6LayerTestCommonFP32, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(Mvn6LayerTestCommonFP32, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ================================= Common params ================================ */

const std::vector<bool> acrossChannels = {true, false};
const std::vector<bool> normalizeVariance = {true, false};
const std::vector<double> epsilon = {0.000000001};
const std::vector<float> epsilonF = {0.0001};
const std::vector<std::string> epsMode = {"inside_sqrt", "outside_sqrt"};
const std::vector<ov::AxisSet> emptyReductionAxes = {{}};
using AxesVec = std::vector<std::vector<int>>;

/* ================================= Param builder utils ================================= */

const auto genMvn6Params = [](auto shapes, auto axes, auto eps) {
    return ::testing::Combine(
            ::testing::ValuesIn(static_shapes_to_test_representation(shapes)), ::testing::Values(ov::element::f16),
            ::testing::Values(ov::element::i32), ::testing::ValuesIn(axes), ::testing::ValuesIn(normalizeVariance),
            ::testing::ValuesIn(eps), ::testing::ValuesIn(epsMode), ::testing::Values(test_utils::TARGET_DEVICE));
};

// less test combinations
const auto genMvn6LessParams = [](auto shape, auto axes, auto eps) {
    bool normVariance = true;  // typical configs
    const std::string epsMode = "inside_sqrt";
    return ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(shape)),
                              ::testing::Values(ov::element::f16), ::testing::Values(ov::element::i32),
                              ::testing::ValuesIn(axes), ::testing::Values(normVariance), ::testing::ValuesIn(eps),
                              ::testing::Values(epsMode), ::testing::Values(test_utils::TARGET_DEVICE));
};

/* ============================ MVN1 tests ============================= */

const std::vector<std::vector<ov::Shape>> inputShapesForOrder = {{{1, 4, 2, 1024}}};

const std::vector<std::vector<ov::Shape>> inputShapes4D = {{{1, 4, 512, 1}}, {{1, 999, 2, 3}}, {{1, 16, 5, 8}},
                                                           {{2, 19, 5, 10}}, {{7, 32, 2, 8}},  {{5, 8, 3, 5}},
                                                           {{4, 41, 6, 9}}
#if 0  // extra shapes
        {{5, 2, 7, 3}},   {{1, 3, 17, 21}}, {{2, 5, 13, 27}}, {{1, 7, 55, 33}}, {{4, 9, 7, 2}},  {{3, 13, 9, 9}}, {{1, 16, 12,
        11}}, {{1, 512, 3, 2}},
#endif
};
const std::vector<std::vector<ov::Shape>> inputShapesForDecomposition = {
        {{2, 3, 20, 35971}}, {{1, 1, 515971, 1}}, {{1, 1, 1, 515971}}, {{1, 12, 20, 35971}}};
const std::vector<std::vector<ov::Shape>> inputShapesForNHWCOpt = {{{1, 16, 4, 32}}, {{1, 32, 4, 16}}};
const std::vector<std::vector<ov::Shape>> inputShapesForBigSize = {{{1, 1, 8, 48}}, {{1, 2, 8, 128}}};

// -------------- MVN1 - NPU3270
INSTANTIATE_TEST_SUITE_P(
        precommit_MVN1_order, Mvn1LayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForOrder)),
                           ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn({ov::Layout("NCHW"), ov::Layout("NCWH"), ov::Layout("NWHC")}),
                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon)),
        Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN1, Mvn1LayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes4D)),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NCHW")),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon)),
                         Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        precommit_MVN1_opt, Mvn1LayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForNHWCOpt)),
                           ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon)),
        Mvn1LayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        precommit_MVN1_bigsize, Mvn1LayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForBigSize)),
                           ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NCHW")),
                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon)),
        Mvn1LayerTest_NPU3720::getTestCaseName);

// -------------- MVN1 Decomposition

INSTANTIATE_TEST_SUITE_P(
        smoke_MVN1_Decomposition, Mvn1LayerTest_NPU3720,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForDecomposition)),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::Layout("NCHW")),
                           ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
                           ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon)),
        Mvn1LayerTest_NPU3720::getTestCaseName);

// -------------- MVN1 - SW

INSTANTIATE_TEST_SUITE_P(precommit_MVN1, Mvn1LayerTest_SW,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes4D)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(emptyReductionAxes),
                                            ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                                            ::testing::ValuesIn(epsilon), ::testing::Values(test_utils::TARGET_DEVICE)),
                         Mvn1LayerTest_SW::getTestCaseName);

// -------------- MVN1 Decomposition

INSTANTIATE_TEST_SUITE_P(
        smoke_MVN1_Decomposition, Mvn1LayerTest_NPU4000_HW,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForDecomposition)),
                           ::testing::Values(ov::element::f32), ::testing::ValuesIn(emptyReductionAxes),
                           ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                           ::testing::ValuesIn(epsilon), ::testing::Values(test_utils::TARGET_DEVICE)),
        Mvn1LayerTest_NPU4000_HW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_MVN1_Decomposition, Mvn1LayerTest_SW,
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesForDecomposition)),
                           ::testing::Values(ov::element::f32), ::testing::ValuesIn(emptyReductionAxes),
                           ::testing::ValuesIn(acrossChannels), ::testing::ValuesIn(normalizeVariance),
                           ::testing::ValuesIn(epsilon), ::testing::Values(test_utils::TARGET_DEVICE)),
        Mvn1LayerTest_SW::getTestCaseName);

// -------------- MVN6 'pseudo' tests : actually testing MVN1 op,
// as innermost-consecutive norm axes config trigger 'ConvertMVN6toMVN1' to pass

const auto pse2D = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{5, 17}}}, AxesVec{{1}}, epsilonF);
const auto pse3D = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{10, 5, 17}}}, AxesVec{{2}}, epsilonF);
const auto pse3DAxes2D =
        genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 48, 48}}}, AxesVec{{1, 2}}, epsilonF);
const auto pse4D = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 48, 48, 32}}}, AxesVec{{-1}}, epsilonF);
const auto pse4DAxis1 =
        genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 20, 112, 112}}}, AxesVec{{1}}, epsilonF);
const auto pse5D =
        genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 32, 20, 20, 20}}}, AxesVec{{2, 3, 4}}, epsilonF);

INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_2D, Mvn6LayerTestCommon, pse2D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_3D, Mvn6LayerTestCommon, pse3D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_3DAxes2D, Mvn6LayerTestCommon, pse3DAxes2D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_4D, Mvn6LayerTestCommon, pse4D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_4DAxis1, Mvn6LayerTestCommon, pse4DAxis1, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(pseudo_MVN6_5D, Mvn6LayerTestCommon, pse5D, Mvn6LayerTestCommon::getTestCaseName);

// -------------- MVN1 Zero-Input test

const auto zeroTestCfg = ::testing::Combine(
        ::testing::Values(static_shapes_to_test_representation(inputShapes4D[0])), ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels), ::testing::Values(true),
        ::testing::ValuesIn(epsilon), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(zero_input, Mvn1ZeroInputLayerTest_NPU3720_HW, zeroTestCfg,
                         Mvn1ZeroInputLayerTest_NPU3720_HW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(zero_input, Mvn1ZeroInputLayerTest_NPU4000_HW, zeroTestCfg,
                         Mvn1ZeroInputLayerTest_NPU4000_HW::getTestCaseName);

/* ============================= MVN6 tests ============================ */

const std::vector<float> bigEps = {0.5};
std::vector<std::vector<ov::Shape>> shapes1D = {{{17}}};
std::vector<std::vector<ov::Shape>> shapes2D = {{{5, 17}}};
std::vector<std::vector<ov::Shape>> shapes3D = {{{10, 5, 17}}};
std::vector<std::vector<ov::Shape>> shapes4D = {{{4, 10, 5, 17}}};
std::vector<std::vector<ov::Shape>> shapes5D = {{{10, 16, 5, 10, 6}}};

// axes values for corresponding ND shapes
std::vector<std::vector<int>> axes1D = {{0}};
std::vector<std::vector<int>> axes2D = {{1}, {0, 1}};
std::vector<std::vector<int>> axes3D = {{0}, {1}, {0, 1}};
std::vector<std::vector<int>> axes4D = {{0},    {2},    {3},       {0, 1},    {0, 2},
                                        {1, 2}, {1, 3}, {0, 1, 2}, {0, 1, 3}, {0, 1, 2, 3}};
std::vector<std::vector<int>> axes5D = {{1}, {1, 2}, {1, 3, 4}, {0, 2, 3}, {0, 1, 3, 4}};

// ND-shape configs
const auto cfg1D = genMvn6Params(shapes1D, axes1D, bigEps);
const auto cfg2D = genMvn6Params(shapes2D, axes2D, bigEps);
const auto cfg3D = genMvn6Params(shapes3D, axes3D, bigEps);
const auto cfg4D = genMvn6LessParams(shapes4D, axes4D, bigEps);
const auto cfg5D = genMvn6LessParams(shapes5D, axes5D, bigEps);

// tiling configs
const auto cfgT0 = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 512, 1219}}}, AxesVec{{1}}, bigEps);
const auto cfgT1 = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{1, 64, 104, 104}}}, AxesVec{{2}}, bigEps);
const auto cfgT2 = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{8, 16, 16, 16, 16}}}, AxesVec{{4}}, bigEps);

// Multi-SHAVEs config
const auto cfgMS = genMvn6LessParams(std::vector<std::vector<ov::Shape>>{{{4, 10, 5, 17}}, {{4, 1, 10, 17}}},
                                     AxesVec{{0}}, bigEps);

// -------------- MVN6 - NPU3720

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_1D, Mvn6LayerTestCommon, cfg1D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_2D, Mvn6LayerTestCommon, cfg2D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_3D, Mvn6LayerTestCommon, cfg3D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_4D, Mvn6LayerTestCommon, cfg4D, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MVN6_5D, Mvn6LayerTestCommon, cfg5D, Mvn6LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(tiling_MVN6_a, Mvn6LayerTestCommon, cfgT0, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_b, Mvn6LayerTestCommon, cfgT1, Mvn6LayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(tiling_MVN6_c, Mvn6LayerTestCommon, cfgT2, Mvn6LayerTestCommon::getTestCaseName);

// -------------- MVN - f32 tests
std::vector<std::vector<int>> axesF32 = {{2} /*MVN6*/, {2, 3} /*MVN1*/};
const auto cfgF32 = ::testing::Combine(
        ::testing::Values(static_shapes_to_test_representation(shapes4D[0])), ::testing::Values(ov::element::f32),
        ::testing::Values(ov::element::i32), ::testing::ValuesIn(axesF32), ::testing::Values(true),
        ::testing::ValuesIn(bigEps), ::testing::Values(epsMode[0]), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_MVN6_fp32, Mvn6LayerTestCommonFP32, cfgF32, Mvn6LayerTestCommonFP32::getTestCaseName);

}  // namespace
