// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"

using namespace ov::test::utils;

using ov::test::utils::InputLayerType;

namespace ov::test::subgraph {

// Test subgraphs (Mul/Add fused into MVN6):
//   MVN6 -> Multiply -> Add
//   MVN6 -> Multiply
//   MVN6 -> Add

using actFlags = std::tuple<bool, bool>;  // <scale, bias> on/off

using FuseMvn6ScaleBiasParams = std::tuple<std::vector<size_t>,  // data shape
                                           std::vector<size_t>,  // scale/bias shape
                                           std::vector<size_t>,  // mvn norm axes
                                           actFlags,             // scale/bias activation flags
                                           InputLayerType>;      // Param/Const

class FuseMvn6ScaleBiasTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<FuseMvn6ScaleBiasParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMvn6ScaleBiasParams> obj) {
        ov::Shape inputShape;
        ov::Shape actShape;
        std::vector<size_t> axes;
        actFlags flags;
        InputLayerType actType;
        bool hasScale, hasBias;
        std::tie(inputShape, actShape, axes, flags, actType) = obj.param;
        std::tie(hasScale, hasBias) = flags;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "IS={" << vec2str(inputShape) << "}" << sep;
        result << "ES={" << vec2str(actShape) << "}" << sep;
        result << "axes={" << vec2str(axes) << "}" << sep;
        result << "hasScale={" << hasScale << "}" << sep;
        result << "hasBias={" << hasBias << "}" << sep;
        result << "actType={" << actType << "}" << sep;
        return result.str();
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "fuse-mvn6-scale-bias=true";
    }

    void SetUp() override {
        ov::Shape inputShape;
        ov::Shape scaleShape;
        ov::Shape biasShape;
        std::vector<size_t> axes;
        actFlags flags;
        InputLayerType actType;
        bool hasScale, hasBias;

        std::tie(inputShape, scaleShape, axes, flags, actType) = GetParam();
        std::tie(hasScale, hasBias) = flags;
        ASSERT_EQ(hasScale || hasBias, true);

        biasShape = scaleShape;
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape, scaleShape, biasShape}));

        ov::ParameterVector params;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape(inputShape));
        params.push_back(param);  // data input

        float eps = 0.000001;  // typical MVN params
        ov::op::MVNEpsMode epsMode = ov::op::MVNEpsMode::INSIDE_SQRT;
        bool normalizeVariance = true;
        auto axesNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto mvn = std::make_shared<ov::op::v6::MVN>(param, axesNode, normalizeVariance, eps, epsMode);

        std::shared_ptr<ov::Node> scaleIn, biasIn;

        auto genActInput = [&](ov::Shape shape, double ctFrom, double ctRange) -> decltype(scaleIn) {
            if (actType == InputLayerType::PARAMETER) {
                auto actParam = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
                params.push_back(actParam);
                return actParam;
            } else {
                ov::test::utils::InputGenerateData genData;
                genData.start_from = ctFrom;
                genData.range = ctRange;
                auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f16, shape, genData);
                return std::make_shared<ov::op::v0::Constant>(tensor);
            }
        };

        if (hasScale) {
            scaleIn = genActInput(scaleShape, -2, 8);
        }
        if (hasBias) {
            biasIn = genActInput(biasShape, -4, 4);
        }

        std::shared_ptr<ov::op::v0::Result> result;
        if (hasScale) {
            auto scale = std::make_shared<ov::op::v1::Multiply>(mvn, scaleIn);
            if (hasBias) {  // SCALE + BIAS
                auto bias = std::make_shared<ov::op::v1::Add>(scale, biasIn);
                result = std::make_shared<ov::op::v0::Result>(bias->output(0));
            } else {  // just SCALE
                result = std::make_shared<ov::op::v0::Result>(scale->output(0));
            }
        } else {  // just BIAS
            auto bias = std::make_shared<ov::op::v1::Add>(mvn, biasIn);
            result = std::make_shared<ov::op::v0::Result>(bias->output(0));
        }
        function = std::make_shared<ov::Model>(result, params, "MVN6ScaleBias");
    }
};

TEST_P(FuseMvn6ScaleBiasTestCommon, NPU3720_HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseMvn6ScaleBiasTestCommon, NPU4000_HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(FuseMvn6ScaleBiasTestCommon, NPU5010_HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::Shape inputShape = {1, 150, 64, 8};
std::vector<std::vector<size_t>> actShape = {
        {1, 150, 64, 8}, {1, 150, 1, 1}, {1, 1, 64, 1}, {1, 1, 1, 8}};  // scale/bias shape
std::vector<size_t> axes = {2, 3};
std::vector<actFlags> actPresent = {std::make_tuple(true, true), std::make_tuple(true, false),
                                    std::make_tuple(false, true)};
std::vector<InputLayerType> actType = {
        InputLayerType::PARAMETER,
        InputLayerType::CONSTANT,
};

const auto testParams =
        ::testing::Combine(::testing::Values(inputShape), ::testing::ValuesIn(actShape), ::testing::Values(axes),
                           ::testing::ValuesIn(actPresent), ::testing::Values(actType[0]));

// Tiling config
ov::Shape tileShape = {1, 1500, 1024};
ov::Shape tileActShape = {1, 1, 1024};
std::vector<size_t> tileAxis = {2};
const auto tileParams =
        ::testing::Combine(::testing::Values(tileShape), ::testing::Values(tileActShape), ::testing::Values(tileAxis),
                           ::testing::Values(actPresent[0]), ::testing::Values(actType[1]));

INSTANTIATE_TEST_SUITE_P(precommit_FuseMvn6ScaleBias, FuseMvn6ScaleBiasTestCommon, testParams,
                         FuseMvn6ScaleBiasTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(tiling_FuseMvn6ScaleBias, FuseMvn6ScaleBiasTestCommon, tileParams,
                         FuseMvn6ScaleBiasTestCommon::getTestCaseName);

}  // namespace
