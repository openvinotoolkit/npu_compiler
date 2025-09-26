//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reshape.hpp"

using namespace ov::test::utils;

using ov::test::utils::InputLayerType;

namespace ov::test::subgraph {

//
// Input subgraph:             |   After 'FuseReshapeMvn' pass:
//                             |
//    [input] (ioShape)        |       ...
//       |                     |        |
//     (Add1) <-- [ct1]        |   (GroupConv, #NHWC) <-- [ct1]
//       |                     |        |
//   (Reshape) {to mvnShape}   |        |
//       |                     |        |
//     (MVN)                   |      (MVN,    #NHWC) {internal-reshape}
//       |                     |        |
//   (Reshape) {to ioShape}    |        |
//       |                     |        |
//     (Add2) <-- [ct2]        |   (GroupConv, #NHWC) <-- [ct2]
//       |                     |        |
//    [output] (ioShape)       |       ...
//

using MvnInternalReshapeParams = std::tuple<ov::Shape,           // first/final shape
                                            ov::Shape,           // mvn shape
                                            bool,                // acrossChannels
                                            bool,                // normalize variance
                                            ov::element::Type>;  // precision

class MvnInternalReshapeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MvnInternalReshapeParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MvnInternalReshapeParams> obj) {
        const auto& [ioShape, mvnShape, acrossChannels, normVariance, prc] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "IOS={" << vec2str(ioShape) << "}" << sep;
        result << "MvnS={" << vec2str(mvnShape) << "}" << sep;
        result << "acrossChannels={" << acrossChannels << "}" << sep;
        result << "norm={" << normVariance << "}" << sep;
        result << "prc={" << prc << "}" << sep;
        return result.str();
    }

    void SetUp() override {
        const auto& [ioShape, mvnShape, acrossChannels, normVariance, prc] = GetParam();
        const auto C = ioShape.at(1);
        const auto K = mvnShape.at(1);
        ASSERT_GT(C, K);
        ASSERT_EQ(C % K, 0);
        inType = outType = prc;
        ov::Shape addShape = {1, C, 1, 1};
        std::vector<float> addVal(C, 1.0f);
        init_input_shapes(ov::test::static_shapes_to_test_representation({ioShape, mvnShape, addShape}));

        ov::ParameterVector params;
        auto input = std::make_shared<ov::op::v0::Parameter>(prc, ov::Shape(ioShape));
        params.push_back(input);

        auto ct1 = ov::opset1::Constant::create(prc, addShape, addVal);
        auto ct2 = ov::opset1::Constant::create(prc, addShape, addVal);
        auto add1 = std::make_shared<ov::op::v1::Add>(params[0], ct1);

        auto reshape1 = buildReshape(add1, mvnShape);
        auto mvn = std::make_shared<ov::op::v0::MVN>(reshape1, acrossChannels, normVariance, 1.0E-6);
        auto reshape2 = buildReshape(mvn, ioShape);
        auto add2 = std::make_shared<ov::op::v1::Add>(reshape2, ct2);

        auto result = std::make_shared<ov::op::v0::Result>(add2->output(0));
        function = std::make_shared<ov::Model>(result, params, "MvnInternalReshape");
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const ov::Shape newShape) {
        auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                std::make_shared<ov::op::v1::Reshape>(param, constNode, false));
        return reshape;
    }
};

TEST_P(MvnInternalReshapeTestCommon, NPU3720_HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MvnInternalReshapeTestCommon, NPU4000_HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

// Note: must use ::f32 for OV reference to produce valid reference
const std::vector<ov::element::Type> precision = {ov::element::f32};

// ioShape[C] must be an integer multiple of mvnShape[C] in order to trigger the internal-reshape fuse
const auto C1 = 32;
const auto W1 = 276640;
ov::Shape mvnShape1 = {1, C1, 1, W1};
std::vector<ov::Shape> ioShape1 = {
        {1, C1 * 5, 1, W1 / 5},
        {1, C1 * 8, 1, W1 / 8},
#if 0  // more test configs
        {1, C1 * 10, 1, W1 / 10},
        {1, C1 * 13, 1, W1 / 13},
        {1, C1 * 20, 1, W1 / 20},
#endif
};
const auto testParams1 =
        ::testing::Combine(::testing::ValuesIn(ioShape1), ::testing::Values(mvnShape1), ::testing::Values(false),
                           ::testing::Values(true), ::testing::ValuesIn(precision));

INSTANTIATE_TEST_SUITE_P(smoke_MvnInternalReshape1, MvnInternalReshapeTestCommon, testParams1,
                         MvnInternalReshapeTestCommon::getTestCaseName);

// Test batched MVN
ov::Shape mvnShape2 = {32, 16, 64, 64};
ov::Shape ioShape2 = {1, 512, 64, 64};
const auto testParams2 =
        ::testing::Combine(::testing::Values(ioShape2), ::testing::Values(mvnShape2), ::testing::Values(true),
                           ::testing::Values(true), ::testing::ValuesIn(precision));

INSTANTIATE_TEST_SUITE_P(smoke_MvnInternalReshape2, MvnInternalReshapeTestCommon, testParams2,
                         MvnInternalReshapeTestCommon::getTestCaseName);

}  // namespace
