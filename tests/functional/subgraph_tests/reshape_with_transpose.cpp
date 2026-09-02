//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::test {

class ReshapeWithTransposeTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        const auto transposeOrder = GetParam();
        const ov::Shape lhsInputShape = {1, 768, 14, 14};
        init_input_shapes(static_shapes_to_test_representation({lhsInputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto add = buildAdd(params.at(0));

        std::vector<size_t> newShape;
        newShape.push_back(lhsInputShape[0]);
        newShape.push_back(lhsInputShape[1]);
        newShape.push_back(lhsInputShape[2] * lhsInputShape[3]);
        const auto reshape = buildReshape(add, newShape);

        const auto lhsTranspose = buildTranspose(reshape, transposeOrder);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lhsTranspose)};

        function = std::make_shared<ov::Model>(results, params, "ReshapeTranspose");
        rel_threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                std::make_shared<ov::op::v1::Reshape>(param, constNode, false));
        return reshape;
    }

    std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& lhs) {
        const auto inShape = lhs.get_shape();
        const auto constShape = ov::Shape{1, inShape.at(1), 1, 1};
        std::vector<float> values(inShape.at(1), 1.f);
        const auto biasConst = ov::op::v0::Constant::create(ov::element::f32, constShape, values);
        return std::make_shared<ov::op::v1::Add>(lhs, biasConst);
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

TEST_P(ReshapeWithTransposeTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ReshapeWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_transpose_add, ReshapeWithTransposeTest_NPU3720, ::testing::ValuesIn(transposes),
                         ReshapeWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace ov::test
