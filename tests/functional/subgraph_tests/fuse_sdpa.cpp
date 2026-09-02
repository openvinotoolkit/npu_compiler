//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_ops/rotary_positional_embeddings.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test {
struct SDPAParams {
    ov::Shape inputQ;
    ov::Shape inputK;
    ov::Shape inputV;
    ov::Shape mask;
};

class FuseSDPATestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<SDPAParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SDPAParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        for (size_t i = 0; i < 4; ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 1;
            in_data.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const ov::Shape newShape) {
        auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                std::make_shared<ov::op::v1::Reshape>(param, constNode, false));
        return reshape;
    }

    void SetUp() override {
        inType = outType = ov::element::f32;
        const auto testParams = GetParam();
        const auto inputQShape = testParams.inputQ;
        const auto inputKShape = testParams.inputK;
        const auto inputVShape = testParams.inputV;
        const auto inputMaskShape = testParams.mask;

        init_input_shapes(ov::test::static_shapes_to_test_representation(
                {inputQShape, inputKShape, inputVShape, inputMaskShape}));

        const auto inputQ = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputK = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputV = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(2));
        const auto inputMask = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(3));

        ov::Shape targetShapeQ = {inputQShape[2], inputQShape[3]};
        ov::Shape targetShapeK = {inputKShape[2], inputKShape[3]};
        ov::Shape targetShapeV = {inputVShape[2], inputVShape[3]};
        ov::Shape targetShapeMatMulQK = {inputQShape[0], inputQShape[1], inputQShape[2], inputKShape[2]};
        ov::Shape targetShapeSoftmax = {inputQShape[2], inputKShape[2]};
        ov::Shape targetShapeSoftmaxV = {inputQShape[0], inputQShape[1], inputQShape[2], inputVShape[2]};

        const auto reshapeQ = buildReshape(inputQ, targetShapeQ);
        const auto reshapeK = buildReshape(inputK, targetShapeK);
        const auto reshapeV = buildReshape(inputV, targetShapeV);

        const auto MatMulQK = std::make_shared<ov::op::v0::MatMul>(reshapeQ, reshapeK, false, true);
        const auto reshapeQK = buildReshape(MatMulQK, targetShapeMatMulQK);
        const auto scaleFactor = sqrt(inputQShape[3]);
        const auto scaleShape = ov::Shape{1, 1, 1, 1};
        const auto scale = ov::op::v0::Constant::create(ov::element::f32, scaleShape, {scaleFactor});
        const auto divide = std::make_shared<ov::opset1::Divide>(reshapeQK, scale);
        const auto add = std::make_shared<ov::opset1::Add>(divide, inputMask);
        const auto softmax = std::make_shared<ov::opset1::Softmax>(add, 3);
        const auto reshapeSoftmax = buildReshape(softmax, targetShapeSoftmax);
        const auto matMulSoftmaxV = std::make_shared<ov::op::v0::MatMul>(reshapeSoftmax, reshapeV, false, true);
        const auto reshapeSoftmaxV = buildReshape(matMulSoftmaxV, targetShapeSoftmaxV);
        const auto result = std::make_shared<ov::op::v0::Result>(reshapeSoftmaxV);
        const ov::ResultVector results{result};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{inputQ, inputK, inputV, inputMask},
                                               "FuseSDPATest");
    }
};

TEST_P(FuseSDPATestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(precommit_FuseSDPA, FuseSDPATestCommon,
                         ::testing::ValuesIn({SDPAParams{
                                 {1, 1, 1, 64}, {1, 1, 64, 64}, {1, 1, 64, 64}, {1, 1, 1, 64}}}),
                         FuseSDPATestCommon::getTestCaseName);
}  // namespace ov::test
