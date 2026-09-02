//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/type/float16.hpp>
#include <openvino/opsets/opset1_decl.hpp>
#include <openvino/opsets/opset3_decl.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"

namespace ov::test {

struct MatMulInputBroadCastTestTestParams {
    ov::Shape input1Shape;
    ov::Shape input2Shape;
};

class MatMulInputBroadCastTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MatMulInputBroadCastTestTestParams> {
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == funcInputs.size(),
                        "Input shapes number does not match with inputs number");

        auto createAndFillTensor = [](ov::Shape inputStaticShape) -> ov::Tensor {
            auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
            const auto totalSize = std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), size_t{1},
                                                   std::multiplies<size_t>());
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            const int64_t upperBound = 127;
            const int64_t lowerBound = -127;
            const int64_t dataRange = upperBound - lowerBound + 1;
            for (size_t i = 0; i < totalSize; i++) {
                inputData[i] = static_cast<ov::float16>((i % dataRange) - upperBound);
            }
            return inputTensor;
        };

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = createAndFillTensor(inputShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode =
                std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::opset1::Reshape>(
                std::make_shared<ov::opset1::Reshape>(param, constNode, false));
        return reshape;
    }

    std::shared_ptr<ov::Node> buildBroadCast(const ov::Output<ov::Node>& param,
                                             const std::vector<size_t>& broadCastShape) {
        auto constNode = std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64,
                                                                ov::Shape{broadCastShape.size()}, broadCastShape);
        const auto broadCast = std::make_shared<ov::opset3::Broadcast>(param, constNode);
        return broadCast;
    }

    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const auto testParams = GetParam();

        const auto input1Shape = testParams.input1Shape;
        const auto input2Shape = testParams.input2Shape;

        init_input_shapes(ov::test::static_shapes_to_test_representation({input1Shape, input2Shape}));

        const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));

        std::vector<size_t> broadCastShape;
        broadCastShape.push_back(input2Shape[0]);
        broadCastShape.push_back(input2Shape[1]);
        broadCastShape.push_back(input1Shape[1] / input2Shape[1]);
        broadCastShape.push_back(input2Shape[3]);
        broadCastShape.push_back(input2Shape[4]);
        const auto broadCast = buildBroadCast(input2, broadCastShape);

        std::vector<size_t> targetShape = {broadCastShape[0], broadCastShape[1] * broadCastShape[2], broadCastShape[3],
                                           broadCastShape[4]};
        const auto reshape = buildReshape(broadCast->output(0), targetShape);

        const auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, reshape, false, true);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2}, "MatMulInputBroadCastTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulInputBroadCastTestTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(MatMulInputBroadCastTestCommon, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulInputBroadCast_NPU4000, MatMulInputBroadCastTestCommon,
                         ::testing::ValuesIn({MatMulInputBroadCastTestTestParams{{1, 24, 1, 64}, {1, 8, 1, 1024, 64}}}),
                         MatMulInputBroadCastTestCommon::getTestCaseName);

}  // namespace ov::test
