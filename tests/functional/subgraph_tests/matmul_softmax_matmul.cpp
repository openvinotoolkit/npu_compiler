// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/type/float16.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

struct TestParams {
    ov::Shape input1Shape;
    ov::Shape input2Shape;
    ov::Shape input3Shape;
    int64_t softmaxAxis;
};

class MatMulSoftmaxMatMulTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<TestParams> {
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(inputShapes.size() == funcInputs.size(),
                        "Input shapes number does not match with inputs number");

        auto createAndFillTensor = [](ov::Shape inputStaticShape) -> ov::Tensor {
            auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
            const auto totalSize =
                    std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            for (size_t i = 0; i < totalSize; i++) {
                // too large values may cause overflow, so we use sin function to generate values in range [-1, 1]
                inputData[i] = std::sin(i);
            }
            return inputTensor;
        };

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = createAndFillTensor(inputShapes[i]);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const auto testParams = GetParam();

        const auto input1Shape = testParams.input1Shape;
        const auto input2Shape = testParams.input2Shape;
        const auto input3Shape = testParams.input3Shape;
        const auto axis = testParams.softmaxAxis;

        init_input_shapes(ov::test::static_shapes_to_test_representation({input1Shape, input2Shape, input3Shape}));

        const auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(0));
        const auto input2 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(1));
        const auto input3 = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.at(2));

        const auto matmul1 = std::make_shared<ov::op::v0::MatMul>(input1, input2, false, true);
        const auto softMax = std::make_shared<ov::op::v8::Softmax>(matmul1, axis);
        const auto matmul2 = std::make_shared<ov::op::v0::MatMul>(softMax, input3, false, false);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul2)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input1, input2, input3},
                                               "MatMulSoftmaxMatMulTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<TestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };

private:
    const double _relativeThreashold = 0.001;
};

TEST_P(MatMulSoftmaxMatMulTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSoftmaxMatMul, MatMulSoftmaxMatMulTestCommon,
                         ::testing::ValuesIn({TestParams{{64, 4, 49, 32}, {64, 4, 49, 32}, {64, 4, 49, 32}, -1},
                                              TestParams{{1, 32, 1, 128}, {1, 32, 1024, 128}, {1, 32, 1024, 128}, -1},
                                              TestParams{{16, 8, 49, 32}, {16, 8, 49, 32}, {16, 8, 49, 32}, -1},
                                              TestParams{{4, 16, 49, 32}, {4, 16, 49, 32}, {4, 16, 49, 32}, -1}}),
                         MatMulSoftmaxMatMulTestCommon::getTestCaseName);

}  // namespace ov::test
