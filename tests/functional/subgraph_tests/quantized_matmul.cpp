//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test;

namespace {

// Quantized 4D MatMul using subtract-multiply pattern for weights quantization
//
//       [input]
//          |
//      (matmul) --- (multiply) --- (subtract) --- (convert to f16) -- [weights_i8]
//                        |              |
//                    [scale]      [zero_point]
//

using MatMulQuantizedTestParams = std::tuple<ov::element::Type>;  // computation precision

class MatMulQuantizedSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<MatMulQuantizedTestParams> {
    void SetUp() override {
        const auto& [float_element_type] = GetParam();

        const ov::Shape inputShape0{1, 8, 16, 64};
        const ov::Shape weightsShape{1, 8, 64, 32};

        init_input_shapes(static_shapes_to_test_representation({inputShape0}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(float_element_type, inputDynamicShapes[0])};

        // Create i8 weights
        const size_t totalWeights = ov::shape_size(weightsShape);
        std::vector<int8_t> weights(totalWeights);
        for (size_t i = 0; i < totalWeights; i++) {
            weights[i] = static_cast<int8_t>((i % 127) - 64);
        }

        auto iWeights = std::make_shared<ov::op::v0::Constant>(ov::element::i8, weightsShape, weights);
        auto fWeights = std::make_shared<ov::op::v0::Convert>(iWeights, float_element_type);

        // Create per-channel zero points and subtract from weights
        const size_t numChannels = weightsShape[3];  // Output channels
        std::vector<int8_t> zpPerChannel(numChannels);
        for (size_t i = 0; i < numChannels; i++) {
            zpPerChannel[i] = static_cast<int8_t>((i % 6) - 3);
        }
        auto iZp =
                std::make_shared<ov::op::v0::Constant>(ov::element::i8, ov::Shape{1, 1, 1, numChannels}, zpPerChannel);
        auto fZp = std::make_shared<ov::op::v0::Convert>(iZp, float_element_type);
        auto zeroPointsSubtracted = std::make_shared<ov::op::v1::Subtract>(fWeights, fZp);

        // Create scale and multiply
        std::vector<float> scalePerChannel;
        scalePerChannel.reserve(numChannels);
        std::vector<float> pattern = {1.0, 2.0, 3.0, 4.0};
        for (size_t i = 0; i < numChannels / pattern.size(); i++) {
            scalePerChannel.insert(scalePerChannel.end(), pattern.begin(), pattern.end());
        }
        auto scale = std::make_shared<ov::op::v0::Constant>(float_element_type, ov::Shape{1, 1, 1, numChannels},
                                                            scalePerChannel);
        auto multiplied = std::make_shared<ov::op::v1::Multiply>(zeroPointsSubtracted, scale);

        const auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], multiplied, false, false);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
        function = std::make_shared<ov::Model>(results, params, "MatMulQuantized");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulQuantizedTestParams> obj) {
        const auto& [float_element_type] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "OutputPrec=" << float_element_type << sep;
        return result.str();
    }
};

const auto basicCases = ::testing::Combine(::testing::Values(ov::element::f16));

INSTANTIATE_TEST_SUITE_P(matMulQuantized, MatMulQuantizedSubGraphTestCommon, basicCases,
                         MatMulQuantizedSubGraphTestCommon::getTestCaseName);

}  // namespace
