//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Functional test exercising the weight-packing feature.
//
// Two instantiations:
//   - convWeightPacking_u2: packing path with u2 weights.
//   - convWeightPacking_i4: packing path with i4 weights.

#include "openvino/opsets/opset6_decl.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace {

struct WeightPackingCase {
    size_t inputChannels;
    size_t outputChannels;
    int8_t zp;       // uniform zero-point applied to all OC (PPE path, no ZP table)
    float scaleMin;  // per-channel scales range from scaleMin to scaleMax
    float scaleMax;
    int8_t weightLo;  // lower bound for the synthetic weight payload
    int8_t weightHi;  // upper bound for the synthetic weight payload
};

template <typename T>
std::vector<T> generateRange(size_t count, T lo, T hi) {
    std::vector<T> values;
    values.reserve(count);
    if (lo == hi || count <= 1) {
        values.assign(count, lo);
        return values;
    }
    for (size_t i = 0; i < count; ++i) {
        values.push_back(static_cast<T>(lo + static_cast<T>((hi - lo) * i / (count - 1))));
    }
    return values;
}

// Parameters: <test case, weight element type, compute precision, target device>
using ConvWeightPackingTestParams = std::tuple<WeightPackingCase, ov::element::Type, ov::element::Type, std::string>;

class ConvWeightPackingTest : public testing::WithParamInterface<ConvWeightPackingTestParams>, public VpuOv2LayerTest {
public:
    void SetUp() override {
        const auto& [testCase, weightType, computePrecision, device] = GetParam();
        targetDevice = device;

        const ov::Shape inputShape{1, testCase.inputChannels, 8, 8};
        const ov::Shape weightsShape{testCase.outputChannels, testCase.inputChannels, 3, 3};
        const ov::Shape perOCShape{testCase.outputChannels, 1, 1, 1};

        const size_t totalWeights = weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3];
        const auto weightsData = generateRange<int8_t>(totalWeights, testCase.weightLo, testCase.weightHi);

        auto intWeights = std::make_shared<ov::op::v0::Constant>(weightType, weightsShape, weightsData);
        auto fpWeights = std::make_shared<ov::opset6::Convert>(intWeights, computePrecision);

        // Uniform ZP (same value for all OC) → areAllZeroPointsEqual() == true in the compiler
        // → no ZP table created → ZP folded into PPE bias. This is the simpler, well-tested path
        const auto zpVec = std::vector<int8_t>(testCase.outputChannels, testCase.zp);
        auto intZp = std::make_shared<ov::op::v0::Constant>(weightType, perOCShape, zpVec);
        auto fpZp = std::make_shared<ov::opset6::Convert>(intZp, computePrecision);
        auto zpSubtracted = std::make_shared<ov::opset6::Subtract>(fpWeights, fpZp);

        // Per-OC scales ensure the scale table is exercised.
        const auto scaleVec = generateRange<float>(testCase.outputChannels, testCase.scaleMin, testCase.scaleMax);
        auto fpScale = std::make_shared<ov::op::v0::Constant>(computePrecision, perOCShape, scaleVec);
        auto dequantizedWeights = std::make_shared<ov::opset6::Multiply>(zpSubtracted, fpScale);

        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(computePrecision, ov::Shape(inputShape))};

        auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], dequantizedWeights->output(0),
                                                              ov::Strides{1, 1},         // strides
                                                              ov::CoordinateDiff{0, 0},  // pads_begin
                                                              ov::CoordinateDiff{0, 0},  // pads_end
                                                              ov::Strides{1, 1}          // dilations
        );

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, params, "ConvWeightPacking");

        // Accuracy tolerance
        rel_threshold = 0.1f;
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvWeightPackingTestParams> obj) {
        const auto& [testCase, weightType, computePrecision, _] = obj.param;
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << "_"
               << "IC=" << testCase.inputChannels << "_OC=" << testCase.outputChannels << "_"
               << "ZP=" << static_cast<int>(testCase.zp) << "_"
               << "scale=" << testCase.scaleMin;
        if (testCase.scaleMin != testCase.scaleMax) {
            result << "_to_" << testCase.scaleMax;
        }
        result << "_WeightType=" << weightType.get_type_name() << "_precision=" << computePrecision.get_type_name();
        return result.str();
    }
};

}  // namespace
