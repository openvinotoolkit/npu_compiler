// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <openvino/opsets/opset14.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "openvino/opsets/opset1.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/single_op/shape_of.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace ov::test;

namespace {

std::shared_ptr<ov::Node> buildMaxPool(const ov::Output<ov::Node>& param) {
    std::vector<uint64_t> poolStridesVec = {1, 1};
    std::vector<uint64_t> poolKernelVec = {1, 1};
    const ov::Strides poolStrides = {1, 1};
    const ov::Shape padsBegin = {1, 1};
    const ov::Shape padsEnd = {1, 1};
    const ov::Shape poolKernel = {3, 3};
    return std::make_shared<ov::op::v1::MaxPool>(param, poolStrides, padsBegin, padsEnd, poolKernel);
}

using SEPDilatedConvTestParams = std::tuple<ov::Shape,           // input shape
                                            std::vector<size_t>  // dilations
                                            >;
class SEPDilatedConvTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<SEPDilatedConvTestParams> {
    void SetUp() override {
        ov::Shape inputShape;
        std::vector<size_t> dilationsParam;
        std::tie(inputShape, dilationsParam) = GetParam();
        abs_threshold = 0.5f;

        const size_t IC = inputShape[1];
        const size_t KY = 3;
        const size_t KX = 3;

        const ov::Shape weightsShape = ov::Shape{IC, 1, 1, KY, KX};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        std::vector<float> values(IC * 1 * 1 * KX * KY, 0.0f);
        for (std::size_t i = 0; i < values.size(); i++) {
            values[i] = std::sin(i);
        }
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {vpux::checked_cast<int64_t>(dilationsParam[0]),
                                               vpux::checked_cast<int64_t>(dilationsParam[0])};
        const ov::CoordinateDiff pads_end = {vpux::checked_cast<int64_t>(dilationsParam[0]),
                                             vpux::checked_cast<int64_t>(dilationsParam[0])};
        const ov::Strides dilations = {dilationsParam[0], dilationsParam[1]};

        auto maxPool = buildMaxPool(params[0]);

        const auto conv = std::make_shared<ov::opset1::GroupConvolution>(maxPool, weights, strides, pads_begin,
                                                                         pads_end, dilations);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, params, "SEPDilatedConv");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<SEPDilatedConvTestParams> obj) {
        std::vector<size_t> inputShape;
        std::vector<size_t> dilationsParam;
        std::tie(inputShape, dilationsParam) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InShape="
               << "inputShape={" << inputShape.at(0) << ", " << inputShape.at(1) << ", " << inputShape.at(2) << ", "
               << inputShape.at(3) << "}_" << sep;
        result << "Dilations={" << dilationsParam.at(0) << ", " << dilationsParam.at(1) << "}" << sep;
        return result.str();
    }
};

TEST_P(SEPDilatedConvTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = "enable-experimental-se-ptrs-operations=true";
    configuration[ov::intel_npu::tiles.name()] = 2;
    run(Platform::NPU4000);
}

std::vector<ov::Shape> inputSizesM = {{1, 64, 32, 32}, {1, 64, 64, 64}};

std::vector<std::vector<size_t>> dilationValsM = {{1, 1}, {2, 2}, {4, 4}};

const auto basicCasesM = ::testing::Combine(::testing::ValuesIn(inputSizesM), ::testing::ValuesIn(dilationValsM));

INSTANTIATE_TEST_SUITE_P(smoke_SEPDilatedConv, SEPDilatedConvTestCommon, basicCasesM,
                         SEPDilatedConvTestCommon::getTestCaseName);

}  // namespace
