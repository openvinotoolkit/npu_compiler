//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include <openvino/core/type/element_type.hpp>
#include "common/print_test_case_name.hpp"
#include "pretty_test_arguments.hpp"
#include "shared_test_classes/single_op/non_max_suppression.hpp"
#include "shared_tests_instances/vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

using DynamicNmsParams =
        std::tuple<std::tuple<ov::test::InputShape, ov::test::InputShape>,  // Shapes for 1st and 2nd inputs
                   InputTypes,                                              // Input precisions
                   int32_t,                                                 // Max output boxes per class
                   float,                                                   // IOU threshold
                   float,                                                   // Score threshold
                   float,                                                   // Soft NMS sigma
                   ov::op::v5::NonMaxSuppression::BoxEncodingType,          // Box encoding
                   bool,                                                    // Sort result descending
                   ov::element::Type,                                       // Output type
                   std::string>;                                            // Device name

class DynamicNmsLayerTest : public testing::WithParamInterface<DynamicNmsParams>, public VpuOv2LayerTest {
public:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size()) << "Number of output tensors mismatch";

        // selected indices compare
        compareTensors(expected[0], actual[0]);
        // selected scores compare
        compareTensors(expected[1], actual[1]);
    }
    void compareTensors(const ov::Tensor& referenceTensor, const ov::Tensor& outputTensor) {
        const auto& referenceShape = referenceTensor.get_shape();
        const auto& outputShape = outputTensor.get_shape();

        ASSERT_EQ(outputShape.size(), 2);
        ASSERT_EQ(outputShape[1], 3) << "Number of columns expected to be 3";
        ASSERT_EQ(outputShape[1], referenceShape[1]) << "Mismatch in number of columns";
        if (referenceShape[0] > 0) {
            ASSERT_GT(outputShape[0], 0) << "Reference tensor has boxes, expected output tensor to be non-empty";
        }

        switch (referenceTensor.get_element_type()) {
        case ov::element::i32:
            compareTypedTensors<int32_t>(referenceTensor, outputTensor);
            break;
        case ov::element::i64:
            compareTypedTensors<int64_t>(referenceTensor, outputTensor);
            break;
        case ov::element::f32:
            compareTypedTensors<float>(referenceTensor, outputTensor);
            break;
        default:
            FAIL() << "Unsupported element type for NMS output comparison";
        }
    }

    void SetUp() override {
        auto [inputShapes, inputTypes, maxOutBoxesPerClass, iouThreshold, scoreThreshold, softNmsSigma, boxEncoding,
              sortResDescend, outType, targetDevice] = this->GetParam();
        VpuOv2LayerTest::targetDevice = targetDevice;
        auto [inBoxesShape, inScoresShape] = inputShapes;
        VpuOv2LayerTest::init_input_shapes({inBoxesShape, inScoresShape});

        auto boxEncodingV9 = boxEncoding == op::v5::NonMaxSuppression::BoxEncodingType::CENTER
                                     ? op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                                     : op::v9::NonMaxSuppression::BoxEncodingType::CORNER;
        auto [paramsType, maxBoxType, thresholdType] = inputTypes;
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(paramsType, VpuOv2LayerTest::inputDynamicShapes[0]),
                std::make_shared<ov::op::v0::Parameter>(paramsType, VpuOv2LayerTest::inputDynamicShapes[1])};
        auto maxOutBoxesPerClassNode = std::make_shared<ov::op::v0::Constant>(
                maxBoxType, ov::Shape{}, std::vector<int32_t>{static_cast<int32_t>(maxOutBoxesPerClass)});
        auto iouThresholdNode =
                std::make_shared<ov::op::v0::Constant>(thresholdType, ov::Shape{}, std::vector<float>{iouThreshold});
        auto scoreThresholdNode =
                std::make_shared<ov::op::v0::Constant>(thresholdType, ov::Shape{}, std::vector<float>{scoreThreshold});
        auto softNmsSigmaNode =
                std::make_shared<ov::op::v0::Constant>(thresholdType, ov::Shape{}, std::vector<float>{softNmsSigma});
        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(
                params[0], params[1], maxOutBoxesPerClassNode, iouThresholdNode, scoreThresholdNode, softNmsSigmaNode,
                boxEncodingV9, sortResDescend, outType);

        VpuOv2LayerTest::function = std::make_shared<ov::Model>(nms, params, "NMS");
    }

    template <typename T>
    static void compareTypedTensors(const ov::Tensor& referenceTensor, const ov::Tensor& outputTensor) {
        size_t rows = std::min(outputTensor.get_shape()[0], referenceTensor.get_shape()[0]);
        size_t cols = outputTensor.get_shape()[1];
        const auto* referenceData = referenceTensor.data<T>();
        const auto* outputData = outputTensor.data<T>();
        for (size_t i = 0; i < rows * cols; ++i) {
            if (std::is_floating_point<T>::value) {
                ASSERT_NEAR(referenceData[i], outputData[i], 1e-3) << "Mismatch at index " << i;
            } else {
                ASSERT_EQ(referenceData[i], outputData[i]) << "Mismatch at index " << i;
            }
        }
    }
};

TEST_P(DynamicNmsLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(DynamicNmsLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

TEST_P(DynamicNmsLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
TEST_P(DynamicNmsLayerTest, NPU5020_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {

std::vector<std::tuple<ov::test::InputShape, ov::test::InputShape>> dynamicInputShapes = {
        {generateTestShape(1, 128_Dyn, 4), generateTestShape(1, 1, 128_Dyn)}};

std::vector<ov::element::Type> paramsType = {
        ov::element::f32,
};
std::vector<ov::element::Type> maxBoxType = {
        ov::element::i32,
};
std::vector<ov::element::Type> thresholdType = {
        ov::element::f16,
};

const std::vector<int32_t> maxOutBoxPerClass = {std::numeric_limits<int32_t>::max()};

const std::vector<float> iouThreshold = {0.3f, 0.7f};
const std::vector<float> scoreThreshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f};  //  0.5f case - Tracking number [E#169560]

const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
};
const std::vector<bool> sortResDesc = {false};  //  true case - Tracking number [E#167730]
const std::vector<ov::element::Type> outType = {ov::element::i64};

const auto nmsDynamicShapes = testing::Combine(
        ::testing::ValuesIn(dynamicInputShapes),
        ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                           ::testing::ValuesIn(thresholdType)),
        ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold),
        ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_DynamicNms, DynamicNmsLayerTest, nmsDynamicShapes, PrintTestCaseName());

}  // namespace
