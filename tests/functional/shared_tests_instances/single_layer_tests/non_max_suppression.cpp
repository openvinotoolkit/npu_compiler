//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include "single_op_tests/non_max_suppression.hpp"
#include <random>
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "shared_tests_instances/vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class NmsLayerTestCommon : public NmsLayerTest, virtual public VpuOv2LayerTest {
public:
protected:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        const auto& ref = expected[0];
        const auto& out = actual[0];
        size_t ref_rows = ref.get_shape()[0];
        size_t out_rows = out.get_shape()[0];
        size_t cols = ref.get_shape()[1];  // always should be 3

        size_t min_rows = std::min(ref_rows, out_rows);

        if (ref.get_element_type() == ov::element::i32) {
            const auto* ref_data = ref.data<int32_t>();
            const auto* out_data = out.data<int32_t>();
            for (size_t i = 0; i < min_rows * cols; ++i) {
                ASSERT_EQ(ref_data[i], out_data[i]) << "Mismatch at index " << i;
            }
        } else if (ref.get_element_type() == ov::element::f32) {
            const auto* ref_data = ref.data<float>();
            const auto* out_data = out.data<float>();
            for (size_t i = 0; i < min_rows * cols; ++i) {
                ASSERT_NEAR(ref_data[i], out_data[i], 1e-3) << "Mismatch at index " << i;
            }
        } else {
            FAIL() << "Unsupported element type for NMS output comparison";
        }
    }

    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }

    void SetUp() override {
        InputShapeParams inShapeParams;
        InputTypes inputTypes;
        int maxOutBoxesPerClass;
        float iouThr, scoreThr, softNmsSigma;
        op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
        op::v9::NonMaxSuppression::BoxEncodingType boxEncoding_v9;
        bool sortResDescend;
        element::Type outType;
        std::tie(inShapeParams, inputTypes, maxOutBoxesPerClass, iouThr, scoreThr, softNmsSigma, boxEncoding,
                 sortResDescend, outType, VpuOv2LayerTest::targetDevice) = this->GetParam();

        boxEncoding_v9 = boxEncoding == op::v5::NonMaxSuppression::BoxEncodingType::CENTER
                                 ? op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                                 : op::v9::NonMaxSuppression::BoxEncodingType::CORNER;

        size_t numBatches, numBoxes, numClasses;
        std::tie(numBatches, numBoxes, numClasses) = inShapeParams;

        ov::element::Type paramsType, maxBoxType, thrType;
        std::tie(paramsType, maxBoxType, thrType) = inputTypes;

        const std::vector<size_t> boxesShape{numBatches, numBoxes, 4}, scoresShape{numBatches, numClasses, numBoxes};
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({boxesShape, scoresShape}));
        ov::ParameterVector params;
        for (const auto& shape : VpuOv2LayerTest::inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(paramsType, shape));
        }
        auto maxOutBoxesPerClassNode = std::make_shared<ov::op::v0::Constant>(
                maxBoxType, ov::Shape{}, std::vector<int32_t>{static_cast<int32_t>(maxOutBoxesPerClass)});
        auto iouThrNode = std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{iouThr});
        auto scoreThrNode = std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{scoreThr});
        auto softNmsSigmaNode =
                std::make_shared<ov::op::v0::Constant>(thrType, ov::Shape{}, std::vector<float>{softNmsSigma});

        auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0], params[1], maxOutBoxesPerClassNode,
                                                                   iouThrNode, scoreThrNode, softNmsSigmaNode,
                                                                   boxEncoding_v9, sortResDescend, outType);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(nms, params, "NMS");
    }
};

TEST_P(NmsLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(NmsLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {

const std::vector<ov::test::InputShapeParams> inShapeParams = {
        ov::test::InputShapeParams{1, 80, 1},    // standard params usage 90% of conformance tests
        ov::test::InputShapeParams{1, 40, 20},   // 1 usage style
        ov::test::InputShapeParams{3, 30, 180},  // for check remain posibility
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 15};
const std::vector<float> iouThreshold = {0.3f, 0.7f};
const std::vector<float> scoreThreshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f};  //  0.5f case - Tracking number [E#169560]
const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
};
const std::vector<bool> sortResDesc = {false};  //  true case - Tracking number [E#167730]
const std::vector<ov::element::Type> outType = {ov::element::i32};
std::vector<ov::element::Type> paramsType = {
        ov::element::f32,
};
std::vector<ov::element::Type> maxBoxType = {
        ov::element::i32,
};
std::vector<ov::element::Type> thrType = {
        ov::element::f16,
};

const auto nmsParams = ::testing::Combine(
        ::testing::ValuesIn(inShapeParams),
        ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                           ::testing::ValuesIn(thrType)),
        ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold),
        ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
        ::testing::ValuesIn(outType), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_NmsLayerTest, NmsLayerTestCommon, nmsParams, NmsLayerTestCommon::getTestCaseName);

const std::vector<ov::test::InputShapeParams> inShapeParamsSmoke = {ov::test::InputShapeParams{2, 9, 12}};
const std::vector<int32_t> maxOutBoxPerClassSmoke = {5};
const std::vector<float> iouThresholdSmoke = {0.3f};
const std::vector<float> scoreThresholdSmoke = {0.3f};
const std::vector<float> sigmaThresholdSmoke = {0.0f};
const std::vector<ov::op::v5::NonMaxSuppression::BoxEncodingType> encodTypeSmoke = {
        ov::op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const auto nmsParamsSmoke =
        testing::Combine(testing::ValuesIn(inShapeParamsSmoke),
                         ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                                            ::testing::ValuesIn(thrType)),
                         ::testing::ValuesIn(maxOutBoxPerClassSmoke), ::testing::ValuesIn(iouThresholdSmoke),
                         ::testing::ValuesIn(scoreThresholdSmoke), ::testing::ValuesIn(sigmaThresholdSmoke),
                         ::testing::ValuesIn(encodTypeSmoke), ::testing::ValuesIn(sortResDesc),
                         ::testing::ValuesIn(outType), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_NmsLayerTest, NmsLayerTestCommon, nmsParamsSmoke,
                         NmsLayerTestCommon::getTestCaseName);

const std::vector<ov::test::InputShapeParams> customInShapeParamsSmoke = {ov::test::InputShapeParams{1, 76726, 1}};
const std::vector<int32_t> customMaxOutBoxPerClassSmoke = {100};
const std::vector<float> customIouThresholdSmoke = {0.5f};
const std::vector<float> customScoreThresholdSmoke = {0.39990234375f};
const auto nmsCustomParamsSmoke = testing::Combine(
        testing::ValuesIn(customInShapeParamsSmoke),
        ::testing::Combine(::testing::ValuesIn(paramsType), ::testing::ValuesIn(maxBoxType),
                           ::testing::ValuesIn(thrType)),
        ::testing::ValuesIn(customMaxOutBoxPerClassSmoke), ::testing::ValuesIn(customIouThresholdSmoke),
        ::testing::ValuesIn(customScoreThresholdSmoke), ::testing::ValuesIn(sigmaThresholdSmoke),
        ::testing::ValuesIn(encodTypeSmoke), ::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(outType),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_custom_NmsLayerTest, NmsLayerTestCommon, nmsCustomParamsSmoke,
                         NmsLayerTestCommon::getTestCaseName);  // Tracking number [E#172848]
}  // namespace
