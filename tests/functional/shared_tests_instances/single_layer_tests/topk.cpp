//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class TopKLayerTestCommon : virtual public TopKLayerTest, virtual public VpuOv2LayerTest {};
class TopK11LayerTestCommon : public TopK11LayerTest, virtual public VpuOv2LayerTest {};
class TopKDDRLayerTestCommon : public TopK11LayerTest, virtual public VpuOv2LayerTest {};
class TopKLayerTest_SW_FP32 : public TopKLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(TopKLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopKLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopKDDRLayerTestCommon, NPU4000_SW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopKLayerTest_SW_FP32, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopKLayerTest_SW_FP32, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopK11LayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopK11LayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
class TopK1LayerTest : public TopKLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> inputShape;
        ov::element::Type modelType;
        int64_t keepK, axis;
        ov::op::v3::TopK::Mode mode;
        ov::op::v3::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, modelType, inputShape, targetDevice) = this->GetParam();
        init_input_shapes(inputShape);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());
        auto k = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ov::op::v3::TopK>(
                std::make_shared<ov::op::v3::TopK>(param, k, axis, mode, sort));

        ov::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "TopK");
    }
};

TEST_P(TopK1LayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace test
}  // namespace ov

using ov::test::TopK11LayerTestCommon;
using ov::test::TopK1LayerTest;
using ov::test::TopKDDRLayerTestCommon;
using ov::test::TopKLayerTest_SW_FP32;
using ov::test::TopKLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypeFP16 = {ov::element::f16};
// SI32 data type is currently unsupported by OpenVINO TopK test environment
const std::vector<ov::element::Type> modelTypes = {ov::element::f32 /*, ov::element::i32*/};

const std::vector<int64_t> axes = {0, 1, 2};

const std::vector<int64_t> k = {1, 5, 10};

const std::vector<ov::op::v3::TopK::Mode> modes = {ov::op::v3::TopK::Mode::MIN, ov::op::v3::TopK::Mode::MAX};

const std::vector<ov::op::v3::TopK::SortType> sortTypes = {
        // The implements of SortType::NONE are different.
        // Reference uses std::nth_element and returns k out-of-order values.
        // Kernel returns k data sorted in values. nth_element causes computation increase.
        // ov::op::v3::TopK::SortType::NONE,
        ov::op::v3::TopK::SortType::SORT_INDICES,
        ov::op::v3::TopK::SortType::SORT_VALUES,
};

const auto paramsConfig = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1, 5}), ::testing::ValuesIn(axes), ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypeFP16),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK, TopKLayerTestCommon, paramsConfig, TopKLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK1, TopK1LayerTest, paramsConfig, TopK1LayerTest::getTestCaseName);

const auto paramsConfigPrecommitFP32 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1}), ::testing::ValuesIn(std::vector<int64_t>{2}),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK_FP32, TopKLayerTest_SW_FP32, paramsConfigPrecommitFP32,
                         TopKLayerTest_SW_FP32::getTestCaseName);

// Tiling tests
const std::vector<int64_t> k_Tilling = {1};
const std::vector<int64_t> axes_Tilling = {1};
const std::vector<ov::op::v3::TopK::Mode> modes_Tilling = {ov::op::v3::TopK::Mode::MAX};
const std::vector<ov::op::v3::TopK::SortType> sortTypes_Tilling = {
        ov::op::v3::TopK::SortType::SORT_INDICES,
};
const std::vector<ov::element::Type> modelTypes_Tilling = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_Tilling, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 5, 512, 512}}}))),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

// K=1 asm optimization tests
INSTANTIATE_TEST_SUITE_P(smoke_TopK_K1, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling),
                                            ::testing::ValuesIn(std::vector<int64_t>{2}),
                                            ::testing::ValuesIn(modes_Tilling),
                                            ::testing::ValuesIn(std::vector<ov::op::v3::TopK::SortType>{
                                                    ov::op::v3::TopK::SortType::NONE}),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>{{{1, 42840, 13}}})),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_K300, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(std::vector<int64_t>{300}),
                                            ::testing::ValuesIn(std::vector<int64_t>{1}),
                                            ::testing::ValuesIn(modes_Tilling),
                                            ::testing::ValuesIn(std::vector<ov::op::v3::TopK::SortType>{
                                                    ov::op::v3::TopK::SortType::SORT_VALUES}),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>{{{1, 3600}}})),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_K21, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(std::vector<int64_t>{1}),
                                            ::testing::ValuesIn(std::vector<int64_t>{3}),
                                            ::testing::ValuesIn(modes_Tilling),
                                            ::testing::ValuesIn(std::vector<ov::op::v3::TopK::SortType>{
                                                    ov::op::v3::TopK::SortType::SORT_VALUES}),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>{{{1, 8, 16, 21}}})),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

}  // namespace

namespace {  // opset v11

INSTANTIATE_TEST_SUITE_P(smoke_TopK11, TopK11LayerTestCommon,
                         ::testing::Combine(::testing::Values(1), ::testing::Values(1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_INDICES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{10, 10, 10}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK11LayerTestCommon::getTestCaseName);

// Can't tile, require DDR
INSTANTIATE_TEST_SUITE_P(smoke_TopK11_DDRAccess, TopKDDRLayerTestCommon,
                         ::testing::Combine(::testing::Values(1), ::testing::Values(-1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 5898240}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKDDRLayerTestCommon::getTestCaseName);

}  // namespace
