//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/extract_image_patches.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ExtractImagePatchesLayerTestCommon : public ExtractImagePatchesTest, virtual public VpuOv2LayerTest {};

TEST_P(ExtractImagePatchesLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ExtractImagePatchesLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(ExtractImagePatchesLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::op::PadType;
using namespace ov::test;

namespace {

const std::vector<PadType> paddingType = {PadType::VALID, PadType::SAME_UPPER, PadType::SAME_LOWER};

// FP16
const auto test1ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 1, 10, 10}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),                   // rates
        ::testing::ValuesIn({PadType::VALID}),                                                   // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke1_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test1ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test2ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 1, 10, 10}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{8, 8}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                           // rates
        ::testing::ValuesIn({PadType::VALID}),                                                   // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke2_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test2ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test3ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 1, 10, 10}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{9, 9}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),                   // rates
        ::testing::ValuesIn({PadType::SAME_UPPER}),                                              // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke3_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test3ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test4ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 2, 5, 5}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                         // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                         // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                         // rates
        ::testing::ValuesIn({PadType::VALID}),                                                 // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke4_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test4ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test5ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 64, 26, 26}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                            // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                            // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                            // rates
        ::testing::ValuesIn({PadType::SAME_LOWER}),                                               // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke5_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test5ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test6ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{2, 1, 10, 10}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                           // rates
        ::testing::ValuesIn({PadType::SAME_UPPER, PadType::SAME_LOWER}),                         // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke6_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test6ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test7ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{2, 3, 13, 37}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                           // rates
        ::testing::ValuesIn({PadType::VALID, PadType::SAME_LOWER}),                              // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke7_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test7ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto test8ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{2, 3, 13, 37}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                           // rates
        ::testing::ValuesIn({PadType::SAME_UPPER}),                                              // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke8_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test8ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

// I32
const auto test9ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 64, 26, 26}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                            // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                            // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                            // rates
        ::testing::ValuesIn(paddingType),                                                         // pad type
        ::testing::ValuesIn({ov::element::i32}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke9_ExtractImagePatches, ExtractImagePatchesLayerTestCommon, test9ExtractImagePatchesParams,
                         ExtractImagePatchesTest::getTestCaseName);

const auto testPrecommitExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 10, 10}})}),  // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                           // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),                           // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),                           // rates
        ::testing::ValuesIn({PadType::VALID}),                                                   // pad type
        ::testing::ValuesIn({ov::element::f16}), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ExtractImagePatches, ExtractImagePatchesLayerTestCommon,
                         testPrecommitExtractImagePatchesParams, ExtractImagePatchesTest::getTestCaseName);

}  // namespace
