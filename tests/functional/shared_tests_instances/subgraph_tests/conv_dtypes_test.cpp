//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_dtypes_test.hpp"

namespace ov {
namespace test {

class ConvDtypesTestCommon : public ConvDtypesTest {};

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::Shape> kDcimShapes = {ov::Shape{1, 128, 32, 32}};
const std::vector<ov::Shape> kSclShapes = {ov::Shape{1, 128, 2, 2}};

// --- U8 input, U8 filter ---

// I U8, W U8 -> O FP16, per-tensor
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U8InU8Filter_PerTensor, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(256),               // input levels
                                                               ::testing::Values(ov::element::f16),  // output
                                                               ::testing::Values(256),               // output levels
                                                               ::testing::Values(false)),            // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// I U8, W U8 -> O FP16, per-axis
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U8InU8Filter_PerAxis, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(256),               // input levels
                                                               ::testing::Values(ov::element::f16),  // output
                                                               ::testing::Values(256),               // output levels
                                                               ::testing::Values(true)),             // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// I U8, W U8 -> O U16, per-tensor
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U8InU8Filter_OutU16_PerTensor, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(256),               // input levels
                                                               ::testing::Values(ov::element::u16),  // output
                                                               ::testing::Values(65536),             // output levels
                                                               ::testing::Values(false)),            // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// --- U16 input, U8 filter ---

// I U16, W U8 -> O FP16, per-tensor (DCIM shape)
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U16InU8Filter_PerTensor, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(65536),             // input levels
                                                               ::testing::Values(ov::element::f16),  // output
                                                               ::testing::Values(256),               // output levels
                                                               ::testing::Values(false)),            // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// I U16, W U8 -> O U16, per-tensor
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U16InU8Filter_OutU16_PerTensor, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(65536),             // input levels
                                                               ::testing::Values(ov::element::u16),  // output
                                                               ::testing::Values(65536),             // output levels
                                                               ::testing::Values(false)),            // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// I U16, W U8 -> O U8, per-tensor
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U16InU8Filter_OutU8_PerTensor, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),  // filter
                                                               ::testing::Values(65536),            // input levels
                                                               ::testing::Values(ov::element::u8),  // output
                                                               ::testing::Values(256),              // output levels
                                                               ::testing::Values(false)),           // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kDcimShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

// I U8, W U8 -> O U16, per-tensor (non-DCIM-aligned width forces SCL execution)
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conv_U8InU8Filter_PerTensor_SCL_Execution, ConvDtypesTestCommon,
                         ::testing::Combine(::testing::Combine(::testing::Values(ov::element::u8),   // filter
                                                               ::testing::Values(256),               // input levels
                                                               ::testing::Values(ov::element::u16),  // output
                                                               ::testing::Values(65536),             // output levels
                                                               ::testing::Values(false)),            // per-axis
                                            ::testing::Values(ov::element::f16), ::testing::ValuesIn(kSclShapes)),
                         ConvDtypesTestCommon::getTestCaseName);

}  // namespace
