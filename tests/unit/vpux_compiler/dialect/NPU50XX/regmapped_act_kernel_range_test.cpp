//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg50XX;

class NPUReg50XX_NpuActKernelRangeTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuActKernelRange, vpux::NPUReg50XX::Descriptors::VpuActKernelRange> {
};

#define TEST_NPU5_ACTKERNELRANGE_REG_FIELD(FieldType, DescriptorMember)                                              \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_NpuActKernelRangeTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU5_ACTKERNELRANGE_REG_FIELD(kernel_entry, kernel_entry)
TEST_NPU5_ACTKERNELRANGE_REG_FIELD(text_window_base, text_window_base)
TEST_NPU5_ACTKERNELRANGE_REG_FIELD(code_size, code_size)
TEST_NPU5_ACTKERNELRANGE_REG_FIELD(kernel_invo_count, kernel_invo_count)

TEST_F(NPUReg50XX_NpuActKernelRangeTest, typeTest) {
    // Field has enum type that can not be converted to uint by the macro as other fields
    const auto value = nn_public::VpuActWLType::WL_DEBUG;
    actual.write<vpux::NPUReg50XX::Fields::type>(value);
    const auto actualValue = static_cast<nn_public::VpuActWLType>(actual.read<vpux::NPUReg50XX::Fields::type>());
    EXPECT_EQ(actualValue, value);

    reference.type = value;
    ASSERT_TRUE(isContentEqual());
}
