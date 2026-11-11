//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

#include <npu_60xx_nnrt.hpp>

using namespace npu60xx;

class NPUReg60XX_NpuActKernelRangeTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuActKernelRange, vpux::NPUReg60XX::Descriptors::VpuActKernelRange> {
};

#define TEST_NPU6_ACTKERNELRANGE_REG_FIELD(FieldType, DescriptorMember)                                              \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_NpuActKernelRangeTest, FieldType, vpux::NPUReg60XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU6_ACTKERNELRANGE_REG_FIELD(kernel_entry, kernel_entry)
TEST_NPU6_ACTKERNELRANGE_REG_FIELD(text_window_base, text_window_base)
TEST_NPU6_ACTKERNELRANGE_REG_FIELD(code_size, code_size)
TEST_NPU6_ACTKERNELRANGE_REG_FIELD(kernel_invo_count, kernel_invo_count)

TEST_F(NPUReg60XX_NpuActKernelRangeTest, typeTest) {
    // Field has enum type that can not be converted to uint by the macro as other fields
    const auto value = nn_public::VpuActWLType::WL_DEBUG;
    actual.write<vpux::NPUReg60XX::Fields::type>(value);
    const auto actualValue = static_cast<nn_public::VpuActWLType>(actual.read<vpux::NPUReg60XX::Fields::type>());
    EXPECT_EQ(actualValue, value);

    reference.type = value;
    ASSERT_TRUE(isContentEqual());
}
