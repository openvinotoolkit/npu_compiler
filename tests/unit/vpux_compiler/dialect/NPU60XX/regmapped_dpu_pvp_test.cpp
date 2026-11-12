//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

#include <npu_60xx_nnrt.hpp>
#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/npu_reg_types.hpp.inc>

using namespace npu60xx;

class NPUReg60XX_VpuDpuPVP :
        public NPUReg_RegisterUnitBase<nn_public::VpuDpuPVP, vpux::NPUReg60XX::Descriptors::VpuDpuPVP> {};

#define TEST_NPU6_DPU_PVP_REG_FIELD(FieldType, DescriptorMember)                                         \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_VpuDpuPVP, FieldType, vpux::NPUReg60XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU6_DPU_PVP_REG_FIELD(VpuDpuPVP_opFPCount, opFPCount)
TEST_NPU6_DPU_PVP_REG_FIELD(VpuDpuPVP_totalOpCount, totalOpCount)

TEST_F(NPUReg60XX_VpuDpuPVP, VpuDpuPVP) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFF;
    actual.write<vpux::NPUReg60XX::Fields::VpuDpuPVP_reserved_1>(value);
    const auto actualValue = actual.read<vpux::NPUReg60XX::Fields::VpuDpuPVP_reserved_1>();

    EXPECT_EQ(actualValue, value);

    std::memset(&reference.reserved[8], 0xFF, 4);

    ASSERT_TRUE(isContentEqual());
}
