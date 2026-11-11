//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

#include <npu_60xx_nnrt.hpp>

using namespace npu60xx;

class NPUReg60XX_VpuBarrierConfigCount :
        public NPUReg_RegisterUnitBase<nn_public::VpuBarrierCountConfig,
                                       vpux::NPUReg60XX::Descriptors::VpuBarrierCountConfig> {};

#define TEST_NPU6_BAR_CFG_REG_FIELD(FieldType, DescriptorMember)                                                     \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_VpuBarrierConfigCount, FieldType, vpux::NPUReg60XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU6_BAR_CFG_REG_FIELD(next_same_id_, next_same_id_)
TEST_NPU6_BAR_CFG_REG_FIELD(producer_count_, producer_count_)
TEST_NPU6_BAR_CFG_REG_FIELD(consumer_count_, consumer_count_)
TEST_NPU6_BAR_CFG_REG_FIELD(real_id_, real_id_)

TEST_F(NPUReg60XX_VpuBarrierConfigCount, BarrierConfigCount) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFF;
    actual.write<vpux::NPUReg60XX::Fields::barcfg_pad_3_>(value);
    const auto actualValue = actual.read<vpux::NPUReg60XX::Fields::barcfg_pad_3_>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad_, 0xFF, 3);

    ASSERT_TRUE(isContentEqual());
}
