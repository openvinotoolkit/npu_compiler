//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

#include <cstring>
#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg50XX;

class NPUReg50XX_VpuBarrierConfigCount :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuBarrierCountConfig,
                                       vpux::NPUReg50XX::Descriptors::VpuBarrierCountConfig> {};

#define TEST_NPU5_BAR_CFG_REG_FIELD(FieldType, DescriptorMember)                                                     \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_VpuBarrierConfigCount, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU5_BAR_CFG_REG_FIELD(next_same_id_, next_same_id_)
TEST_NPU5_BAR_CFG_REG_FIELD(producer_count_, producer_count_)
TEST_NPU5_BAR_CFG_REG_FIELD(consumer_count_, consumer_count_)
TEST_NPU5_BAR_CFG_REG_FIELD(real_id_, real_id_)

TEST_F(NPUReg50XX_VpuBarrierConfigCount, BarrierConfigCount) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::barcfg_pad_3_>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::barcfg_pad_3_>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad_, 0xFF, 3);

    ASSERT_TRUE(isContentEqual());
}
