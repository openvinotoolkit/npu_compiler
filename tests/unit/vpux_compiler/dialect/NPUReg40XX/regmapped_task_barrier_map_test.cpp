//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"

#include <cstring>
#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg40XX_TaskBarrierMap :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuTaskBarrierMap,
                                       vpux::NPUReg40XX::Descriptors::VpuTaskBarrierMap> {};

#define TEST_NPU4_TASK_BAR_MAP_REG_FIELD(FieldType, DescriptorMember)                                         \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg40XX_TaskBarrierMap, FieldType, vpux::NPUReg40XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_next_same_id, next_same_id)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_producer_count, producer_count)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_consumer_count, consumer_count)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_real_id, real_id)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_work_item_idx, work_item_idx)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_enqueue_count, enqueue_count)
TEST_NPU4_TASK_BAR_MAP_REG_FIELD(tb_reserved_next_enqueue_id, reserved)

TEST_F(NPUReg40XX_TaskBarrierMap, TaskBarrierMap) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFF;
    actual.write<vpux::NPUReg40XX::Fields::tb_pad3>(value);
    const auto actualValue = actual.read<vpux::NPUReg40XX::Fields::tb_pad3>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad0_, 0xFF, 3);

    ASSERT_TRUE(isContentEqual());
}
