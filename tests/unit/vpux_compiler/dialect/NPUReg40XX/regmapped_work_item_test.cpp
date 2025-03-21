//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg40XX_WorkItemTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuWorkItem, vpux::NPUReg40XX::Descriptors::WorkItem> {};

#define TEST_NPU4_WORKITEM_REG_FIELD(FieldType, DescriptorMember)                                           \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg40XX_WorkItemTest, FieldType, vpux::NPUReg40XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU4_WORKITEM_REG_FIELD(desc_ptr, wi_desc_ptr)
TEST_NPU4_WORKITEM_REG_FIELD(wi_unit, unit)
TEST_NPU4_WORKITEM_REG_FIELD(wi_sub_unit, sub_unit)
TEST_NPU4_WORKITEM_REG_FIELD(next_workitem_idx, next_workitem_idx)

TEST_F(NPUReg40XX_WorkItemTest, WorkItemTest) {
    // Could not be tested through macro as this field is an enum
    const auto value = npu40xx::nn_public::VpuWorkItem::VpuTaskType::UNKNOWN;
    actual.write<vpux::NPUReg40XX::Fields::wi_type>(value);
    const auto actualValue =
            static_cast<npu40xx::nn_public::VpuWorkItem::VpuTaskType>(actual.read<vpux::NPUReg40XX::Fields::wi_type>());
    EXPECT_EQ(actualValue, value);

    reference.type = value;
    ASSERT_TRUE(isContentEqual());
}
