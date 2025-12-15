//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/types.hpp"

using namespace npu40xx;

#define CREATE_HW_DMA_DESC(field, value)                                                   \
    [] {                                                                                   \
        nn_public::VpuTaskInfo hwVpuTaskInfoDesc;                                          \
        memset(reinterpret_cast<void*>(&hwVpuTaskInfoDesc), 0, sizeof(hwVpuTaskInfoDesc)); \
        hwVpuTaskInfoDesc.field = value;                                                   \
        return hwVpuTaskInfoDesc;                                                          \
    }()

class NPUReg50XX_VpuTaskInfoTest :
        public MLIR_RegMappedNPUReg50XXUnitBase<nn_public::VpuTaskInfo, vpux::NPUReg50XX::RegMapped_VpuTaskInfoType> {};

TEST_P(NPUReg50XX_VpuTaskInfoTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, nn_public::VpuTaskInfo>> TaskInfoFieldSetNPUReg50XX = {
        {{
                 {"ti_desc_ptr", {{"ti_desc_ptr", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(wi_desc_ptr, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"ti_type", {{"ti_type", 1}}},
         },
         CREATE_HW_DMA_DESC(type, nn_public::VpuWorkItem::VpuTaskType::DMA)},
        {{
                 {"ti_unit", {{"ti_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(unit, 0xFF)},
        {{
                 {"ti_sub_unit", {{"ti_sub_unit", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(sub_unit, 0xFF)},
        {{
                 {"ti_linked_list_nodes", {{"ti_linked_list_nodes", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(linked_list_nodes, 0xFFFF)},
        {{
                 {"ti_descr_ref_offset", {{"ti_descr_ref_offset", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(descr_ref_offset, 0xFFFF)},
        {{
                 {"ti_parent_descr_ref_offset", {{"ti_parent_descr_ref_offset", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(parent_descr_ref_offset, 0xFFFF)},
        {{
                 {"ti_enqueueing_task_config", {{"ti_enqueueing_task_config", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(enqueueing_task_config, 0xFFFF)},
        {{
                 {"ti_work_item_ref", {{"ti_work_item_ref", 0xFFFF}}},
         },
         CREATE_HW_DMA_DESC(work_item_ref, 0xFFFF)},

};

INSTANTIATE_TEST_SUITE_P(NPUReg50XX_MappedRegs, NPUReg50XX_VpuTaskInfoTest,
                         testing::ValuesIn(TaskInfoFieldSetNPUReg50XX));
