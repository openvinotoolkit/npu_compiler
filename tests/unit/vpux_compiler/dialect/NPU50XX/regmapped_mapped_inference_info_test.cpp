//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

#include <cstring>
#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg50XX_MappedInferenceInfoTest :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuManagedMappedInferenceInfo,
                                       vpux::NPUReg50XX::Descriptors::VpuManagedMappedInferenceInfo> {};

#define TEST_NPU5_MI_INFO_REG_FIELD(FieldType, DescriptorMember)                                                       \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_MappedInferenceInfoTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR1_MMII_tasks_ref_info, tasks_ref_info.reserved1)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR2_MMII_tasks_ref_info, tasks_ref_info.reserved2)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR3_MMII_tasks_ref_info, tasks_ref_info.reserved3)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceAddr_MMII_tasks_ref_info, tasks_ref_info.address)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceCount_MMII_tasks_ref_info, tasks_ref_info.count)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR1_MMII_vb_mapping, vb_mapping.reserved1)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR2_MMII_vb_mapping, vb_mapping.reserved2)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR3_MMII_vb_mapping, vb_mapping.reserved3)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceAddr_MMII_vb_mapping, vb_mapping.address)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceCount_MMII_vb_mapping, vb_mapping.count)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR1_MMII_barrier_producer_ref_offsets, barrier_producer_ref_offsets.reserved1)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR2_MMII_barrier_producer_ref_offsets, barrier_producer_ref_offsets.reserved2)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR3_MMII_barrier_producer_ref_offsets, barrier_producer_ref_offsets.reserved3)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceAddr_MMII_barrier_producer_ref_offsets, barrier_producer_ref_offsets.address)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceCount_MMII_barrier_producer_ref_offsets, barrier_producer_ref_offsets.count)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR1_MMII_barrier_consumer_ref_offsets, barrier_consumer_ref_offsets.reserved1)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR2_MMII_barrier_consumer_ref_offsets, barrier_consumer_ref_offsets.reserved2)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceR3_MMII_barrier_consumer_ref_offsets, barrier_consumer_ref_offsets.reserved3)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceAddr_MMII_barrier_consumer_ref_offsets, barrier_consumer_ref_offsets.address)
TEST_NPU5_MI_INFO_REG_FIELD(taskReferenceCount_MMII_barrier_consumer_ref_offsets, barrier_consumer_ref_offsets.count)
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_0, ref_info_base_vars[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_1, ref_info_base_vars[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_2, ref_info_base_vars[2])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_3, ref_info_base_vars[3])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_4, ref_info_base_vars[4])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_vars_5, ref_info_base_vars[5])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_0, ref_info_base_invars[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_1, ref_info_base_invars[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_2, ref_info_base_invars[2])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_3, ref_info_base_invars[3])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_4, ref_info_base_invars[4])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_invars_5, ref_info_base_invars[5])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_0, ref_info_base_akr[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_1, ref_info_base_akr[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_2, ref_info_base_akr[2])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_3, ref_info_base_akr[3])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_4, ref_info_base_akr[4])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_akr_5, ref_info_base_akr[5])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_0, ref_info_base_aki[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_1, ref_info_base_aki[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_2, ref_info_base_aki[2])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_3, ref_info_base_aki[3])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_4, ref_info_base_aki[4])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_aki_5, ref_info_base_aki[5])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_dma_from_ddr_0, ref_info_base_dma_from_ddr[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_dma_from_ddr_1, ref_info_base_dma_from_ddr[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_dma_from_cmx_0, ref_info_base_dma_from_cmx[0])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_dma_from_cmx_1, ref_info_base_dma_from_cmx[1])
TEST_NPU5_MI_INFO_REG_FIELD(MMII_ref_info_base_media, ref_info_base_media)
