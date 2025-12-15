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

class NPUReg50XX_MappedInferenceTest :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuMappedInference,
                                       vpux::NPUReg50XX::Descriptors::VpuMappedInference> {};

#define TEST_NPU5_MI_REG_FIELD(FieldType, DescriptorMember)                                                        \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_MappedInferenceTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

#define TEST_NPU5_MI_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)                  \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg50XX_MappedInferenceTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg50XX::Registers::ParentRegType,                   \
                                        vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU5_MI_REG_FIELD(miVpuNNRTApiVer, vpu_nnrt_api_ver)
TEST_NPU5_MI_REG_FIELD(miReserved0, reserved0_)
TEST_NPU5_MI_REG_FIELD(miLogAddrDmaHwp, logaddr_dma_hwp_)
TEST_NPU5_MI_REG_FIELD(miTcReserved1, task_storage_counts_.reserved1)
TEST_NPU5_MI_REG_FIELD(miTcReserved2, task_storage_counts_.reserved2)
TEST_NPU5_MI_REG_FIELD(miTcDmaDDRCount, task_storage_counts_.dma_ddr_count)
TEST_NPU5_MI_REG_FIELD(miTcDmaCMXCount, task_storage_counts_.dma_cmx_count)
TEST_NPU5_MI_REG_FIELD(miDPUInvariantCount, task_storage_counts_.dpu_invariant_count)
TEST_NPU5_MI_REG_FIELD(miTcDPUVariantCount, task_storage_counts_.dpu_variant_count)
TEST_NPU5_MI_REG_FIELD(miTcActRangeCount, task_storage_counts_.act_range_count)
TEST_NPU5_MI_REG_FIELD(miTcActInvoCount, task_storage_counts_.act_invo_count)
TEST_NPU5_MI_REG_FIELD(miTcMediaCount, task_storage_counts_.media_count)
TEST_NPU5_MI_REG_FIELD(miTaskStorageSize, task_storage_size_)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_0, dma_tasks_ddr_[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_0, dma_tasks_ddr_[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_0, dma_tasks_ddr_[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_0, dma_tasks_ddr_[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_0, dma_tasks_ddr_[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_1, dma_tasks_ddr_[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_1, dma_tasks_ddr_[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_1, dma_tasks_ddr_[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_1, dma_tasks_ddr_[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_1, dma_tasks_ddr_[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_2, dma_tasks_ddr_[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_2, dma_tasks_ddr_[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_2, dma_tasks_ddr_[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_2, dma_tasks_ddr_[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_2, dma_tasks_ddr_[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_3, dma_tasks_ddr_[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_3, dma_tasks_ddr_[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_3, dma_tasks_ddr_[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_3, dma_tasks_ddr_[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_3, dma_tasks_ddr_[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_4, dma_tasks_ddr_[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_4, dma_tasks_ddr_[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_4, dma_tasks_ddr_[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_4, dma_tasks_ddr_[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_4, dma_tasks_ddr_[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_DDR_5, dma_tasks_ddr_[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_DDR_5, dma_tasks_ddr_[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_DDR_5, dma_tasks_ddr_[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_DDR_5, dma_tasks_ddr_[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_DDR_5, dma_tasks_ddr_[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_0, dma_tasks_cmx_[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_0, dma_tasks_cmx_[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_0, dma_tasks_cmx_[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_0, dma_tasks_cmx_[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_0, dma_tasks_cmx_[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_1, dma_tasks_cmx_[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_1, dma_tasks_cmx_[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_1, dma_tasks_cmx_[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_1, dma_tasks_cmx_[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_1, dma_tasks_cmx_[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_2, dma_tasks_cmx_[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_2, dma_tasks_cmx_[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_2, dma_tasks_cmx_[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_2, dma_tasks_cmx_[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_2, dma_tasks_cmx_[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_3, dma_tasks_cmx_[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_3, dma_tasks_cmx_[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_3, dma_tasks_cmx_[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_3, dma_tasks_cmx_[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_3, dma_tasks_cmx_[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_4, dma_tasks_cmx_[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_4, dma_tasks_cmx_[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_4, dma_tasks_cmx_[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_4, dma_tasks_cmx_[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_4, dma_tasks_cmx_[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DMA_CMX_5, dma_tasks_cmx_[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DMA_CMX_5, dma_tasks_cmx_[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DMA_CMX_5, dma_tasks_cmx_[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DMA_CMX_5, dma_tasks_cmx_[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DMA_CMX_5, dma_tasks_cmx_[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_0, invariants[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_0, invariants[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_0, invariants[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_0, invariants[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_0, invariants[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_1, invariants[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_1, invariants[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_1, invariants[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_1, invariants[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_1, invariants[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_2, invariants[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_2, invariants[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_2, invariants[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_2, invariants[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_2, invariants[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_3, invariants[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_3, invariants[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_3, invariants[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_3, invariants[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_3, invariants[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_4, invariants[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_4, invariants[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_4, invariants[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_4, invariants[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_4, invariants[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_inv_5, invariants[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_inv_5, invariants[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_inv_5, invariants[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_inv_5, invariants[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_inv_5, invariants[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_0, variants[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_0, variants[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_0, variants[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_0, variants[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_0, variants[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_1, variants[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_1, variants[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_1, variants[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_1, variants[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_1, variants[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_2, variants[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_2, variants[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_2, variants[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_2, variants[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_2, variants[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_3, variants[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_3, variants[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_3, variants[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_3, variants[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_3, variants[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_4, variants[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_4, variants[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_4, variants[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_4, variants[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_4, variants[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_DPU_var_5, variants[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_DPU_var_5, variants[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_DPU_var_5, variants[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_DPU_var_5, variants[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_DPU_var_5, variants[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_0, act_kernel_ranges[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_0, act_kernel_ranges[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_0, act_kernel_ranges[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_0, act_kernel_ranges[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_0, act_kernel_ranges[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_1, act_kernel_ranges[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_1, act_kernel_ranges[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_1, act_kernel_ranges[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_1, act_kernel_ranges[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_1, act_kernel_ranges[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_2, act_kernel_ranges[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_2, act_kernel_ranges[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_2, act_kernel_ranges[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_2, act_kernel_ranges[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_2, act_kernel_ranges[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_3, act_kernel_ranges[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_3, act_kernel_ranges[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_3, act_kernel_ranges[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_3, act_kernel_ranges[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_3, act_kernel_ranges[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_4, act_kernel_ranges[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_4, act_kernel_ranges[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_4, act_kernel_ranges[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_4, act_kernel_ranges[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_4, act_kernel_ranges[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_range_5, act_kernel_ranges[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_range_5, act_kernel_ranges[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_range_5, act_kernel_ranges[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_range_5, act_kernel_ranges[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_range_5, act_kernel_ranges[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_0, act_kernel_invocations[0].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_0, act_kernel_invocations[0].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_0, act_kernel_invocations[0].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_0, act_kernel_invocations[0].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_0, act_kernel_invocations[0].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_1, act_kernel_invocations[1].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_1, act_kernel_invocations[1].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_1, act_kernel_invocations[1].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_1, act_kernel_invocations[1].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_1, act_kernel_invocations[1].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_2, act_kernel_invocations[2].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_2, act_kernel_invocations[2].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_2, act_kernel_invocations[2].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_2, act_kernel_invocations[2].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_2, act_kernel_invocations[2].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_3, act_kernel_invocations[3].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_3, act_kernel_invocations[3].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_3, act_kernel_invocations[3].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_3, act_kernel_invocations[3].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_3, act_kernel_invocations[3].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_4, act_kernel_invocations[4].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_4, act_kernel_invocations[4].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_4, act_kernel_invocations[4].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_4, act_kernel_invocations[4].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_4, act_kernel_invocations[4].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ActKernel_invo_5, act_kernel_invocations[5].reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ActKernel_invo_5, act_kernel_invocations[5].reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ActKernel_invo_5, act_kernel_invocations[5].reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ActKernel_invo_5, act_kernel_invocations[5].address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ActKernel_invo_5, act_kernel_invocations[5].count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_MediaTask, media_tasks.reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_MediaTask, media_tasks.reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_MediaTask, media_tasks.reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_MediaTask, media_tasks.address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_MediaTask, media_tasks.count)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_BarrierConfig, barrier_configs.reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_BarrierConfig, barrier_configs.reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_BarrierConfig, barrier_configs.reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_BarrierConfig, barrier_configs.address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_BarrierConfig, barrier_configs.count)

TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_reserved, shv_rt_configs.reserved)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_runtime_entry, shv_rt_configs.runtime_entry)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_act_rt_window_base, shv_rt_configs.act_rt_window_base)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_0, shv_rt_configs.stack_frames[0])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_1, shv_rt_configs.stack_frames[1])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_2, shv_rt_configs.stack_frames[2])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_3, shv_rt_configs.stack_frames[3])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_4, shv_rt_configs.stack_frames[4])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_5, shv_rt_configs.stack_frames[5])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_6, shv_rt_configs.stack_frames[6])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_7, shv_rt_configs.stack_frames[7])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_8, shv_rt_configs.stack_frames[8])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_9, shv_rt_configs.stack_frames[9])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_10, shv_rt_configs.stack_frames[10])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_11, shv_rt_configs.stack_frames[11])
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_stack_size, shv_rt_configs.stack_size)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_code_window_buffer_size, shv_rt_configs.code_window_buffer_size)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_perf_metrics_mask, shv_rt_configs.perf_metrics_mask)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_runtime_version, shv_rt_configs.runtime_version)
TEST_NPU5_MI_REG_FIELD(MiNNRTCfg_use_schedule_embedded_rt, shv_rt_configs.use_schedule_embedded_rt)
TEST_NPU5_MI_REG_FIELD(MihHwpWorkpointCfgAddr, hwp_workpoint_cfg_addr)

TEST_NPU5_MI_REG_FIELD(taskReferenceR1_ManagedInference, managed_inference.reserved1)
TEST_NPU5_MI_REG_FIELD(taskReferenceR2_ManagedInference, managed_inference.reserved2)
TEST_NPU5_MI_REG_FIELD(taskReferenceR3_ManagedInference, managed_inference.reserved3)
TEST_NPU5_MI_REG_FIELD(taskReferenceAddr_ManagedInference, managed_inference.address)
TEST_NPU5_MI_REG_FIELD(taskReferenceCount_ManagedInference, managed_inference.count)

TEST_F(NPUReg50XX_MappedInferenceTest, dpuPerfModeTest) {
    // Could not be tested through macro as this field is an enum
    const auto value = nn_public::VpuHWPStatMode::INVALID_MODE;
    actual.write<vpux::NPUReg50XX::Fields::MiNNRTCfg_dpu_perf_mode>(value);
    const auto actualValue =
            static_cast<nn_public::VpuHWPStatMode>(actual.read<vpux::NPUReg50XX::Fields::MiNNRTCfg_dpu_perf_mode>());
    EXPECT_EQ(actualValue, value);

    reference.shv_rt_configs.dpu_perf_mode = value;
    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_MappedInferenceTest, pad0Test) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::miPad0>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::miPad0>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad0_, 0xFF, 4);

    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_MappedInferenceTest, nnrtcfgPad) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::MiNNRTCfg_pad_6>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::MiNNRTCfg_pad_6>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.shv_rt_configs.pad_, 0xFF, 6);

    ASSERT_TRUE(isContentEqual());
}
