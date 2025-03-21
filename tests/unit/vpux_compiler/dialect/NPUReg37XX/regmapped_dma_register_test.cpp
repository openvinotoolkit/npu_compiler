//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <npu_37xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.hpp"

using namespace npu37xx;

// nn_public structs which describe hw descriptors for different dialects defined in different headers but have the same
// names (VpuDMATask,VpuDPUInvariant,VpuDPUInvariant,VpuBarrierCountConfig,VpuActKernelInvocation,VpuActKernelRange) and
// are put into the same namespace. It cause problems with gtest lib because NPUReg37XX_RegisterTest class inherit
// TestWithParam class which is parametrized by tested struct from nn_public headers twise:first time in NPU40XX related
// tests, second time in NPUReg37XX related tests. Gtest lib complaining in runtime about duplicate parameterized test
// name That's why we re-define tested hw descriptor as new struct - it helps us to avoid inner gtest conflicts with the
// test for the same descriptor for different dialects
struct Npu37XXDMATask {
    nn_public::VpuDMATask DMA;
};

#define CREATE_HW_DMA_DESC(field, value)                                   \
    [] {                                                                   \
        Npu37XXDMATask hwDMADesc;                                          \
        memset(reinterpret_cast<void*>(&hwDMADesc), 0, sizeof(hwDMADesc)); \
        hwDMADesc.field = value;                                           \
        return hwDMADesc;                                                  \
    }()

class NPUReg37XX_DMARegisterTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37XXDMATask, vpux::NPUReg37XX::RegMapped_DMARegisterType> {};

TEST_P(NPUReg37XX_DMARegisterTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37XXDMATask>> valuesSetDMA = {
        // word 0
        {{
                 {"dma_link_address", {{"dma_link_address", 0x123}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.link_address, 0x123)},
        {{
                 {"dma_watermark", {{"dma_watermark", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.watermark, 1)},
        // word 1
        {{
                 {"dma_cfg_bits", {{"dma_type", 0x3}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.type, 0x3)},
        {{
                 {"dma_cfg_bits", {{"dma_burst_length", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.burst_length, 0xFF)},
        {{
                 {"dma_cfg_bits", {{"dma_critical", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.critical, 1)},
        {{
                 {"dma_cfg_bits", {{"dma_interrupt_en", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.interrupt_en, 1)},
        {{
                 {"dma_cfg_bits", {{"dma_interrupt_trigger", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.interrupt_trigger, 0x7F)},
        {{
                 {"dma_cfg_bits", {{"dma_skip_nr", 0x7F}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.skip_nr, 0x7F)},
        {{
                 {"dma_cfg_bits", {{"dma_order_forced", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.order_forced, 1)},
        {{
                 {"dma_cfg_bits", {{"dma_watermark_en", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.watermark_en, 1)},
        {{
                 {"dma_cfg_bits", {{"dma_dec_en", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.dec_en, 1)},
        {{
                 {"dma_cfg_bits", {{"dma_barrier_en", 1}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.cfg_link.cfg_bits.barrier_en, 1)},
        // word 2
        {{
                 {"dma_src", {{"dma_src", 0x3FFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.src, 0x3FFFFFFFFF)},
        // word 3
        {{
                 {"dma_dst", {{"dma_dst", 0x3FFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.dst, 0x3FFFFFFFFF)},
        // word 4
        {{
                 {"dma_length", {{"dma_length", 0xFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.length, 0xFFFFFF)},
        {{
                 {"dma_num_planes", {{"dma_num_planes", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.num_planes, 0xFF)},
        {{
                 {"dma_task_id", {{"dma_task_id", 0xFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.task_id, 0xFFFFFF)},
        // word 5
        {{
                 {"dma_src_plane_stride", {{"dma_src_plane_stride", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.src_plane_stride, 0xFFFFFFFF)},
        {{
                 {"dma_dst_plane_stride", {{"dma_dst_plane_stride", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.dst_plane_stride, 0xFFFFFFFF)},
        // word 6
        // 2D case
        {{
                 {"dma_attr2d_src_width", {{"dma_attr2d_src_width", 0xFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.attr2d.src_width, 0xFFFFFF)},
        {{
                 {"dma_attr2d_src_stride", {{"dma_attr2d_src_stride", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.attr2d.src_stride, 0xFFFFFFFF)},
        // 1D case
        {{
                 {"dma_barriers1d_prod_mask", {{"dma_barriers_prod_mask", 0xFFFFFFFFFFFFFFFFull}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.barriers1d.prod_mask, 0xFFFFFFFFFFFFFFFFull)},
        // word 7
        // 2D case
        {{
                 {"dma_attr2d_dst_width", {{"dma_attr2d_dst_width", 0xFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.attr2d.dst_width, 0xFFFFFF)},
        {{
                 {"dma_attr2d_dst_stride", {{"dma_attr2d_dst_stride", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.attr2d.dst_stride, 0xFFFFFFFF)},
        // 1D case
        {{
                 {"dma_barriers1d_cons_mask", {{"dma_barriers_cons_mask", 0xFFFFFFFFFFFFFFFFull}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.barriers1d.cons_mask, 0xFFFFFFFFFFFFFFFFull)},
        // word 8 (Used in 2D case only)
        {{
                 {"dma_barriers_prod_mask", {{"dma_barriers_prod_mask", 0xFFFFFFFFFFFFFFFFull}}},
         },
         CREATE_HW_DMA_DESC(DMA.transaction_.barriers.prod_mask, 0xFFFFFFFFFFFFFFFFull)},
        // word 9 (Used in 2D case only)
        {{
                 {"dma_barriers_cons_mask", {{"dma_barriers_cons_mask", 0xFFFFFFFFFFFFFFFFull}}},
         },
         // dma_barriers_sched
         CREATE_HW_DMA_DESC(DMA.transaction_.barriers.cons_mask, 0xFFFFFFFFFFFFFFFFull)},
        {{
                 {"dma_barriers_sched", {{"dma_barriers_sched_start_after", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.barriers_sched_.start_after_, 0xFFFFFFFF)},
        {{
                 {"dma_barriers_sched", {{"dma_barriers_sched_clean_after", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(DMA.barriers_sched_.clean_after_, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_DMARegisterTest, testing::ValuesIn(valuesSetDMA));
