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
// TestWithParam class which is parametrized by tested struct from nn_public headers twice:first time in NPU40XX related
// tests, second time in NPUReg37XX related tests. Gtest lib complaining in runtime about duplicate parameterized test
// name That's why we re-define tested hw descriptor as new struct - it helps us to avoid inner gtest conflicts with the
// test for the same descriptor for different dialects
struct Npu37ActKernelInvocation {
    nn_public::VpuActKernelInvocation actKernelInvo;
};

#define CREATE_HW_DMA_DESC(field, value)                                                       \
    [] {                                                                                       \
        Npu37ActKernelInvocation hwActKernelInvoDesc;                                          \
        memset(reinterpret_cast<void*>(&hwActKernelInvoDesc), 0, sizeof(hwActKernelInvoDesc)); \
        hwActKernelInvoDesc.field = value;                                                     \
        return hwActKernelInvoDesc;                                                            \
    }()

class NPUReg37XX_NpuActKernelInvocationTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37ActKernelInvocation,
                                                vpux::NPUReg37XX::RegMapped_VpuActKernelInvocationType> {};

TEST_P(NPUReg37XX_NpuActKernelInvocationTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37ActKernelInvocation>> actKernelInvoFieldSetNPUReg37XX = {
        {{
                 {"range", {{"range", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.range, 0xFFFFFFFF)},
        {{
                 {"kernel_args", {{"kernel_args", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.kernel_args, 0xFFFFFFFF)},
        {{
                 {"data_window_base", {{"data_window_base", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.data_window_base, 0xFFFFFFFF)},
        {{
                 {"perf_packet_out", {{"perf_packet_out", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.perf_packet_out, 0xFFFFFFFF)},
        {{
                 {"barriers_wait_mask_act", {{"barriers_wait_mask_act", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers.wait_mask_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_post_mask_act", {{"barriers_post_mask_act", 0xFFFFFFFFFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers.post_mask_, 0xFFFFFFFFFFFFFFFF)},
        {{
                 {"barriers_group_mask_act", {{"group_act", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers.group_, 0xFF)},
        {{
                 {"barriers_group_mask_act", {{"mask_act", 0xFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers.mask_, 0xFF)},
        {{
                 {"act_invo_barriers_sched", {{"act_invo_barriers_sched_start_after", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers_sched.start_after_, 0xFFFFFFFF)},
        {{
                 {"act_invo_barriers_sched", {{"act_invo_barriers_sched_clean_after", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.barriers_sched.clean_after_, 0xFFFFFFFF)},
        {{
                 {"invo_index", {{"invo_index", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.invo_index, 0xFFFFFFFF)},
        {{
                 {"invo_tile", {{"invo_tile", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.invo_tile, 0xFFFFFFFF)},
        {{
                 {"kernel_range_index", {{"kernel_range_index", 0xFFFFFFFF}}},
         },
         CREATE_HW_DMA_DESC(actKernelInvo.kernel_range_index, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_NpuActKernelInvocationTest,
                         testing::ValuesIn(actKernelInvoFieldSetNPUReg37XX));
