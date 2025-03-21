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
struct Npu37ActKernelRange {
    nn_public::VpuActKernelRange actKernelRange;
};

#define CREATE_HW_ACT_KERNEL_RANGE_DESC(field, value)                                            \
    [] {                                                                                         \
        Npu37ActKernelRange hwActKernelRangeDesc;                                                \
        memset(reinterpret_cast<void*>(&hwActKernelRangeDesc), 0, sizeof(hwActKernelRangeDesc)); \
        hwActKernelRangeDesc.field = value;                                                      \
        return hwActKernelRangeDesc;                                                             \
    }()

class NPUReg37XX_NpuActKernelRangeTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37ActKernelRange,
                                                vpux::NPUReg37XX::RegMapped_VpuActKernelRangeType> {};

TEST_P(NPUReg37XX_NpuActKernelRangeTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37ActKernelRange>> actKernelRangeFieldSetNPUReg37XX = {
        {{
                 {"type", {{"type", 0x07}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.type, nn_public::VpuActWLType::WL_CACHE_OP_FLUSHINV)},
        {{
                 {"kernel_entry", {{"kernel_entry", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.kernel_entry, 0xFFFFFFFF)},
        {{
                 {"text_window_base", {{"text_window_base", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.text_window_base, 0xFFFFFFFF)},
        {{
                 {"code_size", {{"code_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.code_size, 0xFFFFFFFF)},
        {{
                 {"data_sec_size", {{"data_sec_size", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.data_sec_size, 0xFFFFFFFF)},
        {{
                 {"kernel_invo_count", {{"kernel_invo_count", 0xFFFFFFFF}}},
         },
         CREATE_HW_ACT_KERNEL_RANGE_DESC(actKernelRange.kernel_invo_count, 0xFFFFFFFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_NpuActKernelRangeTest,
                         testing::ValuesIn(actKernelRangeFieldSetNPUReg37XX));
