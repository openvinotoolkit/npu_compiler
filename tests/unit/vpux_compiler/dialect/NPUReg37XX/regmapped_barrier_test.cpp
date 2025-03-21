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
struct Npu37XXBarrierCfg {
    nn_public::VpuBarrierCountConfig barrier;
};

#define CREATE_HW_BARRIER_DESC(field, value)                                       \
    [] {                                                                           \
        Npu37XXBarrierCfg hwBarrierDesc;                                           \
        memset(reinterpret_cast<void*>(&hwBarrierDesc), 0, sizeof(hwBarrierDesc)); \
        hwBarrierDesc.field = value;                                               \
        return hwBarrierDesc;                                                      \
    }()

class NPUReg37XX_NpuBarrierCountConfigTest :
        public MLIR_RegMappedNPUReg37XXUnitBase<Npu37XXBarrierCfg,
                                                vpux::NPUReg37XX::RegMapped_VpuBarrierCountConfigType> {};

TEST_P(NPUReg37XX_NpuBarrierCountConfigTest, CheckFieldsConsistency) {
    this->compare();
}

std::vector<std::pair<MappedRegValues, Npu37XXBarrierCfg>> barrierFieldSetNPUReg37XX = {
        {{
                 {"next_same_id_", {{"next_same_id_", 0xFFFFFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.next_same_id_, 0xFFFFFFFF)},
        {{
                 {"producer_count_", {{"producer_count_", 0xFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.producer_count_, 0xFFFF)},
        {{
                 {"consumer_count_", {{"consumer_count_", 0xFFFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.consumer_count_, 0xFFFF)},
        {{
                 {"real_id_", {{"real_id_", 0xFF}}},
         },
         CREATE_HW_BARRIER_DESC(barrier.real_id_, 0xFF)},
};

INSTANTIATE_TEST_SUITE_P(NPUReg37XX_MappedRegs, NPUReg37XX_NpuBarrierCountConfigTest,
                         testing::ValuesIn(barrierFieldSetNPUReg37XX));
