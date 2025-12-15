//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <npu_40xx_nnrt.hpp>
#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

using namespace npu40xx;
using namespace vpux::NPUReg50XX;

class NPUReg50XX_VpuActKernelInvocationTest :
        public NPUReg_RegisterUnitBase<nn_public::VpuActKernelInvocation,
                                       vpux::NPUReg50XX::Descriptors::VpuActKernelInvocation> {};

#define TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(FieldType, DescriptorMember)         \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_VpuActKernelInvocationTest, FieldType, \
                                   vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

#define TEST_NPU5_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(ParentRegType, FieldType, DescriptorMember)        \
    HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(NPUReg50XX_VpuActKernelInvocationTest, ParentRegType##__##FieldType, \
                                        vpux::NPUReg50XX::Registers::ParentRegType,                          \
                                        vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(range, range)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(kernel_args, kernel_args)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(data_window_base, data_window_base)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(perf_packet_out, perf_packet_out)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(barriers_wait_mask_hi_act, barriers.wait_mask_hi_)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(barriers_wait_mask_lo_act, barriers.wait_mask_lo_)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(barriers_post_mask_hi_act, barriers.post_mask_hi_)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(barriers_post_mask_lo_act, barriers.post_mask_lo_)
TEST_NPU5_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(act_invo_barriers_sched, start_after_, barriers_sched.start_after_)
TEST_NPU5_ACTKERNELINVOCATION_MULTIPLE_REGS_FIELD(act_invo_barriers_sched, clean_after_, barriers_sched.clean_after_)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(invo_tile, invo_tile)
TEST_NPU5_ACTKERNELINVOCATION_REG_FIELD(kernel_range_index, kernel_range_index)
