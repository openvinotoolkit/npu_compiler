//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

#include <cstring>
#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg50XX_NNRTCfgTest :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuNNRTConfig,
                                       vpux::NPUReg50XX::Descriptors::VpuNNRTConfig> {};

#define TEST_NPU5_NN_CFG_REG_FIELD(FieldType, DescriptorMember)                                            \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_NNRTCfgTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_reserved, shv_rt_configs.reserved)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_runtime_entry, shv_rt_configs.runtime_entry)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_act_rt_window_base, shv_rt_configs.act_rt_window_base)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_0, shv_rt_configs.stack_frames[0])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_1, shv_rt_configs.stack_frames[1])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_2, shv_rt_configs.stack_frames[2])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_3, shv_rt_configs.stack_frames[3])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_4, shv_rt_configs.stack_frames[4])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_5, shv_rt_configs.stack_frames[5])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_6, shv_rt_configs.stack_frames[6])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_7, shv_rt_configs.stack_frames[7])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_8, shv_rt_configs.stack_frames[8])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_9, shv_rt_configs.stack_frames[9])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_10, shv_rt_configs.stack_frames[10])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_11, shv_rt_configs.stack_frames[11])
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_stack_size, shv_rt_configs.stack_size)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_code_window_buffer_size, shv_rt_configs.code_window_buffer_size)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_perf_metrics_mask, shv_rt_configs.perf_metrics_mask)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_runtime_version, shv_rt_configs.runtime_version)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_use_schedule_embedded_rt, shv_rt_configs.use_schedule_embedded_rt)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_logAddrDmaHwp, logaddr_dma_hwp)
TEST_NPU5_NN_CFG_REG_FIELD(NNRTCfg_HwpCfgAddr, hwp_workpoint_cfg_addr)

TEST_F(NPUReg50XX_NNRTCfgTest, NNRTDpuPerfModeTest) {
    // Could not be tested through macro as this field is an enum
    const auto value = nn_public::VpuHWPStatMode::INVALID_MODE;
    actual.write<vpux::NPUReg50XX::Fields::NNRTCfg_dpu_perf_mode>(value);
    const auto actualValue =
            static_cast<nn_public::VpuHWPStatMode>(actual.read<vpux::NPUReg50XX::Fields::NNRTCfg_dpu_perf_mode>());
    EXPECT_EQ(actualValue, value);

    reference.shv_rt_configs.dpu_perf_mode = value;
    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_NNRTCfgTest, NNRTCfgPad) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::NNRTCfg_pad_6>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::NNRTCfg_pad_6>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.shv_rt_configs.pad_, 0xFF, 6);

    ASSERT_TRUE(isContentEqual());
}
