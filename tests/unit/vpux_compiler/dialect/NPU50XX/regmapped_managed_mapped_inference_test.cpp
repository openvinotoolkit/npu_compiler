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

class NPUReg50XX_ManagedMappedInferenceTest :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuManagedMappedInference,
                                       vpux::NPUReg50XX::Descriptors::VpuManagedMappedInference> {};

#define TEST_NPU5_MMI_INFO_REG_FIELD(FieldType, DescriptorMember)                    \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_ManagedMappedInferenceTest, FieldType, \
                                   vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU5_MMI_INFO_REG_FIELD(MMI_vpu_nnrt_api_ver, vpu_nnrt_api_ver)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_final_barrier, final_barrier)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_work_item, work_items.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_work_item, work_items.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_work_item, work_items.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_work_item, work_items.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_work_item, work_items.count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_task_configs, task_configs.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_task_configs, task_configs.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_task_configs, task_configs.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_task_configs, task_configs.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_task_configs, task_configs.count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_reserved0_0, reserved0[0].reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_reserved0_0, reserved0[0].reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_reserved0_0, reserved0[0].reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_reserved0_0, reserved0[0].address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_reserved0_0, reserved0[0].count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_reserved0_1, reserved0[1].reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_reserved0_1, reserved0[1].reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_reserved0_1, reserved0[1].reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_reserved0_1, reserved0[1].address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_reserved0_1, reserved0[1].count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_reserved0_2, reserved0[2].reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_reserved0_2, reserved0[2].reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_reserved0_2, reserved0[2].reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_reserved0_2, reserved0[2].address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_reserved0_2, reserved0[2].count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_reserved0_3, reserved0[3].reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_reserved0_3, reserved0[3].reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_reserved0_3, reserved0[3].reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_reserved0_3, reserved0[3].address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_reserved0_3, reserved0[3].count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_barriers_configuration, barriers_configuration.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_barriers_configuration, barriers_configuration.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_barriers_configuration, barriers_configuration.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_barriers_configuration, barriers_configuration.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_barriers_configuration, barriers_configuration.count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_num_of_barrier_reprogrammings, num_of_barrier_reprogrammings.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_num_of_barrier_reprogrammings, num_of_barrier_reprogrammings.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_num_of_barrier_reprogrammings, num_of_barrier_reprogrammings.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_num_of_barrier_reprogrammings, num_of_barrier_reprogrammings.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_num_of_barrier_reprogrammings, num_of_barrier_reprogrammings.count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_initial_barriers, initial_barriers.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_initial_barriers, initial_barriers.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_initial_barriers, initial_barriers.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_initial_barriers, initial_barriers.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_initial_barriers, initial_barriers.count)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_nnrt_config, nnrt_config.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_nnrt_config, nnrt_config.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_nnrt_config, nnrt_config.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_nnrt_config, nnrt_config.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_nnrt_config, nnrt_config.count)

TEST_NPU5_MMI_INFO_REG_FIELD(MMI_actshv_used, actshv_used)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_dpu_used, dpu_used)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_media_used, media_used)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_dma_from_ddr_used, dma_from_ddr_used)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_dma_from_cmx_used, dma_from_cmx_used)

TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR1_MMI_inference_info, inference_info.reserved1)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR2_MMI_inference_info, inference_info.reserved2)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceR3_MMI_inference_info, inference_info.reserved3)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceAddr_MMI_inference_info, inference_info.address)
TEST_NPU5_MMI_INFO_REG_FIELD(taskReferenceCount_MMI_inference_info, inference_info.count)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_barrier_configuration_stride, barrier_configuration_stride)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_bootstrap_workitems_count, bootstrap_workitems_count)
TEST_NPU5_MMI_INFO_REG_FIELD(MMI_model_identifier, model_identifier)

TEST_F(NPUReg50XX_ManagedMappedInferenceTest, barProgrammingMode) {
    // Could not be tested through macro as this field is an enum
    const auto value = nn_public::VpuManagedMappedInference::VpuBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED;
    actual.write<vpux::NPUReg50XX::Fields::MMI_barrier_programming_mode>(value);
    const auto actualValue = static_cast<nn_public::VpuManagedMappedInference::VpuBarrierProgrammingMode>(
            actual.read<vpux::NPUReg50XX::Fields::MMI_barrier_programming_mode>());
    EXPECT_EQ(actualValue, value);

    reference.barrier_programming_mode = value;
    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_ManagedMappedInferenceTest, mmPad0Test) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFF;
    actual.write<vpux::NPUReg50XX::Fields::MMI_pad0_>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::MMI_pad0_>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad0_, 0xFF, 2);

    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_ManagedMappedInferenceTest, mmPad1Test) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::MMI_pad1_>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::MMI_pad1_>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad1_, 0xFF, 3);

    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_ManagedMappedInferenceTest, mmPad2Test) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::MMI_pad2_28>(value);

    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::MMI_pad2_28>();
    EXPECT_EQ(actualValue, value);

    // test only last 4bytes from pad2_
    std::memset(&reference.pad2_[224], 0xFF, 4);

    ASSERT_TRUE(isContentEqual());
}

TEST_F(NPUReg50XX_ManagedMappedInferenceTest, mmPad3Test) {
    // Could not be tested through the macro as this fields are arrays
    const auto value = 0xFFFFFFFF;
    actual.write<vpux::NPUReg50XX::Fields::MMI_pad3_>(value);
    const auto actualValue = actual.read<vpux::NPUReg50XX::Fields::MMI_pad3_>();
    EXPECT_EQ(actualValue, value);

    std::memset(reference.pad3_, 0xFF, 4);

    ASSERT_TRUE(isContentEqual());
}
