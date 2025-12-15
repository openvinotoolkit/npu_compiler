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
using namespace vpux::NPUReg50XX;

class NPUReg50XX_BarrierConfigurationTest :
        public NPUReg_RegisterUnitBase<npu40xx::nn_public::VpuBarrierConfiguration,
                                       vpux::NPUReg50XX::Descriptors::VpuBarrierConfiguration> {};

#define TEST_NPU5_BARRIER_CFG_REG_FIELD(FieldType, DescriptorMember)               \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_BarrierConfigurationTest, FieldType, \
                                   vpux::NPUReg50XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU5_BARRIER_CFG_REG_FIELD(BarrierConfiguration_producerCount, producerCount)
TEST_NPU5_BARRIER_CFG_REG_FIELD(BarrierConfiguration_producerInterruptEnabled, producerInterruptEnabled)
TEST_NPU5_BARRIER_CFG_REG_FIELD(BarrierConfiguration_consumerCount, consumerCount)
TEST_NPU5_BARRIER_CFG_REG_FIELD(BarrierConfiguration_consumerInterruptEnabled, consumerInterruptEnabled)
