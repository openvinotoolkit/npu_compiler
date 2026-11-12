//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"

#include <cstring>
#include <npu_60xx_nnrt.hpp>

using namespace npu60xx;
using namespace vpux::NPUReg60XX;

class NPUReg60XX_BarrierConfigurationTest :
        public NPUReg_RegisterUnitBase<npu60xx::nn_public::VpuBarrierConfiguration,
                                       vpux::NPUReg60XX::Descriptors::VpuBarrierConfiguration> {};

#define TEST_NPU6_BARRIER_CFG_REG_FIELD(FieldType, DescriptorMember)               \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg60XX_BarrierConfigurationTest, FieldType, \
                                   vpux::NPUReg60XX::Fields::FieldType, DescriptorMember, 0)

TEST_NPU6_BARRIER_CFG_REG_FIELD(BarrierConfiguration_producerCount, producerCount)
TEST_NPU6_BARRIER_CFG_REG_FIELD(BarrierConfiguration_producerInterruptEnabled, producerInterruptEnabled)
TEST_NPU6_BARRIER_CFG_REG_FIELD(BarrierConfiguration_consumerCount, consumerCount)
TEST_NPU6_BARRIER_CFG_REG_FIELD(BarrierConfiguration_consumerInterruptEnabled, consumerInterruptEnabled)
