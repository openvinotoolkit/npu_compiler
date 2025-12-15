//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/utils.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace npu40xx;
using namespace vpux::NPUReg40XX;

class NPUReg50XX_BarrierReferenceMapTest :
        public NPUReg_RegisterUnitBase<nn_public::BarrierReferenceMap,
                                       vpux::NPUReg50XX::Descriptors::BarrierReferenceMap> {};

#define TEST_NPU5_BARR_REF_MAP_REG_FIELD(FieldType, DescriptorMember)                                                  \
    HELPER_TEST_NPU_REGISTER_FIELD(NPUReg50XX_BarrierReferenceMapTest, FieldType, vpux::NPUReg50XX::Fields::FieldType, \
                                   DescriptorMember, 0)

TEST_NPU5_BARR_REF_MAP_REG_FIELD(br_physical_barrier, physical_barrier)
TEST_NPU5_BARR_REF_MAP_REG_FIELD(br_producer_count, producer_count)
TEST_NPU5_BARR_REF_MAP_REG_FIELD(br_consumer_count, consumer_count)
TEST_NPU5_BARR_REF_MAP_REG_FIELD(br_producers_ref_offset, producers_ref_offset)
TEST_NPU5_BARR_REF_MAP_REG_FIELD(br_consumers_ref_offset, consumers_ref_offset)
