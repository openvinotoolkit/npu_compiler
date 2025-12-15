//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "common/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"

#include <gtest/gtest.h>
#include <mlir/AsmParser/AsmParser.h>

using namespace vpux;

using DescriptorPrintParseTest = MLIR_UnitBase;

// The printer and parser for the backend descriptors behave differently depending on the build type
// This unit test shows the difference and ensures that the roundtrip mechanism continues to be functional
TEST_F(DescriptorPrintParseTest, Roundtrip) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<NPUReg40XX::NPUReg40XXDialect>();

    constexpr const char* printedDescriptor = []() {
        if constexpr (isDeveloperBuild()) {
            return R"(#NPUReg40XX.VpuActKernelInvocation<
  VpuActKernelInvocation {
    range = UINT 0x200000,
    kernel_args = UINT 0,
    data_window_base = UINT 0,
    perf_packet_out = UINT 0,
    barriers_wait_mask_hi_act {
      UINT barriers_wait_mask_hi_act = 0,
    }
    barriers_wait_mask_lo_act = UINT 1,
    barriers_post_mask_hi_act {
      UINT barriers_post_mask_hi_act = 0,
    }
    barriers_post_mask_lo_act = UINT 2,
    barriers_group_mask_act {
      UINT group_act = 1,
      UINT mask_act = 1,
    }
    act_invo_barriers_sched {
      UINT start_after_ = 0,
      UINT clean_after_ = 0,
    }
    invo_index = UINT 4,
    invo_tile = UINT 0,
    kernel_range_index = UINT 0,
    next_aki_wl_addr = UINT 0,
  } requires 11:4:10
>)";
        } else {
            return R"(#NPUReg40XX<VpuActKernelInvocation"000020000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000002000000000000000101000000000000000000000000000004000000000000000000000000000000">)";
        }
    }();

    NPUReg40XX::Descriptors::VpuActKernelInvocation descriptor;
    descriptor.write<NPUReg40XX::Fields::range>(0x200000);
    descriptor.write<NPUReg40XX::Fields::barriers_wait_mask_hi_act>(0);
    descriptor.write<NPUReg40XX::Fields::barriers_wait_mask_lo_act>(1);
    descriptor.write<NPUReg40XX::Fields::barriers_post_mask_hi_act>(0);
    descriptor.write<NPUReg40XX::Fields::barriers_post_mask_lo_act>(2);
    descriptor.write<NPUReg40XX::Fields::group_act>(1);
    descriptor.write<NPUReg40XX::Fields::mask_act>(1);
    descriptor.write<NPUReg40XX::Registers::act_invo_barriers_sched, NPUReg40XX::Fields::start_after_>(0);
    descriptor.write<NPUReg40XX::Registers::act_invo_barriers_sched, NPUReg40XX::Fields::clean_after_>(0);
    descriptor.write<NPUReg40XX::Fields::invo_index>(4);
    descriptor.write<NPUReg40XX::Fields::invo_tile>(0);
    descriptor.write<NPUReg40XX::Fields::kernel_range_index>(0);
    descriptor.write<NPUReg40XX::Fields::perf_packet_out>(0);
    descriptor.write<NPUReg40XX::Fields::next_aki_wl_addr>(0);
    auto descriptorAttr = NPUReg40XX::VpuActKernelInvocationAttr::get(&ctx, descriptor);
    ASSERT_TRUE(descriptorAttr != nullptr);

    std::string str;
    auto stringStream = llvm::raw_string_ostream(str);
    auto genericAttr = mlir::dyn_cast<mlir::Attribute>(descriptorAttr);
    ASSERT_TRUE(genericAttr != nullptr);
    genericAttr.print(stringStream);
    ASSERT_EQ(str, printedDescriptor);

    auto parsedAttr = mlir::parseAttribute(str, &ctx);
    ASSERT_TRUE(parsedAttr != nullptr);
    auto newDescriptorAttr = mlir::dyn_cast<NPUReg40XX::VpuActKernelInvocationAttr>(parsedAttr);
    ASSERT_TRUE(newDescriptorAttr != nullptr);
    auto newDescriptor = newDescriptorAttr.getRegMapped();
    EXPECT_EQ(descriptor, newDescriptor);
}
