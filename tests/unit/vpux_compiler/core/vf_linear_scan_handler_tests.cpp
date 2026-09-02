//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/scheduling/vf_linear_scan_handler.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>

#include <array>
#include <utility>

// Run cmd: npuUnitTests --gtest_filter="MLIR_VFLinearScanHandler.*"

using namespace vpux;

namespace {

// Tiny helper: produce N distinct mlir::Value instances backed by a single
// trivial function so the handler has unique keys to map records against.
class ValueFactory {
public:
    explicit ValueFactory(size_t n) {
        _ctx.loadDialect<mlir::func::FuncDialect>();
        mlir::OpBuilder builder(&_ctx);
        const auto loc = builder.getUnknownLoc();
        const auto f32 = builder.getF32Type();
        _module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToEnd(_module.getBody());
        llvm::SmallVector<mlir::Type> argTypes(n, f32);
        auto fnType = builder.getFunctionType(argTypes, {});
        auto fn = builder.create<mlir::func::FuncOp>(loc, "main", fnType);
        auto& entry = *fn.addEntryBlock();
        for (size_t i = 0; i < n; ++i) {
            _values.push_back(entry.getArgument(i));
        }
        builder.setInsertionPointToEnd(&entry);
        builder.create<mlir::func::ReturnOp>(loc);
    }

    mlir::Value operator[](size_t i) const {
        return _values[i];
    }

private:
    mlir::MLIRContext _ctx;
    mlir::ModuleOp _module;
    llvm::SmallVector<mlir::Value> _values;
};

using LinearScanT = LinearScan<mlir::Value, VFRegionLinearScanHandler>;
using ExcludedVec = LinearScanT::ReservedAddressAndSizeVector;

}  // namespace

TEST(MLIR_VFLinearScanHandler, BufferDataStorage) {
    ValueFactory vals(3);
    VFRegionLinearScanHandler h(/*defaultAlignment=*/16);

    h.addBufferData(vals[0], BufferCategory::PERSISTENT_CANDIDATE, /*size=*/256, /*alignment=*/64, /*spillWeight=*/100);
    h.addBufferData(vals[1], BufferCategory::TEMPORARY, /*size=*/128, /*alignment=*/32, /*spillWeight=*/-5);
    // Alignment 0 falls back to defaultAlignment.
    h.addBufferData(vals[2], BufferCategory::TEMPORARY, /*size=*/64, /*alignment=*/0, /*spillWeight=*/0);

    EXPECT_EQ(h.getCategory(vals[0]), BufferCategory::PERSISTENT_CANDIDATE);
    EXPECT_EQ(h.getCategory(vals[1]), BufferCategory::TEMPORARY);

    EXPECT_EQ(h.getSize(vals[0]), 256u);
    EXPECT_EQ(h.getAlignment(vals[0]), 64u);
    EXPECT_EQ(h.getSpillWeight(vals[1]), -5);
    EXPECT_EQ(h.getAlignment(vals[2]), 16u);  // fallback
}

TEST(MLIR_VFLinearScanHandler, AliveStateAndAddressBookkeeping) {
    ValueFactory vals(2);
    VFRegionLinearScanHandler h(/*defaultAlignment=*/16);
    h.addBufferData(vals[0], BufferCategory::TEMPORARY, 64, 1, 0);

    EXPECT_FALSE(h.isAlive(vals[0]));
    h.markAsAlive(vals[0]);
    EXPECT_TRUE(h.isAlive(vals[0]));
    EXPECT_ANY_THROW(h.getAddress(vals[0]));

    h.allocated(vals[0], 100);
    EXPECT_EQ(h.getAddress(vals[0]), 100u);
    EXPECT_EQ(h.maxAllocatedSize(), 164u);

    h.freed(vals[0]);
    EXPECT_EQ(h.getAddress(vals[0]), 100u);
    EXPECT_FALSE(h.isAlive(vals[0]));
}

TEST(MLIR_VFLinearScanHandler, SpilledBufferHandling) {
    ValueFactory vals(2);
    VFRegionLinearScanHandler h(/*defaultAlignment=*/16);
    h.addBufferData(vals[0], BufferCategory::TEMPORARY, 64, 1, 0);
    h.addBufferData(vals[1], BufferCategory::TEMPORARY, 64, 1, 0);

    EXPECT_FALSE(h.wasEverSpilled(vals[0]));
    EXPECT_FALSE(h.wasEverSpilled(vals[1]));

    EXPECT_TRUE(h.spilled(vals[0]));
    EXPECT_TRUE(h.wasEverSpilled(vals[0]));
    EXPECT_FALSE(h.wasEverSpilled(vals[1]));
}

TEST(MLIR_VFLinearScanHandler, AllocateWithExcludedRegion) {
    ValueFactory vals(2);
    LinearScanT scan(/*size=*/128, /*reservedVec=*/{}, /*defaultAlignment=*/128u);
    auto& h = scan.handler();
    h.addBufferData(vals[0], BufferCategory::TEMPORARY, /*size=*/40, /*alignment=*/1, /*spillWeight=*/0);
    h.addBufferData(vals[1], BufferCategory::TEMPORARY, /*size=*/80, /*alignment=*/1, /*spillWeight=*/0);

    // Expected memory layout:
    // [0, 64) — excluded region
    // [64, 128) - space for allocation
    const std::array<std::pair<AddressType, AddressType>, 1> excludedArr{{{0u, 64u}}};
    const ExcludedVec excludedLow(excludedArr);

    ASSERT_TRUE(scan.allocWithExcludedRegion(excludedLow, std::array<mlir::Value, 1>{vals[0]},
                                             /*allowSpills=*/false, Partitioner::Direction::Down));

    // vals[0] (40 bytes) buffer allocated at the top of the available space - [88, 128)
    EXPECT_EQ(h.getAddress(vals[0]), 88u);
    EXPECT_GE(h.getAddress(vals[0]), 64u);

    // vals[1] (80 bytes) buffer cannot fit in remaining space [64, 88) (24 bytes) and allocation should fail
    EXPECT_FALSE(scan.allocWithExcludedRegion(excludedLow, std::array<mlir::Value, 1>{vals[1]},
                                              /*allowSpills=*/false, Partitioner::Direction::Down));
}
