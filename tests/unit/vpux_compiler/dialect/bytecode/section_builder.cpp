//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/utils/section_builder.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstddef>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

// Consumers compare bytecode types by their index in the type section, so a second entry for a type already
// there makes that type compare unequal to itself. addType is therefore get-or-create rather than append,
// which is also what keeps getCachedTypeSymName from growing the section while resolving a buffer_type's
// element type.
class MLIR_BytecodeSectionBuilder : public MLIR_UnitBase {
public:
    MLIR_BytecodeSectionBuilder(): ctx(registry) {
        ctx.loadDialect<bytecode::BytecodeDialect>();
    }

    size_t countTypeEntries(mlir::ModuleOp module) {
        size_t count = 0;
        for (auto section : module.getOps<bytecode::TypeSectionOp>()) {
            for (auto entry : section.getContent().getOps<bytecode::TypeOp>()) {
                std::ignore = entry;
                ++count;
            }
        }
        return count;
    }

    mlir::MLIRContext ctx;
};

// Adding the same type twice must return the first entry rather than append a second one, both for a plain
// element type and for a memref, whose buffer_type attribute carries a symbol reference to its element type.
TEST_F(MLIR_BytecodeSectionBuilder, AddingATypeTwiceReusesTheFirstEntry) {
    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);

    bytecode::SectionBuilder sections(module.get());
    ASSERT_EQ(countTypeEntries(module.get()), 0);

    const auto f32 = builder.getF32Type();
    const auto firstF32Sym = sections.addType(f32, loc);
    ASSERT_EQ(countTypeEntries(module.get()), 1);
    EXPECT_EQ(sections.addType(f32, loc), firstF32Sym);
    EXPECT_EQ(countTypeEntries(module.get()), 1);

    const auto memrefType = mlir::MemRefType::get({2, 3}, f32);
    const auto firstMemRefSym = sections.addType(memrefType, loc);
    // The memref's own entry, on top of the f32 its buffer_type refers to
    ASSERT_EQ(countTypeEntries(module.get()), 2);
    EXPECT_EQ(sections.addType(memrefType, loc), firstMemRefSym);
    EXPECT_EQ(countTypeEntries(module.get()), 2);
}

// Because addType is get-or-create, resolving the memref's element type on the way to the lookup returns the
// existing f32 entry instead of appending one, so a lookup leaves the section the size it found it.
TEST_F(MLIR_BytecodeSectionBuilder, CachedTypeLookupForRegisteredMemRefReusesItsEntry) {
    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);

    bytecode::SectionBuilder sections(module.get());

    const auto memrefType = mlir::MemRefType::get({2, 3}, builder.getF32Type());
    const auto addedSym = sections.addType(memrefType, loc);
    // The memref itself plus the f32 its buffer_type refers to
    ASSERT_EQ(countTypeEntries(module.get()), 2);

    EXPECT_EQ(sections.getCachedTypeSymName(memrefType, loc), addedSym);
    EXPECT_EQ(countTypeEntries(module.get()), 2);
}

}  // namespace
