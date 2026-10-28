//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/section.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/utils/bytecode_writer.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstdint>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

// A constant section is laid out by offset: the section header records where each constant starts, and the
// offsets come from getBinarySize(). A constant that serializes fewer bytes than it declares therefore does
// not merely truncate itself, it shifts every section after it and the reader rejects the whole blob.
class MLIR_BytecodeConstantOp : public MLIR_UnitBase {
public:
    MLIR_BytecodeConstantOp(): ctx(registry) {
        ctx.loadDialect<bytecode::BytecodeDialect>();
    }

    // Builds a module holding a constant section with a single constant carrying `value`, serializes the
    // section, and returns the number of bytes it wrote together with the size it declared. The section is
    // the level that matters: it is the one whose declared size the section header table records.
    struct SerializedSize {
        size_t written;
        size_t declared;
    };

    SerializedSize serializeConstant(mlir::DenseElementsAttr value) {
        mlir::OpBuilder builder(&ctx);
        const auto loc = builder.getUnknownLoc();
        mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module->getBody());

        auto section = bytecode::ConstantSectionOp::create(builder, loc, builder.getStringAttr("constant_section"));
        auto& content = section.getContent().emplaceBlock();

        mlir::OpBuilder contentBuilder = mlir::OpBuilder::atBlockEnd(&content);
        bytecode::ConstantOp::create(contentBuilder, loc, builder.getStringAttr("constant"), value);

        bytecode::BytecodeWriter writer(module.get());
        section.serialize(writer);

        return {writer.getBytecodeBuffer().size(), section.getBinarySize()};
    }

    mlir::MLIRContext ctx;
};

// A splat DenseElementsAttr stores one element no matter how many its type holds, so serializing its raw
// data verbatim writes 8 bytes where the section header promised 32.
TEST_F(MLIR_BytecodeConstantOp, SplatConstantFillsItsDeclaredBinarySize) {
    mlir::OpBuilder builder(&ctx);
    const auto type = mlir::RankedTensorType::get({4}, builder.getI64Type());
    const auto splat = mlir::DenseElementsAttr::get(type, static_cast<int64_t>(1));
    ASSERT_TRUE(splat.isSplat());

    const auto size = serializeConstant(splat);
    EXPECT_EQ(size.declared, 32);
    EXPECT_EQ(size.written, size.declared);
}

// The non-splat path is the one that was always correct; pin it so the splat branch cannot regress it.
TEST_F(MLIR_BytecodeConstantOp, NonSplatConstantFillsItsDeclaredBinarySize) {
    mlir::OpBuilder builder(&ctx);
    const auto type = mlir::RankedTensorType::get({4}, builder.getI64Type());
    const auto values = SmallVector<int64_t>{1, 16, 720, 1000};
    const auto dense = mlir::DenseElementsAttr::get(type, ArrayRef<int64_t>(values));
    ASSERT_FALSE(dense.isSplat());

    const auto size = serializeConstant(dense);
    EXPECT_EQ(size.declared, 32);
    EXPECT_EQ(size.written, size.declared);
}

// A single-element constant is splat by definition and must still write exactly one element.
TEST_F(MLIR_BytecodeConstantOp, SingleElementConstantFillsItsDeclaredBinarySize) {
    mlir::OpBuilder builder(&ctx);
    const auto type = mlir::RankedTensorType::get({1}, builder.getI64Type());
    const auto splat = mlir::DenseElementsAttr::get(type, static_cast<int64_t>(42));
    ASSERT_TRUE(splat.isSplat());

    const auto size = serializeConstant(splat);
    EXPECT_EQ(size.declared, 8);
    EXPECT_EQ(size.written, size.declared);
}

}  // namespace
