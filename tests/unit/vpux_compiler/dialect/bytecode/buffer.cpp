//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/buffer.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"
#include "vpux/compiler/dialect/bytecode/IR/ops/register.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>

#include <gtest/gtest.h>

using namespace vpux;

namespace {

TEST(MLIR_BytecodeBufferSubviewOp, RejectsRankAboveBufferTypeLimit) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<bytecode::BytecodeDialect>();

    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    const auto dst = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto src = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto operand = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();

    const auto oversizedRank = static_cast<size_t>(intel_npu::vm::BufferType::MAX_RANK) + 1;
    SmallVector<mlir::Value> operands(oversizedRank, operand);

    auto subview = builder.create<bytecode::BufferSubviewOp>(loc, dst, src, operands, operands, operands);

    mlir::ScopedDiagnosticHandler diagnosticHandler(&ctx, [](mlir::Diagnostic&) {
        return mlir::success();
    });
    EXPECT_TRUE(mlir::failed(mlir::verify(subview.getOperation())));
}

TEST(MLIR_BytecodeBufferCreateOp, AcceptsRank0) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<bytecode::BytecodeDialect>();

    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    const auto dst = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto elemTypeRef = mlir::FlatSymbolRefAttr::get(&ctx, "i64");

    auto create =
            builder.create<bytecode::BufferCreateOp>(loc, dst, elemTypeRef, mlir::ValueRange{}, mlir::ValueRange{});

    EXPECT_TRUE(mlir::succeeded(mlir::verify(create.getOperation())));
}

TEST(MLIR_BytecodeBufferStoreOp, AcceptsRank0) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<bytecode::BytecodeDialect>();

    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    const auto buffer = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto value = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();

    auto store = builder.create<bytecode::BufferStoreOp>(loc, buffer, value, mlir::ValueRange{});

    EXPECT_TRUE(mlir::succeeded(mlir::verify(store.getOperation())));
}

TEST(MLIR_BytecodeBufferStoreOp, RejectsRankAboveBufferTypeLimit) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<bytecode::BytecodeDialect>();

    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    const auto buffer = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto value = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto operand = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();

    const auto oversizedRank = static_cast<size_t>(intel_npu::vm::BufferType::MAX_RANK) + 1;
    SmallVector<mlir::Value> indices(oversizedRank, operand);

    auto store = builder.create<bytecode::BufferStoreOp>(loc, buffer, value, indices);

    mlir::ScopedDiagnosticHandler diagnosticHandler(&ctx, [](mlir::Diagnostic&) {
        return mlir::success();
    });
    EXPECT_TRUE(mlir::failed(mlir::verify(store.getOperation())));
}

TEST(MLIR_BytecodeBufferCreateOp, RejectsRankAboveBufferTypeLimit) {
    mlir::MLIRContext ctx;
    ctx.loadDialect<bytecode::BytecodeDialect>();

    mlir::OpBuilder builder(&ctx);
    const auto loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());

    const auto dst = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();
    const auto elemTypeRef = mlir::FlatSymbolRefAttr::get(&ctx, "i64");
    const auto operand = builder.create<bytecode::VirtualGeneralRegisterOp>(loc).getResult();

    const auto oversizedRank = static_cast<size_t>(intel_npu::vm::BufferType::MAX_RANK) + 1;
    SmallVector<mlir::Value> shape(oversizedRank, operand);
    SmallVector<mlir::Value> strides(oversizedRank, operand);

    auto create = builder.create<bytecode::BufferCreateOp>(loc, dst, elemTypeRef, shape, strides);

    mlir::ScopedDiagnosticHandler diagnosticHandler(&ctx, [](mlir::Diagnostic&) {
        return mlir::success();
    });
    EXPECT_TRUE(mlir::failed(mlir::verify(create.getOperation())));
}

}  // namespace
