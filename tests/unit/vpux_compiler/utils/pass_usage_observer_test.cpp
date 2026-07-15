//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/pass_usage_observer.hpp"
#include "common/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/npu_action_handler.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <fstream>

using namespace vpux;

using MLIR_PassUsageObserver = MLIR_UnitBase;

namespace {
std::string readFile(const std::string& path) {
    std::ifstream file(path);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::string makeTempPath() {
    llvm::SmallString<128> path;
    const auto error = llvm::sys::fs::createTemporaryFile("temp", "txt", path);
    VPUX_THROW_UNLESS(!error, "{0}", error.message());
    return path.str().str();
}

NpuActionHandler makeActionHandler(const std::string& outputFile) {
    auto npuActionHandler = NpuActionHandler();
    npuActionHandler.registerObserver(std::make_unique<PassUsageObserver>(outputFile));
    return npuActionHandler;
}

void testPass(mlir::MLIRContext& ctx, mlir::ModuleOp mod, mlir::PassManager& pm, const std::string& expected) {
    const auto outputFile = makeTempPath();
    VPUX_SCOPE_EXIT {
        llvm::sys::fs::remove(outputFile);
    };

    ctx.registerActionHandler(makeActionHandler(outputFile));
    ASSERT_TRUE(pm.run(mod).succeeded());

    ctx.registerActionHandler(nullptr);
    ASSERT_EQ(readFile(outputFile), expected);
}

}  // namespace

TEST_F(MLIR_PassUsageObserver, PassMakesChanges) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: memref<1x16x4x4xf16>, %output: memref<1x16x4x4xf16>) -> memref<1x16x4x4xf16> {
                return %output : memref<1x16x4x4xf16>
            }
        }
    )";

    auto mod = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(mod);

    mlir::PassManager pm(mod.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>));

    testPass(ctx, mod.get(), pm, "SetMemorySpace\tCHANGED\n");
}

TEST_F(MLIR_PassUsageObserver, PassMakesNoChanges) {
    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<VPUIP::VPUIPDialect>();

    constexpr llvm::StringLiteral inputIR = R"(
        module @test {
            func.func @main(%input: memref<1x16x4x4xf16, @DDR>, %output: memref<1x16x4x4xf16, @DDR>) -> memref<1x16x4x4xf16, @DDR> {
                return %output : memref<1x16x4x4xf16, @DDR>
            }
        }
    )";

    auto mod = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(mod);

    mlir::PassManager pm(mod.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>));

    testPass(ctx, mod.get(), pm, "SetMemorySpace\tSAME\n");
}
