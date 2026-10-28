//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// =============================================================================
// Unit tests for the Shave::LoadExternalKernelResources pass
// (src/vpux_compiler/src/dialect/Shave/transforms/passes/load_external_kernel_resources.cpp).
//
// These specifically exercise the following behavior:
//   - the pass moved from a per-function pass (safeRunOnFunc) to a module pass
//     (safeRunOnModule), so the `loadedKernels` dedup set now persists across
//     every function in the module, not just within one function;
//   - the ifstream that reads the kernel ELF was switched to binary mode;
//   - ExternalKernelOp no longer carries its own kernel_path attribute -- the
//     path is resolved by looking up the op's unique_id in a Custom Layer
//     Config XML supplied to the pass, and the pass soft-skips (no error)
//     when no config path is supplied at all.
// =============================================================================

#include "common/utils.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/Shave/IR/dialect.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <gtest/gtest.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace vpux;

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Writes raw binary content (including embedded NUL / non-ASCII bytes) to a
/// temp file and returns its path. Uses `std::ios::binary` on purpose: the
/// pass under test also reads in binary mode (post-fix), and part of what
/// these tests confirm is that content round-trips byte-for-byte.
std::string writeTempBinary(const std::vector<uint8_t>& bytes) {
    llvm::SmallString<128> path;
    const auto err = llvm::sys::fs::createTemporaryFile("external_kernel", "elf", path);
    VPUX_THROW_UNLESS(!err, "Failed to create temp file: {0}", err.message());
    std::ofstream f(path.str().str(), std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    f.close();
    return path.str().str();
}

/// RAII wrapper that deletes the temp file on destruction.
struct TempFile {
    std::string path;
    explicit TempFile(std::string p): path(std::move(p)) {
    }
    ~TempFile() {
        llvm::sys::fs::remove(path);
    }
};

/// A valid ELF-magic'd payload with extra payload bytes -- including a NUL
/// byte and other non-ASCII bytes past the 4-byte magic -- to prove the pass
/// round-trips arbitrary binary content correctly.
std::vector<uint8_t> validElfBytes() {
    return {0x7F, 'E', 'L', 'F', 0x02, 0x01, 0x01, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF, 0x10};
}

/// Writes a minimal Custom Layer Config XML mapping each (uniqueId, sourcePath)
/// pair to a <CustomLayer> entry, and returns the XML's temp file path.
std::string writeTempConfigXml(const std::vector<std::pair<std::string, std::string>>& kernels) {
    llvm::SmallString<128> path;
    const auto err = llvm::sys::fs::createTemporaryFile("custom_layer_config", "xml", path);
    VPUX_THROW_UNLESS(!err, "Failed to create temp file: {0}", err.message());

    std::ofstream f(path.str().str());
    f << "<CustomLayerCollection target=\"NPU\">\n";
    for (const auto& [uniqueId, sourcePath] : kernels) {
        f << "  <CustomLayer name=\"" << uniqueId << "\">\n";
        f << "    <Kernel type=\"PREBUILT\">\n";
        f << "      <Source path=\"" << sourcePath << "\"/>\n";
        f << "    </Kernel>\n";
        f << "  </CustomLayer>\n";
    }
    f << "</CustomLayerCollection>\n";
    f.close();
    return path.str().str();
}

}  // namespace

// =============================================================================
// LoadExternalKernelResourcesTest
// =============================================================================

struct LoadExternalKernelResourcesTest : MLIR_UnitBase {
    mlir::MLIRContext ctx;
    mlir::OpBuilder builder;
    mlir::OwningOpRef<mlir::ModuleOp> mod;

    LoadExternalKernelResourcesTest(): MLIR_UnitBase(), ctx(), builder(&ctx) {
        // The pass under test calls getShaveBinaryResources(ctx), which lazily
        // getOrLoadDialect<Core::CoreDialect>()'s on first use. MLIR forbids loading a
        // new dialect while a PassManager is running multi-threaded ("Loading a dialect
        // ... while in a multi-threaded execution context" is a fatal LLVM_ERROR, not a
        // catchable exception). Disable multithreading on this context up front, matching
        // the precedent in tests/unit/vpux_compiler/dialect/VPU/mlir_threading.cpp and
        // utils/npu_action_handler_test.cpp.
        ctx.disableMultithreading();

        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<Shave::ShaveDialect>();
        ctx.loadDialect<mlir::func::FuncDialect>();
        ctx.loadDialect<config::ConfigDialect>();

        mod = mlir::ModuleOp::create(builder.getUnknownLoc());

        // Give the module a concrete platform so config::getArch(func) resolves to a
        // real ArchKind. Without this, getArch() returns ArchKind::UNKNOWN (it does not
        // throw), but ShaveBinaryResources::getSwKernelArchString(UNKNOWN) -- called from
        // addCompiledElf() on the success path -- does throw. Only the success/dedup tests
        // below actually reach addCompiledElf(), but the platform is set unconditionally
        // for simplicity.
        (*mod)->setAttr(config::PlatformAttr::name, config::PlatformAttr::get(&ctx, config::Platform::NPU4000));

        // The pass under test calls getShaveBinaryResources(ctx) on its success path, which
        // lazily getOrLoadDialect<Core::CoreDialect>()'s on first use. mlir::PassManager::run()
        // unconditionally calls context->enterMultiThreadedExecution() around the whole pipeline
        // run (see thirdparty/llvm-project/mlir/lib/Pass/Pass.cpp), regardless of whether the
        // context's own multithreading is enabled/disabled -- so disableMultithreading() above
        // does NOT prevent the "Loading a dialect ... while in a multi-threaded execution
        // context" fatal error. The fix is to force Core::CoreDialect to be loaded up front, by
        // calling getShaveBinaryResources() once here, before any PassManager runs.
        getShaveBinaryResources(&ctx);
    }

    /// Creates an empty (no args, no results) function named `name` in the module,
    /// with the builder positioned at the start of its entry block.
    mlir::func::FuncOp addFunc(llvm::StringRef name) {
        builder.setInsertionPointToEnd(mod->getBody());
        auto funcType = builder.getFunctionType({}, {});
        auto func = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(), name, funcType);
        func.setPrivate();
        auto* entryBlock = func.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        return func;
    }

    /// Appends a Shave.ExternalKernelOp (no operands/results) to the function the
    /// builder is currently positioned in.
    Shave::ExternalKernelOp addExternalKernelOp(llvm::StringRef uniqueId) {
        return Shave::ExternalKernelOp::create(builder, builder.getUnknownLoc(), mlir::TypeRange{}, mlir::ValueRange{},
                                               builder.getDictionaryAttr({}), builder.getStringAttr(uniqueId));
    }

    /// Terminates the function the builder is currently positioned in.
    void finalizeFunc() {
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
    }

    /// Runs LoadExternalKernelResources over the test module (with the given Custom
    /// Layer Config XML path, or none) and returns whether the pass manager reported
    /// success.
    bool runPass(const std::string& customKernelConfigPath = "") {
        mlir::PassManager pm(&ctx);
        pm.addPass(Shave::createLoadExternalKernelResourcesPass(customKernelConfigPath, Logger::global()));
        return mlir::succeeded(pm.run(*mod));
    }
};

// =============================================================================
// Tests
// =============================================================================

TEST_F(LoadExternalKernelResourcesTest, T1_Success_LoadsAndRoundTripsBinaryContent) {
    const auto bytes = validElfBytes();
    TempFile tmp{writeTempBinary(bytes)};
    TempFile cfg{writeTempConfigXml({{"kernelA", tmp.path}})};

    addFunc("main");
    addExternalKernelOp("kernelA");
    finalizeFunc();

    ASSERT_TRUE(runPass(cfg.path));

    auto& resources = getShaveBinaryResources(&ctx);
    const auto archStr = ShaveBinaryResources::getSwKernelArchString(config::ArchKind::NPU40XX);

    llvm::ArrayRef<uint8_t> loaded;
    ASSERT_NO_THROW(loaded = resources.getElf("kernelA", archStr));
    ASSERT_EQ(loaded.size(), bytes.size());
    EXPECT_TRUE(std::equal(loaded.begin(), loaded.end(), bytes.begin()));
}

TEST_F(LoadExternalKernelResourcesTest, T2_DedupAcrossModule_SharedConfigEntryServesBothOccurrences) {
    // Regression test: the `loadedKernels` dedup set is declared once per module run (outside the
    // func loop), not once per function. Two functions share the same unique_id, which is served by
    // a single Custom Layer Config entry (the config XML format itself forbids duplicate CustomLayer
    // names) -- the second occurrence must be skipped via dedup rather than re-resolved/re-read.
    const auto bytes = validElfBytes();
    TempFile tmp{writeTempBinary(bytes)};
    TempFile cfg{writeTempConfigXml({{"dupKernel", tmp.path}})};

    addFunc("main1");
    addExternalKernelOp("dupKernel");
    finalizeFunc();

    addFunc("main2");
    addExternalKernelOp("dupKernel");
    finalizeFunc();

    ASSERT_TRUE(runPass(cfg.path));

    auto& resources = getShaveBinaryResources(&ctx);
    const auto archStr = ShaveBinaryResources::getSwKernelArchString(config::ArchKind::NPU40XX);

    llvm::ArrayRef<uint8_t> loaded;
    ASSERT_NO_THROW(loaded = resources.getElf("dupKernel", archStr));
    ASSERT_EQ(loaded.size(), bytes.size());
    EXPECT_TRUE(std::equal(loaded.begin(), loaded.end(), bytes.begin()));
}

TEST_F(LoadExternalKernelResourcesTest, T3_NonexistentKernelPath_SoftFailsWithoutThrowing) {
    // A single op (no dedup involved) whose config-resolved source path does not exist. The pass
    // throws vpux::Exception internally (VPUX_THROW_UNLESS(ifileElf.is_open())),
    // but vpux::ModulePass::runOnOperation() catches std::exception and converts
    // it to signalPassFailure() -- confirmed by reading src/vpux_compiler/src/utils/passes.cpp.
    // So the observable behavior is mlir::failed(), not an exception escaping pm.run().
    TempFile cfg{writeTempConfigXml({{"missingKernel", "/path/does/not/exist/kernel.elf"}})};

    addFunc("main");
    addExternalKernelOp("missingKernel");
    finalizeFunc();

    bool succeeded = true;
    EXPECT_NO_THROW(succeeded = runPass(cfg.path));
    EXPECT_FALSE(succeeded);
}

TEST_F(LoadExternalKernelResourcesTest, T4_ShortFile_SoftFails) {
    // Fewer than 4 bytes: the pass must reject it before even inspecting the
    // (nonexistent) ELF magic, via emitError()+signalPassFailure(), not a throw.
    TempFile tmp{writeTempBinary({0x7F, 'E'})};
    TempFile cfg{writeTempConfigXml({{"shortKernel", tmp.path}})};

    addFunc("main");
    addExternalKernelOp("shortKernel");
    finalizeFunc();

    bool succeeded = true;
    EXPECT_NO_THROW(succeeded = runPass(cfg.path));
    EXPECT_FALSE(succeeded);
}

TEST_F(LoadExternalKernelResourcesTest, T5_BadMagicByteZero_RegressionTest) {
    // Regression test for a bug where the ELF magic check
    // only verified bytes 1-3 ('E','L','F') and never checked that byte 0 is
    // 0x7F. Bytes 1-3 spell "ELF" here, but byte 0 is 0x00 -- this must be
    // rejected as an invalid ELF binary.
    TempFile tmp{writeTempBinary({0x00, 'E', 'L', 'F', 0x01, 0x02, 0x03, 0x04})};
    TempFile cfg{writeTempConfigXml({{"badMagicKernel", tmp.path}})};

    addFunc("main");
    addExternalKernelOp("badMagicKernel");
    finalizeFunc();

    EXPECT_FALSE(runPass(cfg.path));
}

TEST_F(LoadExternalKernelResourcesTest, T6_BadMagicGeneric_SoftFails) {
    // Generic, unambiguously-wrong 4+ byte content.
    TempFile tmp{writeTempBinary({'J', 'U', 'N', 'K', '1', '2', '3', '4'})};
    TempFile cfg{writeTempConfigXml({{"junkKernel", tmp.path}})};

    addFunc("main");
    addExternalKernelOp("junkKernel");
    finalizeFunc();

    EXPECT_FALSE(runPass(cfg.path));
}

TEST_F(LoadExternalKernelResourcesTest, T7_EmptyConfigPath_SoftSkipsWithoutError) {
    // No custom kernel config path supplied at all (the common case for LIT tests that only exercise
    // --init-compiler for unrelated structural passes, with dummy ExternalKernelOps present). The pass
    // must not attempt to resolve or load anything, and must not fail.
    addFunc("main");
    addExternalKernelOp("kernelWithNoConfig");
    finalizeFunc();

    ASSERT_TRUE(runPass());

    auto& resources = getShaveBinaryResources(&ctx);
    const auto archStr = ShaveBinaryResources::getSwKernelArchString(config::ArchKind::NPU40XX);
    llvm::ArrayRef<uint8_t> loaded;
    EXPECT_ANY_THROW(loaded = resources.getElf("kernelWithNoConfig", archStr));
}

TEST_F(LoadExternalKernelResourcesTest, T8_MissingConfigEntry_SoftFails) {
    // A config path is supplied, but it has no entry for this op's unique_id. The pass must reject
    // this via emitError()+signalPassFailure(), not a throw.
    TempFile cfg{writeTempConfigXml({{"someOtherKernel", "/irrelevant/path.elf"}})};

    addFunc("main");
    addExternalKernelOp("kernelNotInConfig");
    finalizeFunc();

    bool succeeded = true;
    EXPECT_NO_THROW(succeeded = runPass(cfg.path));
    EXPECT_FALSE(succeeded);
}
