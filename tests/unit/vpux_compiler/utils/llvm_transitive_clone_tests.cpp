//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/utils/llvm_to_binary.hpp"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace vpux;

struct CloneTestParams {
    llvm::StringLiteral inputIR;
    llvm::StringLiteral entry;
    SmallVector<StringRef> inlinedOps;
};

class LLVMTransitiveCloneTests : public testing::TestWithParam<CloneTestParams> {};

TEST_P(LLVMTransitiveCloneTests, CloneFunctions) {
    const auto params = GetParam();
    const llvm::StringLiteral inputIR = params.inputIR;
    const llvm::StringLiteral entry = params.entry;
    auto registry = vpux::createDialectRegistry();
    auto interfacesRegistry = vpux::createInterfacesRegistry(vpux::VPU::ArchKind::NPU40XX);
    interfacesRegistry->registerInterfaces(registry);

    mlir::MLIRContext ctx(registry);
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(inputIR, &ctx);
    ASSERT_TRUE(module.get() != nullptr);
    auto symref = mlir::FlatSymbolRefAttr::get(&ctx, entry);

    auto moduleBuilder = mlir::OpBuilder::atBlockBegin(module->getBody());
    auto tmpModuleOp = moduleBuilder.create<mlir::ModuleOp>(module->getLoc(), llvm::StringRef("TempModule"));

    transitivelyCloneFunctions(tmpModuleOp, *module, symref);
    llvm::SetVector<StringRef> inlinedOps(params.inlinedOps.begin(), params.inlinedOps.end());

    // Make sure we've cloned all expected functions.
    for (auto opName : params.inlinedOps) {
        auto opRef = mlir::FlatSymbolRefAttr::get(&ctx, opName);
        ASSERT_TRUE(tmpModuleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(opRef) != nullptr);
    }

    // Make sure we didn't clone any other functions.
    auto numFunc = std::distance(tmpModuleOp.getBody()->begin(), tmpModuleOp.getBody()->end());
    ASSERT_TRUE(static_cast<size_t>(numFunc) == params.inlinedOps.size());
}

llvm::StringLiteral testOneFunc = R"(
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @foo(%arg0: !llvm.ptr) attributes {dso_local} {
      llvm.return
    }
    llvm.func @notRequired() {
      llvm.return
    }
  })";

llvm::StringLiteral testOneCall = R"(
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @foo(%arg0: !llvm.ptr) attributes {dso_local} {
      llvm.call @bar() : () -> ()
      llvm.return
    }
    llvm.func @bar() {
      llvm.return
    }
    llvm.func @notRequired() {
      llvm.return
    }
  })";

llvm::StringLiteral testOneCallOneAddroffOneLeaf = R"(
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @foo(%arg0: !llvm.ptr) attributes {dso_local} {
      llvm.call @bar() : () -> ()
      llvm.return
    }
    llvm.func internal @bar() {
      %0 = llvm.mlir.addressof @baz : !llvm.ptr
      llvm.return
    }
    llvm.func @baz() {
      llvm.return
    }
    llvm.func @notRequired() {
      llvm.return
    }
  })";

llvm::StringLiteral testTwoCalls = R"(
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @foo(%arg0: !llvm.ptr) attributes {dso_local} {
      llvm.call @bar() : () -> ()
      llvm.return
    }
    llvm.func @bar() {
      %0 = llvm.mlir.addressof @baz : !llvm.ptr
      llvm.call @baz() : () -> ()
      llvm.return
    }
    llvm.func @baz() {
      llvm.return
    }
    llvm.func @notRequired() {
      llvm.return
    }
  })";

llvm::StringLiteral testOneCallOneAddroffTwoLeafs = R"(
  module @VPU.SW {
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    llvm.func @foo(%arg0: !llvm.ptr) attributes {dso_local} {
      llvm.call @bar() : () -> ()
      %0 = llvm.mlir.addressof @baz : !llvm.ptr
      llvm.return
    }
    llvm.func @bar() {
      llvm.return
    }
    llvm.func @baz() {
      llvm.return
    }
    llvm.func @notRequired() {
      llvm.return
    }
  })";

std::vector<CloneTestParams> cloneParamValues = {{testOneFunc, "foo", {"foo"}},
                                                 {testOneCall, "foo", {"foo", "bar"}},
                                                 {testOneCallOneAddroffOneLeaf, "foo", {"foo", "bar", "baz"}},
                                                 {testTwoCalls, "foo", {"foo", "bar", "baz"}},
                                                 {testOneCallOneAddroffTwoLeafs, "foo", {"foo", "bar", "baz"}}};

INSTANTIATE_TEST_SUITE_P(ShaveCodeGen, LLVMTransitiveCloneTests, testing::ValuesIn(cloneParamValues));
