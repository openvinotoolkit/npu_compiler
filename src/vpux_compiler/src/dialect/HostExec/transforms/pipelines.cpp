//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// registerHostExecPipelines
//

void HostExec::registerHostExecPipelines() {
    mlir::PassPipelineRegistration<>("hostexec-to-llvm",
                                     "Performs full lowering HostExec to LLVM dialect for host compilation",
                                     [](mlir::OpPassManager& pm) {
                                         buildHostExecPipeline(pm);
                                     });
}

void HostExec::buildHostExecPipeline(mlir::OpPassManager& pm, Logger /*log*/) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(HostExec::createSerializeELFToBinaryPass());
    pm.addPass(HostExec::createConvertToLLVMUMDCallsPass());
    pm.addPass(HostExec::createSerializeNetworkMetadataPass());

    // This should be placed after ConvertToLLVMUMDCalls
    // as additional arguments (e.g., L0 command list, command queue, and so on)
    // are added in ConvertToLLVMUMDCalls
    pm.addPass(mlir::LLVM::createRequestCWrappersPass());

    // Lowering to LLVM passes
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    pm.addPass(mlir::createCanonicalizerPass(grc));
}
