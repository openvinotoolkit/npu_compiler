//
// Copyright (C) 2025-2026 Intel Corporation
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
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// registerHostExecPipelines
//

void HostExec::registerHostExecPipelines() {
    mlir::PassPipelineRegistration<HostExec::HostExecOptions>(
            "output-shape-predict", "Predicts the output shapes for host compilation",
            [](mlir::OpPassManager& pm, const HostExec::HostExecOptions&) {
                buildOutputShapePredictPipeline(pm, Logger::global());
            });
}

void HostExec::buildOutputShapePredictPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(HostExec::createExtractReturnShapesPass(log));
    pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    pm.addPass(HostExec::createOutlineDimOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void HostExec::buildBytecodeBackendPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(HostExec::createSerializeELFToBinaryPass(log));
    pm.addPass(bytecode::createSerializeKernelsToBytecodePass(log));
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(HostExec::createInlineMainBatchingCallsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(mlir::createCSEPass());
    pm.addPass(bytecode::createConvertHostcodeToBytecodePass(log));
    pm.addPass(bytecode::createInjectBytecodeMetadataPass(log));
    pm.addPass(bytecode::createConvertIntermediateBytecodeOpsPass(log));
    pm.addPass(bytecode::createDeduplicateAndLowerImmRegisterOpsPass());
    pm.addPass(bytecode::createAllocateBytecodeRegistersPass(log));
}
