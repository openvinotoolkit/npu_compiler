//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils/options.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace vpux {

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUNCEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertIEToVPUM2IPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertLayers2VPUPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertDynamicQuantToVPUNCEPass(Logger log = Logger::global());

void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// LowerVPU2VPUIP
//

//
// Performs full lowering from the VPU Dialect to VPUIP Dialect.
//
// This pipeline performs full IR lowering from VPU Dialect to VPUIP Dialect,
// including Function types, call graph and return operations.
//

std::unique_ptr<mlir::Pass> createOneShotBufferizeVPU2VPUIPPass();
std::unique_ptr<mlir::Pass> createInPlaceBufferizationAnalyzePass();
std::unique_ptr<mlir::Pass> createAdjustDynamicOpsBeforeBufferizationPass();
std::unique_ptr<mlir::Pass> createAddBuffersForNetResults(bool useMemrefForHostFunctionBufferization = false,
                                                          Logger log = Logger::global());

void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, bool enableInPlaceBufferization,
                                 bool useMemrefForHostFunctionBufferization, Logger log = Logger::global());

//
// ShaveCodeGen
//
namespace ShaveCodeGen {
void buildLowerSwLayers2LinalgPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertEltwiseLayers2MathPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandLayersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAffine2LLVMPass(Logger log = Logger::global());
}  // namespace ShaveCodeGen

// ELF back-end lowerings
std::unique_ptr<mlir::Pass> createConvertVPUIP2VPUMI37XXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUMI37XX2VPUASMPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertVPUMI37XX2ELFPass(Logger log = Logger::global());

// NPU40XX ELF specific passes
std::unique_ptr<mlir::Pass> createConvertVPUIP2VPUMI40XXPass(
        Logger log = Logger::global(), bool enableMemorySideCache = false,
        AllocateShaveStackFrames allocateShaveStackFrames = AllocateShaveStackFrames::DISABLED);

std::unique_ptr<mlir::Pass> createConvertVPUMI40XX2VPUASMPass(Logger log = Logger::global(),
                                                              bool disableDmaSwFifo = false);
std::unique_ptr<mlir::Pass> createConvertVPUMI40XX2VPUASMPass(bool enablePWLM, Logger log = Logger::global(),
                                                              bool disableDmaSwFifo = false);

std::unique_ptr<mlir::Pass> createConvertVPUIPDPU2NPUReg40XXPass(
        Logger log = Logger::global(), VPU::DPUDryRunMode dpuDryRunMode = VPU::DPUDryRunMode::NONE);
std::unique_ptr<mlir::Pass> createConvertVPUASM2NPUReg40XXPass(Logger log = Logger::global(),
                                                               uint32_t modelIdentifier = 0);

// Host compile specific passes
void buildLLVMTranslationPipeline(mlir::OpPassManager& pm);

//
// Registration
//

void registerConversionPipelines();
void registerConversionPasses();

}  // namespace vpux
