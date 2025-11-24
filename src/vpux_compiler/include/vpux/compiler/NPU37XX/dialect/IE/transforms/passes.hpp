//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {
namespace IE {
namespace arch37xx {

//
// DefaultHWOptions
//

struct DefaultHWOptions : public IE::DefaultHWOptionsDialectBase, virtual vpux::arch37xx::DefaultHWOptionsDeviceBase {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};
    BoolOption enableConvertToSdpaExtended{*this, "convert-to-sdpa-extended",
                                           llvm::cl::desc("Enable conversion to SDPA extended"), llvm::cl::init(false)};
    BoolOption enableDecomposeGRUSequence{*this, "decompose-gru-sequence",
                                          llvm::cl::desc("Enable decompose-gru-sequence pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantize{*this, "fuse-permute-quantize",
                                         llvm::cl::desc("Enable fuse-permute-quantize pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
    BoolOption enableSwapConvertWithSWOp{*this, "swap-convert-with-sw-op",
                                         llvm::cl::desc("Enable swap-convert-with-sw-op pass"), llvm::cl::init(false)};
    BoolOption mergeUnrolledMatmul{*this, "merge-unrolled-matmul", llvm::cl::desc("Enable merging urolled Matmul ops"),
                                   llvm::cl::init(false)};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetrically quantized weights"),
                                    llvm::cl::init(false)};
    BoolOption enableMatmulMixedPrecisionDecomposition{
            *this, "enable-matmul-mixed-precision-decomposition",
            llvm::cl::desc("Enable mixed precision decomposition for matmul"), llvm::cl::init(true)};
    DoubleOption matmulMixedPrecisionDecompositionRatio{
            *this, "matmul-mixed-precision-decomposition-ratio",
            llvm::cl::desc("Determines when to enable Matmul Mixed Precision Decomposition"
                           "Ratio = (MatMul input size)/(Sum of Inputs of newly added ops by decomposition)"),
            llvm::cl::init(250.0)};

    BoolOption skipUnrollBatch{*this, "skip-unroll-batch", llvm::cl::desc("Skip unroll on batch dimension"),
                               llvm::cl::init(false)};
};

//
// Pipelines
//

void buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                               Logger log = Logger::global());

void buildInitialLowPrecisionTransformationsPipeline(mlir::OpPassManager& pm,
                                                     const IE::LowPrecisionTransformOptions& options,
                                                     Logger log = Logger::global());

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

void buildReferenceSWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// Registration
//

void registerIEPipelines();

}  // namespace arch37xx
}  // namespace IE
}  // namespace vpux
