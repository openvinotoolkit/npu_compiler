//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>

#include <memory>

namespace vpux {
namespace IE {
namespace arch37xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createInsertIdentityPoolBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp = false,
                                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFusePermuteQuantizeExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(const bool seOpsEnabled = false,
                                                               const bool seExperimentalOpsEnabled = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFFTToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSubGRUSequenceToConvPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMixedPrecision(const bool enableFloatInQuantWeightsMixedMode = true,
                                                          Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeNetworkInputConvertPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertWeightsToI8Pass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createProcessAsymmetricZeroPointsForConvolutionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseOutstandingDequant(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWeightsQuantFusedIntoTaskPass(Logger log = Logger::global());

//
// Pipelines
//

void buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                      Logger log = Logger::global());

void buildMemPermutePositioningPipeline(mlir::OpPassManager& pm, const MemPermutePositioningOptions& options,
                                        Logger log = Logger::global());

void buildExpandAndOptimizeActivationChannelsPipeline(mlir::OpPassManager& pm,
                                                      const ExpandActivationChannelsOptions& options,
                                                      Logger log = Logger::global());

void buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

void buildOptimizeMemPermuteAndActivationChannelsExpandPipeline(mlir::OpPassManager& pm,
                                                                const ExpandActivationChannelsOptions& options,
                                                                Logger log = Logger::global());

void buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                               Logger log = Logger::global());

void buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const TransformOptions& options,
                                         Logger log = Logger::global());

void buildInitialLowPrecisionTransformationsPipeline(mlir::OpPassManager& pm,
                                                     const IE::LowPrecisionTransformOptions& options,
                                                     Logger log = Logger::global());

void buildDynamicShapeTransformationsPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions : public IE::DefaultHWOptionsDialectBase, virtual vpux::arch37xx::DefaultHWOptionsDeviceBase {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantize{*this, "fuse-permute-quantize",
                                         llvm::cl::desc("Enable fuse-permute-quantize pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
    BoolOption mergeUnrolledMatmul{*this, "merge-unrolled-matmul", llvm::cl::desc("Enable merging urolled Matmul ops"),
                                   llvm::cl::init(false)};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetricly quantized weight"),
                                    llvm::cl::init(false)};
    Int64Option runtimeDequantizationLimit{
            *this, "runtime-dequantization-limit",
            llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                           "Weights smaller than the limit will be statically dequantized"),
            llvm::cl::init(524'288)};  // 512kb

    BoolOption skipUnrollBatch{*this, "skip-unroll-batch", llvm::cl::desc("Skip unroll on batch dimension"),
                               llvm::cl::init(false)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// AdjustLayout
//

void buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options,
                               Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createPropagateReorderToNCEPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwapMaxPoolWithActivation(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseReordersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseStaticScalePass(Logger log = Logger::global(), bool moveScaleBeforeConcat = true);

//
// Registration
//

void registerIEPipelines();
void registerPasses();

}  // namespace arch37xx
}  // namespace IE
}  // namespace vpux
