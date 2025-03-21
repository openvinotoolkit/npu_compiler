//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace IE {
namespace arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp = false,
                                                                  Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions : public IE::DefaultHWOptionsDialectBase, virtual vpux::arch40xx::DefaultHWOptionsDeviceBase {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantize{*this, "fuse-permute-quantize",
                                         llvm::cl::desc("Enable fuse-permute-quantize pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
    BoolOption mergeUnrolledMatmul{*this, "merge-unrolled-matmul", llvm::cl::desc("Enable merging urolled Matmul ops"),
                                   llvm::cl::init(true)};

    BoolOption enableRuntimeDequant{*this, "enable-runtime-dequant",
                                    llvm::cl::desc("Enable runtime dequantization of asymmetricly quantized weight"),
                                    llvm::cl::init(true)};
    Int64Option runtimeDequantizationLimit{
            *this, "runtime-dequantization-limit",
            llvm::cl::desc("Lower limit on weight size for runtime dequantization"
                           "Weights smaller than the limit will be statically dequantized"),
            llvm::cl::init(524'288)};  // 512kb
};

//
// Pipelines
//

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options,
                            Logger log = Logger::global());

//
// Registration
//

void registerIEPipelines();
void registerPasses();

}  // namespace arch40xx
}  // namespace IE
}  // namespace vpux
