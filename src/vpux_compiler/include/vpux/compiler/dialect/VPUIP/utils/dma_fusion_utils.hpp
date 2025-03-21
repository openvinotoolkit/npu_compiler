//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <functional>

namespace vpux {
namespace VPUIP {

struct StrideInfo {
    bool feasible = false;    // Feasible to fuse grouped DMA. Non feasible cases includes non-linear strides or
                              // variations over clusters
    bool isExplicit = false;  // Requires explicit strides configuration, for example leading stride between
                              // declarations isn't equals to tensor stride. It could because of previous tiling on
                              // other dim, see @DontFuseStridedBuffer2BufferDma test
    vpux::Byte value;         // Explicit stride value
};

using StrideProviderFunc = std::function<StrideInfo(vpux::Logger, SmallVector<VPURT::TaskOp>)>;

bool hasCompatibleTypes(VPUIP::NNDMAOp currentDma, VPUIP::NNDMAOp nextDma);

Const::DeclareOp getCommonConstant(SmallVector<VPURT::TaskOp> tasks);

VPURT::DeclareBufferOp getCommonBuffer(SmallVector<VPURT::TaskOp> tasks, bool input);

void handleDmaFusion(mlir::func::FuncOp funcOp, vpux::Logger log, const StrideProviderFunc& srcStrideProvider,
                     const StrideProviderFunc& dstStrideProvider,
                     const std::function<size_t(SmallVector<VPURT::TaskOp>)>& newPortProvider);

mlir::Value getInput(VPURT::TaskOp taskOp);

mlir::Value getOutput(VPURT::TaskOp taskOp);

}  // namespace VPUIP
}  // namespace vpux
