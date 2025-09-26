//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/reduce_utils.hpp"
#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::VPU {

bool checkStrategyCompatibilityReduce(MultiClusterStrategy strategy, size_t numTiles, ShapeRef inShape,
                                      ArrayRef<int64_t> axesVec) {
    if (strategy == MultiClusterStrategy::Clustering) {
        return true;
    }

    const auto isCompatibleStrategy{[&](auto strategyToCheck, auto dimensionToCheck) {
        return strategy == strategyToCheck && inShape[dimensionToCheck] > static_cast<int64_t>(numTiles) &&
               std::find(axesVec.begin(), axesVec.end(), dimensionToCheck.ind()) == axesVec.end();
    }};

    if (isCompatibleStrategy(MultiClusterStrategy::SplitOverWidth, Dims4D::Act::W)) {
        return true;
    }

    if (isCompatibleStrategy(MultiClusterStrategy::SplitOverHeight, Dims4D::Act::H)) {
        return true;
    }

    if (isCompatibleStrategy(MultiClusterStrategy::SplitOverKernel, Dims4D::Act::C)) {
        return true;
    }

    return false;
}

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "ReduceSumOp requires 1 input and 1 output, but the number of buffers is {0}", buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(operation).count()
                                                          : getTotalCMXFragmentationAwareSize(operation).count();

    return calculateAlignedBuffersMemoryRequirement(config::getArch(operation), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool fitIntoCMXReduce(mlir::Operation* operation, llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMXReduce(operation, buffers, Byte(0));
}

}  // namespace vpux::VPU
