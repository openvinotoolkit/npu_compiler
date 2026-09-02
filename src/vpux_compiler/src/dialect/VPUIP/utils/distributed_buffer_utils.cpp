//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/distributed_buffer_utils.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"

#include <mlir/IR/Diagnostics.h>

using namespace vpux;
using namespace vpux::VPUIP;

namespace {

bool verifyDistributionForShapeQuietly(mlir::MLIRContext* ctx, VPU::DistributionInfoAttr distribution, ShapeRef shape) {
    // Back-inference probes candidate distributions while searching for a legal
    // type. Suppress verifier diagnostics here; the caller will return nullopt
    // for failed candidates and the real op verifier remains responsible for
    // user-visible errors.
    mlir::ScopedDiagnosticHandler diagnosticGuard(ctx, [](mlir::Diagnostic&) {
        return mlir::success();
    });
    const auto emitError = [&]() {
        return mlir::emitError(mlir::UnknownLoc::get(ctx));
    };

    return mlir::succeeded(VPU::verify(emitError, distribution, shape.raw()));
}

std::optional<SmallVector<Shape>> getExplicitPerClusterAttr(mlir::ArrayAttr attr) {
    if (attr == nullptr) {
        return std::nullopt;
    }

    return VPU::arrayAttrToVecOfShapes(attr);
}

std::optional<SmallVector<Shape>> getPerClusterMemoryShapeOffsetsOrNull(VPUIP::DistributedBufferType distributedType) {
    if (distributedType == nullptr) {
        return std::nullopt;
    }

    const auto distribution = distributedType.getDistribution();
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        // Mirror getPerClusterMemoryShapesOrNull: explicit offsets are part of
        // the distribution contract and must be preserved exactly.
        return getExplicitPerClusterAttr(distribution.getMemoryOffsets());
    }

    if (!VPU::getPerClusterMemoryShapes(distributedType.getShape(), distribution).has_value()) {
        return std::nullopt;
    }
    return VPU::getPerClusterMemoryShapeOffsets(distributedType.getShape(), distribution);
}

}  // namespace

std::optional<VPUIP::DistributedBufferType> VPUIP::createDistributedBufferTypeOrNull(
        mlir::MLIRContext* ctx, ShapeRef shape, mlir::Type elementType, mlir::MemRefLayoutAttrInterface order,
        IndexedSymbolAttr memSpace, VPU::DistributionInfoAttr distribution,
        VPUIP::SparsityCompressionAttr sparsityCompression) {
    if (distribution == nullptr || order == nullptr ||
        DimsOrder::fromAffineMap(order.getAffineMap()).numDims() != shape.size()) {
        return std::nullopt;
    }
    if (!verifyDistributionForShapeQuietly(ctx, distribution, shape)) {
        return std::nullopt;
    }

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elementType, order, memSpace, distribution,
                                             sparsityCompression);
}

std::optional<SmallVector<Shape>> VPUIP::getPerClusterMemoryShapesOrNull(VPUIP::DistributedBufferType distributedType) {
    if (distributedType == nullptr) {
        return std::nullopt;
    }

    const auto distribution = distributedType.getDistribution();
    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        // Explicit distributions already carry the memory view. Example:
        // memory_shapes = [[1, 16, 32, 8], [1, 16, 32, 8]] is returned as-is
        // instead of being recomputed from num_tiles.
        return getExplicitPerClusterAttr(distribution.getMemoryShapes());
    }

    return VPU::getPerClusterMemoryShapes(distributedType.getShape(), distributedType.getDistribution());
}

bool VPUIP::isSupportedPerClusterMemoryShapesAndOffsets(VPUIP::DistributedBufferType distributedType) {
    if (distributedType == nullptr) {
        return true;
    }

    return getPerClusterMemoryShapesOrNull(distributedType).has_value() &&
           getPerClusterMemoryShapeOffsetsOrNull(distributedType).has_value();
}
