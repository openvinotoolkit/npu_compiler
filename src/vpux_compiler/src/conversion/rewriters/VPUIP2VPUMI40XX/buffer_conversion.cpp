//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux::vpuip2vpumi40xx {

namespace {

mlir::Value convertITIBuffer(mlir::OpBuilder builder, mlir::Value buffer) {
    auto itiBufferType = mlir::cast<VPUIP::ITIBufferType>(buffer.getType());
    auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(buffer.getDefiningOp());

    auto byteOffset = definingOp.getByteOffset();
    auto swizzlingKey = definingOp.getSwizzlingKey();
    auto buffSec = definingOp.getMemorySpace();

    VPURT::DeclareBufferOp res;

    auto memSpace = itiBufferType.getMemSpace();
    auto tileIndex = memSpace.getIndex().value_or(0);

    auto memType = mlir::MemRefType::get(itiBufferType.getShape().raw(), itiBufferType.getElementType(),
                                         itiBufferType.getLayout(), memSpace);
    if (swizzlingKey.has_value()) {
        res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIndex, byteOffset,
                                                     swizzlingKey.value());
    } else {
        res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIndex, byteOffset);
    }

    return res.getResult();
}

mlir::Value extractFromDistributedBuffer(mlir::OpBuilder builder, mlir::Value buffer, uint32_t tileIndex) {
    auto distributedOutput = mlir::cast<VPUIP::DistributedBufferType>(buffer.getType());

    mlir::Value value;
    auto distribution = distributedOutput.getDistribution();
    auto outputMode = static_cast<std::underlying_type<VPU::DistributionMode>::type>(distribution.getMode().getValue());
    auto duplicatedMode =
            static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::DUPLICATED);
    auto multicastedMode =
            static_cast<std::underlying_type<VPU::DistributionMode>::type>(VPU::DistributionMode::MULTICASTED);
    if ((outputMode & duplicatedMode) || (outputMode & multicastedMode)) {
        auto definingOp = mlir::cast<VPURT::DeclareBufferOp>(buffer.getDefiningOp());

        auto compactType = distributedOutput.getCompactType();

        auto totalClusters = static_cast<size_t>(distribution.getNumClusters().getInt());

        auto byteOffset = definingOp.getByteOffset();
        auto swizzlingKey = definingOp.getSwizzlingKey();
        auto buffSec = definingOp.getMemorySpace();

        VPUX_THROW_WHEN(!definingOp.getSectionIndex().has_value(), "Distributed buffer without section index: {0}",
                        definingOp);

        auto clusters = parseIntArrayAttr<int64_t>(definingOp.getSectionIndex().value());

        VPUX_THROW_WHEN(clusters.size() != totalClusters,
                        "Size of distributed buffer section index ({0}) different than distribution num_clusters ({1})",
                        clusters.size(), totalClusters);

        VPURT::DeclareBufferOp res;

        auto clusters_tileIndex = std::find(clusters.begin(), clusters.end(), tileIndex);

        VPUX_THROW_WHEN(clusters_tileIndex == clusters.end(),
                        "Tile index '{0}' not found in distributed buffer section index array", tileIndex);

        auto currMemLocation = compactType.getMemorySpace().cast<IndexedSymbolAttr>().getLeafNameAttr();
        auto newMemSpace = vpux::IndexedSymbolAttr::get(currMemLocation, static_cast<size_t>(tileIndex));
        auto memType = mlir::MemRefType::get(compactType.getShape(), compactType.getElementType(),
                                             compactType.getLayout(), newMemSpace);
        if (swizzlingKey.has_value()) {
            res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIndex, byteOffset,
                                                         swizzlingKey.value());
        } else {
            res = builder.create<VPURT::DeclareBufferOp>(buffer.getLoc(), memType, buffSec, tileIndex, byteOffset);
        }

        value = res.getResult();
    } else {
        VPUX_THROW("Only distributed buffer with DUPLICATE is accepted as direct output of OP");
    }

    return value;
}

mlir::SmallVector<mlir::Value> unrollDistributedBuffer(mlir::OpBuilder builder, mlir::Value output) {
    auto declareBuffer = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());
    auto clusters = parseIntArrayAttr<int64_t>(declareBuffer.getSectionIndex().value());

    mlir::SmallVector<mlir::Value> results;

    auto distributedOutput = mlir::cast<VPUIP::DistributedBufferType>(output.getType());
    auto distribution = distributedOutput.getDistribution();
    auto totalClusters = static_cast<size_t>(distribution.getNumClusters().getInt());
    results.reserve(totalClusters);

    for (auto cluster : clusters) {
        auto bufferForCluster = extractFromDistributedBuffer(builder, output, cluster);
        results.push_back(bufferForCluster);
    }
    return results;
}

}  // namespace

mlir::SmallVector<mlir::Value> convertOrUnrollBuffer(mlir::OpBuilder builder, mlir::Value output) {
    if (!output) {
        return {};
    }

    auto type = output.getType();
    if (mlir::isa<VPUIP::ITIBufferType>(type)) {
        return {convertITIBuffer(builder, output)};
    } else if (mlir::isa<VPUIP::DistributedBufferType>(type)) {
        return unrollDistributedBuffer(builder, output);
    }
    return {output};
}

mlir::Value convertOrExtractBuffer(mlir::OpBuilder builder, mlir::Value output, uint32_t tileIndex) {
    if (!output) {
        return {};
    }

    auto type = output.getType();
    if (mlir::isa<VPUIP::ITIBufferType>(type)) {
        return convertITIBuffer(builder, output);
    } else if (mlir::isa<VPUIP::DistributedBufferType>(type)) {
        return extractFromDistributedBuffer(builder, output, tileIndex);
    }
    return output;
}

}  // namespace vpux::vpuip2vpumi40xx
