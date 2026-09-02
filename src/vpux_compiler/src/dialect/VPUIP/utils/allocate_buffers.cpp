//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPUIP {
//
// allocateBuffersOfType
//

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                               mlir::Type bufferType, bool individualBuffers) {
    auto createAllocOp = [&](mlir::Type type) {
        if (type == nullptr) {
            return mlir::Value();
        } else if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
            // drop strided layout for memref.alloc, because it doesn't support it
            if (hasStridedLayout(type)) {
                memref = mlir::MemRefType::get(memref.getShape(), memref.getElementType(), nullptr,
                                               memref.getMemorySpace());
            }
            SmallVector<mlir::Value> dynamicSizes;
            if (!memref.hasStaticShape()) {
                dynamicSizes.reserve(memref.getNumDynamicDims());
                const auto bounds = getBounds(memref);
                assert((static_cast<int64_t>(bounds.size()) == memref.getRank()) &&
                       "Bounds are required to create memref.alloc");
                for (int64_t i = 0; i < memref.getRank(); ++i) {
                    if (memref.isDynamicDim(i)) {
                        dynamicSizes.push_back(builder.create<mlir::arith::ConstantIndexOp>(
                                appendLoc(loc, "dynamic{0}", i), bounds[Dim(i)]));
                    }
                }
            }

            return static_cast<mlir::Value>(
                    builder.create<mlir::memref::AllocOp>(loc, memref, dynamicSizes).getMemref());
        } else if (auto distributedBuffer = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(type)) {
            return static_cast<mlir::Value>(
                    builder.create<VPURT::AllocDistributed>(loc, distributedBuffer, nullptr, nullptr).getBuffer());
        }
        VPUX_THROW("Unexpected type to allocate: {0}", type);
    };

    if (mlir::isa<mlir::MemRefType, vpux::VPUIP::DistributedBufferType>(bufferType)) {
        log.trace("Allocating result buffer of type '{0}'", bufferType);
        return {createAllocOp(bufferType)};
    } else if (auto sparseBufferType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(bufferType)) {
        log.trace("Allocating result buffers of type '{0}'", sparseBufferType);

        auto dataBuffer = createAllocOp(sparseBufferType.getData());
        auto sparsityMapBuffer = createAllocOp(sparseBufferType.getSparsityMap());
        auto seTableBuffer = createAllocOp(sparseBufferType.getStorageElementTable());

        if (!individualBuffers) {
            auto groupOp = builder.create<VPUIP::GroupSparseBufferOp>(
                    loc, dataBuffer, sparsityMapBuffer, seTableBuffer,
                    static_cast<bool>(sparseBufferType.getIsWeights()), sparseBufferType.getSparsityCompression(),
                    sparseBufferType.getSeAttr());
            return {groupOp.getOutput()};
        }

        SmallVector<mlir::Value> buffers{dataBuffer};
        if (sparsityMapBuffer != nullptr) {
            buffers.push_back(sparsityMapBuffer);
        }
        if (seTableBuffer != nullptr) {
            buffers.push_back(seTableBuffer);
        }
        return buffers;
    } else if (auto boundedBufferType = mlir::dyn_cast<vpux::VPUIP::BoundedBufferType>(bufferType)) {
        log.trace("Allocating result buffers of type '{0}'", boundedBufferType);

        auto dataBuffer = createAllocOp(boundedBufferType.getData());
        auto dynamicShapeBuffer = createAllocOp(boundedBufferType.getDynamicShape());

        auto groupOp = builder.create<VPUIP::GroupBoundedBufferOp>(loc, dataBuffer, dynamicShapeBuffer);
        return {groupOp.getOutput()};
    }
    VPUX_THROW("Unexpected type to allocate {0}", bufferType);
}

SmallVector<mlir::Value> allocateBuffersOfType(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter,
                                               mlir::Value value, vpux::IndexedSymbolAttr memSpace,
                                               bool individualBuffers) {
    auto bufferType = vpux::getBufferType(value);
    auto resultBufferType = bufferType.changeMemSpace(memSpace);
    return allocateBuffersOfType(log, loc, rewriter, resultBufferType, individualBuffers);
}

//
// allocateBuffers & allocateBuffersForValue using bufferizable interface
//

SmallVector<mlir::Value> allocateBuffersForValue(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                                 mlir::Value value, bool individualBuffers) {
    auto bufferType = vpux::getBufferType(value);
    log.nest().trace("Allocating result buffer of type '{0}' for value type '{1}'", bufferType, value.getType());
    return allocateBuffersOfType(log.nest(), loc, builder, bufferType, individualBuffers);
}

SmallVector<mlir::Value> allocateBuffers(const Logger& log, mlir::Location loc, mlir::OpBuilder& builder,
                                         mlir::ValueRange values, bool individualBuffers) {
    SmallVector<mlir::Value> buffers;
    for (const auto& value : values) {
        const auto valueBuffers = allocateBuffersForValue(log, loc, builder, value, individualBuffers);
        buffers.append(valueBuffers.begin(), valueBuffers.end());
    }
    return buffers;
}

mlir::Value allocateBuffer(const Logger& log, mlir::Location loc, mlir::RewriterBase& rewriter, mlir::Value value,
                           vpux::IndexedSymbolAttr memSpace) {
    auto allocatedBuffers = allocateBuffersOfType(log, loc, rewriter, value, memSpace);
    VPUX_THROW_UNLESS(allocatedBuffers.size() == 1,
                      "allocateBuffersOfType should return only one buffer but {0} buffer provided for {1}",
                      allocatedBuffers.size(), value);
    return allocatedBuffers.front();
}
}  // namespace vpux::VPUIP
