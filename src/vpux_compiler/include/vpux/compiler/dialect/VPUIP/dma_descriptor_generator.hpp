//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <set>
#include <tuple>

namespace vpux {
namespace VPUIP {

class PermuteDmaDescriptorGenerator {
public:
    PermuteDmaDescriptorGenerator(mlir::MLIRContext* ctx, mlir::AffineMap mergedMemPerm, vpux::Logger log);
    virtual ~PermuteDmaDescriptorGenerator() = default;

public:
    // get dma descirptor for dma ops with non distributed output type
    VPUIP::DmaDescriptorAttr generate(ShapeRef mergedInputShape, ShapeRef mergedOutputShape, Byte elemTypeSize) const;
    // get dma descirptor for dma ops with distributed output type
    SmallVector<VPUIP::DmaDescriptorAttr> generate(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                   ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
                                                   Byte elemTypeSize) const;

private:
    VPUIP::DmaDescriptorAttr generateWithTwoAxis(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                 Byte elemTypeSize) const;
    SmallVector<VPUIP::DmaDescriptorAttr> generateWithTwoAxis(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                              ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
                                                              Byte elemTypeSize) const;

    VPUIP::DmaDescriptorAttr generateWithSwapFront(ShapeRef mergedInputShape, Byte elemTypeSize) const;
    VPUIP::DmaDescriptorAttr generateWithSwapBack(ShapeRef mergedInputShape, Byte elemTypeSize) const;

    mlir::MLIRContext* _ctx;
    mlir::AffineMap _mergedMemPerm;
    vpux::Logger _log;
};

class DepthToSpaceDmaDescriptorGenerator {
public:
    DepthToSpaceDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~DepthToSpaceDmaDescriptorGenerator() = default;

public:
    VPUIP::DmaDescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                      vpux::IE::DepthToSpaceMode mode, int64_t blockSize, mlir::IntegerAttr paddedIC,
                                      mlir::IntegerAttr paddedOC) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

class SpaceToDepthDmaDescriptorGenerator {
public:
    SpaceToDepthDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~SpaceToDepthDmaDescriptorGenerator() = default;

public:
    VPUIP::DmaDescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                      vpux::IE::SpaceToDepthMode mode, int64_t blockSize) const;

private:
    VPUIP::DmaDescriptorAttr generateBlocksFirstNCHW2NCHW(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DmaDescriptorAttr generateBlocksFirstNHWC2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DmaDescriptorAttr generateBlocksFirstNCHW2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DmaDescriptorAttr generateDepthFirstNCHW2NCHW(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                         int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DmaDescriptorAttr generateDepthFirstNHWC2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                         int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DmaDescriptorAttr generateDepthFirstNCHW2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                         int64_t elemTypeSize, int64_t blockSize) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

class PerAxisTileDmaDescriptorGenerator {
public:
    PerAxisTileDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~PerAxisTileDmaDescriptorGenerator() = default;

public:
    VPUIP::DmaDescriptorAttr generate(vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t repeats,
                                      int64_t elemTypeSize) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

class ExpandDmaDescriptorGenerator {
public:
    ExpandDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~ExpandDmaDescriptorGenerator() = default;

public:
    VPUIP::DmaDescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                      mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd, int64_t elemTypeSize) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

}  // namespace VPUIP
}  // namespace vpux
