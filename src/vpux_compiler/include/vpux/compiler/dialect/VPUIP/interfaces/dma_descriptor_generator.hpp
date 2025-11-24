//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
namespace VPUIP {

class PermuteDmaDescriptorGenerator {
public:
    PermuteDmaDescriptorGenerator(mlir::MLIRContext* ctx, mlir::AffineMap mergedMemPerm, vpux::Logger log);
    virtual ~PermuteDmaDescriptorGenerator() = default;

public:
    // get dma descirptor for dma ops with non distributed output type
    VPUIP::DMADescriptorAttr generate(ShapeRef mergedInputShape, ShapeRef mergedOutputShape, Byte elemTypeSize) const;
    // get dma descirptor for dma ops with distributed output type
    SmallVector<VPUIP::DMADescriptorAttr> generate(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                   ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
                                                   Byte elemTypeSize) const;

private:
    VPUIP::DMADescriptorAttr generateWithTwoAxis(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                 Byte elemTypeSize) const;
    SmallVector<VPUIP::DMADescriptorAttr> generateWithTwoAxis(ShapeRef mergedInputShape, ShapeRef mergedOutputShape,
                                                              ArrayRef<Shape> mergedSubOutputShapes, Dim tileDim,
                                                              Byte elemTypeSize) const;

    VPUIP::DMADescriptorAttr generateWithSwapFront(ShapeRef mergedInputShape, Byte elemTypeSize) const;
    VPUIP::DMADescriptorAttr generateWithSwapBack(ShapeRef mergedInputShape, Byte elemTypeSize) const;

    mlir::MLIRContext* _ctx;
    mlir::AffineMap _mergedMemPerm;
    vpux::Logger _log;
};

class DepthToSpaceDmaDescriptorGenerator {
public:
    DepthToSpaceDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~DepthToSpaceDmaDescriptorGenerator() = default;

public:
    VPUIP::DMADescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType, ShapeRef inShape,
                                      ShapeRef outShape, vpux::IE::DepthToSpaceMode mode, int64_t blockSize,
                                      mlir::IntegerAttr paddedIC, mlir::IntegerAttr paddedOC) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

class SpaceToDepthDmaDescriptorGenerator {
public:
    SpaceToDepthDmaDescriptorGenerator(mlir::MLIRContext* ctx, vpux::Logger log);
    virtual ~SpaceToDepthDmaDescriptorGenerator() = default;

public:
    VPUIP::DMADescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                      vpux::IE::SpaceToDepthMode mode, int64_t blockSize) const;

private:
    VPUIP::DMADescriptorAttr generateBlocksFirstNCHW2NCHW(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DMADescriptorAttr generateBlocksFirstNHWC2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DMADescriptorAttr generateBlocksFirstNCHW2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                          int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DMADescriptorAttr generateDepthFirstNCHW2NCHW(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                         int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DMADescriptorAttr generateDepthFirstNHWC2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
                                                         int64_t elemTypeSize, int64_t blockSize) const;
    VPUIP::DMADescriptorAttr generateDepthFirstNCHW2NHWC(vpux::ShapeRef inShape, vpux::ShapeRef outShape,
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
    VPUIP::DMADescriptorAttr generate(vpux::ShapeRef inShape, vpux::ShapeRef outShape, int64_t repeats,
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
    VPUIP::DMADescriptorAttr generate(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                      mlir::ArrayAttr padsBegin, mlir::ArrayAttr padsEnd, int64_t elemTypeSize) const;

private:
    mlir::MLIRContext* _ctx;
    vpux::Logger _log;
};

}  // namespace VPUIP
}  // namespace vpux
