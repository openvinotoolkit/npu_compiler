//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux {
namespace VPUIP {

// Replace permute with DMA
std::optional<Shape> getPermuteDMAInputShape(NDTypeInterface inType, NDTypeInterface outType, mlir::AffineMap memPerm,
                                             vpux::Logger log);
std::optional<Shape> getPermuteDMAOutputShape(NDTypeInterface inType, NDTypeInterface outType, mlir::AffineMap memPerm,
                                              vpux::Logger log);
std::optional<SmallVector<Shape>> getPermuteDMASubInputShapes(VPU::ArchKind arch, NDTypeInterface inType,
                                                              NDTypeInterface outType, mlir::AffineMap memPerm,
                                                              int64_t dmaPortCount, vpux::Logger log);
SmallVector<vpux::Shape> getPermuteDMASubOutputShapes(SmallVector<vpux::Shape> subInputShapes,
                                                      vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                                      mlir::AffineMap memPerm);
// Get the real permutation map
mlir::AffineMap getPermuteDMAMergedMemPerm(vpux::NDTypeInterface inType, mlir::AffineMap memPerm);

// Get the numPlane dim of the merged input shape
Dim getPermuteDMANumPlaneDim(vpux::NDTypeInterface inType, mlir::AffineMap memPerm);

// Check if is a [d0, d1, d2] -> [d2, d1, d0] permutation that needs to be split in order to use permuteDMA
bool isSplitNeededForPermuteDMA(vpux::NDTypeInterface inType, mlir::AffineMap memPerm);

// Get Tiled dim of the distributed output of PermuteDMA op
std::optional<Dim> getTileDimForPermuteDMA(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType,
                                           mlir::AffineMap memPerm, VPUIP::DistributedBufferType distributedOutputType,
                                           vpux::Logger log);
SmallVector<DimArr> getPermuteDMAOutputMergedDimList(vpux::NDTypeInterface outputType, ShapeRef mergedOutputShape);

// Check the tile axis is compatible for generating the DMA descriptor
// For PermuteDMA op under cluster tiling, the distributed output may be tiled on different axis. It will cause the
// different value of DMA descriptors. Note that for 3 axis permutation, it only support duplicated output for now.
bool doesPermuteDMATileDimSupportWrapInCluster(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType,
                                               mlir::AffineMap memPerm,
                                               VPUIP::DistributedBufferType distributedOutputType, vpux::Logger log);

std::optional<mlir::AffineMap> getMemPermFromSwKernel(VPUIP::SwKernelOp swKernelTask);
bool isMemPermSwKernel(VPUIP::SwKernelOp swKernelTask);

// Replace DepthToSpace with DMA
using DepthToSpaceReturnType = std::tuple<IE::DepthToSpaceModeAttr, mlir::IntegerAttr, IE::ChannelPaddingAttr>;
std::optional<DepthToSpaceReturnType> getDepthToSpaceSwKernelAttr(VPUIP::SwKernelOp swKernelTask);
bool isDepthToSpaceSwKernel(VPUIP::SwKernelOp swKernelTask);
bool isCompatibleWithMultiClusterNNDMA(VPU::DepthToSpaceOp op, vpux::ShapeRef nTilesOnDim);

// Replace SpaceToDepth with DMA
std::optional<std::pair<IE::SpaceToDepthModeAttr, mlir::IntegerAttr>> getSpaceToDepthSwKernelAttr(
        VPUIP::SwKernelOp swKernelTask);
bool isSpaceToDepthSwKernel(VPUIP::SwKernelOp swKernelTask);

// Replace PerAxisTile with DMA

struct PerAxisTileAttr {
    mlir::IntegerAttr axis;
    mlir::IntegerAttr repeats;
};

bool isTileSwKernel(VPUIP::SwKernelOp swKernelTask);
bool isPerAxisTileSwKernel(VPUIP::SwKernelOp swKernelTask);
std::optional<VPUIP::PerAxisTileAttr> getPerAxisTileSwKernelAttr(VPUIP::SwKernelOp swKernelTask);

// All of PerAxisTileDMA should with 3D shape rank to simplify the Descriptor
// This function merge the original shape to 3D and returns the input and output merged shape
std::pair<vpux::Shape, vpux::Shape> getPerAxisTileDMAMergedShape(vpux::NDTypeInterface inType,
                                                                 vpux::NDTypeInterface outType, int64_t axis,
                                                                 int64_t tiles);
SmallVector<vpux::Shape> getPerAxisTileDMASubShapes(VPU::ArchKind arch, vpux::ShapeRef shape);

// Public interface
bool doesSWLayerFitIntoCMX(mlir::Operation* op, vpux::Logger log);
bool isLegalConvertToDMA(mlir::Operation* op, vpux::Logger log, bool checkCMXSize = true);
bool isLegalAndBeneficialConvertToDMA(mlir::Operation* op, vpux::Logger log = Logger::global());
vpux::NDTypeInterface changeShapeWithMemShape(vpux::NDTypeInterface* type, vpux::ShapeRef newMemShape, DimsOrder order);
VPUIP::DMADescriptorAttr updateNumPlanes(VPUIP::DMADescriptorAttr dmaDescriptor, int64_t numPlane);

VPURT::DeclareBufferOp createNewDeclareBuffer(mlir::PatternRewriter& rewriter, mlir::Operation* insertionPoint,
                                              VPURT::DeclareBufferOp declBuff, vpux::NDTypeInterface newType,
                                              int64_t offset);

inline VPURT::DeclareBufferOp createNewDeclareBuffer(mlir::PatternRewriter& rewriter, mlir::Operation* insertionPoint,
                                                     VPURT::DeclareBufferOp declBuff, vpux::NDTypeInterface newType,
                                                     Byte offset) {
    return createNewDeclareBuffer(rewriter, insertionPoint, declBuff, newType, offset.count());
}
}  // namespace VPUIP
}  // namespace vpux
