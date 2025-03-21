//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace VPU {

OverlapDistributionParams getExplicitOverlapParamsForSWOpInput(SWOpInterface swOp, ShapeRef outShape,
                                                               ArrayRef<int64_t> numTiles, ArrayRef<int64_t> alignment);

DistributionInfoAttr getSWExplicitDistributionInfoAttr(SWOpInterface swOp, ShapeRef shape,
                                                       DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                       mlir::IntegerAttr numClusters, mlir::ArrayAttr alignment,
                                                       mlir::UnitAttr uniformDistributedSegments,
                                                       const vpux::VPU::OverlapDistributionParams& overlapParams);
DistributionInfoAttr getNCEExplicitDistributionInfoAttr(NCEOpInterface nceOp, ShapeRef shape,
                                                        VPU::DistributionMode distributionMode,
                                                        mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                        mlir::ArrayAttr alignment,
                                                        mlir::UnitAttr uniformDistributedSegments,
                                                        const vpux::VPU::OverlapDistributionParams& overlapParams);
DistributionInfoAttr getConcatExplicitDistributedAttr(ShapeRef shape, VPU::DistributionMode distributionMode,
                                                      mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                      mlir::ArrayAttr alignment,
                                                      mlir::UnitAttr uniformDistributedSegments,
                                                      const vpux::VPU::OverlapDistributionParams& overlapParams,
                                                      mlir::MLIRContext* ctx);
DistributionInfoAttr getConcatExplicitDistributedAttrForNewShape(VPU::DistributionInfoAttr originDistribution,
                                                                 ShapeRef newShape, mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSliceLikeOps(VPU::DistributionInfoAttr distributionWithProperAlignment,
                                                         ArrayRef<int64_t> sliceShape, ArrayRef<int64_t> originShape,
                                                         mlir::MLIRContext* ctx);
DistributionInfoAttr getSegmentedExplicitDistrAttrForSliceLikeOps(VPU::DistributionInfoAttr distributionAttr,
                                                                  ArrayRef<int64_t> sliceOutputShape,
                                                                  mlir::ArrayAttr explicitOutputShapes,
                                                                  mlir::MLIRContext* ctx);
DistributionInfoAttr getNonOverlappedDistributedAttr(ShapeRef shape, VPU::DistributionModeAttr distrModeAttr,
                                                     mlir::ArrayAttr numTiles, mlir::IntegerAttr numClusters,
                                                     mlir::ArrayAttr alignment,
                                                     mlir::UnitAttr uniformDistributedSegments, mlir::MLIRContext* ctx);
NDTypeInterface changeShapeElemTypeForDuplicatedDistributedBuffers(NDTypeInterface buff, ShapeRef shape,
                                                                   mlir::Type elemType);

DistributionInfoAttr getExplicitDistrAttrForSparseData(VPU::DistributionInfoAttr denseDataDistribution,
                                                       ShapeRef dataShape, VPU::SEAttr seAttr, mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSparsityMap(VPU::DistributionInfoAttr denseDataDistribution,
                                                        ShapeRef sparsityMapShape, mlir::UnitAttr isWeights,
                                                        mlir::MLIRContext* ctx);
DistributionInfoAttr getExplicitDistrAttrForSETable(VPU::DistributionInfoAttr denseDataDistribution,
                                                    const size_t seSize, mlir::MLIRContext* ctx);

//
DistributionInfo getSWExplicitDistributionInfo(VPU::SWOpInterface swOp, ShapeRef shape,
                                               VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                               const int64_t numClusters, ArrayRef<int64_t> alignment,
                                               bool uniformDistributedSegments,
                                               const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getNCEExplicitDistributionInfo(VPU::NCEOpInterface nceOp, ShapeRef shape,
                                                     VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                     const int64_t numClusters, ArrayRef<int64_t> alignment,
                                                     bool uniformDistributedSegments,
                                                     const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getConcatExplicitDistributedNative(ShapeRef shape, VPU::DistributionMode distributionMode,
                                                         ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                         ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                         const vpux::VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo getExplicitDistrNativeForSliceLikeOps(
        const VPU::DistributionInfo& distributionWithProperAlignment, ArrayRef<int64_t> sliceShape,
        ArrayRef<int64_t> originShape);

VPU::DistributionInfo getSegmentedExplicitDistrNativeForSliceLikeOps(const VPU::DistributionInfo& distribution,
                                                                     ArrayRef<int64_t> sliceOutputShape,
                                                                     ArrayRef<SmallVector<int64_t>> explicitShapes);

DistributionInfo getNonOverlappedDistributedNative(ShapeRef shape, VPU::DistributionMode distrMode,
                                                   ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                   ArrayRef<int64_t> alignment, bool uniformDistributedSegments);

VPU::DistributionInfo getConcatExplicitDistributedNativeForNewShape(const VPU::DistributionInfo& originDistribution,
                                                                    vpux::ShapeRef newShape);

DistributionInfoAttr getExplicitDistrAttrForActualDataFromSparseType(mlir::Type origType);

}  // namespace VPU
}  // namespace vpux
