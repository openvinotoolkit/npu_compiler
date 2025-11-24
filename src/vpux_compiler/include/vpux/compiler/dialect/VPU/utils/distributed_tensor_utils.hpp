//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distribution_info.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
enum class MultiClusterStrategy : uint64_t;
class ClusteredOpInterface;
class CopyOp;
class DistributedTensorType;
class DistributedTypeInterface;
class NCEOpInterface;
class SiblingOpsAnalysis;
class SparseTensorType;
class SWOpInterface;
class UnrolledTypeOp;
class GatherDMAOp;
struct OverlapDistributionParams;
}  // namespace vpux::VPU

namespace vpux::VPUIP {
enum class NCETaskType : uint64_t;
class DistributedBufferType;
}  // namespace vpux::VPUIP

namespace vpux {
namespace VPU {

constexpr int64_t KMB_DPU_CHANNELS_ALIGNMENT = 16;
constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";
const SmallVector<int64_t> DISTRIBUTED_C_ALIGNMENT = SmallVector<int64_t>{1, 16, 1, 1};
const SmallVector<int64_t> DISTRIBUTED_N_ALIGNMENT = SmallVector<int64_t>{16, 1, 1, 1};

using TensorDistributionMap = llvm::DenseMap<mlir::Type, VPU::DistributionInfo>;

VPU::DistributionInfoAttr updateSliceLikeOpsAlignment(mlir::MLIRContext* ctx, vpux::ShapeRef inShape,
                                                      vpux::ShapeRef sliceShape,
                                                      VPU::DistributionInfoAttr originDistribution);
bool isSOCSegmentedOp(mlir::Operation* op);
bool isSOCSegmentedSWOp(mlir::Operation* op);
bool isSOCSegmentedNCEOp(mlir::Operation* op);
bool inputProducersCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSegmentedInputCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSOKSegmentedOutputCompatible(mlir::Operation* op);
bool hasDistributedTypesIO(mlir::Operation* op);
int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation,
                                                  bool uniformDistributedSegments = true);
int64_t getNumberOfClustersForSpatialDim(int64_t outputSpatialDim, int64_t numClustersForCompilation,
                                         bool uniformDistributedSegments = true);
SmallVector<int64_t> getActivationTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                 int64_t numClustersAvailableForCompilation,
                                                 VPU::MultiClusterStrategy strategy,
                                                 vpux::NDTypeInterface inputType = nullptr);
std::optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::ClusteredOpInterface clusteredOp,
                                                                 int64_t numClusters,
                                                                 VPU::MultiClusterStrategy strategy,
                                                                 vpux::NDTypeInterface inputType = nullptr,
                                                                 vpux::NDTypeInterface outputType = nullptr);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                             int64_t numClustersAvailableForCompilation,
                                             VPU::MultiClusterStrategy strategy,
                                             vpux::NDTypeInterface outputType = nullptr);
std::optional<SmallVector<int64_t>> getOutputTensorAlignment(VPU::MultiClusterStrategy strategy);
std::optional<vpux::NDTypeInterface> adjustOutputAlignmentForSOH(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface originalDistType);

SmallVector<int64_t> getWeightsTensorNumTiles(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface tensorType,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                   vpux::NDTypeInterface tensorType,
                                                   int64_t numClustersAvailableForCompilation,
                                                   VPU::MultiClusterStrategy strategy);
DistributionMode getActivationTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                     VPU::MultiClusterStrategy strategy);
DistributionMode getActivationTensorDistributionMode(VPU::GatherDMAOp op, VPU::MultiClusterStrategy strategy,
                                                     mlir::Value operand);
DistributionMode getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getOutputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                 VPU::MultiClusterStrategy strategy, vpux::NDTypeInterface outputType);

int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth, bool isInputSparse);
int64_t getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse,
                                     config::ArchKind arch);
bool isSOHSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool DWTypeOp,
                         config::ArchKind arch);
bool isSOGSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool DWTypeOp,
                         config::ArchKind arch);

vpux::VPU::CopyOp createDistributedCopyIn(mlir::PatternRewriter& rewriter, VPU::ClusteredOpInterface clusteredOp,
                                          mlir::Value input, vpux::NDTypeInterface inputTensorDistributedTensorType);

vpux::VPU::UnrolledTypeOp createDistributedUnrolledTypeIn(mlir::PatternRewriter& rewriter,
                                                          VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                          vpux::NDTypeInterface inputTensorDistributedTensorType);

vpux::NDTypeInterface getDistributedTypeFromInput(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy,
                                                  const bool hasExplicitDistributedAttr,
                                                  SiblingOpsAnalysis& siblingsAnalysis);

vpux::NDTypeInterface getSwDistributedTypeForOpOperand(VPU::ClusteredOpInterface clusteredOp, mlir::OpOperand& operand,
                                                       SiblingOpsAnalysis& siblingsAnalysis,
                                                       bool hasExplicitDistributedAttr);

bool getUniformDistributedSegments(VPU::ClusteredOpInterface clusteredOp, ArrayRef<int64_t> shape,
                                   VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                   ArrayRef<int64_t> alignment);

VPU::DistributedTensorType createExplicitDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                       vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       int64_t numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments, bool hasExplicitDistributedAttr,
                                                       const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       int64_t numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments, ArrayRef<int64_t> kernel = {},
                                                       VPU::PaddingAttr pad = nullptr, ArrayRef<int64_t> stride = {},
                                                       bool equalComputeAndMemoryView = false);

VPU::SparseTensorType createSparseTensorDistributedType(
        VPU::ClusteredOpInterface clusteredOp, VPU::SparseTensorType sparseInputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
        bool hasExplicitDistributedAttr, const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(mlir::Operation* viewLikeOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       int64_t optimalNumberOfClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments, ArrayRef<int64_t> kernel = {},
                                                       VPU::PaddingAttr pad = nullptr, ArrayRef<int64_t> stride = {});

VPU::DistributedTensorType createDistributedTensorType(VPU::SWOpInterface swOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       int64_t numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments);
VPU::DistributedTensorType createDistributedTensorType(VPU::GatherDMAOp gatherDMAOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       int64_t numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, mlir::Value operand, vpux::NDTypeInterface inputType,
        int64_t numClusters, vpux::NDTypeInterface tiledOutputType = nullptr,
        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             int64_t numClusters);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        ArrayRef<vpux::NDTypeInterface> inputTypes = {}, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
        bool hasExplicitDistributedAttr = false,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, mlir::Value operand, vpux::NDTypeInterface inputType,
        int64_t numClusters, VPU::MultiClusterStrategy customStrategy, ArrayRef<int64_t> customAlignment = {},
        vpux::NDTypeInterface tiledOutputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             int64_t numClusters,
                                                             VPU::MultiClusterStrategy customStrategy);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, ArrayRef<vpux::NDTypeInterface> inputTypes = {},
        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()), bool hasExplicitDistributedAttr = false,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

vpux::NDTypeInterface getDistributedOutputTensorType(
        VPU::ClusteredOpInterface clusteredOp, int64_t numClusters, VPU::MultiClusterStrategy strategy,
        vpux::NDTypeInterface outputTensorType, bool hasExplicitDistributedAttr, bool alignForSOH = true,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

vpux::NDTypeInterface getDistributedOutputTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                     vpux::NDTypeInterface outputTensorType,
                                                     SiblingOpsAnalysis& siblingsAnalysis,
                                                     VPU::MultiClusterStrategy strategy,
                                                     const bool hasExplicitDistributedAttr);
// Get distributed output type for clustered op and vf op
vpux::NDTypeInterface getDistributedOutputType(mlir::Operation* op);

// Get distributed input type for clustered op and vf op
vpux::NDTypeInterface getDistributedInputType(mlir::Operation* op, mlir::Value operand);

bool hasSpillDueToIncompatibleDistributionMode(VPU::DistributedTensorType distributedInType,
                                               VPU::DistributedTensorType distributedOutType);

bool isSegmentedOverlappedAxisSameAsSliceAxis(mlir::ArrayAttr numTiles, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> sliceShape);

bool isSegmentedOverlappedAxisSameAsSliceAxis(ArrayRef<int64_t> numTiles, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> sliceShape);

bool isSegmentedLikeDistributionMode(vpux::NDTypeInterface sourceType, const VPU::DistributionInfo& sourceDistribution);

mlir::Type getCompactTypeFromDistributed(mlir::Type originalType);

Shape getLargestClusterOutputShape(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
bool isDWOpAndNeedsAlign(config::ArchKind arch, VPUIP::NCETaskType nceTaskType);
bool isEltwiseOpAndNeedsAlign(VPU::ClusteredOpInterface nceOp);
bool isSWOpChannelAlignmentCompatible(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                      vpux::NDTypeInterface outputType);

bool isSWOpWithAlignedChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType = nullptr,
                                 vpux::NDTypeInterface outputType = nullptr);
bool isWeightsDequant(mlir::Operation* origOp);

VPU::DistributedTensorType composeDistributedType(VPU::ClusteredOpInterface permuteOp,
                                                  VPU::DistributedTensorType distType, vpux::NDTypeInterface ndType,
                                                  mlir::ArrayAttr tileOverDim,
                                                  const OverlapDistributionParams& fusedOverlapParams,
                                                  bool enableExplicitDistributionInfoAttr = false,
                                                  bool equalComputeAndMemoryView = false);

mlir::Operation* getNextCompressConv(mlir::Operation* nceOp);
mlir::Type fuseOverlapParams(VPU::ClusteredOpInterface permuteOp, VPU::DistributedTensorType distType,
                             mlir::Operation* nextConv, bool enableExplicitDistributionInfoAttr = false);

SmallVector<int64_t> getNonOneDimInds(ArrayRef<int64_t> inputArray);

/**
 * @brief OVERLAPPED cluster tiling is only supported for dimensions H and W
 *        If it is actually SEGMENTED, this function can be used to replace the mode with SEGMENTED
 */
mlir::FailureOr<VPU::DistributionInfoAttr> legalizeCastedDistribution(VPU::DistributionInfoAttr castedDistribution,
                                                                      mlir::MLIRContext* ctx);
mlir::FailureOr<VPU::DistributionInfo> legalizeCastedDistribution(VPU::DistributionInfo& castedDistribution);

//
// Create DistributionInfoAttr
//

VPU::DistributionInfo createDistributionInfo(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType,
                                             DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                             int64_t numClusters, ArrayRef<int64_t> alignment,
                                             bool uniformDistributedSegments, bool hasExplicitDistributedAttr,
                                             const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributionInfo createDistributionInfo(VPU::NCEOpInterface nceOp, DistributionMode distributionMode,
                                             ArrayRef<int64_t> numTiles, int64_t numClusters,
                                             ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                             ArrayRef<int64_t> kernel = {},
                                             const std::optional<VPU::Padding>& pad = std::nullopt,
                                             ArrayRef<int64_t> stride = {}, bool equalComputeAndMemoryView = false);

VPU::DistributionInfo createDistributionInfo(mlir::Operation* viewLikeOp, DistributionMode distributionMode,
                                             ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters,
                                             ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                             ArrayRef<int64_t> kernel = {},
                                             const std::optional<VPU::Padding>& pad = std::nullopt,
                                             ArrayRef<int64_t> stride = {});

VPU::DistributionInfo createDistributionInfo(VPU::SWOpInterface swOp, DistributionMode distributionMode,
                                             ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters,
                                             ArrayRef<int64_t> alignment, bool uniformDistributedSegments);

VPU::DistributionInfo createDistributionInfo(VPU::GatherDMAOp gatherDMAOp, DistributionMode distributionMode,
                                             ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters,
                                             ArrayRef<int64_t> alignment, bool uniformDistributedSegments);

VPU::DistributionInfo composeDistributedAttr(VPU::ClusteredOpInterface permuteOp, VPU::DistributedTensorType distType,
                                             vpux::NDTypeInterface ndType, mlir::ArrayAttr tileOverDim,
                                             const OverlapDistributionParams& fusedOverlapParams,
                                             bool enableExplicitDistributionInfoAttr = false,
                                             bool equalComputeAndMemoryView = false);

TensorDistributionMap getOutputDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                      vpux::NDTypeInterface outputType, int64_t numClusters,
                                                      VPU::MultiClusterStrategy customStrategy,
                                                      SiblingOpsAnalysis& siblingsAnalysis,
                                                      ArrayRef<vpux::NDTypeInterface> inputTypes = {},
                                                      const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
                                                      bool hasExplicitDistributedAttr = false);

TensorDistributionMap getActivationDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp, mlir::Value operand,
                                                          vpux::NDTypeInterface inputType, int64_t numClusters,
                                                          vpux::NDTypeInterface tiledOutputType = nullptr,
                                                          const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

TensorDistributionMap getOutputDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                      vpux::NDTypeInterface outputType, int64_t numClusters,
                                                      ArrayRef<vpux::NDTypeInterface> inputTypes = {},
                                                      const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
                                                      bool hasExplicitDistributedAttr = false);

TensorDistributionMap getActivationDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp, mlir::Value operand,
                                                          vpux::NDTypeInterface inputType, int64_t numClusters,
                                                          SiblingOpsAnalysis& siblingsAnalysis,
                                                          vpux::NDTypeInterface tiledOutputType = nullptr,
                                                          const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

TensorDistributionMap getOutputDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                      vpux::NDTypeInterface outputType, int64_t numClusters,
                                                      SiblingOpsAnalysis& siblingsAnalysis,
                                                      ArrayRef<vpux::NDTypeInterface> inputTypes = {},
                                                      const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
                                                      bool hasExplicitDistributedAttr = false);

TensorDistributionMap getActivationDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp, mlir::Value operand,
                                                          vpux::NDTypeInterface inputType, int64_t numClusters,
                                                          VPU::MultiClusterStrategy customStrategy,
                                                          SiblingOpsAnalysis& siblingsAnalysis,
                                                          ArrayRef<int64_t> customAlignment = {},
                                                          vpux::NDTypeInterface tiledOutputType = nullptr,
                                                          const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

TensorDistributionMap getFilterDistributionAttrFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                      int64_t numClusters, VPU::MultiClusterStrategy customStrategy);

vpux::Byte getTotalAllocSizeWithDistribution(vpux::NDTypeInterface type, const VPU::DistributionInfo& distribution);

vpux::Byte getTotalAllocSizeWithDistribution(vpux::NDTypeInterface type, const TensorDistributionMap& distributions);

vpux::NDTypeInterface getDistributedTypeFromDistributionMap(vpux::NDTypeInterface type,
                                                            const TensorDistributionMap& distributionMap);

TensorDistributionMap getDistributionMapFromDistributedType(vpux::NDTypeInterface type);

/**
 * @brief SEP DW.Conv has strict channel restrictions. A SEP DW.Conv workload must have 16/32/64
 *        channel size and workload must start from 0 offset in cluster.
 *        As such, when dividing channels for SOK multiclustering purposes, each individual
 *        cluster must respect the above mentioned conditions. This util ensures that.
 *
 *        E.g. 144 channels divided into 3 clusters:
 *                * non-SEP DW.Conv would have [48, 48, 48] as a valid distribution;
 *                * SEP DW.Conv must have [64, 64, 16] to be legal.
 */
mlir::FailureOr<OverlapDistributionParams> getSupportedPerClusterShapesAndOffsetsForSEPDWConv(
        VPU::ClusteredOpInterface clusteredOp, ShapeRef shape, int64_t numClusters, Dim tileDim, bool isBroadcasted);

mlir::LogicalResult sameLayout(VPU::DistributedTensorType inDistributedType,
                               VPU::DistributedTensorType outDistributedType, LogCb logCb = emptyLogCb);
mlir::LogicalResult sameLayout(VPUIP::DistributedBufferType inDistributedType,
                               VPUIP::DistributedBufferType outDistributedType, LogCb logCb = emptyLogCb);

bool arePerClusterDistributionMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface srcType,
                                                         VPU::DistributionInfo& sourceDistribution,
                                                         vpux::NDTypeInterface targetType,
                                                         VPU::DistributionInfo& targetDistribution);

bool arePerClusterMemoryShapeAndOffsetsEqual(vpux::NDTypeInterface sourceType,
                                             const VPU::DistributionInfo& sourceDistribution,
                                             const VPU::DistributionInfo& targetDistribution);

mlir::LogicalResult areDistributionsCompatible(vpux::NDTypeInterface srcType, VPU::DistributionInfo& sourceAttr,
                                               vpux::NDTypeInterface targetType, VPU::DistributionInfo& targetAttr,
                                               const bool allowDifferentPerClusterMemoryView = false);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult areDistributionAttrsCompatible(T sourceType, T targetType,
                                                   const bool allowDifferentPerClusterMemoryView = false) {
    auto inDistribution = VPU::DistributionInfo::getClassFromAttr(sourceType.getDistribution());
    auto outDistribution = VPU::DistributionInfo::getClassFromAttr(targetType.getDistribution());
    auto inType = mlir::cast<vpux::NDTypeInterface>(sourceType);
    auto outType = mlir::cast<vpux::NDTypeInterface>(targetType);
    return areDistributionsCompatible(inType, inDistribution, outType, outDistribution,
                                      allowDifferentPerClusterMemoryView);
}

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::LogicalResult isDistributedCastCompatible(T inDistributedType, T outDistributedType, LogCb logCb = emptyLogCb) {
    if (inDistributedType.getShape() != outDistributedType.getShape()) {
        logCb(formatv("Mismatch between shapes for input ({0}) and output ({1}).", inDistributedType.getShape(),
                      outDistributedType.getShape()));
        return mlir::failure();
    }

    if (areDistributionElementTypesCompatible(inDistributedType.getElementType(), outDistributedType.getElementType())
                .failed()) {
        logCb(formatv("Mismatch between element types for input ({0}) and output ({1}).",
                      inDistributedType.getElementType(), outDistributedType.getElementType()));
        return mlir::failure();
    }

    if (inDistributedType.getMemSpace() != outDistributedType.getMemSpace()) {
        logCb(formatv("Mismatch between memspaces for input ({0}) and output ({1}).", inDistributedType.getMemSpace(),
                      outDistributedType.getMemSpace()));
        return mlir::failure();
    }

    const auto sameLayoutCheck = sameLayout(inDistributedType, outDistributedType, logCb);
    if (sameLayoutCheck.failed()) {
        return mlir::failure();
    }

    auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());
    auto outDistribution = VPU::DistributionInfo::getClassFromAttr(outDistributedType.getDistribution());
    auto inType = mlir::cast<vpux::NDTypeInterface>(inDistributedType);
    auto outType = mlir::cast<vpux::NDTypeInterface>(outDistributedType);
    if (areDistributionsCompatible(inType, inDistribution, outType, outDistribution).failed()) {
        logCb(formatv("Mismatch between distributionAttr for input ({0}) and output ({1}).",
                      inDistributedType.getDistribution(), outDistributedType.getDistribution()));
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace VPU
}  // namespace vpux
