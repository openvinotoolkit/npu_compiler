//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/shave_controls_dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <algorithm>

using namespace vpux;

namespace {

constexpr int64_t GDN_CHUNK = 64;
constexpr int64_t GDN_NUM_DPU_DESCRIPTORS = 6;
constexpr int64_t GDN_MAX_WEIGHT_TABLE_N = 128;

mlir::Type getAuxiliaryBufferType(mlir::ModuleOp moduleOp, mlir::Value query, mlir::Value value) {
    const auto qType = mlir::cast<vpux::NDTypeInterface>(query.getType());
    const auto vType = mlir::cast<vpux::NDTypeInterface>(value.getType());
    const auto D = qType.getShape().raw().back();
    const auto Dv = vType.getShape().raw().back();
    const int64_t C = GDN_CHUNK;
    const auto arch = config::getArch(moduleOp);

    if (mlir::ShapedType::isDynamic(D) || mlir::ShapedType::isDynamic(Dv) || arch == config::ArchKind::UNKNOWN) {
        return mlir::RankedTensorType::get({1, 1, 1, 1}, getUInt8Type(query.getContext()));
    }

    const int64_t fp32WorkBytes = (Dv * D + 2 * C * D + 2 * C * Dv + 2 * C * C) * static_cast<int64_t>(sizeof(float));

    const int64_t oneDpuDescriptorBytes = checked_cast<int64_t>(
            VPU::getDpuDebugDataSize(arch) + VPU::getDPUVariantDataSize(arch) + VPU::getDPUInvariantDataSize(arch));
    const int64_t dpuDescriptorBytes = GDN_NUM_DPU_DESCRIPTORS * oneDpuDescriptorBytes;
    const int64_t weightTableBytes =
            GDN_NUM_DPU_DESCRIPTORS * GDN_MAX_WEIGHT_TABLE_N * 4 * static_cast<int64_t>(sizeof(int32_t));
    const int64_t DvA = (Dv + 15) & ~static_cast<int64_t>(15);
    const int64_t DA = (D + 15) & ~static_cast<int64_t>(15);
    const int64_t shatElems = DvA * std::max<int64_t>(D, 32);
    const int64_t fp16StagingElems =
            2 * C * D + 3 * C * C + shatElems + 3 * C * DvA + DA * C + Dv * C + Dv * DA + DvA * C;
    const int64_t fp16StagingBytes = fp16StagingElems * static_cast<int64_t>(sizeof(int16_t));

    const int64_t padBytes = 64 * 4;
    const int64_t perShaveRaw = fp32WorkBytes + dpuDescriptorBytes + weightTableBytes + fp16StagingBytes + padBytes;
    const int64_t perShaveBytes = (perShaveRaw + 63) & ~static_cast<int64_t>(63);
    const int64_t numShaves = config::getNumOfEnginesOnTile(moduleOp, config::ExecutorKind::SHAVE_ACT);
    const int64_t numBytes = perShaveBytes * numShaves;
    return mlir::RankedTensorType::get({1, 1, 1, numBytes}, getUInt8Type(query.getContext()));
}

}  // namespace

void vpux::VPU::GatedDeltaNetOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value query,
                                       mlir::Value key, mlir::Value value, mlir::Value recurrent_state,
                                       mlir::Value gate, mlir::Value beta, mlir::UnitAttr fuse_qk_l2norm,
                                       mlir::FloatAttr q_l2_norm_eps, mlir::FloatAttr k_l2_norm_eps) {
    auto block = odsBuilder.getInsertionBlock();
    const auto moduleOp = getModuleOp(block->getParentOp());
    const auto auxBuffType = getAuxiliaryBufferType(moduleOp, query, value);
    auto scratch = VPU::createEmptyAuxiliaryBuffer(odsBuilder, odsState.location, auxBuffType);
    build(odsBuilder, odsState, query, key, value, recurrent_state, gate, beta, scratch, fuse_qk_l2norm, q_l2_norm_eps,
          k_l2_norm_eps, /*multiClusterStrategy=*/nullptr);
}

mlir::LogicalResult vpux::VPU::GatedDeltaNetOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatedDeltaNetOpAdaptor gdn(operands, attrs, prop);
    if (mlir::failed(gdn.verify(loc))) {
        return mlir::failure();
    }

    const auto queryType = mlir::cast<vpux::NDTypeInterface>(gdn.getQuery().getType());
    const auto valueType = mlir::cast<vpux::NDTypeInterface>(gdn.getValue().getType());
    const auto stateType = mlir::cast<vpux::NDTypeInterface>(gdn.getRecurrentState().getType());

    inferredReturnTypes.push_back(valueType.changeElemType(queryType.getElementType()));
    inferredReturnTypes.push_back(stateType);

    return mlir::success();
}

//
// SWOpInterface
//

bool vpux::VPU::GatedDeltaNetOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() >= 9 && buffers.size() <= 10,
                      "GatedDeltaNetOp requires 6 inputs, a scratch and 2 outputs (the scratch may also be "
                      "mirrored as a result), but the number of buffers is {0}",
                      buffers.size());

    SmallVector<Byte> buffersSize;
    std::transform(buffers.begin(), buffers.end(), std::back_inserter(buffersSize), [](const auto buffer) {
        return buffer.getTotalAllocSize();
    });

    auto totalAvailableCMXSize = reservedMem.count() == 0 ? getTotalCMXSize(getOperation()).count()
                                                          : getTotalCMXFragmentationAwareSize(getOperation()).count();

    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(getOperation()), buffersSize).count() +
                   reservedMem.count() <=
           totalAvailableCMXSize;
}

bool vpux::VPU::GatedDeltaNetOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

llvm::LogicalResult vpux::VPU::GatedDeltaNetOp::verify() {
    const auto moduleOp = getModuleOp(getOperation()->getParentOp());
    if (config::getArch(getOperation()) == config::ArchKind::UNKNOWN) {
        return mlir::success();
    }
    const auto D = getShape(getQuery()).raw().back();
    const auto Dv = getShape(getValue()).raw().back();
    if (mlir::ShapedType::isDynamic(D) || mlir::ShapedType::isDynamic(Dv)) {
        return emitOpError("expects static qk head size (D) and v head size (Dv) because the scratch-buffer size "
                           "depends on them");
    }
    if (D > 128) {
        return emitOpError("expects qk head size (D) <= 128, got ") << D;
    }
    if (Dv > 128) {
        return emitOpError("expects v head size (Dv) <= 128, got ") << Dv;
    }
    const auto queryShape = getShape(getQuery());
    const auto valueShape = getShape(getValue());
    if (queryShape.size() != 4 || valueShape.size() != 4) {
        return emitOpError("expects 4D query/value, got ") << queryShape.size() << " and " << valueShape.size();
    }
    const auto qkH = queryShape[Dims4D::Act::H];
    const auto vH = valueShape[Dims4D::Act::H];
    if (qkH <= 0 || vH <= 0 || vH % qkH != 0) {
        return emitOpError("expects positive qk_H (") << qkH << ") and v_H (" << vH << ") with v_H divisible by qk_H";
    }
    auto scratchType = mlir::cast<NDTypeInterface>(getScratch().getType());
    auto expectedType = mlir::cast<NDTypeInterface>(getAuxiliaryBufferType(moduleOp, getQuery(), getValue()));
    return VPU::compareTypes(getOperation()->getLoc(), scratchType, expectedType);
}

SmallVector<mlir::OpOperand*> vpux::VPU::GatedDeltaNetOp::getAuxiliaryBuffers() {
    return {&getScratchMutable()};
}

bool vpux::VPU::GatedDeltaNetOp::supportCycleCostCalculation() {
    return false;
}

//
// TilingBuilderOpInterface
//

// Tile only batch (N) and the value head (H): seq (C) is the recurrence, qk head-size (W) the reduction; GQA keeps
// heads whole.
SmallVector<int64_t> vpux::VPU::GatedDeltaNetOp::getMaxNumTiles() {
    auto* op = getOperation();
    SmallVector<int64_t> excludedAxes = {Dims4D::Act::C.ind(), Dims4D::Act::W.ind()};
    const auto qH = getShape(getQuery())[Dims4D::Act::H];
    const auto vH = getShape(getValue())[Dims4D::Act::H];
    if (mlir::ShapedType::isDynamic(qH) || mlir::ShapedType::isDynamic(vH) || qH <= 0 || vH <= 0 || (vH % qH) != 0) {
        excludedAxes.push_back(Dims4D::Act::H.ind());
    }
    return vpux::getMaxNumTiles(op, false, false, getMaxNumTilesWithAxesExclusion(op, excludedAxes));
}

vpux::InputTiling vpux::VPU::GatedDeltaNetOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto setDim = [&outputTile](TileInfo& tile, Dim dim, Dim outDim, int64_t div = 1) {
        VPUX_THROW_UNLESS(div == 1 || (outputTile.shape[outDim] % div == 0 && outputTile.offsets[outDim] % div == 0),
                          "GatedDeltaNet tiling must align to group size {0} (H tile shape/offset must be divisible)",
                          div);
        tile.shape[dim] = outputTile.shape[outDim] / div;
        tile.offsets[dim] = outputTile.offsets[outDim] / div;
        tile.axis[dim] = outputTile.axis[outDim];
    };
    const auto N = Dims4D::Act::N, C = Dims4D::Act::C, H = Dims4D::Act::H;
    const auto qHeads = getShape(getQuery())[H];
    const auto vHeads = getShape(getValue())[H];
    const int64_t groupSize = (!mlir::ShapedType::isDynamic(qHeads) && !mlir::ShapedType::isDynamic(vHeads) &&
                               qHeads > 0 && vHeads % qHeads == 0)
                                      ? vHeads / qHeads
                                      : 1;

    TileInfo qTile(getShape(getQuery())), kTile(getShape(getKey())), vTile(getShape(getValue()));
    TileInfo sTile(getShape(getRecurrentState())), gTile(getShape(getGate())), bTile(getShape(getBeta()));
    for (auto* tile : {&qTile, &kTile, &vTile, &sTile, &gTile, &bTile}) {
        setDim(*tile, N, N);
    }
    // value/gate/beta carry the head at H, the transposed state at C, and query/key fold it into qk heads.
    setDim(qTile, H, H, groupSize);
    setDim(kTile, H, H, groupSize);
    setDim(vTile, H, H);
    setDim(sTile, C, H);
    setDim(gTile, H, H);
    setDim(bTile, H, H);
    TileInfo scratchTile(getShape(getScratch()));
    return InputTiling{{std::move(qTile), std::move(kTile), std::move(vTile), std::move(sTile), std::move(gTile),
                        std::move(bTile), std::move(scratchTile)}};
}

vpux::OutputTiling vpux::VPU::GatedDeltaNetOp::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger) {
    const auto setDim = [&firstOutputTile](TileInfo& tile, Dim dim, Dim outDim) {
        tile.shape[dim] = firstOutputTile.shape[outDim];
        tile.offsets[dim] = firstOutputTile.offsets[outDim];
        tile.axis[dim] = firstOutputTile.axis[outDim];
    };
    TileInfo stateTile(getShape(getRecurrentState()));
    setDim(stateTile, Dims4D::Act::N, Dims4D::Act::N);
    setDim(stateTile, Dims4D::Act::C, Dims4D::Act::H);
    return OutputTiling{firstOutputTile, std::move(stateTile)};
}

vpux::TileInfo vpux::VPU::GatedDeltaNetOp::getMainOutputTile(mlir::OpResult /*secondaryOutput*/,
                                                             const vpux::TileInfo& secondaryOutputTile,
                                                             vpux::Logger /*log*/) {
    // Inverse of getOutputTiling: map the output_state tile back to the main output tile.
    const auto setDim = [&secondaryOutputTile](TileInfo& tile, Dim dim, Dim fromDim) {
        tile.shape[dim] = secondaryOutputTile.shape[fromDim];
        tile.offsets[dim] = secondaryOutputTile.offsets[fromDim];
        tile.axis[dim] = secondaryOutputTile.axis[fromDim];
    };
    TileInfo outputTile(getShape(getOutput()));
    setDim(outputTile, Dims4D::Act::N, Dims4D::Act::N);
    setDim(outputTile, Dims4D::Act::H, Dims4D::Act::C);
    return outputTile;
}

void vpux::VPU::GatedDeltaNetOp::adjustAttrs(const TilingInfo&, const TileInfo&) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::GatedDeltaNetOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(getOperation(), tilingMode, std::move(log));
}

//
// ClusteredOpInterface
//

bool vpux::VPU::GatedDeltaNetOp::isOperationSplitOverHeightCompatible(const vpux::TileInfo&) {
    if (VPU::getGatedDeltaNetHeadGroupSize(*this) == 0) {
        return false;
    }
    auto tileOp = config::getTileExecutor(getOperation()->getParentOfType<mlir::ModuleOp>());
    const auto numTiles = (tileOp != nullptr) ? tileOp.getCount() : 1;
    return getShape(getQuery())[Dims4D::Act::H] >= numTiles;
}

bool vpux::VPU::GatedDeltaNetOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t numTiles) {
    if (numTiles == 0) {
        return false;
    }
    if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return true;
    }
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        return isOperationSplitOverHeightCompatible(vpux::TileInfo(ShapeRef()));
    }
    return false;
}

vpux::VPU::DistributionInfo vpux::VPU::GatedDeltaNetOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams, const std::optional<ArrayRef<int64_t>>) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

vpux::NDTypeInterface vpux::VPU::GatedDeltaNetOp::getDistributedTypeForOpOperand(mlir::OpOperand& operand,
                                                                                 bool hasExplicitDistributedAttr,
                                                                                 SiblingOpsAnalysis& siblingsAnalysis) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto* ctx = clusteredOp->getContext();

    if (operand.get() == getScratch()) {
        return getDistributedTypeFromInput(clusteredOp, operand.get(), VPU::DistributionMode::DUPLICATED, {}, {},
                                           VPU::MultiClusterStrategy::Clustering, hasExplicitDistributedAttr,
                                           siblingsAnalysis);
    }

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(getOutput()), strategy);

    const auto rank = mlir::cast<NDTypeInterface>(operand.get().getType()).getShape().size();
    SmallVector<int64_t> numTiles(rank, 1);
    const auto axis = VPU::getGatedDeltaNetHeadAxis(*this, operand.get());
    numTiles[axis] = numClusters;

    const auto alignment = VPU::getGatedDeltaNetHeadAlignment(*this, operand.get());
    const auto alignmentAttr = alignment.empty() ? nullptr : getIntArrayAttr(ctx, alignment);

    const auto distributionMode = VPU::getSWInputTensorDistributionMode(*this, strategy, operand.get());
    return getDistributedTypeFromInput(clusteredOp, operand.get(), distributionMode, getIntArrayAttr(ctx, numTiles),
                                       alignmentAttr, strategy, hasExplicitDistributedAttr, siblingsAnalysis);
}

vpux::NDTypeInterface vpux::VPU::GatedDeltaNetOp::getDistributedTypeForOpResult(mlir::Value result,
                                                                                VPU::MultiClusterStrategy strategy,
                                                                                SiblingOpsAnalysis& siblingsAnalysis,
                                                                                bool hasExplicitDistributedAttr) {
    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(getOperation());
    auto* ctx = clusteredOp->getContext();
    const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(getOutput()), strategy);

    const auto rank = mlir::cast<NDTypeInterface>(result.getType()).getShape().size();
    SmallVector<int64_t> numTiles(rank, 1);
    const auto axis = VPU::getGatedDeltaNetHeadAxis(*this, result);
    numTiles[axis] = numClusters;

    const auto alignment = VPU::getGatedDeltaNetHeadAlignment(*this, result);
    const auto alignmentAttr = alignment.empty() ? nullptr : getIntArrayAttr(ctx, alignment);

    return getDistributedTypeFromInput(clusteredOp, result, VPU::DistributionMode::SEGMENTED,
                                       getIntArrayAttr(ctx, numTiles), alignmentAttr, strategy,
                                       hasExplicitDistributedAttr, siblingsAnalysis);
}
