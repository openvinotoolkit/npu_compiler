//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <mlir/IR/BuiltinAttributes.h>

using namespace vpux;

namespace {
//
// ConvertToPermuteCast
//
class ConvertToPermuteCast final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    using mlir::OpRewritePattern<VPU::MemPermuteOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp memPermuteOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(VPU::MemPermuteOp memPermuteOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(memPermuteOp.getInput());
    const auto inShape = getShape(memPermuteOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    if (!isTrivialPermute(inMemShape, memPermuteOp.getMemPerm())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::PermuteCastOp>(memPermuteOp, memPermuteOp.getInput(),
                                                    memPermuteOp.getDstOrderAttr(), memPermuteOp.getMemPermAttr());
    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::VPU::MemPermuteOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MemPermuteOpAdaptor mem_permute(operands, attrs, prop);
    if (mlir::failed(mem_permute.verify(loc))) {
        return mlir::failure();
    }

    VPU::inferPermuteReturnTypes(mem_permute.getInput(), mem_permute.getMemPerm(), mem_permute.getDstOrder(),
                                 inferredReturnTypes);

    return mlir::success();
}

InputTiling vpux::VPU::MemPermuteOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    mlir::AffineMap memPerm = getMemPerm();
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    const auto inShape = getShape(getInput());
    const auto inOrder = DimsOrder::fromValue(getInput());
    const auto outOrder = DimsOrder::fromValue(getOutput());
    auto curTile = outputTile;
    for (auto ind : irange(inShape.size())) {
        // take in consideration input and output shape vector order not map with memory order
        auto idxOrdIn = inOrder.dimAt(perm.dimAt(ind).ind());
        auto idxOrdOut = outOrder.dimAt(ind);
        curTile.shape[idxOrdIn] = outputTile.shape[idxOrdOut];
        curTile.offsets[idxOrdIn] = outputTile.offsets[idxOrdOut];
        curTile.axis[idxOrdIn] = outputTile.axis[idxOrdOut];
    }
    return TilingInfo{curTile};
}

void vpux::VPU::MemPermuteOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
    // Do nothing
}

mlir::FailureOr<OutputTiling> vpux::VPU::MemPermuteOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

//
// canonicalize
//

void vpux::VPU::MemPermuteOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertToPermuteCast>(context);
}

//
// ClusteredOpInterface
//

void vpux::VPU::MemPermuteOp::build(::mlir::OpBuilder& odsBuilder, ::mlir::OperationState& odsState,
                                    ::mlir::Value input, ::mlir::AffineMapAttr dstOrder,
                                    ::mlir::AffineMapAttr memPerm) {
    build(odsBuilder, odsState, input, dstOrder, memPerm, nullptr);
}

bool vpux::VPU::MemPermuteOp::checkStrategyCompatibility(VPU::MultiClusterStrategy strategy, size_t) {
    auto inputType = mlir::cast<NDTypeInterface>(getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(getOutput().getType());
    if (VPUIP::satisfiesOptimizedMemPermute(config::getArch(getOperation()), inputType, outputType)) {
        // Optimal MemPermute kernel is most performant with SOH
        // Should remove this experimental condition when shave cost is supported
        // Track E#170850
        return strategy == VPU::MultiClusterStrategy::Clustering ||
               strategy == VPU::MultiClusterStrategy::SplitOverHeight;
    }
    return strategy == VPU::MultiClusterStrategy::Clustering ||
           strategy == VPU::MultiClusterStrategy::SplitOverKernel ||
           strategy == VPU::MultiClusterStrategy::SplitOverHeight;
}

vpux::VPU::DistributionInfo vpux::VPU::MemPermuteOp::getExplicitDistributionInfoAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getSWExplicitDistributionInfo(mlir::cast<VPU::SWOpInterface>(getOperation()), shape, distributionMode,
                                              numTiles, numClusters, alignment, uniformDistributedSegments,
                                              overlapParams);
}

//
// SWOpInterface
//

bool vpux::VPU::MemPermuteOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers, Byte reservedMem) {
    VPUX_THROW_UNLESS(buffers.size() == 2,
                      "MemPermuteOp requires 1 input and 1 output, but the number of buffers is {0}", buffers.size());

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

bool vpux::VPU::MemPermuteOp::fitIntoCMX(llvm::ArrayRef<vpux::NDTypeInterface> buffers) {
    return fitIntoCMX(buffers, Byte(0));
}

bool vpux::VPU::MemPermuteOp::supportCycleCostCalculation() {
    return false;
}
