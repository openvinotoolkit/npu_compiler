//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_to_pool_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTMEMPERMUTETOOPPASS
#define GEN_PASS_DEF_CONVERTMEMPERMUTETOOPPASS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const uint32_t levelCount = 4;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

bool isLegalConvertToPool(IE::MemPermuteOp memPermuteOp, mlir::AffineMap memPermMap, mlir::MLIRContext* ctx,
                          int64_t numClusters, StringRef debugName, Logger log) {
    auto inputType = mlir::cast<NDTypeInterface>(memPermuteOp.getInput().getType());
    auto outputType = mlir::cast<NDTypeInterface>(memPermuteOp.getOutput().getType());
    auto arch = config::getArch(memPermuteOp);

    return vpux::isLegalConvertToPool(inputType, outputType, memPermuteOp.getInput().getDefiningOp(), memPermMap, ctx,
                                      numClusters, debugName, arch, log);
}

//
// ConvertMemPermuteToMaxPool
//

class ConvertMemPermuteToMaxPool final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertMemPermuteToMaxPool(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _numClusters(numClusters), _log(log) {
        this->setDebugName("ConvertMemPermuteToMaxPool");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult ConvertMemPermuteToMaxPool::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check whether it is legal to convert
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto memPerm = origOp.getMemPerm();
    const auto arch = config::getArch(origOp);
    if (!vpux::isLegalConvertToPool(inputType, outputType, origOp.getInput().getDefiningOp(), memPerm,
                                    rewriter.getContext(), _numClusters, getDebugName(), arch, _log.nest())) {
        return matchFailed(_log.nest(), rewriter, origOp, "Not legal to convert MemPermute to Pool");
    }

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    // Populate the target shape following NCHW order of dimensions.
    // Physical layout NHWC corresponds to logical layout NCHW.
    const Shape targetInShape = {inMemShape[MemDim(0)], inMemShape[MemDim(3)], inMemShape[MemDim(1)],
                                 inMemShape[MemDim(2)]};

    auto ctx = rewriter.getContext();

    const auto nhwcOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(inShape.size()), ctx);
    auto inPermuteCastOp = rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), origOp.getInput(),
                                                              DimsOrder::NHWC.toAffineMap(ctx), identityMap);
    inferReturnTypes(inPermuteCastOp, InferShapedTypeMode::ALL);

    const auto targetOrder = vpux::getNHWCOutputLayout(DimsOrder::fromAffineMap(memPerm));

    // Calculate the inputType of maxPoolOp
    Shape poolInLogicShape(inShape.size());
    auto poolInOrder = DimsOrder::NHWC;
    for (const auto idx : irange(inShape.size())) {
        poolInLogicShape[poolInOrder.dimAt(idx)] = inMemShape[MemDim(idx)];
    }
    auto poolInputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

    const auto IC = poolInLogicShape[Dims4D::Act::C];
    const auto alignedChannel = VPU::NCEInvariant::getAlignment(poolInputType.getElementType());
    mlir::Value latestPooling = nullptr;

    if (IC % alignedChannel == 0) {
        const auto maxPoolOutType =
                mlir::cast<vpux::NDTypeInterface>(inPermuteCastOp.getResult().getType()).changeDimsOrder(targetOrder);
        auto maxPool = IE::createIdentityMaxPool(inPermuteCastOp.getResult(), maxPoolOutType, rewriter);
        auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(maxPool);
        VPUX_THROW_WHEN(alignInterface == nullptr, "{0} don't have aligninterface.", origOp);
        latestPooling = maxPool->getResult(0);
    } else {
        auto conversionMap = vpux::calculateConversions(targetInShape, alignedChannel, targetOrder);
        auto latestInput = inPermuteCastOp.getResult();
        for (const auto& elem : conversionMap | indexed) {
            const auto idx = elem.index();
            const auto& item = elem.value();
            auto shapeCastTmp =
                    rewriter.createOrFold<IE::ShapeCastOp>(appendLoc(origOp.getLoc(), std::to_string(idx)), latestInput,
                                                           getIntArrayAttr(ctx, item.first.raw()));
            const auto layoutCastType = mlir::cast<vpux::NDTypeInterface>(shapeCastTmp.getType());
            const auto outType = layoutCastType.changeDimsOrder(item.second);
            auto maxPool = IE::createIdentityMaxPool(shapeCastTmp, outType, rewriter);
            latestPooling = maxPool->getResult(0);
            auto inLayoutCast =
                    rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), maxPool->getResult(0), nhwcOrderAttr);
            latestInput = inLayoutCast.getOutput();
        }
    }

    auto dstOrder = DimsOrder::fromValue(origOp.getOutput());
    auto outPermuteCastOp =
            rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), latestPooling, dstOrder.toAffineMap(ctx), identityMap);
    inferReturnTypes(outPermuteCastOp, InferShapedTypeMode::ALL);

    auto dstShape = getShape(origOp.getOutput());
    auto outShapeCastOp = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), outPermuteCastOp.getResult(),
                                                                 getIntArrayAttr(ctx, dstShape.raw()));

    rewriter.replaceOp(origOp, outShapeCastOp);

    return mlir::success();
}

//
// ConvertMemPermuteWithPermuteCast
//
// Permute cast MemPermuteOp to make it feasible to convert to pool:
//       Input: 1024x1x1x128xf16#NCHW                            Input: 1024x1x1x128xf16#NCHW
//           |                                   ==>                 |
//  MemPermute: 128x1024x1x1xf16#NHWC                      PermuteCast: 1x1024x1x128xf16#NCHW
//           |    (mem_perm: [d3, d1, d2, d0])                       |    (mem_perm: [d1, d0, d2, d3])
//                                                          MemPermute: 1x1024x128x1xf16#NHWC
//                                                                   |    (mem_perm: [d0, d3, d2, d1])
//                                                         PermuteCast: 128x1024x1x1xf16#NHWC
//

class ConvertMemPermuteWithPermuteCast final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertMemPermuteWithPermuteCast(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, int64_t numClusters,
                                     Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _numClusters(numClusters), _log(log) {
        this->setDebugName("ConvertMemPermuteWithPermuteCast");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult ConvertMemPermuteWithPermuteCast::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto ctx = rewriter.getContext();
    const int64_t SUPPORTED_RANK = 4;

    if (IE::hasDynamicTensors(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp has dynamic tensors");
    }
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto memPerm = origOp.getMemPerm();
    const auto arch = config::getArch(origOp);
    if (vpux::isLegalConvertToPool(inputType, outputType, origOp.getInput().getDefiningOp(), memPerm, ctx, _numClusters,
                                   getDebugName(), arch, _log.nest())) {
        return matchFailed(_log.nest(), rewriter, origOp, "Op is already legal to convert to Pool");
    }
    auto [mergedPermutation, mergedMemShape] = vpux::getMergedPermutationAndShape(inputType, memPerm, SUPPORTED_RANK);
    extendPermutationAndShape(mergedPermutation, mergedMemShape, SUPPORTED_RANK);
    auto mergedLogicShape = inputType.getDimsOrder().toLogicalOrder(MemShape(mergedMemShape));
    auto newMemPermAttr = mlir::AffineMap::getPermutationMap(ArrayRef(mergedPermutation), ctx);
    if (mergedLogicShape == inputType.getShape() && memPerm == newMemPermAttr) {
        return matchFailed(_log.nest(), rewriter, origOp, "No need to permute cast");
    }
    auto hasValidInPermutationMap =
            vpux::tryToFindPermutationForPermuteCast(inputType, inputType.getDimsOrder(), mergedLogicShape, ctx);
    if (!hasValidInPermutationMap.has_value()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Can not convert to inPermuteCastOp");
    }
    // Check whether it is legal to convert with new memPerm
    auto newMemPermuteInput =
            inferNewTypeWithMemPerm(inputType, hasValidInPermutationMap.value(), inputType.getDimsOrder());
    auto newMemPermuteOutput = inferNewTypeWithMemPerm(newMemPermuteInput, newMemPermAttr, outputType.getDimsOrder());
    if (!vpux::isLegalConvertToPool(newMemPermuteInput, newMemPermuteOutput, nullptr, newMemPermAttr, ctx, _numClusters,
                                    getDebugName(), arch, _log.nest())) {
        return matchFailed(_log.nest(), rewriter, origOp, "Not legal to convert MemPermute to Pool");
    }
    auto hasValidOutPermutationMap = vpux::tryToFindPermutationForPermuteCast(
            newMemPermuteOutput, outputType.getDimsOrder(), getShape(origOp.getResult()), ctx);
    if (!hasValidOutPermutationMap.has_value()) {
        return matchFailed(_log.nest(), rewriter, origOp, "Can not convert to outPermuteCastOp");
    }

    // Create in PermuteCastOp
    auto inPermuteCast = rewriter.create<IE::PermuteCastOp>(
            origOp.getLoc(), origOp.getInput(), mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx)),
            mlir::AffineMapAttr::get(hasValidInPermutationMap.value()));
    // Create new MemPermuteOp
    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp.getLoc(), inPermuteCast.getOutput(),
                                                             origOp.getDstOrder(), newMemPermAttr);
    // Create out PermuteCastOp
    auto outPermuteCast =
            rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), newMemPermuteOp.getOutput(),
                                               mlir::AffineMapAttr::get(outputType.getDimsOrder().toAffineMap(ctx)),
                                               mlir::AffineMapAttr::get(hasValidOutPermutationMap.value()));
    rewriter.replaceOp(origOp, outPermuteCast.getOutput());

    return mlir::success();
}

//
// ConvertMemPermuteWithShapeCast
//
// Shape cast MemPermuteOp to make it feasible to convert to pool:
//       Input: 2x256x16x64xf16#NCHW                             Input: 2x256x16x64xf16#NCHW
//           |                                   ==>                 |
//  MemPermute: 2x16x64x256xf16#NCHW                         ShapeCast: 1x2x256x1024xf16#NCHW
//           |    (mem_perm: [d3, d1, d2, d0])                       |
//                                                          MemPermute: 1x2x1024x256xf16#NCHW
//                                                                   |    (mem_perm: [d0, d1, d3, d2])
//                                                           ShapeCast: 2x16x64x256xf16#NCHW
//

class ConvertMemPermuteWithShapeCast final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertMemPermuteWithShapeCast(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, int64_t numClusters,
                                   Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _numClusters(numClusters), _log(log) {
        this->setDebugName("ConvertMemPermuteWithShapeCast");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult ConvertMemPermuteWithShapeCast::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto ctx = rewriter.getContext();
    const int64_t SUPPORTED_RANK = 4;

    if (IE::hasDynamicTensors(origOp)) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp has dynamic tensors");
    }
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto memPerm = origOp.getMemPerm();
    const auto arch = config::getArch(origOp);
    if (vpux::isLegalConvertToPool(inputType, outputType, origOp.getInput().getDefiningOp(), memPerm, ctx, _numClusters,
                                   getDebugName(), arch, _log.nest())) {
        return matchFailed(_log.nest(), rewriter, origOp, "Op is already legal to convert to Pool");
    }
    bool isPerAxisQuant = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inputType.getElementType());
    if (isPerAxisQuant) {
        return matchFailed(_log.nest(), rewriter, origOp, "Can not reshape for per axis quant");
    }
    auto [mergedPermutation, mergedMemShape] = vpux::getMergedPermutationAndShape(inputType, memPerm, SUPPORTED_RANK);
    extendPermutationAndShape(mergedPermutation, mergedMemShape, SUPPORTED_RANK);
    auto mergedLogicShape = inputType.getDimsOrder().toLogicalOrder(MemShape(mergedMemShape));
    auto newMemPermAttr = mlir::AffineMap::getPermutationMap(ArrayRef(mergedPermutation), ctx);
    if (mergedLogicShape == inputType.getShape() && memPerm == newMemPermAttr) {
        return matchFailed(_log.nest(), rewriter, origOp, "No need to shape cast");
    }

    auto module = getModuleOp(origOp.getOperation());
    const auto dmaPortNum = config::getAvailableExecutor(module, config::ExecutorKind::DMA_NN).getCount();
    auto newMemPermuteInput = inputType.changeShape(mergedLogicShape);
    auto newMemPermuteOutput = inferNewTypeWithMemPerm(newMemPermuteInput, newMemPermAttr, outputType.getDimsOrder());

    if (!vpux::isLegalConvertToPool(newMemPermuteInput, newMemPermuteOutput, nullptr, newMemPermAttr, ctx, _numClusters,
                                    getDebugName(), arch, _log.nest()) &&
        !(VPUIP::isBeneficialForUsingPermuteDMA(config::getArch(origOp.getOperation()), newMemPermuteInput,
                                                newMemPermuteOutput, newMemPermAttr, dmaPortNum, _log) &&
          inputType.getShape()[Dims4D::Act::N] != 1)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Not legal to convert MemPermute to Pool or PermuteDMA");
    }

    // Create input ShapeCast
    auto inShapeCast = vpux::IE::buildShapeCast(origOp.getLoc(), origOp.getInput(), mergedLogicShape.raw(), rewriter);
    // Create new MemPermuteOp
    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp.getLoc(), inShapeCast.getResult(),
                                                             origOp.getDstOrder(), newMemPermAttr);
    // change shape back
    auto outShapeCast = vpux::IE::buildShapeCast(origOp.getLoc(), newMemPermuteOp.getOutput(),
                                                 getShape(origOp.getResult()), rewriter);
    rewriter.replaceOp(origOp, outShapeCast);

    return mlir::success();
}

//
// ConvertMemPermuteWithExpand
//
// Insert Expand before MemPermuteOp to align channel and convert it to pool:
//       Input: 1x16x1575x72xf16#NCHW
//           |
//  MemPermute: 1x16x72x1575xf16#NCHW
//           |    (mem_perm: [d0, d1, d3, d2])
//
// Convert to:
//
//       Input: 1x16x1575x72xf16#NCHW
//           |
//      Expand: 1x16x1575x80xf16#NCHW
//           |    (pads_end: [0, 0, 0, 8])
//  MemPermute: 1x16x80x1575xf16#NCHW
//           |    (mem_perm: [d0, d1, d3, d2])
//      Slice: 1x16x72x1575xf16#NCHW

class ConvertMemPermuteWithExpand final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertMemPermuteWithExpand(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx, benefit), _numClusters(numClusters), _log(log) {
        this->setDebugName("ConvertMemPermuteWithExpand");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult ConvertMemPermuteWithExpand::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto ctx = rewriter.getContext();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    if (inputType.getRank() != int64_t(4)) {
        return matchFailed(_log.nest(), rewriter, origOp, "Only supports 4D shape rank");
    }
    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());
    const auto memPerm = origOp.getMemPerm();
    const auto arch = config::getArch(origOp);
    if (vpux::isLegalConvertToPool(inputType, outputType, origOp.getInput().getDefiningOp(), memPerm, ctx, _numClusters,
                                   getDebugName(), arch, _log.nest())) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "It is legal to convert MemPermute to Pool, no need to align channel");
    }

    const auto inMemShape = inputType.getMemShape().raw();
    const auto inMemW = inMemShape[Dims4D::Act::W.ind()];
    const auto alignedChannel = VPU::NCEInvariant::getAlignment(inputType.getElementType());

    const auto isLegalMemPerm = (DimsOrder::fromAffineMap(memPerm) == DimsOrder::NCWH) ||
                                (DimsOrder::fromAffineMap(memPerm) == DimsOrder::NWCH);
    if (!isLegalMemPerm || inMemW % alignedChannel == 0) {
        return matchFailed(_log.nest(), rewriter, origOp,
                           "memPerm is not supported, or channel alignment is not required");
    }

    const auto inExpandDim = inputType.getDimsOrder().dimAt(inputType.getRank() - 1);
    const auto inExpandSize = alignedChannel - (inMemW % alignedChannel);
    const auto shapeRank = inputType.getRank();

    // E#164080 for more details:
    // The experiment indicates that performance is highly influenced by three factors
    // 1. Memory H size: Larger H size improves HW efficiency and evenly distributing H across clusters
    // 2. Expand size: A smaller expand size is preferable as it minimizes unnecessary data movement
    // 3. MemPermute total size: A larger total size is preferable for ODU permute
    constexpr Byte PERMUTE_SIZE_THRESHOLD = 128_KB;
    const auto isLargeDataSize = inputType.getTotalAllocSize().count() >= PERMUTE_SIZE_THRESHOLD.count();
    const auto isLargeMemH = inMemShape[Dims4D::Act::C.ind()] >= _numClusters * 2;
    const auto isSmallExpandSize = inExpandSize <= (alignedChannel / 2);
    if (!isLargeDataSize || !isLargeMemH || !isSmallExpandSize) {
        return matchFailed(_log.nest(), rewriter, origOp, "Is not benefit convert to MaxPool");
    }

    // Insert Expand to align channel
    auto padsBeginAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(shapeRank, 0));
    auto padsEndVal = SmallVector<int64_t>(shapeRank, 0);
    padsEndVal[inExpandDim.ind()] = inExpandSize;
    auto padsEndAttr = getIntArrayAttr(ctx, padsEndVal);
    auto inExpandOp = rewriter.create<IE::ExpandOp>(appendLoc(origOp.getLoc(), "expand"), origOp.getInput(),
                                                    padsBeginAttr, padsEndAttr);

    // Create and check new MemPermute
    auto newMemPermuteOp = rewriter.create<IE::MemPermuteOp>(origOp.getLoc(), inExpandOp.getResult(),
                                                             origOp.getDstOrderAttr(), origOp.getMemPermAttr());

    if (!isLegalConvertToPool(newMemPermuteOp, memPerm, ctx, _numClusters, getDebugName(), _log.nest())) {
        rewriter.eraseOp(newMemPermuteOp);
        rewriter.eraseOp(inExpandOp);
        return matchFailed(_log.nest(), rewriter, origOp, "Not legal to convert new MemPermute to Pool");
    }

    // Insert Slice to get real data
    auto staticOffsetsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>(shapeRank, 0));
    auto staticSizesAttr = getIntArrayAttr(ctx, to_small_vector(getShape(origOp.getOutput())));
    auto outSliceOp = rewriter.create<IE::SliceOp>(appendLoc(origOp.getLoc(), "slice"), newMemPermuteOp.getOutput(),
                                                   staticOffsetsAttr, staticSizesAttr);

    _log.nest().trace("Convert MemPermute {1} to MaxPool with Expand", origOp->getLoc());

    rewriter.replaceOp(origOp, outSliceOp);

    return mlir::success();
}

//
// ConvertMemPermuteToPermuteQuantize
//

class ConvertMemPermuteToPermuteQuantize final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    ConvertMemPermuteToPermuteQuantize(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("ConvertMemPermuteToPermuteQuantize");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertMemPermuteToPermuteQuantize::matchAndRewrite(IE::MemPermuteOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto inOrder = inType.getDimsOrder();
    auto memPerm = origOp.getMemPerm();
    bool inPermuteCastRequired = false;
    bool outPermuteCastRequired = false;
    if (IE::canConvertToNCHWInOrderWithPermuteCast(inType, memPerm)) {
        // There is a chance to convert memPermuteOp to permuteQuantizeOp after inserting a permuteCastOp for input
        inType = inType.changeDimsOrder(DimsOrder::NCHW);
        memPerm = vpux::getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NHWC, origOp.getContext());
        inPermuteCastRequired = true;

        const auto outOrder = outType.getDimsOrder();
        if (outOrder != DimsOrder::NHWC) {
            // There is a chance to convert memPermuteOp to permuteQuantizeOp after inserting permuteCastOp for output
            outType = outType.changeDimsOrder(DimsOrder::NHWC);
            outType = outType.changeShape(inType.getShape());
            outPermuteCastRequired = true;
        }
    }

    const auto isLegalReorderOp = [&]() {
        if (!IE::isLegalReorderLikeToPermuteQuantize(inType, outType, _log)) {
            _log.trace("it's not Reorder-like op");
            return false;
        }

        if (DimsOrder::fromAffineMap(memPerm) != DimsOrder::NHWC) {
            _log.trace("Unsupported mem permute {0}", origOp.getMemPerm());
            return false;
        }

        if (inType.getShape() != outType.getShape()) {
            _log.trace("The input and output shape is not identical");
            return false;
        }

        const auto alignment = VPU::NCEInvariant::getAlignment(inType.getElementType());
        const auto inShape = inType.getShape();

        // Avoid introducing Expand and Slice
        if (inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W] % alignment != 0) {
            _log.trace("Unable to adjust the input shape for op {0} at {1}, ExpandOp may be introduced",
                       origOp->getName(), origOp->getLoc());
            return false;
        }
        return true;
    };

    if (!isLegalReorderOp()) {
        return matchFailed(_log.nest(), rewriter, origOp, "illegal Reorder op");
    }

    const auto& ctx = origOp.getContext();
    auto curInput = origOp.getInput();
    // Insert permuteCastOp for input
    if (inPermuteCastRequired) {
        const auto inMemPerm = vpux::getPermutationFromOrders(inOrder, DimsOrder::NCHW, ctx);
        auto inPermuteCastOp =
                rewriter.create<IE::PermuteCastOp>(appendLoc(origOp->getLoc(), "PermuteCast"), origOp.getInput(),
                                                   DimsOrder::NCHW.toAffineMap(origOp->getContext()), inMemPerm);
        curInput = inPermuteCastOp.getResult();
    }

    const auto dstElemTypeAttr = mlir::TypeAttr::get(outType.getElementType());
    const auto noPadBeginEnd = SmallVector<int64_t>(outType.getRank(), 0);

    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            takeOpLoc(origOp, "PermuteQuantize"), outType, curInput,
            mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)), mlir::AffineMapAttr::get(memPerm),
            dstElemTypeAttr, getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    _log.trace("convert to PermuteQuantize {0}", origOp->getLoc());

    // Insert pemuteCastOp for output
    if (outPermuteCastRequired) {
        auto identityMap =
                mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(inType.getShape().size()), ctx);
        auto outPermuteCastOp = rewriter.create<IE::PermuteCastOp>(origOp.getLoc(), permuteQuantizeOp.getOutput(),
                                                                   origOp.getDstOrder(), identityMap);
        inferReturnTypes(outPermuteCastOp, InferShapedTypeMode::ALL);

        rewriter.replaceOp(origOp, outPermuteCastOp);
    } else {
        rewriter.replaceOp(origOp, permuteQuantizeOp.getOutput());
    }

    return mlir::success();
}

//
// ConvertMemPermuteToOpPass
//

class ConvertMemPermuteToOpPass final : public IE::impl::ConvertMemPermuteToOpPassBase<ConvertMemPermuteToOpPass> {
public:
    explicit ConvertMemPermuteToOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertMemPermuteToOpPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    // As a priority, convert ReorderOp-like MemPermuteOp to PermuteQuantizeOp
    mlir::RewritePatternSet pqPatterns(&ctx);
    pqPatterns.add<ConvertMemPermuteToPermuteQuantize>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(pqPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    auto tileOp = config::getTileExecutor(func);
    auto numClusters = tileOp.getCount();

    // For HW limitation, only N = 1 NHWC and channel aligned MemPermute can be converted to MaxPool directly.
    // And for those can not be converted directly, first do some conversions, and then convert to MaxPool.
    mlir::RewritePatternSet patterns(&ctx);
    // PermuteCast works much better than ShapeCast later on during tiling/multiclustering/VF passes
    patterns.add<ConvertMemPermuteWithPermuteCast>(&ctx, benefitLevels[0], numClusters, _log);
    patterns.add<ConvertMemPermuteWithShapeCast>(&ctx, benefitLevels[1], numClusters, _log);
    patterns.add<ConvertMemPermuteWithExpand>(&ctx, benefitLevels[2], numClusters, _log);
    patterns.add<ConvertMemPermuteToMaxPool>(&ctx, benefitLevels[3], numClusters, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> IE::createConvertMemPermuteToOpPass(Logger log) {
    return std::make_unique<ConvertMemPermuteToOpPass>(log);
}
