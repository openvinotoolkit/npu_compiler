//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTTODMA
#define GEN_PASS_DEF_CONVERTTODMA
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

//
// ConvertToDMAPass
//

class ConvertToDMAPass final : public VPUIP::impl::ConvertToDMABase<ConvertToDMAPass> {
public:
    explicit ConvertToDMAPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPUIP::VPUIPDialect>();
    }

public:
    class ExpandConverter;
    class SwKernelMemPermuteConverter;
    class SwKernelDepthToSpaceConverter;
    class SwKernelSpaceToDepthConverter;
    class SwKernelPerAxisTileConverter;
    class UpsamplingOpConverter;

private:
    void safeRunOnFunc() final;
};

//
//  SwKernelMemPermuteConverter
//

class ConvertToDMAPass::SwKernelMemPermuteConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelMemPermuteConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelMemPermOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value createPermuteCastForDimsOrderConsistent(mlir::Value input, vpux::NDTypeInterface inType,
                                                    vpux::NDTypeInterface outType, mlir::Location loc,
                                                    mlir::PatternRewriter& rewriter) {
    auto inMemShape = inType.getMemShape();
    auto inPermuteCastShape = outType.getDimsOrder().toLogicalOrder(inMemShape);
    auto inPermuteCastType = outType;
    inPermuteCastType = inPermuteCastType.changeShape(ShapeRef(inPermuteCastShape));
    return rewriter.create<VPUIP::PermuteCastOp>(
            loc, inPermuteCastType, input,
            mlir::AffineMapAttr::get(outType.getDimsOrder().toAffineMap(rewriter.getContext())),
            mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(rewriter.getContext())));
}

VPUIP::GenericReshapeOp createGenericReshape(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                             vpux::NDTypeInterface outType, mlir::AffineMap mergedPerm,
                                             mlir::PatternRewriter& rewriter) {
    auto inGenReshapeInput = input;
    auto inGenReshapeType = input.getType().cast<vpux::NDTypeInterface>();

    if (outType.getDimsOrder() != inGenReshapeType.getDimsOrder()) {
        auto inPermuteCastOp = createPermuteCastForDimsOrderConsistent(input, inGenReshapeType, outType,
                                                                       swKernelOp->getLoc(), rewriter);
        inGenReshapeInput = inPermuteCastOp;
        inGenReshapeType = inGenReshapeInput.getType().cast<vpux::NDTypeInterface>();
    }

    auto inGenReshapeMemShape = Shape(inGenReshapeType.getMemShape().raw());
    vpux::Shape inGenReshapeNewMemShape;

    if (mergedPerm == DimsOrder::NHCW.toAffineMap(rewriter.getContext()) ||
        mergedPerm == DimsOrder::HCNW.toAffineMap(rewriter.getContext())) {
        inGenReshapeNewMemShape = Shape({1, inGenReshapeMemShape[Dims4D::Act::N] * inGenReshapeMemShape[Dims4D::Act::C],
                                         inGenReshapeMemShape[Dims4D::Act::H], inGenReshapeMemShape[Dims4D::Act::W]});
    } else if (mergedPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        inGenReshapeNewMemShape = Shape({1, inGenReshapeMemShape[Dims4D::Act::N], inGenReshapeMemShape[Dims4D::Act::C],
                                         inGenReshapeMemShape[Dims4D::Act::H] * inGenReshapeMemShape[Dims4D::Act::W]});
    } else if (mergedPerm == DimsOrder::CWNH.toAffineMap(rewriter.getContext())) {
        inGenReshapeNewMemShape =
                Shape({1, inGenReshapeMemShape[Dims4D::Act::N] * inGenReshapeMemShape[Dims4D::Act::C],
                       inGenReshapeMemShape[Dims4D::Act::H] * inGenReshapeMemShape[Dims4D::Act::W], 1});
    } else if (mergedPerm == DimsOrder::WCHN.toAffineMap(rewriter.getContext())) {
        inGenReshapeNewMemShape = Shape({1, inGenReshapeMemShape[Dims4D::Act::N],
                                         inGenReshapeMemShape[Dims4D::Act::C] * inGenReshapeMemShape[Dims4D::Act::H],
                                         inGenReshapeMemShape[Dims4D::Act::W]});
    } else if (mergedPerm == DimsOrder::NCHW.toAffineMap(rewriter.getContext())) {
        inGenReshapeNewMemShape = Shape(outType.getMemShape().raw());
    } else {
        VPUX_THROW("Unsupport MergedPerm {0}", mergedPerm);
    }
    inGenReshapeType =
            VPUIP::changeShapeWithMemShape(&inGenReshapeType, inGenReshapeNewMemShape, outType.getDimsOrder());

    return rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), inGenReshapeType, inGenReshapeInput);
}

VPUIP::PermuteDMAOp createPermuteDMA(VPUIP::SwKernelOp swKernelOp, mlir::Value input, vpux::NDTypeInterface inType,
                                     DimsOrder dimsOrderDMA, vpux::NDTypeInterface outType,
                                     mlir::PatternRewriter& rewriter) {
    auto memPermDMA = dimsOrderDMA.toAffineMap(rewriter.getContext());
    auto permDMAType = inType;
    auto permDMAMemShape = Shape(inType.getMemShape().raw());
    auto permDMANewMemShape =
            Shape({permDMAMemShape[dimsOrderDMA.toPermutation()[0]], permDMAMemShape[dimsOrderDMA.toPermutation()[1]],
                   permDMAMemShape[dimsOrderDMA.toPermutation()[2]], permDMAMemShape[dimsOrderDMA.toPermutation()[3]]});
    permDMAType = VPUIP::changeShapeWithMemShape(&permDMAType, permDMANewMemShape, outType.getDimsOrder());
    auto permMemRefType = permDMAType.dyn_cast<mlir::MemRefType>();
    VPUX_THROW_WHEN(permMemRefType == nullptr, "Unexpected output type for first VPUIP::permuteDMAOp at '{0}'",
                    swKernelOp.getLoc());
    auto allocPermuteOp = rewriter.create<mlir::memref::AllocOp>(swKernelOp->getLoc(), permMemRefType);

    return rewriter.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), input, allocPermuteOp,
                                                mlir::AffineMapAttr::get(memPermDMA), nullptr);
}

//
// Convert MemPermute NCHW->NHCW to 2 permuteDMAs
// Permute pattern: [d0, d1, d2, d3] -> [d0, d2, d1, d3]
// For example:
//            Input            :    6x4x8x512xf16#NCHW
//              |
//           MemPermute        :    memPerm: (d0, d1, d2, d3) -> (d0, d2, d1, d3)
//              |
//            Output           :    6x8x4x512xf16#NCHW
// Convert to:
//            Input            :    6x4x8x512xf16#NCHW
//              |
//         GenericReshape 1    :    1x24x8x512xf16#NCHW
//              |
//         PermuteDMA 1        :    1x8x24x512xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 2    :    1x8x6x2048xf16#NCHW
//              |
//         PermuteDMA 2        :    1x6x8x2048xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 3    :    6x8x4x512xf16#NCHW
//              |
//            Output           :    6x8x4x512xf16#NCHW
//
VPUIP::GenericReshapeOp convertMemPermuteNHCWAsDMA(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                                   vpux::NDTypeInterface outType, mlir::Value outputBuf,
                                                   mlir::PatternRewriter& rewriter) {
    // Create genericReshapeOp for first permuteDMAOp
    const auto mergedPerm = DimsOrder::NHCW.toAffineMap(rewriter.getContext());
    auto inGenReshapeOp = createGenericReshape(swKernelOp, input, outType, mergedPerm, rewriter);
    auto inGenReshapeType = inGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create first permuteDMAOp: permutation is [d0, d2, d1, d3]
    auto dimsOrderDMA = DimsOrder::NHCW;
    auto memPermDMA = dimsOrderDMA.toAffineMap(rewriter.getContext());
    auto firstPermDmaOp =
            createPermuteDMA(swKernelOp, inGenReshapeOp, inGenReshapeType, dimsOrderDMA, outType, rewriter);
    auto firstPermDMAType = firstPermDmaOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create genericReshapeOp for second permuteDMAOp
    auto midGenReshapeType = firstPermDMAType;
    auto outTypeMemShape = Shape(outType.getMemShape().raw());
    auto midGenReshapeNewMemShape = Shape({1, outTypeMemShape[Dims4D::Act::C], outTypeMemShape[Dims4D::Act::N],
                                           outTypeMemShape[Dims4D::Act::H] * outTypeMemShape[Dims4D::Act::W]});
    midGenReshapeType =
            VPUIP::changeShapeWithMemShape(&midGenReshapeType, midGenReshapeNewMemShape, outType.getDimsOrder());
    auto midGenReshapeOp =
            rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), midGenReshapeType, firstPermDmaOp);

    // Create second permuteDMAOp: permutation is [d0, d1, d3, d2]
    auto secondPermDmaOp = rewriter.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), midGenReshapeOp, outputBuf,
                                                                mlir::AffineMapAttr::get(memPermDMA), nullptr);

    // Create genericReshapeOp for output
    auto genricReshape = rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), outType, secondPermDmaOp);
    return genricReshape;
}

//
// Convert MemPermute NCHW->WCHN to 2 permuteDMAs
// Permute pattern: [d0, d1, d2, d3] -> [d3, d1, d2, d0]
// For example:
//            Input            :    2x4x121x3xf16#NCHW
//              |
//           MemPermute        :    memPerm: (d0, d1, d2, d3) -> (d3, d1, d2, d0)
//              |
//            Output           :    3x4x121x2xf16#NCHW
// Convert to:
//            Input            :    2x4x121x3xf16#NCHW
//              |
//         GenericReshape 1    :    1x2x484x3xf16#NCHW
//              |
//         PermuteDMA 1        :    1x484x2x3xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 2    :    1x968x3x1xf16#NCHW
//              |
//         PermuteDMA 2        :    1x3x968x1xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 3    :    3x4x121x2xf16#NCHW
//              |
//            Output           :    3x4x121x2xf16#NCHW
//
VPUIP::GenericReshapeOp convertMemPermuteWCHNAsDMA(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                                   mlir::PatternRewriter& rewriter) {
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    // Create genericReshapeOp for first permuteDMAOp
    const auto mergedPerm = DimsOrder::WCHN.toAffineMap(rewriter.getContext());
    auto inGenReshapeOp = createGenericReshape(swKernelOp, input, outType, mergedPerm, rewriter);
    auto inGenReshapeType = inGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create first permuteDMAOp: permutation is [d0, d2, d1, d3]
    auto firstPermDmaOp =
            createPermuteDMA(swKernelOp, inGenReshapeOp, inGenReshapeType, DimsOrder::NHCW, outType, rewriter);
    auto firstPermDMAType = firstPermDmaOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create genericReshapeOp for second permuteDMAOp
    auto midGenReshapeType = firstPermDMAType;
    auto outTypeMemShape = Shape(outType.getMemShape().raw());
    auto midGenReshapeNewMemShape = Shape(
            {1, outTypeMemShape[Dims4D::Act::C] * outTypeMemShape[Dims4D::Act::H] * outTypeMemShape[Dims4D::Act::W],
             outTypeMemShape[Dims4D::Act::N], 1});
    midGenReshapeType =
            VPUIP::changeShapeWithMemShape(&midGenReshapeType, midGenReshapeNewMemShape, outType.getDimsOrder());
    auto midGenReshapeOp =
            rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), midGenReshapeType, firstPermDmaOp);
    auto midGenReshapeOutType = midGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    // Create second permuteDMAOp: permutation is [d0, d2, d1, d3]
    auto secondPermDmaOp =
            createPermuteDMA(swKernelOp, midGenReshapeOp, midGenReshapeOutType, DimsOrder::NHCW, outType, rewriter);

    // Create genericReshapeOp for output
    return rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), outType, secondPermDmaOp);
}

//
// Convert MemPermute NCHW->HCNW or NCHW->NWHC to 3 permuteDMAs
// MemPermute NCHW->HCNW, Permute pattern: [d0, d1, d2, d3] -> [d2, d1, d0, d3]
// MemPermute NCHW->NWHC, Permute pattern: [d0, d1, d2, d3] -> [d0, d3, d2, d1]
// For example, MemPermute NCHW->HCNW:
//            Input                            :    6x4x8x512xf16#NCHW
//              |
//           MemPermute                        :    memPerm: (d0, d1, d2, d3) -> (d2, d1, d0, d3)
//              |
//            Output                           :    8x4x6x512xf16#NCHW
// Convert to:
//            Input                            :    6x4x8x512xf16#NCHW
//              |
//         GenericReshape 1                    :    1x24x8x512xf16#NCHW
//              |
//         PermuteDMA 1                        :    1x8x24x512xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 2                    :    8x6x4x512xf16#NCHW
//              |
//         ConvertMemPermuteNHCWAsDMA          :    8x4x6x512xf16#NCHW
//              |
//            Output                           :    8x4x6x512xf16#NCHW
//
VPUIP::GenericReshapeOp convertMemPermuteHCNWOrNWHC(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                                    mlir::AffineMap mergedPerm, mlir::PatternRewriter& rewriter) {
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputBuf = swKernelOp.getOperand(1);

    // Create genericReshapeOp for first permuteDMAOp
    auto inGenReshapeOp = createGenericReshape(swKernelOp, input, outType, mergedPerm, rewriter);
    auto inGenReshapeType = inGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create first permuteDMAOp
    auto dimsOrderDMA =
            mergedPerm == DimsOrder::HCNW.toAffineMap(rewriter.getContext()) ? DimsOrder::NHCW : DimsOrder::NCWH;
    auto firstPermDmaOp =
            createPermuteDMA(swKernelOp, inGenReshapeOp, inGenReshapeType, dimsOrderDMA, outType, rewriter);
    auto firstPermDMAType = firstPermDmaOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create genericReshapeOp for later permuteDMAOp
    auto midGenReshapeType = firstPermDMAType;
    auto outTypeMemShape = Shape(outType.getMemShape().raw());
    auto midGenReshapeMemShape = Shape(firstPermDMAType.getMemShape().raw());
    auto midGenReshapeNewMemShape = Shape({outTypeMemShape[Dims4D::Act::N], outTypeMemShape[Dims4D::Act::H],
                                           outTypeMemShape[Dims4D::Act::C], outTypeMemShape[Dims4D::Act::W]});
    midGenReshapeType =
            VPUIP::changeShapeWithMemShape(&midGenReshapeType, midGenReshapeNewMemShape, outType.getDimsOrder());
    auto midGenReshapeOp =
            rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), midGenReshapeType, firstPermDmaOp);

    // Convert next MemPermute NCHW->NHCW to 2 permuteDMAs
    return convertMemPermuteNHCWAsDMA(swKernelOp, midGenReshapeOp, outType, outputBuf, rewriter);
}

//
// Convert MemPermute NCHW->CWNH to 3 permuteDMAs
// MemPermute NCHW->CWNH, Permute pattern: [d0, d1, d2, d3] -> [d1, d3, d0, d2]
// For example, MemPermute NCHW->CWNH:
//            Input                            :    256x4x256x4xf16#NCHW
//              |
//           MemPermute                        :    memPerm: (d0, d1, d2, d3) -> (d1, d3, d0, d2)
//              |
//            Output                           :    4x4x256x256xf16#NCHW
// Convert to:
//            Input                            :    256x4x256x4xf16#NCHW
//              |
//         ConvertMemPermuteNHCWAsDMA          :    256x256x4x4xf16#NCHW
//              |
//         GenericReshape 1                    :    1x65536x16x1xf16#NCHW
//              |
//         PermuteDMA 3                        :    1x16x65536x1xf16#NCHW ([1, 0]: HW->WH)
//              |
//         GenericReshape 2                    :    4x4x256x256xf16#NCHW
//              |
//            Output                           :    4x4x256x256xf16#NCHW
//
VPUIP::GenericReshapeOp convertMemPermuteCWNH(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                              mlir::PatternRewriter& rewriter) {
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    // Convert Mempermute NCHW->NHCW to 2 permuteDMAs
    auto inShape = input.getType().cast<vpux::NDTypeInterface>().getShape().toValues();
    auto newInShape = inShape;
    newInShape[Dims4D::Act::C] = inShape[Dims4D::Act::H];
    newInShape[Dims4D::Act::H] = inShape[Dims4D::Act::C];

    auto firstPermuteOutType = input.getType().cast<vpux::NDTypeInterface>().changeShape(newInShape);
    auto firstPermuteOutBuf = swKernelOp.getOperand(1);
    firstPermuteOutBuf.setType(firstPermuteOutType);
    auto midPermuteOp =
            convertMemPermuteNHCWAsDMA(swKernelOp, input, firstPermuteOutType, firstPermuteOutBuf, rewriter);

    // Create genericReshapeOp for 3rd permuteDMAOp
    const auto mergedPerm = DimsOrder::CWNH.toAffineMap(rewriter.getContext());
    auto inGenReshapeOp = createGenericReshape(swKernelOp, midPermuteOp, outType, mergedPerm, rewriter);
    auto inGenReshapeType = inGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();

    // Create 3rd permuteDMAOp
    auto dimsOrderDMA = DimsOrder::NHCW;
    auto thirdPermDmaOp =
            createPermuteDMA(swKernelOp, inGenReshapeOp, inGenReshapeType, dimsOrderDMA, outType, rewriter);

    const auto outMergedPerm = DimsOrder::NCHW.toAffineMap(rewriter.getContext());
    return createGenericReshape(swKernelOp, thirdPermDmaOp, outType, outMergedPerm, rewriter);
}

//
// Convert MemPermute NCHW->HNWC to 2 permuteDMAs
// MemPermute NCHW->HNWC, Permute pattern: [d0, d1, d2, d3] -> [d2, d0, d3, d1]
// For example, MemPermute NCHW->HNWC:
//            Input                            :    4x4x256x256xf16#NCHW
//              |
//           MemPermute                        :    memPerm: (d0, d1, d2, d3) -> (d2, d0, d3, d1)
//              |
//            Output                           :    256x4x256x4xf16#NCHW
// Convert to:
//            Input                            :    4x4x256x256xf16#NCHW
//              |
//         GenericReshape 1                    :    1x16x256x256xf16#NCHW
//              |
//         PermuteDMA 1                        :    1x256x16x256xf16#NCHW ([1, 0, 2]: HWC->WHC)
//              |
//         GenericReshape 2                    :    1x1024x4x256xf16#NCHW
//              |
//         PermuteDMA 2                        :    1x1024x256x4xf16#NCHW ([0, 2, 1]: HWC->HCW)
//              |
//         GenericReshape 3                    :    256x4x256x4xf16#NCHW
//              |
//            Output                           :    256x4x256x4xf16#NCHW
//
VPUIP::GenericReshapeOp convertMemPermuteHNWCAsDMA(VPUIP::SwKernelOp swKernelOp, mlir::Value input,
                                                   mlir::PatternRewriter& rewriter) {
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    // Create genericReshapeOp for first permuteDMAOp
    const auto mergedPerm = DimsOrder::NHCW.toAffineMap(rewriter.getContext());
    auto inGenReshapeOp = createGenericReshape(swKernelOp, input, outType, mergedPerm, rewriter);
    auto inGenReshapeType = inGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    // Create first permuteDMAOp: permutation is [d0, d2, d1, d3]
    auto dimsOrderDMA = DimsOrder::NHCW;
    auto firstPermDmaOp =
            createPermuteDMA(swKernelOp, inGenReshapeOp, inGenReshapeType, dimsOrderDMA, outType, rewriter);
    auto firstPermDMAType = firstPermDmaOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    // Create genericReshapeOp for second permuteDMAOp
    auto midGenReshapeType = firstPermDMAType;
    auto outTypeMemShape = Shape(outType.getMemShape().raw());
    auto midGenReshapeNewMemShape = Shape({1, outTypeMemShape[Dims4D::Act::N] * outTypeMemShape[Dims4D::Act::C],
                                           outTypeMemShape[Dims4D::Act::W], outTypeMemShape[Dims4D::Act::H]});
    midGenReshapeType =
            VPUIP::changeShapeWithMemShape(&midGenReshapeType, midGenReshapeNewMemShape, outType.getDimsOrder());
    auto midGenReshapeOp =
            rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), midGenReshapeType, firstPermDmaOp);
    auto midGenReshapeOutType = midGenReshapeOp.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
    // Create second permuteDMAOp: permutation is [d0, d1, d3, d2]
    auto secondDimsOrderDMA = DimsOrder::NCWH;
    auto secondPermDmaOp =
            createPermuteDMA(swKernelOp, midGenReshapeOp, midGenReshapeOutType, secondDimsOrderDMA, outType, rewriter);
    // Create genericReshapeOp for output
    return rewriter.create<VPUIP::GenericReshapeOp>(swKernelOp->getLoc(), outType, secondPermDmaOp);
}

mlir::LogicalResult ConvertToDMAPass::SwKernelMemPermuteConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isMemPermSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got Mempermute SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto memPerm = VPUIP::getMemPermFromSwKernel(swKernelOp);
    VPUX_THROW_UNLESS(memPerm.has_value(), "Cannot extract mem_perm attribute from permute SwKernel '{0}'.",
                      swKernelOp.getLoc());

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);

    const auto inType = swKernelOp.getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto input = swKernelOp.getOperand(0);
    const auto outputBuf = swKernelOp.getOperand(1);
    // Check for inversed permutation which needs split into 2 consecutive permuteDMAs
    // e.g. pattern [d0, d1, d2, d3] -> [d0, d3, d2, d1]
    auto mergedPerm = vpux::VPUIP::getPermuteDMAMergedMemPerm(inType, memPerm.value());
    if (!VPUIP::isSplitNeededForPermuteDMA(inType, memPerm.value())) {
        rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(swKernelOp, input, outputBuf,
                                                         mlir::AffineMapAttr::get(memPerm.value()), nullptr);

        _log.nest().trace("Rewrite Mempermute SwKernel '{0}' at '{1}' to PermuteDMA.", swKernelOp->getName(),
                          swKernelOp->getLoc());
        return mlir::success();
    } else if (mergedPerm == DimsOrder::NHCW.toAffineMap(rewriter.getContext())) {
        // Convert MemPermute NCHW->NHCW to 2 permuteDMAs
        auto newOp = convertMemPermuteNHCWAsDMA(swKernelOp, input,
                                                swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>(),
                                                outputBuf, rewriter);
        rewriter.replaceOp(swKernelOp, newOp.getOutput());

        return mlir::success();
    } else if (mergedPerm == DimsOrder::HCNW.toAffineMap(rewriter.getContext()) ||
               mergedPerm == DimsOrder::NWHC.toAffineMap(rewriter.getContext())) {
        // Convert MemPermute NCHW->HCNW or NCHW->NWHC to 3 permuteDMAs
        auto newOp = convertMemPermuteHCNWOrNWHC(swKernelOp, input, mergedPerm, rewriter);
        rewriter.replaceOp(swKernelOp, newOp.getOutput());

        return mlir::success();
    } else if (mergedPerm == DimsOrder::CWNH.toAffineMap(rewriter.getContext())) {
        // Convert MemPermute NCHW->CWNH to 3 permuteDMAs
        auto newOp = convertMemPermuteCWNH(swKernelOp, input, rewriter);
        rewriter.replaceOp(swKernelOp, newOp.getOutput());

        return mlir::success();
    } else if (mergedPerm == DimsOrder::HNWC.toAffineMap(rewriter.getContext())) {
        // Convert MemPermute NCHW->HNWC to 2 permuteDMAs
        auto newOp = convertMemPermuteHNWCAsDMA(swKernelOp, input, rewriter);
        rewriter.replaceOp(swKernelOp, newOp.getOutput());

        return mlir::success();
    } else if (mergedPerm == DimsOrder::WHC.toAffineMap(rewriter.getContext()) &&
               memPerm.value() == DimsOrder::WCHN.toAffineMap(rewriter.getContext())) {
        // Convert MemPermute NCHW->WCHN to 2 permuteDMAs
        auto newOp = convertMemPermuteWCHNAsDMA(swKernelOp, input, rewriter);
        rewriter.replaceOp(swKernelOp, newOp.getOutput());

        return mlir::success();
    }
    _log.nest().trace("Split into 2 permuteDMA: memPerm {0}", memPerm);

    auto permuteMemRefType = swKernelOp.getOperand(1).getType().dyn_cast<mlir::MemRefType>();
    VPUX_THROW_WHEN(permuteMemRefType == nullptr, "Unexpected output type for VPUIP.SwKernelOp at '{0}'",
                    swKernelOp->getLoc());

    // 3 types of memPermute can be supported:
    // a) NHWC -> NCHW; b) NWCH -> NCHW; c) NCWH -> NHWC; d) NCHW->CWNH
    // The first 2 types cannot be replaced with the 2 consecutive permuteDMAs directly,
    // so a permuteCast is required before the 2 permuteDMAs.
    // The dst order of permuteCast can be reversedly derived from the final dst order

    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto outShapedType = outType.cast<vpux::NDTypeInterface>();
    auto outOrder = outType.getDimsOrder();
    // The 2nd permuteDMA is [d0, d2, d3, d1] -> [d0, d3, d2, d1], permutation is [d0, d2, d1, d3]
    // The inversed permutation is [d0, d2, d1, d3]
    // dstOutOrder here is actually the inOrder for the second permuation
    auto inversePermLast = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 2, 1, 3}, rewriter.getContext());
    auto dstOutOrder = applyPermutation(outOrder, DimsOrder::fromAffineMap(inversePermLast));
    // The 1st permuteDMA is [d0, d1, d2, d3] -> [d0, d2, d3, d1], permutation is [d0, d2, d3, d1]
    // The inverse permuation is [d0, d3, d1, d2]]
    auto inversePermFirst =
            mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 3, 1, 2}, rewriter.getContext());
    auto permuteCastDstOrder = applyPermutation(dstOutOrder, DimsOrder::fromAffineMap(inversePermFirst));
    auto permuteCastOutType = outShapedType.changeDimsOrder(permuteCastDstOrder);
    _log.nest().trace("Deduced permuteCastDstOrder = {0} from outOrder {1}", permuteCastDstOrder, outOrder);
    auto permuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(
            swKernelOp->getLoc(), permuteCastOutType, input,
            mlir::AffineMapAttr::get(permuteCastDstOrder.toAffineMap(rewriter.getContext())),
            mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(rewriter.getContext())));

    // create the 1st permuteDMA Op [d0, d1, d2, d3] -> [d0, d2, d3, d1], permutation is [d0, d2, d3, d1]
    auto memPermFirst = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 2, 3, 1}, rewriter.getContext());
    auto newPermuteMemRefType = mlir::MemRefType::get(
            permuteMemRefType.getShape(), permuteMemRefType.getElementType(),
            dstOutOrder.toAffineMap(rewriter.getContext()),
            IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0));

    auto allocPermuteOp = rewriter.create<mlir::memref::AllocOp>(swKernelOp->getLoc(), newPermuteMemRefType);
    auto permuteDmaOp = rewriter.create<VPUIP::PermuteDMAOp>(swKernelOp->getLoc(), permuteCastOp, allocPermuteOp,
                                                             mlir::AffineMapAttr::get(memPermFirst), nullptr);

    // create the 2nd permuteDMA Op [d0, d2, d3, d1] -> [d0, d3, d2, d1], permutation is [d0, d2, d1, d3]
    auto memPermLast = mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 2, 1, 3}, rewriter.getContext());
    rewriter.replaceOpWithNewOp<VPUIP::PermuteDMAOp>(swKernelOp, permuteDmaOp.getOutput(), outputBuf,
                                                     mlir::AffineMapAttr::get(memPermLast), nullptr);
    _log.nest().trace("Rewrite Mempermute SwKernel '{0}' at '{1}' to 2 PermuteDMA ops.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// SwKernelDepthToSpaceConverter
//

class ConvertToDMAPass::SwKernelDepthToSpaceConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelDepthToSpaceConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelDepthToSpaceOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelDepthToSpaceConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isDepthToSpaceSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got DepthToSpace SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto depthToSpaceAttrs = VPUIP::getDepthToSpaceSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(depthToSpaceAttrs.has_value(),
                      "Cannot extract depthToSpace attribute from depthToSpace SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = std::get<0>(depthToSpaceAttrs.value());
    auto blockSizeAttr = std::get<1>(depthToSpaceAttrs.value());
    auto paddedChannel = std::get<2>(depthToSpaceAttrs.value());

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);
    auto input = swKernelOp.getOperand(0);
    auto outputBuf = swKernelOp.getOperand(1);

    rewriter.replaceOpWithNewOp<VPUIP::DepthToSpaceDMAOp>(swKernelOp, input, outputBuf, blockSizeAttr, modeAttr,
                                                          nullptr, paddedChannel);

    _log.nest().trace("Rewrite DepthToSpace SwKernel '{0}' at '{1}' to DepthToSpaceDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// SwKernelSpaceToDepthConverter
//

class ConvertToDMAPass::SwKernelSpaceToDepthConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelSpaceToDepthConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelSpaceToDepthOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelSpaceToDepthConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isSpaceToDepthSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got SpaceToDepth SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());

    auto spaceToDepthAttrs = VPUIP::getSpaceToDepthSwKernelAttr(swKernelOp);
    VPUX_THROW_UNLESS(spaceToDepthAttrs.has_value(),
                      "Cannot extract spaceToDepth attribute from spaceToDepth SwKernel '{0}'.", swKernelOp.getLoc());
    auto modeAttr = spaceToDepthAttrs.value().first;
    auto blockSizeAttr = spaceToDepthAttrs.value().second;

    VPUX_THROW_UNLESS(swKernelOp->getNumOperands() == 2, "Unexpected operand number for VPUIP.SwKernelOp at '{0}'",
                      swKernelOp);
    auto input = swKernelOp.getOperand(0);
    auto outputBuf = swKernelOp.getOperand(1);

    rewriter.replaceOpWithNewOp<VPUIP::SpaceToDepthDMAOp>(swKernelOp, input, outputBuf, blockSizeAttr, modeAttr,
                                                          nullptr);

    _log.nest().trace("Rewrite SpaceToDepth SwKernel '{0}' at '{1}' to SpaceToDepthDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// ExpandConverter
//

class ConvertToDMAPass::ExpandConverter final : public mlir::OpRewritePattern<VPUIP::ExpandOp> {
public:
    ExpandConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ExpandOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::ExpandConverter");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandOp depthToSpaceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::ExpandConverter::matchAndRewrite(VPUIP::ExpandOp expandOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Expand '{0}' at '{1}'", expandOp->getName(), expandOp->getLoc());

    const auto inputType = mlir::cast<NDTypeInterface>(expandOp.getInput().getType());
    VPUX_THROW_WHEN(mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
                            vpux::isFloat8Quantized(inputType.getElementType()),
                    "Only integral type ExpandOp can be converted to DMA, but got '{0}'", inputType.getElementType());

    const auto outputType = mlir::cast<NDTypeInterface>(expandOp.getOutput().getType());
    _log.nest().trace("inType: '{0}', outType: '{1}', padBegin: '{2}', padEnd: '{3}'", inputType, outputType,
                      expandOp.getPadsBegin(), expandOp.getPadsEnd());

    rewriter.replaceOpWithNewOp<VPUIP::ExpandDMAOp>(expandOp, expandOp.getInput(), expandOp.getOutputBuff(),
                                                    expandOp.getPadsBeginAttr(), expandOp.getPadsEndAttr(), nullptr);

    _log.nest().trace("Rewrite Expand '{0}' at '{1}' to ExpandDMAOp.", expandOp->getName(), expandOp->getLoc());

    return mlir::success();
}

//
// SwKernelPerAxisTileConverter
//

class ConvertToDMAPass::SwKernelPerAxisTileConverter final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    SwKernelPerAxisTileConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
        setDebugName("ConvertToDMAPass::SwKernelPerAxisTileOp");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::SwKernelPerAxisTileConverter::matchAndRewrite(
        VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const {
    if (!VPUIP::isTileSwKernel(swKernelOp)) {
        return mlir::failure();
    }

    _log.trace("Got Tile SwKernel '{0}' at '{1}'", swKernelOp->getName(), swKernelOp->getLoc());
    const auto ctx = swKernelOp->getContext();

    const auto inType = swKernelOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outType = swKernelOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    VPUX_THROW_UNLESS(inType.getRank() == outType.getRank(), "Tile Op has different input '{0}' output '{1}' rank",
                      inType, outType);

    auto lastResult = swKernelOp->getOperand(0);
    const auto inShape = inType.getShape();
    const auto outShape = outType.getShape();
    // If Tile Op repeats at more than one Axis, will convert to PerAxisTile
    // e.g. Input 1x2x3x4, Output 1x4x9x16, repeats [1x2x3x4]
    // Convert to 3 sub PerAxisTileDMA Op:
    // Sub Op 0: Input 1x2x3x4, Output 1x4x3x4,  repeats [1x2x1x1]
    // Sub Op 1: Input 1x4x3x4, Output 1x4x9x4,  repeats [1x1x3x1]
    // Sub Op 2: Input 1x4x9x4, Output 1x4x9x16, repeats [1x1x1x4]
    auto diffDims = IE::getDiffInOutSizeDims(inShape, outShape);
    auto calculateRequiredCMX = [](const auto& inputType, const auto& outputType) {
        auto requiredCMX = Byte(0);
        if (inputType.getMemoryKind() == vpux::VPU::MemoryKind::CMX_NN) {
            requiredCMX += inputType.getTotalAllocSize();
        }
        if (outputType.getMemoryKind() == vpux::VPU::MemoryKind::CMX_NN) {
            requiredCMX += outputType.getTotalAllocSize();
        }
        return requiredCMX;
    };
    for (size_t i = 0; i < checked_cast<size_t>(diffDims.size()); ++i) {
        auto currRepeatDim = diffDims[i];
        auto lastInType = lastResult.getType().cast<vpux::NDTypeInterface>();
        auto newOutShape = to_small_vector(lastInType.getShape());
        newOutShape[currRepeatDim.ind()] = outShape[currRepeatDim];
        auto newMemRefOutputType = outType.changeShape(ShapeRef(newOutShape));
        if (i < checked_cast<size_t>(diffDims.size()) - 1) {
            auto nextRepeatDim = diffDims[i + 1];
            // Infer the output shape for the next PerAxisTile
            auto nextOutShape = std::move(newOutShape);
            nextOutShape[nextRepeatDim.ind()] = outShape[nextRepeatDim];
            auto nextMemRefOutputType = outType.changeShape(ShapeRef(nextOutShape));

            // The input shape of the next PerAxisTile is the output shape of the current PerAxisTile
            auto nextMemRefInputType = newMemRefOutputType;

            // Calculate the required CMX memory
            auto requiredCMX = calculateRequiredCMX(nextMemRefInputType, nextMemRefOutputType);

            // If the next PerAxisTile can't fit in CMX, the current PerAxisTile should put output in DDR
            if (requiredCMX > VPU::getTotalCMXSize(swKernelOp)) {
                newMemRefOutputType = newMemRefOutputType.changeMemSpace(VPU::MemoryKind::DDR);
            }
        }
        auto outputBuffer = rewriter.create<mlir::memref::AllocOp>(swKernelOp->getLoc(),
                                                                   newMemRefOutputType.cast<mlir::MemRefType>());

        VPUX_THROW_UNLESS(outShape[currRepeatDim] % inShape[currRepeatDim] == 0 &&
                                  outShape[currRepeatDim] / inShape[currRepeatDim] > 1,
                          "Unexpect Tile Op inshape '{0}' outShape '{1}' rank", inShape, outShape);
        const auto repeats = outShape[currRepeatDim] / inShape[currRepeatDim];
        const auto repeatsAttr = mlir::IntegerAttr::get(getInt64Type(ctx), repeats);
        const auto axisAttr = mlir::IntegerAttr::get(getInt64Type(ctx), currRepeatDim.ind());

        lastResult = rewriter.create<VPUIP::PerAxisTileDMAOp>(swKernelOp->getLoc(), lastResult, outputBuffer, axisAttr,
                                                              repeatsAttr, nullptr)
                             .getResult();
    }

    VPUX_THROW_UNLESS(lastResult != swKernelOp->getOperand(0), "Unexpect Tile Op at '{0}', not find repeats Axis",
                      swKernelOp->getLoc());
    rewriter.replaceOp(swKernelOp, lastResult);

    _log.nest().trace("Rewrite PerAxisTile SwKernel '{0}' at '{1}' to PerAxisTileDMA.", swKernelOp->getName(),
                      swKernelOp->getLoc());
    return mlir::success();
}

//
// UpsamplingOpConverter
//

class ConvertToDMAPass::UpsamplingOpConverter final : public mlir::OpRewritePattern<VPUIP::UpsamplingOp> {
public:
    UpsamplingOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::UpsamplingOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::UpsamplingOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToDMAPass::UpsamplingOpConverter::matchAndRewrite(VPUIP::UpsamplingOp origOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto* ctx = origOp.getContext();

    const auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>();
    // If the output of UpsamplingOp can fix into CMX and at DDR
    // Will move it into CMX and Copy back to DDR for better performance
    const auto isOutputBenefitMoveIntoCMX = (outputType.getMemoryKind() == VPU::MemoryKind::DDR) &&
                                            (outputType.getTotalAllocSize() < VPU::getTotalCMXSize(origOp));

    auto outputMemRefType = mlir::cast<mlir::MemRefType>(outputType);
    auto outputBuffer = origOp.getOutputBuff();
    if (isOutputBenefitMoveIntoCMX) {
        auto newOutputType =
                outputType.changeMemSpace(IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0));
        outputMemRefType = newOutputType.cast<mlir::MemRefType>();
        outputBuffer = rewriter.create<mlir::memref::AllocOp>(origOp.getLoc(), outputMemRefType);
    }

    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto zeroType =
            mlir::cast<NDTypeInterface>(mlir::MemRefType::get(outputType.getShape().raw(), inputType.getElementType()))
                    .changeDimsOrder(inputType.getDimsOrder());
    auto constZeros = Const::createZerosConst(rewriter, origOp.getLoc(), mlir::cast<mlir::MemRefType>(zeroType));
    auto copyZeroOp = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), constZeros, outputBuffer);

    const auto origFactors = parseIntArrayAttr<int64_t>(origOp.getUpsamplingFactor());
    // The `upsampling_factor` exist in `UpsamplingOp` with order [W, H, C]
    // Convert it to [N, C, H, W] in `UpsamplingDMAOp`
    VPUX_THROW_UNLESS(origFactors.size() == 3, "Get unexpect upsampling factor");
    SmallVector<int64_t> newFactors = {1, origFactors[2], origFactors[1], origFactors[0]};

    auto outputVal =
            rewriter.create<VPUIP::UpsamplingDMAOp>(origOp.getLoc(), origOp.getInput(), copyZeroOp.getOutput(),
                                                    getIntArrayAttr(ctx, newFactors), /*dma_descriptor*/ nullptr,
                                                    /*expand*/ nullptr, getIntAttr(ctx, 0),
                                                    /*is_out_of_order=*/nullptr, /*is_critical*/ nullptr,
                                                    /*dma_hwp_id*/ nullptr, /*profilingMetadata=*/nullptr)
                    .getOutput();

    _log.trace("Create UpsamplingDMA Op {0} with output buffer at {1}.", outputVal, outputMemRefType.getMemorySpace());

    if (isOutputBenefitMoveIntoCMX) {
        outputVal = rewriter.create<VPUIP::CopyOp>(origOp.getLoc(), outputVal, origOp.getOutputBuff());
    }

    origOp.replaceAllUsesWith(outputVal);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToDMAPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPUIP::SwKernelOp, VPUIP::UpsamplingOp>(op)) {
            return true;
        }
        if (VPUIP::hasBoundedBuffers(op)) {
            return true;
        }

        if (!VPUIP::isLegalAndBeneficialConvertToDMA(op, _log)) {
            return true;
        }

        const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

        if (vpux::isSubByteType(inputType.getElementType())) {
            return true;
        }

        if (mlir::isa<VPUIP::SwKernelOp>(op)) {
            return false;
        }

        return inputType.getMemoryKind() != VPU::MemoryKind::DDR || outputType.getMemoryKind() != VPU::MemoryKind::DDR;
    });
    target.addIllegalOp<VPUIP::ExpandOp>();
    target.addLegalOp<VPUIP::ExpandDMAOp>();

    target.addLegalOp<mlir::memref::AllocOp>();
    target.addLegalOp<VPUIP::CopyOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<VPUIP::UpsamplingDMAOp>();
    target.addIllegalOp<VPUIP::UpsamplingOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExpandConverter>(&ctx, _log);
    patterns.add<SwKernelMemPermuteConverter>(&ctx, _log);
    patterns.add<SwKernelDepthToSpaceConverter>(&ctx, _log);
    patterns.add<SwKernelSpaceToDepthConverter>(&ctx, _log);
    patterns.add<SwKernelPerAxisTileConverter>(&ctx, _log);
    patterns.add<UpsamplingOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToDMACMXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertToDMAPass(Logger log) {
    return std::make_unique<ConvertToDMAPass>(log);
}
