//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_to_pool_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>

using namespace vpux;

namespace {

bool isBeneficialToConvert(ShapeRef shape) {
    // If the MemPermute is legal to be converted to a pooling op. Need to compare with the DMA implementation.
    // Experimental data shows a linear correlation between inference time and permute data size for both ODU permute
    // and DMA permute with different slopes.
    // Experimental Constraint: utilize DMA conversion when data size is less than the threshold
    return shape.totalSize() >= PERMUTE_TO_POOLING_THRESHOLD;
}

}  // namespace

DimsOrder vpux::getNHWCOutputLayout(const DimsOrder& memPermute) {
    // To use NCE accelerate Permutation, we always cast the input tensor's layout to NHWC based on physical layout.
    //  In this way, we only need consider the below 5 cases:
    //
    //                  NHWC (Case 0)
    //                   |
    //      NHCW  NWCH  NWHC  NCWH  NCHW
    // Case   1    2     3     4     5
    //
    const std::unordered_map<DimsOrder, DimsOrder> permuteToLayout = {{DimsOrder::NCWH, DimsOrder::NHCW},
                                                                      {DimsOrder::NHWC, DimsOrder::NWCH},
                                                                      {DimsOrder::NHCW, DimsOrder::NWHC},
                                                                      {DimsOrder::NWHC, DimsOrder::NCWH},
                                                                      {DimsOrder::NWCH, DimsOrder::NCHW}};
    const auto configIter = permuteToLayout.find(memPermute);
    VPUX_THROW_WHEN(configIter == permuteToLayout.end(), "The permute layout {0} not supported.", memPermute);
    return configIter->second;
}

SmallVector<std::pair<Shape, DimsOrder>> vpux::calculateConversions(ShapeRef originInputShape,
                                                                    const int64_t alignedChannel,
                                                                    const DimsOrder& targetOrder) {
    //
    //               NWCH (Case 2)
    //                 |
    //      NHCW  NWHC  NCWH  NCHW
    // Case   1    3     4     5
    //
    const std::unordered_map<DimsOrder, DimsOrder> dimHLayoutToPerm = {{DimsOrder::NHCW, DimsOrder::NWHC},
                                                                       {DimsOrder::NWHC, DimsOrder::NCWH},
                                                                       {DimsOrder::NCWH, DimsOrder::NHCW},
                                                                       {DimsOrder::NCHW, DimsOrder::NHWC}};

    //
    //          NCHW (Case 5)
    //             |
    //      NHCW  NWHC  NCWH
    // Case   1    3     4
    //
    const std::unordered_map<DimsOrder, DimsOrder> dimWLayoutToPerm = {{DimsOrder::NHCW, DimsOrder::NHCW},
                                                                       {DimsOrder::NWHC, DimsOrder::NWHC},
                                                                       {DimsOrder::NCWH, DimsOrder::NCWH}};

    const auto dimHAligned = (originInputShape[Dims4D::Act::H] % alignedChannel) == 0;
    const auto dimWAligned = (originInputShape[Dims4D::Act::W] % alignedChannel) == 0;
    const auto dimWCAligned =
            ((originInputShape[Dims4D::Act::W] * originInputShape[Dims4D::Act::C]) % alignedChannel) == 0;
    const auto dimHCAligned =
            ((originInputShape[Dims4D::Act::H] * originInputShape[Dims4D::Act::C]) % alignedChannel) == 0;
    SmallVector<std::pair<Shape, DimsOrder>> newMaxPoolOrder;

    auto getMaxPoolTargetDimOrder =
            [targetOrder](const std::unordered_map<DimsOrder, DimsOrder>& dimsLayoutToPermConfig) {
                const auto layoutPermute = dimsLayoutToPermConfig.find(targetOrder);
                VPUX_THROW_WHEN(layoutPermute == dimsLayoutToPermConfig.end(), "The layout should be considered.");
                return getNHWCOutputLayout(layoutPermute->second);
            };

    auto calculateSingleDimConversion = [&](const bool mergedAlign, const bool dimAligned,
                                            const DimsOrder& fromDimOrder, const DimsOrder& toDimOrder,
                                            const std::unordered_map<DimsOrder, DimsOrder>& layout2Perm) -> bool {
        if (!mergedAlign) {
            newMaxPoolOrder.clear();
            return false;  // Failed
        }
        Shape castShape = {
                originInputShape[fromDimOrder.dimAt(0)], alignedChannel, originInputShape[fromDimOrder.dimAt(1)],
                originInputShape[fromDimOrder.dimAt(2)] * originInputShape[fromDimOrder.dimAt(3)] / alignedChannel};

        newMaxPoolOrder.push_back({castShape, DimsOrder::NWCH});
        if (targetOrder == toDimOrder) {
            return false;
        }
        if (dimAligned) {
            castShape = {originInputShape[toDimOrder.dimAt(0)], originInputShape[toDimOrder.dimAt(3)],
                         originInputShape[toDimOrder.dimAt(1)], originInputShape[toDimOrder.dimAt(2)]};
            newMaxPoolOrder.push_back({castShape, getMaxPoolTargetDimOrder(layout2Perm)});
            return false;
        }
        return true;
    };

    auto needFollowProcess =
            calculateSingleDimConversion(dimWCAligned, dimHAligned, DimsOrder::NHWC, DimsOrder::NWCH, dimHLayoutToPerm);
    if (!needFollowProcess) {
        return newMaxPoolOrder;
    }
    needFollowProcess =
            calculateSingleDimConversion(dimHCAligned, dimWAligned, DimsOrder::NWCH, DimsOrder::NCHW, dimWLayoutToPerm);
    if (needFollowProcess) {
        // If need more process, the layout conversion will be like: NCHW -> NHWC.
        // And NHWC is input layout, so we can't convert this MemPermute to MaxPool.
        newMaxPoolOrder.clear();
    }
    return newMaxPoolOrder;
}

bool vpux::isLegalConvertToPool(NDTypeInterface inputType, NDTypeInterface outputType, mlir::Operation* parentOp,
                                mlir::AffineMap memPermMap, mlir::MLIRContext* ctx, int64_t numClusters,
                                llvm::StringRef debugName, config::ArchKind arch, const Logger& log) {
    // Pooling op does not support dynamic shapes,
    // so we fail transformation if any of the input or output shapes are dynamic.
    auto isDynamic = [](NDTypeInterface type) {
        return mlir::isa<Core::BoundedTensorType>(type) || mlir::isa<Core::DynamicDimsMaskTensorType>(type);
    };

    if (isDynamic(inputType) || isDynamic(outputType)) {
        log.trace("MemPermuteOp has dynamic tensors");
        return false;
    }

    const auto inputElementType = inputType.getElementType();
    if (const auto inputQuantType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputElementType)) {
        const bool is16BitsQuantization = (inputQuantType.getStorageType().getIntOrFloatBitWidth() == 16);
        if (is16BitsQuantization) {
            log.trace("NCE MaxPool does not support quantized 16 bits input");
            return false;
        }
        const bool isSubByteQuantization = (inputQuantType.getStorageType().getIntOrFloatBitWidth() < 8);
        if (isSubByteQuantization) {
            log.trace("NCE MaxPool does not support sub-byte quantized input");
            return false;
        }
    }

    const auto inShape = inputType.getShape();
    const auto inMemShape = inputType.getMemShape();

    // E-128307: Replace with using a robust NCE-Op supported datatype checking mechanism
    const auto elementType = outputType.getElementType();
    if (elementType.isSignedInteger() || elementType.isUnsignedInteger()) {
        log.trace("NCE MaxPool does not support signed or unsigned integer");
        return false;
    }
    if (mlir::isa<mlir::FloatType>(elementType) && mlir::cast<mlir::FloatType>(elementType).getIntOrFloatBitWidth() !=
                                                           mlir::Float16Type::get(ctx).getIntOrFloatBitWidth()) {
        log.trace("NCE MaxPool does not support float type width different from 16 bits");
        return false;
    }
    if (isTrivialPermute(inMemShape, memPermMap)) {
        log.trace("MemPermuteOp is actually a permute cast");
        return false;
    }

    const auto memPerm = DimsOrder::fromAffineMap(memPermMap);
    if (memPerm.dimAt(0) != Dims4D::Act::N) {
        log.trace("MemPermuteOp with dim N changed dim position");
        return false;
    }

    if (auto expandOp = mlir::dyn_cast_or_null<IE::ExpandOp>(parentOp)) {
        auto inType = mlir::cast<NDTypeInterface>(expandOp.getInput().getType());
        auto outType = mlir::cast<NDTypeInterface>(expandOp.getResult().getType());
        const auto isExpandAtChannel = inType.getShape()[Dims4D::Act::C] != outType.getShape()[Dims4D::Act::C];
        if (expandOp->hasOneUse() && isExpandAtChannel && inType.getDimsOrder() == DimsOrder::NCHW &&
            !IE::isEligibleConvertToConv(expandOp, log, debugName)) {
            // For expand which will be lowered into DMA op, there is an optimization in another pass later which will
            // fuse pattern `input(NCHW) -> Expand -> Permute` into a single DMA op. So skip mempermute optimization
            // here.
            log.trace("MemPermuteOp will be fused with parent Expand op in later pass");
            return false;
        }
    }

    if (memPerm == DimsOrder::NHCW) {
        // Attention -> [Slice/PermuteCast]+ -> NHCW MemPermute: prefer DMA so it fuses with Slice.
        if (mlir::isa_and_nonnull<IE::SliceOp, IE::PermuteCastOp>(parentOp)) {
            auto p = parentOp;
            while (mlir::isa_and_nonnull<IE::SliceOp, IE::PermuteCastOp>(p)) {
                p = p->getOperand(0).getDefiningOp();
            }
            if (mlir::isa_and_nonnull<IE::AttentionOp>(p)) {
                log.trace("NHCW MemPermute after Attention->Slice: prefer PermuteDMA");
                return false;
            }
        }
        if (!isBeneficialToConvert(inShape)) {
            return false;
        }
    }

    // Populate the target shape following NCHW order of dimensions.
    // Physical layout NHWC corresponds to logical layout NCHW.
    const Shape targetInShape = {inMemShape[MemDim(0)], inMemShape[MemDim(3)], inMemShape[MemDim(1)],
                                 inMemShape[MemDim(2)]};
    const auto targetOrder = getNHWCOutputLayout(memPerm);

    // Calculate the inputType of maxPoolOp
    const auto poolInType = inferNewTypeWithMemPerm(
            inputType, getPermutationFromOrders(DimsOrder::NCHW, DimsOrder::NCHW, ctx), DimsOrder::NHWC);
    const auto poolInLogicShape = poolInType.getShape();
    if (poolInLogicShape[Dims4D::Act::N] != 1) {
        log.trace("MaxPoolOp with dim N > 1");
        return false;
    }

    const auto IC = poolInLogicShape[Dims4D::Act::C];
    const auto alignedChannel = VPU::NCEInvariant::getAlignment(outputType.getElementType());
    if (IC % alignedChannel != 0) {
        // Not use shapeCast for per axis type
        bool isPerAxisQuant = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(poolInType.getElementType());
        if (isPerAxisQuant) {
            log.trace("Can not reshape for per axis quant");
            return false;
        }
        auto conversionMap = calculateConversions(targetInShape, alignedChannel, targetOrder);
        auto hasSmallHeightNum = [&](const std::pair<Shape, DimsOrder>& map) {
            const int64_t PERFORMANT_HEIGHT_NUM_OF_PER_CLUSTER = 4;
            return map.first[Dims4D::Act::H] < numClusters * PERFORMANT_HEIGHT_NUM_OF_PER_CLUSTER;
        };
        bool hasToSplitOnDimC = llvm::any_of(conversionMap, hasSmallHeightNum);
        // If new MaxPool has to be split on Dim C which is the inner most dimension,
        // it is not performant because of strided DMA.
        // For the case of memPerm DimsOrder::NCWH, there may exist maxPool conversions size > 2
        const auto isNotPerformant = (memPerm == DimsOrder::NHCW || memPerm == DimsOrder::NCWH) &&
                                     (hasToSplitOnDimC || conversionMap.size() > 2);
        if (conversionMap.empty() || isNotPerformant) {
            log.trace("Channels of an IE.MaxPool are not aligned or the Conversion is not performant.");
            return false;
        }
    } else {
        const auto poolOutType = mlir::cast<vpux::NDTypeInterface>(poolInType).changeDimsOrder(targetOrder);
        // If types exist per axis quantize, check if both types are consistent
        auto inputPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(poolInType.getElementType());
        auto outputPerAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(poolOutType.getElementType());
        if ((inputPerAxisType || outputPerAxisType) && poolInType.getElementType() != poolOutType.getElementType()) {
            log.trace("NCE MaxPool does not support inconsistent element type");
            return false;
        }
    }

    if (VPUIP::satisfiesOptimizedMemPermute(arch, inputType, outputType)) {
        log.trace("Software memPermute is more efficient");
        return false;
    }
    return true;
}
