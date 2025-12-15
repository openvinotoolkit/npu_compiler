//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

bool vpux::isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'", memPerm,
                      inShape);

    SmallVector<int64_t> nonTrivialPerm;

    for (auto ind : irange(inShape.size())) {
        const auto inDim = MemDim(perm.dimAt(ind).ind());

        if (inShape[inDim] == 1) {
            continue;
        }

        nonTrivialPerm.push_back(inDim.ind());
    }

    if (nonTrivialPerm.empty()) {
        return true;
    }

    for (auto ind : irange<size_t>(1, nonTrivialPerm.size())) {
        if (nonTrivialPerm[ind] < nonTrivialPerm[ind - 1]) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> vpux::getPermutateDims(MemShapeRef inShape, mlir::AffineMap memPerm) {
    const auto perm = DimsOrder::fromAffineMap(memPerm);
    VPUX_THROW_UNLESS(inShape.size() == perm.numDims(), "Permutation '{0}' is not compatible with shape '{1}'", memPerm,
                      inShape);

    SmallVector<int64_t> permutateDims;

    for (auto ind : irange(inShape.size())) {
        const auto inDim = MemDim(perm.dimAt(ind).ind());

        if (inShape[inDim] == 1) {
            continue;
        }

        permutateDims.push_back(inDim.ind());
    }

    for (auto ind : irange<size_t>(1, permutateDims.size())) {
        if (permutateDims[ind] > permutateDims[ind - 1]) {
            permutateDims.clear();
            return permutateDims;
        }
    }

    return permutateDims;
}

bool vpux::isTrivialReorder(DimsOrder inOrder, DimsOrder outOrder, ShapeRef shape) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    const auto shapeIsOne = [&](const Dim& perm) -> bool {
        return shape[perm] == 1;
    };
    // Ignore dim whose shape is one
    inPerm.erase(std::remove_if(inPerm.begin(), inPerm.end(), shapeIsOne), inPerm.end());
    outPerm.erase(std::remove_if(outPerm.begin(), outPerm.end(), shapeIsOne), outPerm.end());

    return inPerm == outPerm;
}

bool vpux::isTrivialReorder(IE::ReorderOp origOp) {
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto inShape = getShape(origOp.getInput());

    return isTrivialReorder(inOrder, outOrder, inShape);
}

mlir::AffineMap vpux::getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx) {
    auto inPerm = inOrder.toPermutation();
    auto outPerm = outOrder.toPermutation();
    SmallVector<uint32_t> memPerm(inPerm.size());
    for (auto p : outPerm | indexed) {
        memPerm[p.index()] = static_cast<uint32_t>(inOrder.dimPos(p.value()));
    }

    return mlir::AffineMap::getPermutationMap(ArrayRef(memPerm), ctx);
}

DimsOrder vpux::applyPermutation(const DimsOrder srcOrder, const DimsOrder dstOrder) {
    const auto srcPermutation = srcOrder.toPermutation();
    const auto dstPermutation = dstOrder.toPermutation();
    DimArr result;
    const auto getDimAt = [&](const Dim& perm) -> Dim {
        return srcOrder.dimAt(perm.ind());
    };
    std::transform(dstPermutation.begin(), dstPermutation.end(), std::back_inserter(result), getDimAt);
    return DimsOrder::fromPermutation(result);
}

// change order like from CNHW to NCHW
DimsOrder vpux::moveD0ToTheFront(DimsOrder inOrder) {
    SmallVector<vpux::Dim> perm = {Dim(0)};
    auto permutation = inOrder.toPermutation();
    std::copy_if(permutation.begin(), permutation.end(), std::back_inserter(perm), [](const Dim dim) {
        return dim != Dim(0);
    });
    return DimsOrder::fromPermutation(ArrayRef(perm));
}

// Normalize permutation vector
// Example: [1, 3, 7, 6] -> [0, 1, 3, 2]
void normalizePermutation(SmallVector<uint32_t>& vec) {
    SmallVector<uint32_t> sorted(vec);
    llvm::DenseMap<uint32_t, uint32_t> helper;
    std::sort(sorted.begin(), sorted.end());

    for (size_t i = 0; i < sorted.size(); ++i) {
        helper.insert(std::make_pair(sorted[i], checked_cast<uint32_t>(i)));
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = helper[vec[i]];
    }
}

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> vpux::getMergedPermutationAndShape(NDTypeInterface inputType,
                                                                                          mlir::AffineMap permutation,
                                                                                          int64_t rank) {
    auto memShape = to_small_vector(inputType.getMemShape());
    auto origPermVec = DimsOrder::fromAffineMap(permutation).toPermutation();

    // Example of origPermVec
    // origPermVec = [d0, d1, d2, d3] -> [d1, d2, d3, d0];
    // origPermVec[0] = 1, origPermVec[1] = 2, origPermVec[2] = 3, origPermVec[3] = 0
    SmallVector<uint32_t> permVec;
    SmallVector<int64_t> shapeVec;

    for (auto d : origPermVec) {
        // Dims with size 1 are dropped
        if (memShape[d.ind()] != 1) {
            permVec.push_back(checked_cast<uint32_t>(d.ind()));
        }
    }

    for (auto dimSize : memShape) {
        // Dims with size 1 are dropped
        if (dimSize != 1) {
            shapeVec.push_back(dimSize);
        }
    }

    normalizePermutation(permVec);

    if ((int64_t)shapeVec.size() < rank) {
        return std::make_pair(permVec, shapeVec);
    }

    // Merge dims that are adjacent before and after permutation
    // Example:
    // memShape =[2, 4, 25, 255, 255]
    // permVec = [d0, d1, d2, d3, d4] -> [d0, d4, d1, d2, d3]
    //
    // mergedShape = [2, 25500, 255]
    // mergedPermutation = [d0, d1, d2] -> [d0, d2, d1]
    SmallVector<uint32_t> mergedPermutation;
    SmallVector<int64_t> mergedShape;

    std::map<uint32_t, int64_t> mergedShapeMap;
    size_t j = 0;
    int64_t remainingRank = permVec.size();
    for (size_t i = 0; i < checked_cast<size_t>(permVec.size()); i = j) {
        int64_t dimSize = shapeVec[permVec[i]];
        for (j = i + 1; j < checked_cast<size_t>(permVec.size()) && (permVec[j - 1] + 1 == permVec[j]); ++j) {
            if (remainingRank < rank) {
                break;
            }
            dimSize *= shapeVec[permVec[j]];
            remainingRank--;
        }

        mergedShapeMap.insert(std::make_pair(permVec[i], dimSize));
        mergedPermutation.push_back(permVec[i]);
    }

    // Keys iterated in ascending order
    for (const auto& p : mergedShapeMap) {
        mergedShape.push_back(p.second);
    }

    // Normalize vectors
    normalizePermutation(mergedPermutation);

    return std::make_pair(mergedPermutation, mergedShape);
}

void vpux::extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape,
                                     int64_t targetRank) {
    const int64_t TENSOR_4D_RANK = 4;
    // Function Description:
    // This function adjusts the permutation and shape vectors to match a specified target rank.
    //
    // Considerations for optimal performance:
    // 1. Dimension N should be set to 1:
    //    - It allows the possibility of transforming mempermute operations into MaxPool for enhanced performance
    //    - If N > 1, the operation defaults to PermuteDMA
    // 2. For a 2D to 4D case, it's advantageous to extend dimensions N and H by 1:
    //    - This adjustment aligns with the NCE's requirements for 4D tensor operations, ensuring compatibility and
    //    potentially improving processing efficiency
    //
    // Example:
    // - 2D to 4D case: shape: [128, 256], permutation: [d1, d0]
    //   will be transformed to: shape: [1, 128, 1, 256], permutation: [d0, d3, d2, d1]
    // - 3D to 4D case: shape: [4, 128, 256], permutation: [d1, d0, d2]
    //   will be transformed to: shape: [1, 4, 128, 256], permutation: [d0, d2, d1, d3]
    int64_t padSize = targetRank - checked_cast<int64_t>(permutation.size());
    if (targetRank == TENSOR_4D_RANK && shape.size() == 2) {
        permutation = SmallVector<uint32_t>{0, 3, 2, 1};
        shape = SmallVector<int64_t>{1, shape[0], 1, shape[1]};
    } else if (padSize > 0) {
        auto paddedPermutation = SmallVector<uint32_t>(targetRank);
        auto paddedShape = SmallVector<int64_t>(targetRank);
        for (int64_t i = 0; i < targetRank; ++i) {
            paddedPermutation[i] = i < padSize ? i : permutation[i - padSize] + padSize;
            paddedShape[i] = i < padSize ? 1 : shape[i - padSize];
        }
        permutation.assign(paddedPermutation);
        shape.assign(paddedShape);
    }
}

IE::LayerWithPermuteInterface vpux::getFusableLayerWithPermuteInterface(mlir::Operation* op) {
    auto inputOp = op->getOperand(0).getDefiningOp();
    if (auto quantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(inputOp)) {
        auto outElemType = quantizeCastOp.getOutput().getType().getElementType();
        if (quantizeCastOp->hasOneUse() && mlir::isa<mlir::quant::UniformQuantizedType>(outElemType)) {
            inputOp = quantizeCastOp.getInput().getDefiningOp();
        }
    }
    return mlir::dyn_cast_or_null<IE::LayerWithPermuteInterface>(inputOp);
}

NDTypeInterface vpux::inferNewTypeWithMemPerm(NDTypeInterface oldType, mlir::AffineMap memPerm,
                                              const DimsOrder& dstOrder) {
    const auto oldMemShape = oldType.getMemShape();
    const auto oldOrder = oldType.getDimsOrder();
    const auto newMemShape = applyPerm(oldMemShape, memPerm);
    const auto newShape = dstOrder.toLogicalOrder(newMemShape);
    auto elemType = oldType.getElementType();
    if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType)) {
        const auto origAxis = perAxisType.getQuantizedDimension();
        const auto inMemAxis = oldOrder.dimPos(Dim(origAxis));
        const auto outMemAxis = DimsOrder::fromAffineMap(memPerm).dimPos(Dim(inMemAxis));
        const auto outAxis = dstOrder.dimAt(outMemAxis);
        elemType = changeAxis(perAxisType, outAxis.ind());
    }
    return oldType.changeDimsOrder(dstOrder).changeShapeElemType(newShape, elemType);
}

mlir::FailureOr<VPU::DistributionInfo> vpux::applyPermutationOnDistributionInfo(
        vpux::NDTypeInterface inType, const VPU::DistributionInfo& inDistribution, mlir::AffineMap memPerm,
        DimsOrder srcOrder, DimsOrder dstOrder, ShapeRef srcShape, ShapeRef dstShape) {
    auto permuteAxisOfArray = [&](ArrayRef<int64_t> arr) -> SmallVector<int64_t> {
        // At VPUIP level, VPU.LayoutCast gets lowered to VPUIP.PermuteCast.
        // LayoutCast will have same in/out shape but different orders, which cannot be handled
        // the same way as the VPU.PermuteCast ops which have the same memory shape between input
        // and output even if orders and logical shapes differ. In such a case, applying the
        // `toMemoryOrder -> applyPerm -> toLogicalOrder` transformations will not permute the
        // distributed attr correctly.
        if (arr.empty()) {
            return SmallVector<int64_t>(arr);
        }
        if (srcShape == dstShape) {
            return SmallVector<int64_t>(arr);
        }

        const auto arrInMemOrder = srcOrder.toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, memPerm);
        auto arrPermutedInLogicalOrder = dstOrder.toLogicalOrder(arrPermutedInMemOrder).raw();

        return arrPermutedInLogicalOrder;
    };

    auto numTiles = permuteAxisOfArray(inDistribution.getNumTiles());
    auto alignment = permuteAxisOfArray(inDistribution.getAlignment());

    auto permutePerClusterShapesOffsets =
            [&](ArrayRef<SmallVector<int64_t>> inPerClusterShapesOffsetsVec) -> SmallVector<SmallVector<int64_t>> {
        if (inPerClusterShapesOffsetsVec.empty()) {
            return SmallVector<SmallVector<int64_t>>(inPerClusterShapesOffsetsVec);
        }
        SmallVector<SmallVector<int64_t>> outComputeShapesVec{};
        outComputeShapesVec.reserve(inPerClusterShapesOffsetsVec.size());
        std::transform(inPerClusterShapesOffsetsVec.begin(), inPerClusterShapesOffsetsVec.end(),
                       std::back_inserter(outComputeShapesVec), [&](const SmallVector<int64_t>& shapesOffsets) {
                           return permuteAxisOfArray(shapesOffsets);
                       });

        return outComputeShapesVec;
    };

    auto computeShapes = permutePerClusterShapesOffsets(inDistribution.getComputeShapes());
    auto computeOffsets = permutePerClusterShapesOffsets(inDistribution.getComputeOffsets());
    auto memoryShapes = permutePerClusterShapesOffsets(inDistribution.getMemoryShapes());
    auto memoryOffsets = permutePerClusterShapesOffsets(inDistribution.getMemoryOffsets());

    auto distribution = VPU::DistributionInfo(
            inDistribution.getDistributionMode(), numTiles, inDistribution.getKernel(), inDistribution.getStrides(),
            inDistribution.getPadding(), inDistribution.getNumClusters(), alignment,
            inDistribution.hasUniformDistributedSegments(), computeShapes, computeOffsets, memoryShapes, memoryOffsets,
            inDistribution.hasEqualMemoryAndComputeView());

    if (inDistribution.getDistributionMode() != VPU::DistributionMode::OVERLAPPED) {
        return distribution;
    }

    if (VPU::isOverlappedOverH(distribution) || VPU::isOverlappedOverW(distribution)) {
        return distribution;
    }

    if (VPU::isSegmentedLikeDistributionMode(inType, inDistribution)) {
        return VPU::legalizeCastedDistribution(distribution);
    }
    return mlir::failure();
}

// for a given input and a output requirement(outOrdr and outShape), the function is trying to find a permutation that
// can use permuteCastOp to convert input to output requirement.
std::optional<mlir::AffineMap> vpux::tryToFindPermutationForPermuteCast(NDTypeInterface inputType, DimsOrder outOrder,
                                                                        ShapeRef outShape, mlir::MLIRContext* ctx) {
    const auto inMemShape = inputType.getMemShape().raw();
    const auto outMemShape = outOrder.toMemoryOrder(outShape).raw();

    auto hasSameLogicShape = [&] {
        SmallVector<int64_t> inShape(inMemShape);
        SmallVector<int64_t> outShape(outMemShape);
        if (inShape.size() != outShape.size()) {
            return false;
        }

        for (auto dim : inShape) {
            auto it = std::find(outShape.begin(), outShape.end(), dim);
            if (it != outShape.end()) {
                outShape.erase(it);
            } else {
                return false;
            }
        }
        return true;
    };

    // logic shape need to be the same, like input shape[1x2x3x4], output shape is [2x1x3x4]
    if (!hasSameLogicShape()) {
        return std::nullopt;
    }

    // Try to find a permutation from the mem shape. For each dimension in input, find the position
    // in output, then store it to permutation map. Like if in mem shape is [10, 20, 30, 40],
    // out mem shape is [10, 20, 40, 30], then the permutation is {0, 1, 3, 2}.
    SmallVector<int64_t> permutation;
    for (auto outShape : outMemShape) {
        for (auto inShape : inMemShape | indexed) {
            if (inShape.value() == outShape &&
                std::find(permutation.begin(), permutation.end(), inShape.index()) == permutation.end()) {
                permutation.push_back(inShape.index());
                break;
            }
        }
    }

    if (permutation.size() != inMemShape.size()) {
        return std::nullopt;
    }

    auto permutationMap = mlir::AffineMap::getPermutationMap(permutation, ctx);
    const auto memShape = inputType.getMemShape();
    if (!isTrivialPermute(memShape, permutationMap)) {
        return std::nullopt;
    }
    if (applyPerm(memShape, permutationMap) != outOrder.toMemoryOrder(outShape)) {
        return std::nullopt;
    }

    return permutationMap;
}

// for a given input and a output requirement(outOrdr and outShape), the function is trying to find a permutation that
// can use permuteCastOp to convert input to output requirement.
std::optional<IE::PermuteCastOp> vpux::tryToFindPermuteCastOp(mlir::Location loc, mlir::Value input, DimsOrder outOrder,
                                                              ShapeRef outShape, mlir::PatternRewriter& rewriter) {
    const auto ctx = rewriter.getContext();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto hasValidPermutationMap =
            tryToFindPermutationForPermuteCast(inputType, outOrder, outShape, rewriter.getContext());
    if (hasValidPermutationMap.has_value()) {
        return rewriter.create<IE::PermuteCastOp>(loc, input, mlir::AffineMapAttr::get(outOrder.toAffineMap(ctx)),
                                                  mlir::AffineMapAttr::get(hasValidPermutationMap.value()));
    } else {
        return std::nullopt;
    }
}

/**
 * @brief Infers the dimension after applying a permutation.
 *
 * This function calculates the new dimension in the destination order after applying a permutation
 * to a given dimension in the source order.
 *
 * @param dim The original dimension in the source order.
 * @param srcOrder The source order of dimensions.
 * @param dstOrder The destination order of dimensions.
 * @param perm The affine map representing the permutation.
 * @return The new dimension in the destination order after applying the permutation.

    For example, given a dimension Dim(1) representing C in the source order [NWHC], we want to calculate the output
    dimension in the destination order [NCWH] after applying a permutation (0, 1, 2, 3)
    -> (0, 2, 3, 1).

    Steps:
    1. Identify Source Memory Dimension:
    In this case, Dim(1) corresponds to MemDim(3) in the source order [NWHC].

    2. Apply Permutation:
    The permutation (0, 1, 2, 3) -> (0, 2, 3, 1) is applied to the source memory dimension.
    MemDim(3) is changed to dimension position 2 after applying the permutation.

    3. Determine Destination Logical Dimension:
    In the destination order [NCWH], dimension position 2 corresponds to logical dimension Dim(3) (which represents W).

    The function returns Dim(3) as the result, indicating that the logical dimension C in the source order [NWHC] maps
    to logical dimension W in the destination order [NCWH] after applying the permutation.
 */
Dim vpux::inferDimAfterPermutation(Dim dim, DimsOrder srcOrder, DimsOrder dstOrder, mlir::AffineMap perm) {
    const auto srcMemDim = srcOrder.toMemDim(dim);
    const auto dstDimPos = DimsOrder::fromAffineMap(perm).dimPos(Dim(srcMemDim.ind()));
    return dstOrder.dimAt(dstDimPos);
}
