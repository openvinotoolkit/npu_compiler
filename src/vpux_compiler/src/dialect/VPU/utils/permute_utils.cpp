//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

#include <mlir/Support/LLVM.h>

using namespace vpux;

Shape VPU::inferShapeThroughPermute(ShapeRef origShape, NDTypeInterface srcType, NDTypeInterface dstType,
                                    mlir::AffineMap memPerm) {
    const auto arrInMemOrder = srcType.getDimsOrder().toMemoryOrder(origShape);
    const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, memPerm);
    return Shape(dstType.getDimsOrder().toLogicalOrder(arrPermutedInMemOrder).raw());
}

mlir::FailureOr<VPU::DistributionInfo> VPU::applyPermutationOnDistributionInfo(
        vpux::NDTypeInterface inType, const VPU::DistributionInfo& inDistribution, mlir::AffineMap memPerm,
        const DimsOrder& srcOrder, const DimsOrder& dstOrder, ShapeRef srcShape, ShapeRef dstShape) {
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
        const auto arrPermutedInMemOrder = vpux::applyPerm(arrInMemOrder, memPerm);
        auto arrPermutedInLogicalOrder = dstOrder.toLogicalOrder(arrPermutedInMemOrder).raw();

        return arrPermutedInLogicalOrder;
    };

    auto numTiles = permuteAxisOfArray(inDistribution.getNumTiles());
    auto alignment = permuteAxisOfArray(inDistribution.getAlignment());
    auto memoryNumTiles = inDistribution.getMemoryNumTiles().has_value()
                                  ? std::make_optional(permuteAxisOfArray(inDistribution.getMemoryNumTiles().value()))
                                  : std::nullopt;

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
            inDistribution.hasEqualMemoryAndComputeView(), memoryNumTiles);

    if (!(VPU::bitEnumContainsAny(inDistribution.getDistributionMode(), VPU::DistributionMode::OVERLAPPED))) {
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

mlir::FailureOr<SmallVector<Dim>> VPU::remapDimsThroughInputPermuteCast(mlir::func::FuncOp funcOp,
                                                                        ArrayRef<Dim> inputDims) {
    auto remappedDims = SmallVector<Dim>(inputDims.begin(), inputDims.end());

    auto isTensorArg = [](mlir::BlockArgument arg) {
        return mlir::isa<vpux::NDTypeInterface>(arg.getType()) || mlir::isa<mlir::RankedTensorType>(arg.getType());
    };

    auto hasTensorInputPermuteCast = [&](mlir::func::FuncOp op) {
        return llvm::any_of(op.getArguments(), [&](mlir::BlockArgument arg) {
            if (!isTensorArg(arg)) {
                return false;
            }
            return llvm::any_of(arg.getUsers(), [](mlir::Operation* userOp) {
                return mlir::isa<VPU::PermuteCastOp>(userOp);
            });
        });
    };

    const auto tensorArgCount = llvm::count_if(funcOp.getArguments(), [&](mlir::BlockArgument arg) {
        return isTensorArg(arg);
    });

    if (tensorArgCount > 1 && hasTensorInputPermuteCast(funcOp)) {
        funcOp.emitError("VPU::remapDimsThroughInputPermuteCast: unsupported multi-input function with input "
                         "VPU.PermuteCast. Supported cases are: (1) any number of inputs without input "
                         "VPU.PermuteCast, or (2) single-input function with input VPU.PermuteCast");
        return mlir::failure();
    }

    auto inputTensorArgIt = llvm::find_if(funcOp.getArguments(), [&](mlir::BlockArgument arg) {
        return isTensorArg(arg);
    });
    if (inputTensorArgIt == funcOp.getArguments().end()) {
        return remappedDims;
    }

    auto inputArg = *inputTensorArgIt;
    for (auto* userOp : inputArg.getUsers()) {
        auto permuteCastOp = mlir::dyn_cast<VPU::PermuteCastOp>(userOp);
        if (permuteCastOp == nullptr) {
            continue;
        }
        if (!inputArg.hasOneUse()) {
            permuteCastOp.emitError(
                    "VPU::remapDimsThroughInputPermuteCast: unsupported case where input tensor argument "
                    "with VPU.PermuteCast has multiple uses");
            return mlir::failure();
        }
        auto srcType = mlir::dyn_cast<vpux::NDTypeInterface>(userOp->getOperand(0).getType());
        auto dstType = mlir::dyn_cast<vpux::NDTypeInterface>(userOp->getResult(0).getType());
        auto memPerm = permuteCastOp.getMemPerm();
        if (srcType == nullptr || dstType == nullptr) {
            break;
        }

        for (auto& remappedDim : remappedDims) {
            remappedDim =
                    inferDimAfterPermutation(remappedDim, srcType.getDimsOrder(), dstType.getDimsOrder(), memPerm);
        }
        break;
    }

    return remappedDims;
}

std::optional<VPU::MemPermuteAlignmentInfo> VPU::getMemPermuteAlignmentInfo(NDTypeInterface inType,
                                                                            NDTypeInterface outType,
                                                                            mlir::AffineMap memPerm) {
    const auto elemBitWidth = static_cast<int64_t>(outType.getElemTypeSize().count());
    if (elemBitWidth >= CHAR_BIT) {
        return std::nullopt;
    }

    const auto outShape = getBoundedShape(outType);
    const auto inShape = getBoundedShape(inType);
    const auto outOrder = outType.getDimsOrder();
    const auto inOrder = inType.getDimsOrder();
    const auto rank = static_cast<int64_t>(outOrder.numDims());

    auto permInvMap = mlir::inversePermutation(memPerm);
    SmallVector<int64_t> permInv(rank);
    for (int64_t i = 0; i < rank; ++i) {
        permInv[i] = static_cast<int64_t>(mlir::cast<mlir::AffineDimExpr>(permInvMap.getResult(i)).getPosition());
    }

    SmallVector<int64_t> outInnerProducts(rank, 1);
    for (int64_t pos = rank - 2; pos >= 0; --pos) {
        outInnerProducts[pos] = outInnerProducts[pos + 1] * outShape[outOrder.dimAt(pos + 1)];
    }
    SmallVector<int64_t> inInnerProducts(rank, 1);
    for (int64_t pos = rank - 2; pos >= 0; --pos) {
        inInnerProducts[pos] = inInnerProducts[pos + 1] * inShape[inOrder.dimAt(pos + 1)];
    }

    return MemPermuteAlignmentInfo{
            elemBitWidth, rank, outOrder, std::move(permInv), std::move(outInnerProducts), std::move(inInnerProducts)};
}

std::optional<VPU::MemPermuteAlignmentInfo> VPU::getMemPermuteAlignmentInfo(VPU::MemPermuteOp memPermuteOp) {
    auto inType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getInput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(memPermuteOp.getOutput().getType());
    return getMemPermuteAlignmentInfo(inType, outType, memPermuteOp.getMemPerm());
}

int64_t VPU::MemPermuteAlignmentInfo::getAlignmentForMemPos(int64_t memPos) const {
    const auto outPlaneBits = outInnerProducts[memPos] * elemBitWidth;
    const auto inPlaneBits = inInnerProducts[permInv[memPos]] * elemBitWidth;
    if (outPlaneBits % CHAR_BIT == 0 && inPlaneBits % CHAR_BIT == 0) {
        return 1;
    }
    const auto outAlign = CHAR_BIT / std::gcd(outPlaneBits, static_cast<int64_t>(CHAR_BIT));
    const auto inAlign = CHAR_BIT / std::gcd(inPlaneBits, static_cast<int64_t>(CHAR_BIT));
    return std::lcm(outAlign, inAlign);
}

bool VPU::doesMemPermuteNeedAlignment(VPU::MemPermuteOp memPermuteOp) {
    auto info = getMemPermuteAlignmentInfo(memPermuteOp);
    if (!info) {
        return false;
    }

    for (int64_t pos = 0; pos < info->rank; ++pos) {
        if (info->getAlignmentForMemPos(pos) > 1) {
            return true;
        }
    }
    return false;
}

std::optional<SmallVector<int64_t>> VPU::getSubbyteAlignmentForMemPermute(VPU::MemPermuteOp memPermuteOp,
                                                                          ShapeRef divisors) {
    auto info = getMemPermuteAlignmentInfo(memPermuteOp);
    if (!info) {
        return std::nullopt;
    }

    SmallVector<int64_t> alignment(divisors.size(), 1);
    bool needsAlignment = false;
    for (int64_t pos = 0; pos < info->rank; ++pos) {
        const auto outDim = info->outOrder.dimAt(pos);
        if (divisors[outDim] <= 1) {
            continue;
        }

        const auto align = info->getAlignmentForMemPos(pos);
        if (align > 1) {
            alignment[outDim.ind()] = align;
            needsAlignment = true;
        }
    }

    return needsAlignment ? std::optional(alignment) : std::nullopt;
}
