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
