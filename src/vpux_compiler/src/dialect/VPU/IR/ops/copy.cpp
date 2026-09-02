//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::CopyOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::CopyOpAdaptor copyOp(operands, attrs, prop);
    if (mlir::failed(copyOp.verify(loc))) {
        return mlir::failure();
    }

    const auto ndInType = mlir::dyn_cast<vpux::NDTypeInterface>(copyOp.getInput().getType());
    if (ndInType == nullptr) {
        return errorAt(loc, "CopyOp operand must have vpux::NDTypeInterface type");
    }

    IndexedSymbolAttr outMemSpace = nullptr;
    if (copyOp.getOutMemSpace().has_value()) {
        outMemSpace = copyOp.getOutMemSpace().value();
    }
    const auto outType = ndInType.changeMemSpace(outMemSpace);

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

bool vpux::VPU::areTypesCompatibleForCopy(mlir::TypeRange lhs, mlir::TypeRange rhs) {
    constexpr IE::TypeComparisonMode elemComparisonModes = IE::TypeComparisonMode::ALLOW_DISTRIBUTED_OUTPUT;
    constexpr bool checkDimsOrder = true;
    constexpr bool checkMemSpace = true;
    // Note: the below is mostly a copy-paste of vpux::areTypesCompatible()
    // logic with slight variation in the way shape comparison is done.

    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (const auto p : zip(lhs, rhs)) {
        auto lhsOrigType = std::get<0>(p);
        auto rhsOrigType = std::get<1>(p);

        if (lhsOrigType.getTypeID() != rhsOrigType.getTypeID()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, (IE::TypeComparisonMode::ALLOW_GROUPED_OUTPUT |
                                                             IE::TypeComparisonMode::ALLOW_DISTRIBUTED_OUTPUT))) {
                const auto oneIsGrouped = (mlir::isa<vpux::GroupedTypeInterface>(lhsOrigType) &&
                                           !mlir::isa<vpux::GroupedTypeInterface>(rhsOrigType)) ||
                                          (!mlir::isa<vpux::GroupedTypeInterface>(lhsOrigType) &&
                                           mlir::isa<vpux::GroupedTypeInterface>(rhsOrigType));
                const auto oneIsDistributed = (mlir::isa<vpux::VPU::DistributedTensorType>(lhsOrigType) &&
                                               !mlir::isa<vpux::VPU::DistributedTensorType>(rhsOrigType)) ||
                                              (!mlir::isa<vpux::VPU::DistributedTensorType>(lhsOrigType) &&
                                               mlir::isa<vpux::VPU::DistributedTensorType>(rhsOrigType));

                if (!oneIsGrouped && !oneIsDistributed) {
                    return false;
                }
            } else {
                return false;
            }
        }

        auto lhsType = mlir::dyn_cast<NDTypeInterface>(lhsOrigType);
        auto rhsType = mlir::dyn_cast<NDTypeInterface>(rhsOrigType);

        if (lhsType == nullptr || rhsType == nullptr) {
            return false;
        }

        if (lhsType.getShape() != rhsType.getShape()) {
            // if static shapes do not match, check bounded shapes instead
            const auto lhsBoundedShape = getBoundedShape(lhsType);
            const auto rhsBoundedShape = getBoundedShape(rhsType);
            if (lhsBoundedShape != rhsBoundedShape) {
                return false;
            }
        }

        if (lhsType.getElementType() != rhsType.getElementType()) {
            if (IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::STRICT_EQUAL)) {
                return false;
            }

            const auto lhsQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(lhsType.getElementType());
            const auto rhsQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(rhsType.getElementType());

            if (lhsQuantizedType && rhsQuantizedType) {
                if ((lhsQuantizedType.getExpressedType() != rhsQuantizedType.getExpressedType()) ||
                    (lhsQuantizedType.getStorageType() != rhsQuantizedType.getStorageType())) {
                    if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_DIFFERENT_QUANT)) {
                        return false;
                    }
                }
            } else if (!IE::bitEnumContainsAny(elemComparisonModes, IE::TypeComparisonMode::ALLOW_MIXED_PRECISION)) {
                return false;
            }
        }

        if (checkDimsOrder) {
            const auto order1 = lhsType.getDimsOrder();
            const auto order2 = rhsType.getDimsOrder();

            if (order1 != order2) {
                return false;
            }
        }

        if (checkMemSpace) {
            const auto memSpace1 = lhsType.getMemSpace();
            const auto memSpace2 = rhsType.getMemSpace();

            if (memSpace1 != memSpace2) {
                // Allow different memory spaces only if both types are in DDR, since a null value also represents DDR
                if (!(lhsType.getMemoryKind() == VPU::MemoryKind::DDR &&
                      rhsType.getMemoryKind() == VPU::MemoryKind::DDR)) {
                    return false;
                }
            }
        }

        const bool lhsIsBounded = mlir::isa<Core::BoundedTensorType>(lhsOrigType);
        const bool rhsIsBounded = mlir::isa<Core::BoundedTensorType>(rhsOrigType);
        if (lhsIsBounded && rhsIsBounded) {
            auto lhsBoundedType = mlir::cast<Core::BoundedTensorType>(lhsOrigType);
            auto rhsBoundedType = mlir::cast<Core::BoundedTensorType>(rhsOrigType);
            if (lhsBoundedType.getBounds() != rhsBoundedType.getBounds()) {
                return false;
            }
        }
    }

    return true;
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::CopyOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseCopies final : public mlir::OpRewritePattern<VPU::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseCopies::matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerCopyOp = origOp.getInput().getDefiningOp<VPU::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }
    // The I/O types of this CopyOp chain should not contain Distributed types
    auto isDistributedType = [](mlir::Value val) {
        auto distributedIf = mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(val.getType());
        return distributedIf != nullptr && distributedIf.containsDistributedTypes();
    };
    if (isDistributedType(producerCopyOp.getInput()) || isDistributedType(producerCopyOp.getOutput()) ||
        isDistributedType(origOp.getInput()) || isDistributedType(origOp.getOutput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, producerCopyOp.getInput(), origOp.getOutMemSpaceAttr());
    return mlir::success();
}

/// @brief Finds and eliminates sequences of surplus Copies that effectively leave the Type unchanged
/// @details The pattern of surplus Copy chains can appear when two consequent operations are ClusterTiled the same way:
/// for example, when a (Conv)->(Conv) chain is all split-over-height
/// @example The expected pattern is:
/// %0 = VPU.Copy(!DistributedTensor0) -> !Tensor0
/// %1 = VPU.Copy(%0) -> !DistributedTensor0
///
/// Action expected: replace the two CopyOp sequence with a DistributedCastOp
class EliminateCopyPairs final : public mlir::OpRewritePattern<VPU::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult EliminateCopyPairs::matchAndRewrite(VPU::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    // Its input should be produced another Copy operation
    auto producerCopyOp = origOp.getInput().getDefiningOp<VPU::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }

    // The I/O types of this CopyOp-chain should be similar
    auto producerInput = producerCopyOp.getInput();
    auto output = origOp.getOutput();

    if (producerInput.getType() != output.getType()) {
        const auto inDistributedTypeInterface =
                mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(producerInput.getType());
        const auto outDistributedTypeInterface = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(output.getType());

        if (inDistributedTypeInterface == nullptr || outDistributedTypeInterface == nullptr ||
            !inDistributedTypeInterface.containsDistributedTypes() ||
            !outDistributedTypeInterface.containsDistributedTypes()) {
            return mlir::failure();
        }

        if (VPU::isDistributedCastCompatible(mlir::cast<vpux::VPU::DistributedTensorType>(
                                                     inDistributedTypeInterface.getDistributedTypes().front()),
                                             mlir::cast<vpux::VPU::DistributedTensorType>(
                                                     outDistributedTypeInterface.getDistributedTypes().front()))
                    .failed()) {
            return mlir::failure();
        }

        const auto distributedCastOp =
                rewriter.create<VPU::DistributedCastOp>(origOp.getLoc(), output.getType(), producerInput);

        rewriter.replaceOp(origOp, distributedCastOp->getResult(0));
        return mlir::success();
    }
    // If Input of producerCopy == Output of consumer Copy, then both Copy's ops can be removed
    rewriter.replaceAllUsesWith(origOp.getOutput(), producerCopyOp.getInput());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::CopyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseCopies>(ctx);
    results.add<EliminateCopyPairs>(ctx);
}
