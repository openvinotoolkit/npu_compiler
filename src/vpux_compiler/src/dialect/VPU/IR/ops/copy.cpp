//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

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
