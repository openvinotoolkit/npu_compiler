//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

// for a given input and a output requirement(outOrdr and outShape), the function is trying to find a permutation that
// can use permuteCastOp to convert input to output requirement.
std::optional<IE::PermuteCastOp> IE::tryToFindPermuteCastOp(mlir::Location loc, mlir::Value input,
                                                            const DimsOrder& outOrder, ShapeRef outShape,
                                                            mlir::PatternRewriter& rewriter) {
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

IE::LayerWithPermuteInterface IE::getFusableLayerWithPermuteInterface(mlir::Operation* op) {
    auto inputOp = op->getOperand(0).getDefiningOp();
    if (auto quantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(inputOp)) {
        auto outElemType = quantizeCastOp.getOutput().getType().getElementType();
        if (quantizeCastOp->hasOneUse() && mlir::isa<mlir::quant::UniformQuantizedType>(outElemType)) {
            inputOp = quantizeCastOp.getInput().getDefiningOp();
        }
    }
    return mlir::dyn_cast_or_null<IE::LayerWithPermuteInterface>(inputOp);
}

bool IE::isTrivialReorder(IE::ReorderOp origOp) {
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    const auto inShape = getShape(origOp.getInput());

    return isTrivialReorder(inOrder, outOrder, inShape);
}

bool IE::isTrivialTranspose(IE::TransposeOp origOp) {
    if (!origOp.getOrderValue().has_value()) {
        return false;
    }

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    const auto perm = inOrder.toAffineMap(origOp.getContext()).compose(origOp.getOrderValue().value());
    return isTrivialPermute(inMemShape, perm);
}

bool IE::isTrivialMemPermute(IE::MemPermuteOp origOp) {
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    return isTrivialPermute(inMemShape, origOp.getMemPerm());
}

std::optional<mlir::AffineMap> IE::createTrivialAffineMap(int64_t rank, mlir::MLIRContext* ctx) {
    if (rank == 1) {
        return DimsOrder::C.toAffineMap(ctx);
    } else if (rank == 2) {
        return DimsOrder::NC.toAffineMap(ctx);
    } else if (rank == 3) {
        return DimsOrder::CHW.toAffineMap(ctx);
    } else if (rank == 4) {
        return DimsOrder::NCHW.toAffineMap(ctx);
    }

    return std::nullopt;
}
