//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

namespace vpux::IE {

struct ConsumerChain {
    using OpAndDimPair = std::pair<mlir::Operation*, Dim>;

    // Each Slice is paired with the (possibly empty) chain of view-like ops that sit between the conv-like
    // producer and that particular Slice. Different Slice users are allowed to be reached through different
    // view-like chains (or through no view-like ops at all); reconstruction must replay each Slice's own chain.
    struct SliceWithViewChain {
        SliceOp slice;
        SmallVector<OpAndDimPair> viewLikeChain;
    };
    SmallVector<SliceWithViewChain> slices;
};

//
// FuseConvLikeOpsWithSlice
//

//  Input(1x16x8x8)    Filter (64x16x1x1)         Input(1x16x8x8)      Filter(32x16x1x1)
//          \            /                                 \               /
//         Convolution(1x64x8x8)            To            Convolution(1x32x8x8)
//                  |
//            Slice(1x32x8x8)

// Or

//  Input(1x16x8x8)          Filter (64x16x1x1)
//          \                   /
//           Convolution(1x64x8x8)
//               |         |
//    Slice(1x32x8x8)     Slice(1x32x8x8)

//   To

//    Filter(32x16x1x1)            Input(1x16x8x8)         Filter(32x16x1x1)
//                 \                 /        \               /
//               Convolution(1x32x8x8)       Convolution(1x32x8x8)

template <class ConvLike>
class FuseWithSliceBase : public mlir::OpRewritePattern<ConvLike> {
public:
    FuseWithSliceBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConvLike>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConvLike origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual bool doesSliceMeetRequirement(IE::SliceOp sliceOp, mlir::Operation* convLikeOp, Dim convChannelDim,
                                          const Dim& dimToSlice) const;
    virtual bool checkOpEligibility(mlir::Operation* /*origOp*/) const {
        return true;
    }

    virtual Dim getOutputChannelDim(ConvLike op) const = 0;
    virtual Dim getWeightsChannelDim(ConvLike op) const = 0;
    virtual mlir::Operation* sliceOpAtOutputChannel(ConvLike op, int64_t channelOffset, int64_t channelSize,
                                                    int64_t sliceIdx, mlir::PatternRewriter& rewriter) const = 0;

    virtual bool isSupportedViewOp(mlir::Operation* /*op*/) const {
        return false;
    }

    virtual bool isFusingBeneficial(mlir::Operation* /*origOp*/, const ConsumerChain& /*consumers*/) const {
        return true;
    }

    mlir::FailureOr<ConsumerChain> hasRequiredSliceUsers(mlir::Operation* origOp, const Dim& opChannelDim) const;
    Logger _log;
};

class FuseConvWithSlice final : public FuseWithSliceBase<IE::ConvolutionOp> {
public:
    FuseConvWithSlice(mlir::MLIRContext* ctx, Logger log): FuseWithSliceBase<IE::ConvolutionOp>(ctx, log) {
    }

private:
    bool checkOpEligibility(mlir::Operation* origOp) const override;
    Dim getOutputChannelDim(IE::ConvolutionOp /*op*/) const override {
        return Dims4D::Act::C;
    };
    Dim getWeightsChannelDim(IE::ConvolutionOp /*op*/) const override {
        return Dims4D::Filter::OC;
    };

    mlir::Operation* sliceOpAtOutputChannel(IE::ConvolutionOp op, int64_t channelOffset, int64_t channelSize,
                                            int64_t sliceIdx, mlir::PatternRewriter& rewriter) const override;
};

class FuseMatMulWithSlice final : public FuseWithSliceBase<IE::MatMulOp> {
public:
    FuseMatMulWithSlice(mlir::MLIRContext* ctx, Logger log): FuseWithSliceBase<IE::MatMulOp>(ctx, log) {
    }

private:
    bool isSupportedViewOp(mlir::Operation* op) const override {
        return mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp, IE::PermuteCastOp>(op);
    }

    bool isFusingBeneficial(mlir::Operation* origOp, const ConsumerChain& consumers) const override;

    bool checkOpEligibility(mlir::Operation* origOp) const override;
    Dim getOutputChannelDim(IE::MatMulOp op) const override {
        const auto outputRank = mlir::cast<NDTypeInterface>(op.getResult().getType()).getRank();
        return Dim(outputRank - 1);  // last dim is the output channel for MatMul
    };
    Dim getWeightsChannelDim(IE::MatMulOp op) const override {
        const auto filterRank = mlir::cast<NDTypeInterface>(op.getInput2().getType()).getRank();
        if (!op.getTransposeB()) {
            return Dim(filterRank - 1);  // last dim is the output channel when input2 is [K, N]
        }
        return Dim(filterRank - 2);  // second-to-last dim is the output channel when input2 is [N, K]
    };
    mlir::Operation* sliceOpAtOutputChannel(IE::MatMulOp op, int64_t channelOffset, int64_t channelSize,
                                            int64_t sliceIdx, mlir::PatternRewriter& rewriter) const override;
};

class FuseFullyConnectedWithSlice final : public FuseWithSliceBase<IE::FullyConnectedOp> {
public:
    FuseFullyConnectedWithSlice(mlir::MLIRContext* ctx, Logger log): FuseWithSliceBase<IE::FullyConnectedOp>(ctx, log) {
    }

private:
    bool isSupportedViewOp(mlir::Operation* op) const override {
        return mlir::isa_and_present<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp, IE::PermuteCastOp>(op);
    }

    bool isFusingBeneficial(mlir::Operation* origOp, const ConsumerChain& consumers) const override;

    bool checkOpEligibility(mlir::Operation* origOp) const override;
    Dim getOutputChannelDim(IE::FullyConnectedOp op) const override {
        const auto outputRank = mlir::cast<NDTypeInterface>(op.getResult().getType()).getRank();
        return Dim(outputRank - 1);  // last dim is the output channel for FullyConnected
    };
    Dim getWeightsChannelDim(IE::FullyConnectedOp op) const override {
        const auto filterRank = mlir::cast<NDTypeInterface>(op.getWeights().getType()).getRank();
        return Dim(filterRank - 2);
    };
    mlir::Operation* sliceOpAtOutputChannel(IE::FullyConnectedOp op, int64_t channelOffset, int64_t channelSize,
                                            int64_t sliceIdx, mlir::PatternRewriter& rewriter) const override;
};
}  // namespace vpux::IE
