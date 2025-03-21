//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux {

//
// IfRewrite
//

class IfRewrite final : public mlir::OpRewritePattern<IE::IfOp> {
public:
    IfRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::IfOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::IfOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// CTCGreedyDecoderSeqLenRewrite
//

class CTCGreedyDecoderSeqLenRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// ProposalRewrite
//

class ProposalRewrite final : public mlir::OpRewritePattern<IE::ProposalOp> {
public:
    ProposalRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ProposalOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// StubRewrite
//

class StubRewrite final : public mlir::OpRewritePattern<IE::StubOp> {
public:
    StubRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::StubOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteNonMaxSuppression
//

class NonMaxSuppressionRewrite final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    NonMaxSuppressionRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::NonMaxSuppressionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// InterpolateRewrite
//

// IE.Interpolate -> VPU.Interpolate
class InterpolateRewrite : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    TopKRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// NormalizeL2Rewrite
//

class NormalizeL2Rewrite final : public mlir::OpRewritePattern<IE::NormalizeL2Op> {
public:
    NormalizeL2Rewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::NormalizeL2Op>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NormalizeL2Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// LSTMCellRewrite
//

class LSTMCellRewrite final : public mlir::OpRewritePattern<IE::LSTMCellOp> {
public:
    LSTMCellRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LSTMCellOp>(ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// LSTMSequenceRewrite
//

class LSTMSequenceRewrite final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    LSTMSequenceRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// LSTMGatesRewrite
//

class LSTMGatesRewrite final : public mlir::OpRewritePattern<IE::LSTMGatesOp> {
public:
    LSTMGatesRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LSTMGatesOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LSTMGatesOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// GroupConvolutionRewrite
//

class GroupConvolutionRewrite final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteEmbeddingBagPackedSum
//

class EmbeddingBagPackedSumRewrite final : public mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp> {
public:
    EmbeddingBagPackedSumRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagPackedSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// EmbeddingSegmentsSum
//

class EmbeddingSegmentsSumRewriter : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    EmbeddingSegmentsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// EmbeddingBagOffsetsSum
//

class EmbeddingBagOffsetsSumRewriter final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    EmbeddingBagOffsetsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

class AccumulateRewrite final : public mlir::OpRewritePattern<IE::AccumulateOp> {
public:
    AccumulateRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AccumulateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteGRUCell
//

class GRUCellRewrite final : public mlir::OpRewritePattern<IE::GRUCellOp> {
public:
    GRUCellRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GRUCellOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteExperimentalDetectronROIFeatureExtractor
//

class ExperimentalDetectronROIFeatureExtractorRewrite final :
        public mlir::OpRewritePattern<IE::ExperimentalDetectronROIFeatureExtractorOp> {
public:
    ExperimentalDetectronROIFeatureExtractorRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExperimentalDetectronROIFeatureExtractorOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExperimentalDetectronROIFeatureExtractorOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TransposedConvRewrite
//

class TransposedConvRewrite final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    TransposedConvRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DynamicReshapeRewrite
//

class DynamicReshapeRewrite final : public mlir::OpRewritePattern<IE::DynamicReshapeOp> {
public:
    DynamicReshapeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DynamicReshapeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DynamicTileRewrite
//

class DynamicTileRewrite final : public mlir::OpRewritePattern<IE::DynamicTileOp> {
public:
    DynamicTileRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::DynamicTileOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicTileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace vpux
