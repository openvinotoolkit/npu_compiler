//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
//
// ConvToNCE
//

class ConvToNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("ConvToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// MatMulToNCE
//

class MatMulToNCE final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::MatMulOp>(ctx), _arch(arch), _log(log) {
        setDebugName("MatMulToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// DepthConvToNCE
//

class DepthConvToNCE final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    DepthConvToNCE(mlir::MLIRContext* ctx, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _arch(arch), _log(log) {
        setDebugName("DepthConvToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    config::ArchKind _arch;
    Logger _log;
};

//
// MaxPoolToNCE
//

class MaxPoolToNCE final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolToNCE(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

//
// AveragePoolToNCE
//

class AveragePoolToNCE final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolToNCE(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolToNCE");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

//
// PermuteQuantizeToNCEPermute
//

class PermuteQuantizeToNCEPermute final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PermuteQuantizeToNCEPermute(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        setDebugName("PermuteQuantizeToNCEPermute");
    }

    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const override;

private:
    Logger _log;
};

//
// EltwiseToNCE
//

template <class ConcreteOp>
class EltwiseToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::EltwiseType opType, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::EltwiseType _opType;
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    auto nceOp = rewriter.create<VPU::NCEEltwiseOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput1(), origOp.getInput2(),
            VPU::EltwiseTypeAttr::get(this->getContext(), _opType), ppeAttr,
            /*multi_cluster_strategyAttr=*/nullptr,
            /*is_inplace=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}
}  // namespace vpux
