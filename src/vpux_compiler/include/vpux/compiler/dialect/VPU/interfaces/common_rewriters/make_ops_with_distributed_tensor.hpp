//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <memory>

namespace vpux {
namespace VPU {

class ClusteredOpRewriter final : public mlir::OpInterfaceRewritePattern<VPU::ClusteredOpInterface> {
public:
    using LegalOpSelectionFn = std::function<bool(VPU::ClusteredOpInterface)>;
    ClusteredOpRewriter(
            mlir::MLIRContext* ctx, const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            LegalOpSelectionFn isOpLegalToRewrite, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::ClusteredOpInterface>(ctx),
              _log(log),
              _typeLookup(typeLookup),
              _inputTypeLookup(inputTypeLookup),
              _isOpLegalToRewrite(std::move(isOpLegalToRewrite)) {
        setDebugName("ClusteredOpRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::ClusteredOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& _typeLookup;
    const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& _inputTypeLookup;
    LegalOpSelectionFn _isOpLegalToRewrite;
};

//
// NCEEltwiseRewriterRewrite
//

class NCEEltwiseRewriter final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseRewriter(
            mlir::MLIRContext* ctx, const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx),
              _log(log),
              _typeLookup(typeLookup),
              _inputTypeLookup(inputTypeLookup) {
        setDebugName("NCEEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& _typeLookup;
    const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& _inputTypeLookup;
};

}  // namespace VPU
}  // namespace vpux
