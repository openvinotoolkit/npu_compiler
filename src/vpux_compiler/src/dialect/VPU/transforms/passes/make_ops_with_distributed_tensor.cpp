//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/make_ops_with_distributed_tensor_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MAKEOPSWITHDISTRIBUTEDTENSOR
#define GEN_PASS_DEF_MAKEOPSWITHDISTRIBUTEDTENSOR
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// MakeOpsWithDistributedTensorPass
//

class MakeOpsWithDistributedTensorPass final :
        public VPU::impl::MakeOpsWithDistributedTensorBase<MakeOpsWithDistributedTensorPass> {
public:
    MakeOpsWithDistributedTensorPass(Logger log): _enableExplicitDistributionInfoAttr(false) {
        Base::initLogger(log, Base::getArgumentName());
    };

    explicit MakeOpsWithDistributedTensorPass(bool enableExplicitDistributionInfoAttr, Logger log)
            : _enableExplicitDistributionInfoAttr(enableExplicitDistributionInfoAttr) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableExplicitDistributionInfoAttr = false;
    void safeRunOnFunc() final;
    static void insertDistributedInputTypes(
            VPU::ClusteredOpInterface clusteredOp, bool hasExplicitDistributedAttr,
            SiblingOpsAnalysis& siblingsAnalysis,
            llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup);
};

mlir::LogicalResult MakeOpsWithDistributedTensorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableExplicitDistributionInfoAttr.hasValue()) {
        _enableExplicitDistributionInfoAttr = enableExplicitDistributionInfoAttr.getValue();
        return mlir::success();
    }

    return mlir::success();
}

void MakeOpsWithDistributedTensorPass::insertDistributedInputTypes(
        VPU::ClusteredOpInterface clusteredOp, bool hasExplicitDistributedAttr, SiblingOpsAnalysis& siblingsAnalysis,
        llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup) {
    llvm::DenseMap<int, vpux::NDTypeInterface> operandLookup;
    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        auto origOp = mlir::cast<VPU::SWOpInterface>(clusteredOp.getOperation());
        const auto strategy = clusteredOp.getMultiClusterStrategy().value();
        auto* ctx = clusteredOp->getContext();
        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(origOp->getResult(0)), strategy);
        for (auto& operand : origOp->getOpOperands()) {
            const auto operandType = operand.get().getType().cast<vpux::NDTypeInterface>();
            const auto activationTensorDistributionMode =
                    getSWInputTensorDistributionMode(clusteredOp, strategy, operandType);
            const auto activationTensorNumTiles =
                    getIntArrayAttr(ctx, getSWInputTensorNumTiles(clusteredOp, numClusters, strategy, operandType));

            // Input alignment is possibly needed to keep compatibility and avoid spilling
            // Only support:
            //       NCE_DPU (non SOH/SOHOverlapped)
            //          |
            //       NCE_SW  (Clustering/SOK)
            const auto activationAlignment =
                    getActivationTensorAlignment(clusteredOp, numClusters, strategy, operandType);
            const auto activationAlignmentAttr =
                    activationAlignment.has_value() ? getIntArrayAttr(ctx, activationAlignment.value()) : nullptr;

            operandLookup.insert(std::make_pair(
                    operand.getOperandNumber(),
                    getDistributedTypeFromInput(clusteredOp, operand.get(), activationTensorDistributionMode,
                                                activationTensorNumTiles, activationAlignmentAttr, strategy,
                                                hasExplicitDistributedAttr, siblingsAnalysis)));
        }
    } else {
        for (auto& operand : clusteredOp->getOpOperands()) {
            operandLookup.insert(std::make_pair(
                    operand.getOperandNumber(),
                    clusteredOp.getDistributedTypeForOpOperand(operand, hasExplicitDistributedAttr, siblingsAnalysis)));
        }
    }
    inputTypeLookup.insert(std::make_pair(clusteredOp.getOperation(), operandLookup));
}

//
// safeRunOnModule
//

void MakeOpsWithDistributedTensorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface> typeLookup;
    llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>> inputTypeLookup;
    auto& siblingsAnalysis = getAnalysis<SiblingOpsAnalysis>();
    func->walk([&](VPU::ClusteredOpInterface clusteredOp) {
        const auto strategyAttr = clusteredOp.getMultiClusterStrategy();
        if (!strategyAttr.has_value()) {
            return;
        }
        auto strategy = strategyAttr.value();

        for (const auto& opResult : clusteredOp->getResults()) {
            typeLookup.insert(std::make_pair(
                    opResult,
                    getDistributedOutputTensorType(clusteredOp, opResult.getType().cast<vpux::NDTypeInterface>(),
                                                   siblingsAnalysis, strategy, _enableExplicitDistributionInfoAttr)));
        }
        insertDistributedInputTypes(clusteredOp, _enableExplicitDistributionInfoAttr, siblingsAnalysis,
                                    inputTypeLookup);
    });

    mlir::RewritePatternSet patterns(&ctx);
    auto strategy = vpux::VPU::createMakeOpsWithDistributedTensorStrategy(func, typeLookup, inputTypeLookup,
                                                                          _enableExplicitDistributionInfoAttr);
    // Both ACT Shaves and DPUs are grouped together in NCE clusters, in a symmetric manner.
    // Each NCE cluster has the same amount of DPUs and ACT shaves.
    // Thus shaves have the availability for distributing across clusters similar to DPUs.
    strategy->addPatterns(patterns, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op)) {
            if (op->hasAttr(multiClusterStrategy))
                return false;
        }

        return true;
    });

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMakeOpsWithDistributedTensorPass
//

std::unique_ptr<mlir::Pass> VPU::createMakeOpsWithDistributedTensorPass(bool enableExplicitDistributionInfoAttr,
                                                                        Logger log) {
    return std::make_unique<MakeOpsWithDistributedTensorPass>(enableExplicitDistributionInfoAttr, log);
}
