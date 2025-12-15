//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/core/dialect.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_MAKEDISTRIBUTEDCOPIES
#define GEN_PASS_DEF_MAKEDISTRIBUTEDCOPIES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

bool isDistributedType(mlir::Value val) {
    auto distributedIf = mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(val.getType());
    return distributedIf != nullptr && distributedIf.containsDistributedTypes();
}

//
// OptimizeShapeCastDistributedCopies
//
// Identify any "ShapeCast" operation surrounded by "UnrolledType" operations. Insert the necessary copies to ensure
// they cancel each other out, thereby eliminating redundant copies after canonicalization.
// Additionally, overlapped data can be managed using halo regions or ITI buffers, leveraging subsequent VPU passes.
/*
    Before:                       After:                        After (canonicalized):
                                         ClusterOp
                                             |
           ClusterOp              DistributedCopy(CMX2DDR)
               |                             |
    DistributedCopy(CMX2DDR)           Copy(DDR2CMX)            ClusterOp
               |                             |                      |
           ShapeCast          ->         ShapeCast          ->  ShapeCast
               |                             |                      |
         Copy(DDR2CMX)            DistributedCopy(CMX2DDR)      ClusterOp
               |                             |
           ClusterOp                   Copy(DDR2CMX)
                                             |
                                         ClusterOp
*/

class OptimizeShapeCastDistributedCopies final : public mlir::OpRewritePattern<VPU::ShapeCastOp> {
public:
    OptimizeShapeCastDistributedCopies(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ShapeCastOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult OptimizeShapeCastDistributedCopies::matchAndRewrite(VPU::ShapeCastOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto prevUTOp = origOp.getSource().getDefiningOp<VPU::UnrolledTypeOp>();
    auto nextUTOp = mlir::dyn_cast_or_null<VPU::UnrolledTypeOp>(*origOp.getResult().getUsers().begin());

    auto prevUTOpInputDistType = prevUTOp.getInput().getType();
    auto nextUTOpOutputDistType = nextUTOp.getOutput().getType();

    auto prevUTOpInputDistTypeInterface = mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(prevUTOpInputDistType);
    auto nextUTOpOutputDistTypeInterface =
            mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(nextUTOpOutputDistType);

    if (prevUTOpInputDistTypeInterface == nullptr || nextUTOpOutputDistTypeInterface == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Failed to retrieve distributed type interfaces");
    }

    auto prevUTOpInputDistTensorType =
            mlir::cast<VPU::DistributedTensorType>(*prevUTOpInputDistTypeInterface.getDistributedTypes().begin());

    // Get updated distribution based on the distribution after the shape cast, using the shape before the shape cast
    auto updatedDistAttr = VPUIP::getDistributedAttrAfterShapeCast<VPU::DistributedTensorType>(
            nextUTOpOutputDistTypeInterface, prevUTOpInputDistTensorType.getShape(), config::getArch(origOp));

    _log.trace("[{0}] Updating output type of: {1}\n\tOld Distribution: {2}\n\tNew Distribution: {3}", getDebugName(),
               prevUTOp.getInput(), prevUTOpInputDistTensorType.getDistribution(), updatedDistAttr);

    auto newDistType = nextUTOpOutputDistTypeInterface.changeShapeForExplicitDistribution(
            prevUTOpInputDistTensorType.getShape(), updatedDistAttr);

    // Update "prevUTOp" input with the new distribution type
    prevUTOp.getInput().setType(newDistType);

    rewriter.setInsertionPoint(origOp);
    auto distInputCopyOp = rewriter.create<VPU::UnrolledTypeOp>(takeOpLoc(origOp, "shapecast_copy_in"), newDistType,
                                                                origOp.getSource());

    // Clone the original "ShapeCastOp" and update its input and output types
    auto* newOp = rewriter.clone(*origOp);
    newOp->setOperand(0, distInputCopyOp.getResult());
    newOp->getResult(0).setType(nextUTOpOutputDistType);

    auto distOutputCopyOp = rewriter.create<VPU::UnrolledTypeOp>(takeOpLoc(newOp, "shapecast_copy_out"),
                                                                 nextUTOp.getInput().getType(), newOp->getResult(0));

    rewriter.replaceOp(origOp, distOutputCopyOp);

    return mlir::success();
}

//
// UnrolledTypeToCopyConversion
//

class UnrolledTypeToCopyConversion final : public mlir::OpRewritePattern<VPU::UnrolledTypeOp> {
public:
    UnrolledTypeToCopyConversion(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::UnrolledTypeOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::UnrolledTypeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrolledTypeToCopyConversion::matchAndRewrite(VPU::UnrolledTypeOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const bool isDistributedInput = isDistributedType(origOp.getInput());
    const bool isDistributedOutput = isDistributedType(origOp.getOutput());

    if (!isDistributedInput && !isDistributedOutput) {
        rewriter.replaceOp(origOp, origOp.getInput());
        return mlir::success();
    }

    IndexedSymbolAttr memSpace = nullptr;
    if (!isDistributedInput && isDistributedOutput) {
        memSpace = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN));
    }

    rewriter.replaceOpWithNewOp<VPU::CopyOp>(origOp, origOp.getType(), origOp.getInput(), memSpace);
    return mlir::success();
}

//
// MakeDistributedCopiesPass
//

class MakeDistributedCopiesPass final : public VPU::impl::MakeDistributedCopiesBase<MakeDistributedCopiesPass> {
public:
    MakeDistributedCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    };

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MakeDistributedCopiesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    // TODO: The scf/affine/tensor dialects are explicitly marked as legal because, in the case of the HostCompile
    // pipeline, this pass is executed on the main function, which contains host-side code as well. Ideally, this pass
    // should not operate on the main function in the HostCompile pipeline. This will be refactored in the future.
    // Track: E#168311
    bool hostCompileMode = (config::getCompilationMode(func) == config::CompilationMode::HostCompile);
    auto entryPointFunc = vpux::net::findEntryPointFunc(func, _log);
    if (hostCompileMode && (func == entryPointFunc)) {
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPU::UnrolledTypeOp>();
    target.addLegalDialect<Core::CoreDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalOp<VPU::CopyOp>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::affine::AffineDialect>();
    target.addLegalDialect<mlir::cf::ControlFlowDialect>();
    target.addLegalOp<mlir::tensor::ExtractSliceOp>();
    target.addLegalOp<mlir::tensor::InsertSliceOp>();
    target.addLegalOp<mlir::tensor::EmptyOp>();
    target.addLegalOp<mlir::cf::AssertOp>();
    target.addLegalOp<mlir::tensor::CastOp>();

    target.addDynamicallyLegalOp<VPU::ShapeCastOp>([&](VPU::ShapeCastOp shapeCast) -> bool {
        if (isDistributedType(shapeCast.getSource())) {
            return true;
        }
        auto hasSameUTOpUsers = llvm::all_of(shapeCast->getUsers(), [&](mlir::Operation* user) {
            auto curUTUser = mlir::dyn_cast_or_null<VPU::UnrolledTypeOp>(user);
            if (curUTUser == nullptr) {
                return false;
            }
            return curUTUser.getOutput().getType() == shapeCast->user_begin()->getResult(0).getType();
        });
        if (!hasSameUTOpUsers) {
            return true;
        }

        auto prevUTOp = shapeCast.getSource().getDefiningOp<VPU::UnrolledTypeOp>();
        auto nextUTOp = mlir::dyn_cast_or_null<VPU::UnrolledTypeOp>(*shapeCast.getResult().getUsers().begin());

        if (prevUTOp == nullptr || nextUTOp == nullptr) {
            return true;
        }

        auto prevUTOpInputDistTensorType =
                mlir::dyn_cast_or_null<VPU::DistributedTensorType>(prevUTOp.getInput().getType());
        auto nextUTOpOutputDistTensorType =
                mlir::dyn_cast_or_null<VPU::DistributedTensorType>(nextUTOp.getOutput().getType());

        // Current optimization only targets I/O with "OVERLAPPED" distribution mode
        if (prevUTOpInputDistTensorType.getDistribution().getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            nextUTOpOutputDistTensorType.getDistribution().getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
            return true;
        }

        const auto inputTilingAxis =
                VPUIP::getSpecificAxisFromAttr(nextUTOpOutputDistTensorType.getDistribution().getNumTiles());
        VPUX_THROW_WHEN(inputTilingAxis == -1, "cannot get input tiling axis");

        // Skip if 'clusteringDimChanges'
        if (prevUTOpInputDistTensorType.getShape().raw()[inputTilingAxis] !=
            nextUTOpOutputDistTensorType.getShape().raw()[inputTilingAxis]) {
            return true;
        }

        // If the previous operation is an inplace eltwise operation
        // then we should skip the optimization because inplace eltwise requires output has the same distribution as
        // input
        auto prevEltwiseOp = prevUTOp.getInput().getDefiningOp<VPU::NCEEltwiseOp>();
        if ((prevEltwiseOp != nullptr) && (prevEltwiseOp.getIsInplace().value_or(false))) {
            return true;
        }

        // Skip the optimization in case the prevClusteredOp with updated distribution could not fit into CMX
        auto prevUTInputClusteredOp =
                mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(prevUTOp.getInput().getDefiningOp());
        auto nextUTOutputClusteredOp =
                mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(*nextUTOp.getOutput().getUsers().begin());
        if (prevUTInputClusteredOp != nullptr && nextUTOutputClusteredOp != nullptr) {
            SmallVector<Byte> buffersSize{};
            // Calculate required CMX size for inputs
            for (auto input : prevUTInputClusteredOp->getOperands()) {
                auto inputDistTensorType = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(input.getType());
                if (inputDistTensorType != nullptr) {
                    buffersSize.push_back(VPU::getTotalAllocSizeWithDistribution(
                            input.getType(),
                            VPU::DistributionInfo::getClassFromAttr(inputDistTensorType.getDistribution())));
                }
            }

            // Calculate required CMX size for output which updated distribution
            auto nextClusteredOpInput = nextUTOutputClusteredOp->getOperands()[0];
            auto newOutputDistTensorType =
                    mlir::dyn_cast_or_null<VPU::DistributedTensorType>(nextClusteredOpInput.getType());
            if (newOutputDistTensorType != nullptr) {
                buffersSize.push_back(VPU::getTotalAllocSizeWithDistribution(
                        nextClusteredOpInput.getType(),
                        VPU::DistributionInfo::getClassFromAttr(newOutputDistTensorType.getDistribution())));
            }

            const auto totalRequiredCMXSize = vpux::VPU::calculateAlignedBuffersMemoryRequirement(
                                                      config::getArch(prevUTInputClusteredOp), buffersSize)
                                                      .count();
            const auto totalAvailableCMXSize = getTotalCMXSize(prevUTInputClusteredOp).count();
            return totalRequiredCMXSize > totalAvailableCMXSize;
        }

        return false;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<OptimizeShapeCastDistributedCopies>(&ctx, _log);
    patterns.add<UnrolledTypeToCopyConversion>(&ctx, _log);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMakeDistributedCopiesPass
//

std::unique_ptr<mlir::Pass> VPU::createMakeDistributedCopiesPass(Logger log) {
    return std::make_unique<MakeDistributedCopiesPass>(log);
}
