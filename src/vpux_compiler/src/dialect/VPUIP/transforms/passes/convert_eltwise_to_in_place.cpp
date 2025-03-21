//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTELTWISETOINPLACE
#define GEN_PASS_DEF_CONVERTELTWISETOINPLACE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool areInOutDistributionsCompatible(VPUIP::NCEClusterTaskOp clusterTaskOp, AliasesInfo& aliasesInfo,
                                     mlir::Value& inputRootBuff, mlir::Value& originalInputRootBuff,
                                     VPUIP::DistributedBufferType inRootBuffDistributedType,
                                     VPUIP::DistributedBufferType inDistributedType,
                                     VPUIP::DistributedBufferType outDistributedType, Logger log) {
    if (inRootBuffDistributedType.getDistribution() == outDistributedType.getDistribution()) {
        return true;
    }

    if (!VPU::areDistributionAttrsCompatible(inDistributedType, outDistributedType, true).succeeded()) {
        log.trace("Incompatible input/output dist modes {0} {1}", inDistributedType, outDistributedType);
        return false;
    }

    mlir::OpBuilder builder(clusterTaskOp);
    builder.setInsertionPointAfterValue(inputRootBuff);

    // Distributed input of NCEClusterTilingOp is compatible with its output, but Distributed input root buffer type is
    // not (e.g. NCE.Permute (SOW) -> ViewOp -> NCE.Eltwise (SOH)). Insert ViewOp.
    if (VPU::areDistributionAttrsCompatible(inRootBuffDistributedType, outDistributedType, true).failed()) {
        auto supportView = builder.create<VPUIP::ViewOp>(clusterTaskOp.getLoc(), inDistributedType, inputRootBuff);
        aliasesInfo.addAlias(originalInputRootBuff, supportView.getResult());
        inputRootBuff = supportView.getResult();
    } else {
        auto distributedCastOp =
                builder.create<VPUIP::DistributedCastOp>(clusterTaskOp.getLoc(), outDistributedType, inputRootBuff);
        aliasesInfo.addAlias(originalInputRootBuff, distributedCastOp.getOutput());
        inputRootBuff = distributedCastOp.getOutput();
    }

    return true;
}

void makeInPlaceEltwise(VPUIP::NCEClusterTaskOp clusterTaskOp, AliasesInfo& aliasesInfo, Logger log) {
    auto eltwiseAllInputs = SmallVector<mlir::Value>{clusterTaskOp.getInput(), clusterTaskOp.getWeights()};

    // Get the root output buffer of the clusterTaskOp
    auto getOutputRootBuffOfNCEClusterTiling = [](mlir::Operation* innerOp) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp->getParentOp())) {
            return VPUIP::getLayerOutputs(nceClustOp)[0];
        }
        return VPUIP::getLayerOutputs(innerOp)[0];
    };

    auto outputRootBuff = getOutputRootBuffOfNCEClusterTiling(clusterTaskOp);
    mlir::Value suitableInputToOverwrite = nullptr;
    mlir::Value potentialOverwriteableInput = nullptr;

    for (auto input : eltwiseAllInputs) {
        log.trace("Checking input at `{0}`", input.getLoc());

        auto inputBuff = VPUIP::getTopBufferOfNCEClusterTiling(clusterTaskOp, input);
        auto inputRootBuff = *aliasesInfo.getRoots(inputBuff).begin();

        auto nestLog = log.nest(2);
        if (inputRootBuff == outputRootBuff) {
            nestLog.trace("Output already replaced with input");
            return;
        }

        const auto inInterface = inputBuff.getType().dyn_cast<NDTypeInterface>();
        const auto outInterface = outputRootBuff.getType().dyn_cast<NDTypeInterface>();
        if (!isCompatibleForInplaceOp(inInterface, outInterface, nestLog)) {
            continue;
        }

        // Ensure element type and tensor size compatibility
        if (inInterface.getElementType() != outInterface.getElementType() ||
            inInterface.getTotalAllocSize() > outInterface.getTotalAllocSize()) {
            mlir::OpBuilder builder(clusterTaskOp);
            builder.setInsertionPointAfterValue(inputRootBuff);
            auto supportView =
                    builder.create<VPUIP::ViewOp>(inputBuff.getLoc(), outputRootBuff.getType(), inputRootBuff);
            aliasesInfo.addAlias(inputRootBuff, supportView.getResult());
            inputRootBuff = supportView.getResult();
        }

        const auto inRootDataBuffType = VPUIP::extractDataType(inputRootBuff).cast<NDTypeInterface>();
        const auto outBuffType = VPUIP::extractDataType(outputRootBuff).cast<NDTypeInterface>();
        auto originalInputRootBuff = inputRootBuff;
        if (inRootDataBuffType.getShape() != outBuffType.getShape() &&
            inRootDataBuffType.getDimsOrder() == outBuffType.getDimsOrder() &&
            inRootDataBuffType.getNumElements() == outBuffType.getNumElements()) {
            mlir::OpBuilder shapeCastBuilder(clusterTaskOp);
            shapeCastBuilder.setInsertionPointAfterValue(inputRootBuff);
            auto shapeCastOp = shapeCastBuilder.create<VPUIP::ShapeCastOp>(clusterTaskOp.getLoc(), inputRootBuff,
                                                                           outBuffType.getShape());
            aliasesInfo.addAlias(inputRootBuff, shapeCastOp.getResult());
            inputRootBuff = shapeCastOp.getResult();
        }

        // Ensure distribution compatibility
        const auto inRootBuffDistributedType =
                VPUIP::extractDataType(inputRootBuff).dyn_cast<VPUIP::DistributedBufferType>();
        const auto inDistributedType = VPUIP::extractDataType(inputBuff).dyn_cast<VPUIP::DistributedBufferType>();
        const auto outDistributedType = VPUIP::extractDataType(outputRootBuff).dyn_cast<VPUIP::DistributedBufferType>();
        if (inDistributedType != nullptr && outDistributedType != nullptr && inRootBuffDistributedType != nullptr) {
            if (!areInOutDistributionsCompatible(clusterTaskOp, aliasesInfo, inputRootBuff, originalInputRootBuff,
                                                 inRootBuffDistributedType, inDistributedType, outDistributedType,
                                                 nestLog)) {
                continue;
            }
        }

        // Try to pick the input that does not have any other consumer apart from the in place Eltwise.
        // If the other input does not meet the criteria for overwrite, save this input to use as back-up,
        // as it could be legalized by inserting a spill right before eltwise, at a later stage.
        if (!VPUIP::isEltwiseTheOnlyConsumer(clusterTaskOp, inputBuff, /*checkThroughCopyOps=*/true, log)) {
            potentialOverwriteableInput = inputRootBuff;
            nestLog.trace("This input is used by another operation, try next input.");
            continue;
        }

        suitableInputToOverwrite = inputRootBuff;
        break;
    }

    // If pass was unable to convert eltwise then compilation must fail since
    // operation was not tiled to fit into CMX.
    VPUX_THROW_WHEN((potentialOverwriteableInput == nullptr) && (suitableInputToOverwrite == nullptr),
                    "Failed to convert Eltwise to in-place Eltwise {0}", clusterTaskOp.getLoc());

    auto replacementInputRootBuff =
            (suitableInputToOverwrite != nullptr) ? suitableInputToOverwrite : potentialOverwriteableInput;
    outputRootBuff.replaceAllUsesWith(replacementInputRootBuff);

    const auto getEltwiseResult = [&]() {
        if (auto nceClustOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(clusterTaskOp->getParentOp())) {
            return nceClustOp->getResult(0);
        } else {
            return clusterTaskOp->getResult(0);
        }
    };

    const auto eltwiseResult = getEltwiseResult();
    aliasesInfo.remove(eltwiseResult);
    aliasesInfo.addAlias(replacementInputRootBuff, eltwiseResult);

    log.trace("Eltwise input Replaced with output {0}", replacementInputRootBuff);
}

//
// ConvertEltwiseToInPlacePass
//

class ConvertEltwiseToInPlacePass final : public VPUIP::impl::ConvertEltwiseToInPlaceBase<ConvertEltwiseToInPlacePass> {
public:
    explicit ConvertEltwiseToInPlacePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertEltwiseToInPlacePass::safeRunOnFunc() {
    auto& aliasesInfo = getAnalysis<AliasesInfo>();
    auto func = getOperation();

    const auto isEltwiseInplaceCandidate = [](VPUIP::NCEClusterTaskOp op) {
        if (op.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
            return false;
        }
        return op.getIsInplace().value_or(false);
    };

    func->walk([&](VPUIP::NCEClusterTaskOp op) {
        if (isEltwiseInplaceCandidate(op)) {
            _log.trace("Found inplace eltwise at {0}", op.getLoc());
            makeInPlaceEltwise(op, aliasesInfo, _log);
        }
    });
}

}  // namespace

//
// createConvertEltwiseToInPlacePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertEltwiseToInPlacePass(Logger log) {
    return std::make_unique<ConvertEltwiseToInPlacePass>(log);
}
