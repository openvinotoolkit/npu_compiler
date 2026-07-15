//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::IE {
#define GEN_PASS_DECL_RUNF16TOF32CONVERTONDPU
#define GEN_PASS_DEF_RUNF16TOF32CONVERTONDPU
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

//
// RunF16ToF32ConvertOnDPUPass
//

class RunF16ToF32ConvertOnDPUPass final : public IE::impl::RunF16ToF32ConvertOnDPUBase<RunF16ToF32ConvertOnDPUPass> {
public:
    explicit RunF16ToF32ConvertOnDPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    void replaceWithIdentityPool(IE::ConvertOp convert);
    void fuseWithParentDPUOp(IE::ConvertOp convert, mlir::Operation* parentOp);
};

void RunF16ToF32ConvertOnDPUPass::fuseWithParentDPUOp(IE::ConvertOp convert, mlir::Operation* parentOp) {
    _log.nest().debug("F16 -> F32 Convert will be fused with parent DPU op at loc {0}", parentOp->getLoc());

    auto parentOpOutputType = mlir::cast<NDTypeInterface>(parentOp->getResult(0).getType());
    auto outElemType = mlir::cast<NDTypeInterface>(convert.getOutput().getType()).getElementType();

    parentOp->getResult(0).setType(parentOpOutputType.changeElemType(outElemType));

    convert->replaceAllUsesWith(parentOp->getResults());
    convert->erase();
}

bool isInShapePerfForNCEAvgPool(Shape inShape) {
    const int64_t maxSizeLimit = 8192;
    const int64_t minSizeLimit = 512;
    const int64_t heightAndWidth = inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W];
    return (inShape.size() == 4) && (inShape[Dims4D::Act::C] == 1) && (inShape[Dims4D::Act::N] == 1) &&
           (heightAndWidth <= maxSizeLimit) && (heightAndWidth > minSizeLimit);
}

void RunF16ToF32ConvertOnDPUPass::replaceWithIdentityPool(IE::ConvertOp convert) {
    mlir::OpBuilder builder(convert);
    if (IE::hasDynamicShape(convert)) {
        _log.nest().trace("Case with dynamic shapes not supported.");
        return;
    }

    auto inputType = mlir::cast<NDTypeInterface>(convert.getInput().getType());
    auto inputShape = inputType.getShape().raw();
    if (inputShape.size() != 4) {
        _log.nest().trace("Case with rank != 4 not supported.");
        return;
    }

    if (isInShapePerfForNCEAvgPool(inputShape)) {
        _log.nest().trace("F16 -> F32 Convert will be replaced by identity AvgPool.");
        auto replacementAvgPool = IE::createIdentityAvgPool(convert.getInput(), convert.getOutput().getType(), builder,
                                                            takeOpLoc(convert, "as_avgpool"));
        convert->replaceAllUsesWith(replacementAvgPool->getResults());
    }
}

void RunF16ToF32ConvertOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto parentCheck = strategyFactory->getFuseConvertToDPUChecker();

    auto func = getOperation();
    auto nestedLog = _log.nest();
    SmallVector<IE::ConvertOp> f16Tof32Converts = {};
    for (auto convertOp : func.getOps<IE::ConvertOp>()) {
        _log.debug("Got '{0}' at '{1}'", convertOp->getName(), convertOp->getLoc());

        auto inputElemType = mlir::cast<NDTypeInterface>(convertOp.getInput().getType()).getElementType();
        auto outputElemType = mlir::cast<NDTypeInterface>(convertOp.getOutput().getType()).getElementType();

        if (!mlir::isa<mlir::Float16Type>(inputElemType) || !mlir::isa<mlir::Float32Type>(outputElemType)) {
            nestedLog.trace("Not a FP16 -> FP32 Convert.");
            continue;
        }

        auto parentOp = convertOp.getInput().getDefiningOp();
        if (parentOp == nullptr || !parentOp->hasOneUse()) {
            nestedLog.trace("No parent op or parent has more than one use.");
            if (parentCheck->isConvertOnDPUBeneficial()) {
                replaceWithIdentityPool(convertOp);
            }
            continue;
        }

        auto convertOutputType = convertOp.getOutput().getType().getElementType();
        if (mlir::failed(VPU::NCEInvariant::isSupported(parentOp))) {
            auto interpolateOp = mlir::dyn_cast<IE::InterpolateOp>(parentOp);
            if (interpolateOp == nullptr ||
                !IE::isFusingConvertIntoBilinearInterpolateOnDpuBeneficial(interpolateOp, convertOutputType)) {
                nestedLog.trace("Parent op of type {0} at loc {1} is not a supported DPU op.", parentOp->getName(),
                                parentOp->getLoc());
                if (parentCheck->isConvertOnDPUBeneficial()) {
                    replaceWithIdentityPool(convertOp);
                }
                continue;
            }
        }

        if (!parentCheck->isFusionToParentDPUOpSupported(parentOp, nestedLog)) {
            continue;
        }

        // Data movement ops (Roll, Pad) propagate input element type to output unchanged.
        //  Fusing a FP16->FP32 Convert
        // into such ops when batch != 1 would set their result to FP32 while inferReturnTypes
        // derives FP16 from the FP16 data input, causing a type mismatch during verification.
        // When batch == 1 the op runs on DPU via the SE/NCE path and fusion is beneficial.
        if (mlir::isa<IE::RollOp, IE::PadOp>(parentOp)) {
            const auto parentInputShape = getShape(parentOp->getOperand(0));
            if (parentInputShape[Dims4D::Act::N] != 1) {
                nestedLog.trace("Skipping fusion for data movement op '{0}' with batch={1} - op will not run on DPU.",
                                parentOp->getName(), parentInputShape[Dims4D::Act::N]);
                continue;
            }
        }

        const auto inputShape = getShape(parentOp->getOperand(0));
        // This will cause an error, because of EnsureNCEOpsSizeRequirementsPass.
        // This can be unrolled into Convs -> FP32 -> Eltwise Add.
        // FP32 input is not supported.
        if (inputShape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            continue;
        }

        f16Tof32Converts.emplace_back(convertOp);
    }

    for (auto eligibleConvert : llvm::make_early_inc_range(f16Tof32Converts)) {
        auto parentOp = eligibleConvert.getInput().getDefiningOp();
        fuseWithParentDPUOp(eligibleConvert, parentOp);
    }
}

//
// RunF16ToF32ConvertOnDPU
//

std::unique_ptr<mlir::Pass> vpux::IE::createRunF16ToF32ConvertOnDPUPass(Logger log) {
    return std::make_unique<RunF16ToF32ConvertOnDPUPass>(log);
}
