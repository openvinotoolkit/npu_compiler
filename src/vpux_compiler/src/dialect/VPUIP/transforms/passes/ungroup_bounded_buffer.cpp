//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>
#include <vpux/utils/core/error.hpp>
#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNGROUPBOUNDEDBUFFERS
#define GEN_PASS_DEF_UNGROUPBOUNDEDBUFFERS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// UngroupBoundedBuffers
//

class UngroupCopyOp final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    UngroupCopyOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupCopyOp::matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getInput());
    auto ungroupOutput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getOutputBuff());

    auto copyData = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), ungroupInput.getData(), ungroupOutput.getData());
    auto copyShape = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), ungroupInput.getDynamicShape(),
                                                    ungroupOutput.getDynamicShape());
    rewriter.replaceOpWithNewOp<VPUIP::GroupBoundedBufferOp>(origOp, copyData.getOutput(), copyShape.getOutput());

    return mlir::success();
}

class UngroupConcatViewOp final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    UngroupConcatViewOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupConcatViewOp::matchAndRewrite(VPUIP::ConcatViewOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> dataResults;
    SmallVector<mlir::Value> shapeResults;

    // Iterate over all inputs and create UngroupBoundedBufferOp for each.
    for (auto input : origOp.getInputs()) {
        auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), input);
        dataResults.push_back(ungroupInput.getData());
        shapeResults.push_back(ungroupInput.getDynamicShape());
    }
    auto ungroupOutput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getOutputBuff());

    // Iterate over the ungrouped inputs and create individual ConcatViewOps for data and shapes.
    auto dataConcatOp = rewriter.create<VPUIP::ConcatViewOp>(appendLoc(origOp->getLoc(), "_data"), dataResults,
                                                             ungroupOutput.getData());
    auto shapeConcatOp = rewriter.create<VPUIP::ConcatViewOp>(appendLoc(origOp->getLoc(), "_shape"), shapeResults,
                                                              ungroupOutput.getDynamicShape());

    // Group the individual results back into a single BoundedBuffer.
    rewriter.replaceOpWithNewOp<VPUIP::GroupBoundedBufferOp>(origOp, dataConcatOp.getOutput(),
                                                             shapeConcatOp.getOutput());

    return mlir::success();
}

class UngroupConvertDMAOp final : public mlir::OpRewritePattern<VPUIP::ConvertDMAOp> {
public:
    UngroupConvertDMAOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConvertDMAOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConvertDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupConvertDMAOp::matchAndRewrite(VPUIP::ConvertDMAOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto ungroupInput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getInput());
    auto ungroupOutput = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), origOp.getOutputBuff());

    auto copyData =
            rewriter.create<VPUIP::ConvertDMAOp>(origOp->getLoc(), ungroupInput.getData(), ungroupOutput.getData());
    auto copyShape = rewriter.create<VPUIP::CopyOp>(origOp->getLoc(), ungroupInput.getDynamicShape(),
                                                    ungroupOutput.getDynamicShape());
    rewriter.replaceOpWithNewOp<VPUIP::GroupBoundedBufferOp>(origOp, copyData.getOutput(), copyShape.getOutput());

    return mlir::success();
}

class UngroupSwKernelOp final : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    UngroupSwKernelOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UngroupSwKernelOp::matchAndRewrite(VPUIP::SwKernelOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto swKernelRuns = origOp.getBody().getOps<VPUIP::SwKernelRun>();
    auto swKernelRunNum = std::distance(swKernelRuns.begin(), swKernelRuns.end());
    VPUX_THROW_UNLESS(swKernelRunNum > 0, "UngroupSwKernelOp: Got wrong number of VPUIP::SwKernelRun {0}",
                      swKernelRunNum);

    auto ungroupBuffers = [&](auto buffers, auto& swKernelBuffers, auto& swKernelDynamicShapes,
                              auto& swKernelDynamicShapesMap) {
        auto numBuffers = buffers.size();
        for (size_t i = 0; i < numBuffers; i++) {
            auto buffer = buffers[i];
            auto boundedBuffer = mlir::dyn_cast<VPUIP::BoundedBufferType>(buffer.getType());
            if (boundedBuffer != nullptr) {
                auto ungroupBuffer = rewriter.create<VPUIP::UngroupBoundedBufferOp>(origOp->getLoc(), buffer);
                swKernelBuffers.push_back(ungroupBuffer.getData());

                // Avoid duplicating data. This information remains unchanged.
                if (i < numBuffers / swKernelRunNum) {
                    swKernelDynamicShapesMap.push_back(swKernelDynamicShapes.size());
                    swKernelDynamicShapes.push_back(ungroupBuffer.getDynamicShape());
                }
            } else {
                swKernelBuffers.push_back(buffer);
                if (i < numBuffers / swKernelRunNum) {
                    swKernelDynamicShapesMap.push_back(ABSENT_DIMS_FLAG);
                }
            }
        }
    };

    SmallVector<mlir::Value> swKernelOperands;
    SmallVector<mlir::Value> swKernelDynamicInputShapes;
    SmallVector<int32_t> swKernelDynamicInputShapesMap;

    ungroupBuffers(origOp.getInputs(), swKernelOperands, swKernelDynamicInputShapes, swKernelDynamicInputShapesMap);

    SmallVector<mlir::Value> swKernelOutputBuffs;
    SmallVector<mlir::Value> swKernelDynamicOutputShapes;
    SmallVector<int32_t> swKernelDynamicOutputShapesMap;

    ungroupBuffers(origOp.getOutputBuffs(), swKernelOutputBuffs, swKernelDynamicOutputShapes,
                   swKernelDynamicOutputShapesMap);

    auto tileIndex = origOp.getTileIndexAttr();
    auto swKernelOp = rewriter.create<VPUIP::SwKernelOp>(origOp->getLoc(), swKernelOperands, swKernelOutputBuffs,
                                                         swKernelDynamicInputShapes, swKernelDynamicInputShapesMap,
                                                         swKernelDynamicOutputShapes, swKernelDynamicOutputShapesMap,
                                                         origOp.getKernelFunction(), tileIndex);
    auto args = kernelArgsRange(origOp);
    auto swKernelRun = *swKernelRuns.begin();
    initSwKernel(swKernelOp, swKernelOperands, swKernelOutputBuffs, args, _log.nest(),
                 swKernelRunNum > 1 ? swKernelRun : nullptr);

    SmallVector<mlir::Value> newResults;
    const auto kernelName = getSwKernelEntryName(origOp);
    // TODO: refactoring ticket E#145673
    if (kernelName == "dynamic_reshape" && (parseIntAttr<int64_t>(args[0]) != 0)) {
        // dynamic_reshape will only propagate shape
        auto dataOperand = swKernelOperands[0];
        auto shapeResult = swKernelOp.getResult(1);
        auto groupOp = rewriter.create<VPUIP::GroupBoundedBufferOp>(swKernelOp.getLoc(), dataOperand, shapeResult);
        newResults.push_back(groupOp.getOutput());
    } else {
        size_t dynamicShapeIndex = 0;
        for (auto resultIndex : irange(origOp.getNumResults())) {
            if (mlir::isa<VPUIP::BoundedBufferType>(origOp.getResult(resultIndex).getType())) {
                size_t outDynShapesNum = swKernelOp.getDynamicOutputShapes().size();
                VPUX_THROW_UNLESS(outDynShapesNum > 0, "UngroupSwKernelOp: Got wrong number of DynamicOutputShapes {0}",
                                  outDynShapesNum);

                // Since we do not modify getDynamicOutputShapes, it will only contain the original values without
                // duplication. Therefore, we need to reuse the values.
                if (dynamicShapeIndex >= outDynShapesNum && swKernelRunNum > 1) {
                    auto groupOp = rewriter.create<VPUIP::GroupBoundedBufferOp>(
                            swKernelOp.getLoc(), swKernelOp.getResult(resultIndex),
                            swKernelOp.getDynamicOutputShapes()[dynamicShapeIndex % outDynShapesNum]);
                    newResults.push_back(groupOp.getOutput());
                } else {
                    auto groupOp = rewriter.create<VPUIP::GroupBoundedBufferOp>(
                            swKernelOp.getLoc(), swKernelOp.getResult(resultIndex),
                            swKernelOp.getDynamicOutputShapes()[dynamicShapeIndex]);
                    newResults.push_back(groupOp.getOutput());
                }
                dynamicShapeIndex++;
            } else {
                newResults.push_back(swKernelOp.getResult(resultIndex));
            }
        }
    }
    rewriter.replaceOp(origOp, newResults);

    return mlir::success();
}

class UngroupBoundedBuffers final : public VPUIP::impl::UngroupBoundedBuffersBase<UngroupBoundedBuffers> {
public:
    explicit UngroupBoundedBuffers(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UngroupBoundedBuffers::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalCopyOp = [](VPUIP::CopyOp copyOp) {
        return !VPUIP::isBoundedBufferType(copyOp.getInput()) || !VPUIP::isBoundedBufferType(copyOp.getOutput());
    };
    auto isLegalConcatViewOp = [](VPUIP::ConcatViewOp concatViewOp) {
        bool areOperandsBoundedBuffers = VPUIP::isBoundedBufferType(concatViewOp.getOutput());

        return !areOperandsBoundedBuffers;
    };
    auto isLegalConvertDMAOp = [](VPUIP::ConvertDMAOp ConvertDMAOp) {
        bool areBothOperandsBoundedBuffers = mlir::isa<VPUIP::BoundedBufferType>(ConvertDMAOp.getInput().getType()) &&
                                             mlir::isa<VPUIP::BoundedBufferType>(ConvertDMAOp.getOutput().getType());

        return !areBothOperandsBoundedBuffers;
    };
    auto isLegalSwKernelOp = [](VPUIP::SwKernelOp op) {
        const auto isBoundedBuffer = [](mlir::Value value) {
            return VPUIP::isBoundedBufferType(value);
        };
        const auto hasDynamicInputs = llvm::any_of(op.getInputs(), isBoundedBuffer);
        const auto hasDynamicOutputs = llvm::any_of(op.getOutputBuffs(), isBoundedBuffer);

        return !hasDynamicInputs && !hasDynamicOutputs;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPUIP::CopyOp>(isLegalCopyOp);
    target.addDynamicallyLegalOp<VPUIP::ConcatViewOp>(isLegalConcatViewOp);
    target.addDynamicallyLegalOp<VPUIP::ConvertDMAOp>(isLegalConvertDMAOp);
    target.addDynamicallyLegalOp<VPUIP::SwKernelOp>(isLegalSwKernelOp);
    target.addLegalOp<VPUIP::GroupBoundedBufferOp>();
    target.addLegalOp<VPUIP::UngroupBoundedBufferOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UngroupCopyOp>(&ctx, _log);
    patterns.add<UngroupConcatViewOp>(&ctx, _log);
    patterns.add<UngroupSwKernelOp>(&ctx, _log);
    patterns.add<UngroupConvertDMAOp>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUngroupBoundedBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupBoundedBuffersPass(Logger log) {
    return std::make_unique<UngroupBoundedBuffers>(log);
}
