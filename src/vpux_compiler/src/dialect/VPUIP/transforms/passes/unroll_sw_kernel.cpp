//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/bounded_buffer.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNROLLSWKERNEL
#define GEN_PASS_DEF_UNROLLSWKERNEL
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

bool hasMultiSwKernelRun(VPUIP::SwKernelOp swKernelOp) {
    auto swKernelRun = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    return std::distance(swKernelRun.begin(), swKernelRun.end()) > 1;
}

bool isOperandFromList(mlir::ValueRange rangeList, mlir::Value operand) {
    return llvm::find(rangeList, operand) != rangeList.end();
}

SmallVector<mlir::Value> getOuterMostMappingOperand(VPUIP::SwKernelRun swKernelRun, bool isDynamic) {
    auto swKernelOp = swKernelRun->getParentOfType<VPUIP::SwKernelOp>();
    VPUX_THROW_WHEN(swKernelOp == nullptr, "Cannot find VPUIP.SwKernelOp at '{0}'", swKernelRun->getLoc());

    SmallVector<mlir::Value> outerMostOperands;
    auto swKernelOpOperands = swKernelOp.getOperands();

    auto getOuterOperand = [&](mlir::BlockArgument blockArg) {
        auto index = blockArg.getArgNumber();
        if (isDynamic) {
            auto dynShapesInStart = llvm::find_if(swKernelOpOperands, [&](auto operand) {
                return isOperandFromList(swKernelOp.getDynamicInputShapes(), operand);
            });
            // Since we only require Inputs and OutputBuffs, we need to consider the following order of
            // operands in the case of dynamic shapes: inputs, dynamicInputShapes, outputBuffs, dynamicOutputShapes.
            auto dynShapesStartInIndex = std::distance(swKernelOpOperands.begin(), dynShapesInStart);
            if (index >= dynShapesStartInIndex) {
                index += swKernelOp.getDynamicInputShapes().size();
            }
        }
        return swKernelOp->getOperand(index);
    };

    for (auto operand : swKernelRun->getOperands()) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand);
        VPUX_THROW_WHEN(blockArg == nullptr, "Matching argument was not identified");
        outerMostOperands.push_back(getOuterOperand(blockArg));
    }

    return outerMostOperands;
}

VPUIP::SwProfilingMetadataAttr getUpdatedSwProfilingMetadataAttr(VPUIP::SwProfilingMetadataAttr attr, size_t tileId,
                                                                 std::optional<size_t> maybeClusterId) {
    const size_t bufferId = attr.getBufferId().getInt();
    const size_t bufferOffset = attr.getBufferOffset().getInt();
    const size_t clusterSize = attr.getClusterSize().getInt();
    const size_t dataIndex = attr.getDataIndex().getInt();
    return vpux::getSwProfilingMetaAttr(attr.getContext(), bufferId, bufferOffset, clusterSize, dataIndex + tileId,
                                        tileId, maybeClusterId);
}

//
// SwKernelRewriter
//

class SwKernelRewriter : public mlir::OpRewritePattern<VPUIP::SwKernelOp> {
public:
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp swKernelOp, mlir::PatternRewriter& rewriter) const override;
    SwKernelRewriter(mlir::MLIRContext* ctx, bool enableSwKernelFifoPerShaveEngine, Logger log)
            : mlir::OpRewritePattern<VPUIP::SwKernelOp>(ctx),
              _log(log),
              _ctx(ctx),
              _enableSwKernelFifoPerShaveEngine(enableSwKernelFifoPerShaveEngine) {
        setDebugName("SwKernelRewriter");
    }

    bool needUnroll(VPUIP::SwKernelOp swKernelOp) const;
    VPURT::TaskOp createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                  VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter, size_t index) const;

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
    bool _enableSwKernelFifoPerShaveEngine;
};

mlir::LogicalResult SwKernelRewriter::matchAndRewrite(VPUIP::SwKernelOp swKernelOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (!needUnroll(swKernelOp)) {
        if (_enableSwKernelFifoPerShaveEngine && !swKernelOp.getListIndex().has_value()) {
            swKernelOp.setListIndexAttr(getIntAttr(_ctx, 0));
        }
        return mlir::failure();
    }

    auto vpurtTask = swKernelOp->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_UNLESS(vpurtTask != nullptr, "Can't get VPURT task operation");

    rewriter.setInsertionPointAfter(vpurtTask);
    auto swKernelRunList = swKernelOp.getBody().getOps<VPUIP::SwKernelRun>();
    for (auto swKernelRunTuple : swKernelRunList | indexed) {
        auto newTaskOp =
                createNewTaskOp(swKernelOp, swKernelRunTuple.value(), vpurtTask, rewriter, swKernelRunTuple.index());
        _log.trace("create new task op: {0}", newTaskOp);
    }
    rewriter.eraseOp(vpurtTask);
    return mlir::success();
}

bool SwKernelRewriter::needUnroll(VPUIP::SwKernelOp swKernelOp) const {
    auto hasMultiSwKernelRunFlag = hasMultiSwKernelRun(swKernelOp);
    return hasMultiSwKernelRunFlag;
}

VPURT::TaskOp SwKernelRewriter::createNewTaskOp(VPUIP::SwKernelOp swKernelOp, VPUIP::SwKernelRun swKernelRun,
                                                VPURT::TaskOp origTaskOp, mlir::PatternRewriter& rewriter,
                                                size_t index) const {
    auto opLoc = swKernelOp->getLoc();
    auto isDynamic = VPUIP::hasBoundedBuffers(swKernelOp) || VPUIP::hasUngroupedBoundedBuffers(swKernelOp);

    auto outerOperand = getOuterMostMappingOperand(swKernelRun, isDynamic);
    auto iter = llvm::find_if(outerOperand, [&](auto operand) {
        return isOperandFromList(swKernelOp.getOutputBuffs(), operand);
    });
    VPUX_THROW_WHEN(iter == outerOperand.end(), "Cannot find operand for output buffer at '{0}'", opLoc);

    auto outBufferStartIndex = std::distance(outerOperand.begin(), iter);

    SmallVector<mlir::Value> swKernelInputDynamicShapes, swKernelOutputDynamicShapes;
    SmallVector<int32_t> swKernelInputDynamicShapesMap, swKernelOutputDynamicShapesMap;

    auto newInputs = SmallVector<mlir::Value>(outerOperand.begin(), outerOperand.begin() + outBufferStartIndex);
    auto newOutBuffers = SmallVector<mlir::Value>(outerOperand.begin() + outBufferStartIndex, outerOperand.end());

    if (isDynamic) {
        auto fullInputShapes = swKernelOp.getDynamicInputShapes();
        auto fullOutputShapes = swKernelOp.getDynamicOutputShapeBuffs();
        auto fullInputShapesMap = swKernelOp.getDynamicInputShapesMap().value_or(ArrayRef<int32_t>());
        auto fullOutputShapesMap = swKernelOp.getDynamicOutputShapesMap().value_or(ArrayRef<int32_t>());

        swKernelInputDynamicShapes = to_small_vector(fullInputShapes);
        swKernelOutputDynamicShapes = to_small_vector(fullOutputShapes);

        swKernelInputDynamicShapesMap = to_small_vector(fullInputShapesMap);
        swKernelOutputDynamicShapesMap = to_small_vector(fullOutputShapesMap);
    }

    mlir::Value newProfilingBuffer = nullptr;
    VPUIP::SwProfilingMetadataAttr maybeProfMeta = nullptr;
    if (auto profilingBuffer = swKernelOp.getProfilingData()) {
        // In case task has profiling buffer as SwKernelOp operand, each individual SwKernelRun will be given
        // its own chunk of this buffer. Buffer size was properly prepared by act-shave-profiling pass with
        // enough space for each SwKernelRun inside
        VPUX_THROW_WHEN(swKernelOp.getProfilingMetadataAttr() == nullptr, "Missed profilingMetadata for '{0}'",
                        swKernelOp);
        maybeProfMeta =
                getUpdatedSwProfilingMetadataAttr(swKernelOp.getProfilingMetadataAttr(), index, /*clusterId=*/0);
        auto profilingBufferDecl = profilingBuffer.getDefiningOp<VPURT::DeclareBufferOp>();
        auto profilingBufferNDType = mlir::cast<vpux::NDTypeInterface>(profilingBuffer.getType());

        int64_t numEl =
                VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES / vpux::Byte(profilingBufferNDType.getElemTypeSize()).count();
        if (auto distrProfilingType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(profilingBuffer.getType())) {
            numEl *= distrProfilingType.getDistribution().getNumClusters().getInt();
        }

        const auto newMemType = profilingBufferNDType.changeShape({numEl});
        auto newProfilingBufferDecl = rewriter.create<VPURT::DeclareBufferOp>(
                profilingBufferDecl->getLoc(), newMemType, profilingBufferDecl.getSectionAttr(),
                profilingBufferDecl.getSectionIndexAttr(),
                getIntAttr(_ctx,
                           profilingBufferDecl.getByteOffset() + VPUIP::HW_ACT_SHAVE_PROFILING_SIZE_BYTES * index),
                nullptr);

        newProfilingBuffer = newProfilingBufferDecl.getBuffer();
    }
    // NOTE: use of cluster, tile and list indexes is not consistent. When tile is used in combination with
    // cluster it is understood that cluster identifies NCE unit and tile identifies its SHAVE sub unit
    // (typically used in profiling). When tile index is used in combination with list index it is the tile that
    // identifies NCE unit and list index identifies its SHAVE sub-unit (typically used in HAS and in backend).
    opLoc = appendLoc(opLoc, "tile_{0}", index);

    VPUIP::SwKernelOp newSwKernelOp = [&] {
        if (isDynamic) {
            return VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                    rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), opLoc, newInputs,
                    newOutBuffers, swKernelInputDynamicShapes, swKernelInputDynamicShapesMap,
                    swKernelOutputDynamicShapes, swKernelOutputDynamicShapesMap, newProfilingBuffer,
                    swKernelOp.getKernelFunctionAttr(), swKernelOp.getTileIndexAttr(), swKernelOp.getInputStridesAttr(),
                    swKernelOp.getOutputStridesAttr());
        }
        return VPURT::wrapIntoTaskOp<VPUIP::SwKernelOp>(
                rewriter, origTaskOp.getWaitBarriers(), origTaskOp.getUpdateBarriers(), opLoc, newInputs, newOutBuffers,
                newProfilingBuffer, swKernelOp.getKernelFunctionAttr(), swKernelOp.getTileIndexAttr(),
                swKernelOp.getInputStridesAttr(), swKernelOp.getOutputStridesAttr());
    }();

    if (maybeProfMeta != nullptr) {
        newSwKernelOp.setProfilingMetadataAttr(maybeProfMeta);
    }
    if (_enableSwKernelFifoPerShaveEngine) {
        newSwKernelOp.setListIndexAttr(getIntAttr(_ctx, index));
    }
    VPUIP::initSwKernel(newSwKernelOp, swKernelRun, _log);
    return newSwKernelOp->getParentOfType<VPURT::TaskOp>();
}

bool verifySwKernelOpsWithDedicatedFifoPerShaveEngine(mlir::func::FuncOp func, Logger log) {
    bool hasDefinedListIndexAttr;
    size_t swKerOpCount = 0;
    size_t swKerOpCountWithListAttr = 0;
    func.walk([&](VPURT::TaskOp taskOp) {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(taskOp.getInnerTaskOp())) {
            swKerOpCount++;
            if (swKernelOp.getListIndex().has_value()) {
                swKerOpCountWithListAttr++;
            }
        }
    });

    if (swKerOpCount == 0) {
        return true;
    }
    hasDefinedListIndexAttr = swKerOpCountWithListAttr > 0;
    auto isListIndexSetForAllOrNoneOps = (swKerOpCount == swKerOpCountWithListAttr || !hasDefinedListIndexAttr);
    if (!isListIndexSetForAllOrNoneOps) {
        log.warning("Inconsistent use of listIndex attribute. SwKernelOp count: {0}, SwKernelOp count with listIndex "
                    "attribute: {1}",
                    swKerOpCount, swKerOpCountWithListAttr);
    }

    return isListIndexSetForAllOrNoneOps;
}

//
// UnrollSwKernelPass
//

class UnrollSwKernelPass final : public VPUIP::impl::UnrollSwKernelBase<UnrollSwKernelPass> {
public:
    explicit UnrollSwKernelPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollSwKernelPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    bool swKernelFifoPerShaveEngine = enableSwKernelFifoPerShaveEngine.hasValue()
                                              ? enableSwKernelFifoPerShaveEngine.getValue()
                                              : config::isFifoPerShaveEngineEnabled(func);
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<SwKernelRewriter>(&ctx, swKernelFifoPerShaveEngine, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));

    VPUX_THROW_UNLESS(verifySwKernelOpsWithDedicatedFifoPerShaveEngine(func, _log),
                      "Inconsistent use of listIndex attribute");
}
}  // namespace

//
// createUnrollSwKernelPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUnrollSwKernelPass(Logger log) {
    return std::make_unique<UnrollSwKernelPass>(log);
}
