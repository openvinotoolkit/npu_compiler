//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/BuiltinAttributes.h>
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_UNROLLFLASHSDPA
#define GEN_PASS_DEF_UNROLLFLASHSDPA
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class FlashSDPARewrite final : public mlir::OpRewritePattern<VPU::FlashSDPAOp> {
public:
    FlashSDPARewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::FlashSDPAOp>(ctx), _log(log) {
        setDebugName("FlashSDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value createSlice(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value value, Dim dimension,
                        int64_t beginOffset, int64_t endOffset, const Logger& log) {
    auto shape = getShape(value);

    auto sliceOffset = Shape(shape.size(), 0);
    sliceOffset[dimension] = checked_cast<int64_t>(beginOffset);
    auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), sliceOffset);

    auto sliceSize = Shape(shape);
    sliceSize[dimension] = endOffset - beginOffset;
    auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sliceSize);

    log.trace("Created SliceOp with offset {0} and size {1} at {2}", sliceOffset, sliceSize, loc);
    return rewriter.create<VPU::SliceOp>(loc, value, offsetsAttr, sizesAttr);
}

mlir::LogicalResult FlashSDPARewrite::matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();
    auto ctx = rewriter.getContext();

    const auto kvNumBlocksAttr = origOp.getKvNumBlocksAttr();
    VPUX_THROW_UNLESS(kvNumBlocksAttr != nullptr,
                      "Expected to have kv_num_blocks attribute set to be able to tile {0} at {1}", origOp->getName(),
                      origOp->getLoc());

    const auto kvNumBlocks = parseIntAttr<int64_t>(kvNumBlocksAttr);

    const auto clearTilingAttr = [&] {
        origOp.setKvNumBlocksAttr(nullptr);
    };
    rewriter.modifyOpInPlace(origOp, clearTilingAttr);

    if (kvNumBlocks < 2) {
        log.trace("No need to tile the operation");
        return mlir::success();
    }

    // Tiling parameters
    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];

    // MatMul computed as DPU DWConv from SHAVE that requires channel alignment
    // Because we use NCHW layout for the input tensors, the channel dimension is actually the width
    // Second MatMul has Attention scores and Values tensors as an input, with "channels" == sourceSeqLen
    // So we must align SourceSeqLen dimension to have a correct WeightsTable
    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    const auto tileSize = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);

    // Partial values that are chained through FlashSDPAOp
    auto out = origOp.getInputRunningOutput();
    auto max = origOp.getInputRunningMax();
    auto sum = origOp.getInputRunningSum();

    // Padding on SequenceLength is 0 for all operations except the last one
    auto zeroPadAttr = getIntAttr(rewriter, 0);

    // Initial Query tensor slice that will be scaled by the first FlashSDPAOp
    auto query = origOp.getQuery();

    auto i = int64_t{0};
    while (true) {
        log.trace("Unrolling {0} - {1} / {2} times", origOp->getName(), i + 1, kvNumBlocks);
        auto beginOffset = i * tileSize;
        auto endOffset = std::min(beginOffset + tileSize, sourceSeqLen);

        auto keySlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "key_slice_{0}", i), origOp.getKey(),
                                    Dims4D::Act::H, beginOffset, endOffset, log);
        auto valueSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "value_slice_{0}", i), origOp.getValue(),
                                      Dims4D::Act::W, beginOffset, endOffset, log);

        auto attentionMaskSlice = mlir::Value{nullptr};
        if (origOp.getAttentionMask() != nullptr) {
            attentionMaskSlice = createSlice(rewriter, appendLoc(origOp->getLoc(), "attention_mask_slice_{0}", i),
                                             origOp.getAttentionMask(), Dims4D::Act::W, beginOffset, endOffset, log);
        }

        auto isHeadAttr = mlir::BoolAttr::get(ctx, i == 0);
        auto isTailAttr = mlir::BoolAttr::get(ctx, i + 1 == kvNumBlocks);

        auto sourceSeqLenPadSize = (i + 1 == kvNumBlocks) ? origOp.getSourceSeqLenPadSizeAttr() : zeroPadAttr;

        auto tileLoc = appendLoc(origOp->getLoc(), "flash_sdpa_kv_tile_{0}", i);
        auto tiledOp = rewriter.create<VPU::FlashSDPAOp>(tileLoc, query, keySlice, valueSlice, out, max, sum,
                                                         attentionMaskSlice, origOp.getScale(), sourceSeqLenPadSize,
                                                         isHeadAttr, isTailAttr, /*kvNumBlocksAttr*/ nullptr,
                                                         origOp.getMultiClusterStrategyAttr());

        log.trace("Unrolled {0} - {1}", tiledOp->getName(), tiledOp->getResult(0));

        if (i + 1 == kvNumBlocks) {
            rewriter.replaceOp(origOp, tiledOp);

            return mlir::success();
        }

        // Propagate intermediate values to the next FlashSDPAOp
        out = tiledOp.getResultRunningOutput();
        max = tiledOp.getResultRunningMax();
        sum = tiledOp.getResultRunningSum();
        query = tiledOp.getResultQuery();

        i++;
    }
}

//
// UnrollFlashSDPA
//

class UnrollFlashSDPA final : public VPU::impl::UnrollFlashSDPABase<UnrollFlashSDPA> {
public:
    explicit UnrollFlashSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void UnrollFlashSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto isLegal = [](VPU::FlashSDPAOp op) {
        return op.getKvNumBlocksAttr() == nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::FlashSDPAOp>(isLegal);
    target.addLegalOp<VPU::SliceOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FlashSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUnrollFlashSDPAPass
//
std::unique_ptr<mlir::Pass> vpux::VPU::createUnrollFlashSDPAPass(Logger log) {
    return std::make_unique<UnrollFlashSDPA>(log);
}
