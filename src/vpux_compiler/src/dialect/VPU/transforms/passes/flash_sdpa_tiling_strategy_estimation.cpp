//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/IR/PatternMatch.h>
#include <cmath>
#include <functional>
#include <optional>

namespace vpux::VPU {
#define GEN_PASS_DECL_FLASHSDPATILINGSTRATEGYESTIMATION
#define GEN_PASS_DEF_FLASHSDPATILINGSTRATEGYESTIMATION
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

// Estimate if we apply isolated tiling with the given Query number of slices
// how many times we would have to tile on Key/Value to fit everything in CMX
std::optional<int64_t> estimateRequiredKvNumBlocks(VPU::FlashSDPAOp origOp, ArrayRef<NDTypeInterface> tiledTensors) {
    if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, Byte(0), /*kvNumBlocks*/ 1)) {
        return 1;
    }

    const auto keyShape = getShape(origOp.getKey());
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];

    const auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    const auto elemType = keyType.getElementType();
    const auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    auto kvNumBlocks = int64_t{1};
    auto dimSize = sourceSeqLen;
    while (dimSize > alignment) {
        kvNumBlocks = divUp(sourceSeqLen, dimSize - alignment);
        dimSize = alignValUp(divUp(sourceSeqLen, kvNumBlocks), alignment);

        if (origOp.fitIntoCMXAfterKeyValueTiling(tiledTensors, Byte(0), kvNumBlocks)) {
            return kvNumBlocks;
        }
    }

    // No split on Key/Value was found that would fit CMX
    return std::nullopt;
}

mlir::LogicalResult FlashSDPARewrite::matchAndRewrite(VPU::FlashSDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    auto log = _log.nest();

    auto resultShape = getShape(origOp.getResult(0));
    auto resultRank = resultShape.size();
    if (resultRank < 2) {
        return errorAt(origOp, "Output shape must at least have a rank 2, got {0}", resultRank);
    }

    auto tilingDims = getTileDimOrder(origOp, TilingMode::ISOLATED, log);
    auto strategy = SmallVector<int64_t>(tilingDims.size(), 1);
    log.trace("Tiling space size '{0}' with order '{1}'", strategy, tilingDims);

    auto tiledResultShape = Shape(resultShape);

    auto kvNumBlocks = std::optional<int64_t>{};
    for (auto tilingIndex : irange(strategy.size())) {
        auto curTilingDim = tilingDims[tilingIndex];
        const auto tilingDimSize = resultShape[curTilingDim];

        auto curDimSize = tilingDimSize;
        auto numDivisions = int64_t{1};

        while (true) {
            auto swOp = mlir::cast<VPU::SWOpInterface>(origOp.getOperation());
            auto tiledTensors = getAllOperandsSwInterface(swOp, TileInfo{tiledResultShape}, log);
            kvNumBlocks = estimateRequiredKvNumBlocks(origOp, tiledTensors);

            const auto qNumBlocks = accumulate(strategy, int64_t{1}, std::multiplies<int64_t>{});
            if (kvNumBlocks.has_value() && qNumBlocks >= kvNumBlocks.value()) {
                break;
            }

            if (curDimSize <= 1) {
                break;
            }

            numDivisions = divUp(tilingDimSize, curDimSize - 1);
            curDimSize = divUp(tilingDimSize, numDivisions);

            tiledResultShape[curTilingDim] = curDimSize;
            strategy[tilingIndex] = numDivisions;
        }
    }

    if (!kvNumBlocks.has_value()) {
        return errorAt(origOp, "Failed to estimate tiling for FlashSDPA operation. Tensors will not fit CMX.");
    }

    log.trace("Computed tiling: Query {0} - {1} times, KeyValue - {2} times", tilingDims, strategy, kvNumBlocks);

    // Save Key/Value tiling attribute to propagate the tiling decision to each FlashSDPAOp.
    const auto setKvNumBlocks = [&] {
        origOp.setKvNumBlocks(kvNumBlocks);
    };
    rewriter.modifyOpInPlace(origOp, setKvNumBlocks);

    return mlir::success();
}

//
// FlashSDPATilingStrategyEstimation
//

class FlashSDPATilingStrategyEstimation final :
        public VPU::impl::FlashSDPATilingStrategyEstimationBase<FlashSDPATilingStrategyEstimation> {
public:
    explicit FlashSDPATilingStrategyEstimation(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void FlashSDPATilingStrategyEstimation::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto isLegal = [](VPU::FlashSDPAOp op) {
        return op.getKvNumBlocksAttr() != nullptr;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<VPU::FlashSDPAOp>(isLegal);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FlashSDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFlashSDPATilingStrategyEstimationPass
//

std::unique_ptr<mlir::Pass> VPU::createFlashSDPATilingStrategyEstimationPass(Logger log) {
    return std::make_unique<FlashSDPATilingStrategyEstimation>(log);
}
