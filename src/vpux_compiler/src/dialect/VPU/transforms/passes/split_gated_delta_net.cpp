//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

#include <algorithm>
#include <utility>

namespace vpux::VPU {
#define GEN_PASS_DECL_SPLITGATEDDELTANET
#define GEN_PASS_DEF_SPLITGATEDDELTANET
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

constexpr int64_t SEQ_DIM = 1;

//
// SplitGatedDeltaNet
//

class SplitGatedDeltaNet final : public mlir::OpRewritePattern<VPU::GatedDeltaNetOp> {
public:
    SplitGatedDeltaNet(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::GatedDeltaNetOp>(ctx), _log(std::move(log)) {
        setDebugName("SplitGatedDeltaNet");
    }

    mlir::LogicalResult matchAndRewrite(VPU::GatedDeltaNetOp op, mlir::PatternRewriter& rewriter) const final;

private:
    bool fitIntoCMX(ArrayRef<Byte> bufferSizes, int64_t totalAvailableCMXSize, config::ArchKind archKind) const;
    mlir::FailureOr<int64_t> getNumSplits(VPU::GatedDeltaNetOp op) const;
    void splitOverSeq(VPU::GatedDeltaNetOp op, mlir::PatternRewriter& rewriter, int64_t numSplits) const;

    Logger _log;
};

bool SplitGatedDeltaNet::fitIntoCMX(ArrayRef<Byte> bufferSizes, int64_t totalAvailableCMXSize,
                                    config::ArchKind archKind) const {
    auto sizes = to_small_vector(bufferSizes);
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(archKind, sizes).count() <= totalAvailableCMXSize;
}

// Head-axis tiling scheme for one GatedDeltaNet operand/result, mirroring getDistributedTypeForOpOperand/Result:
// scratch is kept whole (DUPLICATED), recurrent_state/output_state are tiled on the head axis C, and every activation
// (query/key/value/gate/beta/output) on the head axis H. Used for both the multi-cluster split and the head tiling.
SmallVector<int64_t> gdnNumTiles(VPU::GatedDeltaNetOp op, mlir::Value value, int64_t numTilesOnHeads) {
    const auto rank = mlir::cast<NDTypeInterface>(value.getType()).getShape().size();
    SmallVector<int64_t> numTiles(rank, 1);
    if (value == op.getScratch()) {
        return numTiles;
    }
    numTiles[VPU::getGatedDeltaNetHeadAxis(op, value)] = numTilesOnHeads;
    return numTiles;
}

NDTypeInterface applyTiling(NDTypeInterface type, ArrayRef<int64_t> numTiles, ArrayRef<int64_t> alignment = {}) {
    auto shape = to_small_vector(type.getShape());
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = divUp(shape[i], numTiles[i]);
        if (!alignment.empty() && alignment[i] > 1) {
            shape[i] = std::min(alignValUp(shape[i], alignment[i]), type.getShape().raw()[i]);
        }
    }
    return type.changeShape(ShapeRef(shape));
}

mlir::FailureOr<int64_t> SplitGatedDeltaNet::getNumSplits(VPU::GatedDeltaNetOp op) const {
    const auto operands = op.getOperands();
    const auto results = op.getResults();

    const auto module = op->getParentOfType<mlir::ModuleOp>();
    auto tileExec = config::getTileExecutor(module);
    const int64_t numClusters = (tileExec != nullptr) ? tileExec.getCount() : 1;
    const bool tiled = op.getMultiClusterStrategy().has_value() &&
                       op.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight;

    const auto isSeqVarying = [&](mlir::Value value) {
        return value == op.getQuery() || value == op.getKey() || value == op.getValue() || value == op.getGate() ||
               value == op.getBeta() || value == op.getOutput();
    };

    // Buffer sizes of the op that reaches the kernel, not of the one this pass produces: the chunk is tiled further
    // over the heads by TilingStrategyAssignment and then spread over the clusters. The scratch is seq-independent
    // but CMX-resident and DUPLICATED, so it stays in at full size.
    const auto totalSizes = [&](int64_t chunkSeqLen, int64_t headTiles) -> SmallVector<Byte> {
        SmallVector<Byte> sizes;
        sizes.reserve(operands.size() + results.size());
        const auto addBuf = [&](mlir::Value value) {
            auto type = mlir::cast<NDTypeInterface>(value.getType());
            if (isSeqVarying(value)) {
                auto shape = to_small_vector(type.getShape());
                shape[SEQ_DIM] = chunkSeqLen;
                type = type.changeShape(ShapeRef(shape));
            }
            const auto alignment = VPU::getGatedDeltaNetHeadAlignment(op, value);
            type = applyTiling(type, gdnNumTiles(op, value, headTiles), alignment);
            if (tiled) {
                type = applyTiling(type, gdnNumTiles(op, value, numClusters), alignment);
            }
            sizes.push_back(type.getTotalAllocSize());
        };
        for (const auto& operand : operands) {
            addBuf(operand);
        }
        for (const auto& result : results) {
            addBuf(result);
        }
        return sizes;
    };

    const int64_t seqLength = getShape(op.getQuery())[Dim(SEQ_DIM)];
    const int64_t vHeads = getShape(op.getValue())[Dims4D::Act::H];
    const int64_t headGroup = std::max<int64_t>(VPU::getGatedDeltaNetHeadGroupSize(op), 1);
    const int64_t numShaves = config::getNumOfEnginesOnTile(module, config::ExecutorKind::SHAVE_ACT);
    const auto totalAvailableCMXSize = getTotalCMXSize(op).count();
    const auto archKind = config::getArch(op);

    // Splitting the sequence is serial (the chunks hand the state over one after another), head tiling is not. So
    // only split as far as the downstream head tiling cannot finish off, and keep at least numShaves heads per
    // cluster, past which the head tiles just leave shaves idle.
    const auto fitsAfterHeadTiling = [&](int64_t chunkSeqLen, bool keepShavesFed) {
        for (int64_t headTiles = 1; headTiles <= vHeads; ++headTiles) {
            const int64_t largestTile = divUp(vHeads, headTiles);
            const int64_t smallestTile = vHeads - (headTiles - 1) * largestTile;
            if (smallestTile <= 0) {
                // The head axis cannot be divided into that many tiles.
                break;
            }
            if (tiled && smallestTile < numClusters * headGroup) {
                // checkStrategyCompatibility would drop the op back to a single cluster.
                continue;
            }
            const int64_t headsPerCluster = tiled ? divUp(largestTile, numClusters) : largestTile;
            if (keepShavesFed && headTiles > 1 && headsPerCluster < numShaves) {
                continue;
            }
            if (fitIntoCMX(totalSizes(chunkSeqLen, headTiles), totalAvailableCMXSize, archKind)) {
                return true;
            }
        }
        return false;
    };

    // Fewest splits the downstream head tiling can finish off; numSplits == 1 leaves it the whole op. Fall back to
    // starving the shaves only if no split avoids it.
    for (const bool keepShavesFed : {true, false}) {
        for (int64_t numSplits = 1; numSplits <= seqLength; ++numSplits) {
            if (fitsAfterHeadTiling(divUp(seqLength, numSplits), keepShavesFed)) {
                return numSplits;
            }
        }
    }

    return mlir::failure();
}

void SplitGatedDeltaNet::splitOverSeq(VPU::GatedDeltaNetOp op, mlir::PatternRewriter& rewriter,
                                      int64_t numSplits) const {
    const auto loc = op.getLoc();
    auto* ctx = rewriter.getContext();
    const int64_t seqLength = getShape(op.getQuery())[Dim(SEQ_DIM)];

    int64_t sliceCnt = 0;
    const auto sliceSeq = [&](mlir::Value in, int64_t start, int64_t len) -> mlir::Value {
        const auto inShape = getShape(in);
        auto offsets = SmallVector<int64_t>(inShape.size(), 0);
        auto sizes = to_small_vector(inShape.raw());
        offsets[SEQ_DIM] = start;
        sizes[SEQ_DIM] = len;
        return rewriter
                .create<VPU::SliceOp>(appendLoc(loc, "gdn_split_slice_{0}", sliceCnt++), in,
                                      getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                .getResult();
    };

    const int64_t splitSize = seqLength / numSplits;
    const int64_t remainder = seqLength % numSplits;

    mlir::Value state = op.getRecurrentState();
    SmallVector<mlir::Value> outputs;
    int64_t start = 0;
    for (int64_t c = 0; c < numSplits; ++c) {
        const int64_t len = (c < remainder) ? (splitSize + 1) : splitSize;

        auto chunk = rewriter.create<VPU::GatedDeltaNetOp>(
                appendLoc(loc, "gdn_split_chunk_{0}", c), sliceSeq(op.getQuery(), start, len),
                sliceSeq(op.getKey(), start, len), sliceSeq(op.getValue(), start, len), state,
                sliceSeq(op.getGate(), start, len), sliceSeq(op.getBeta(), start, len), op.getFuseQkL2normAttr(),
                op.getQL2NormEpsAttr(), op.getKL2NormEpsAttr());
        chunk.getScratchMutable().assign(op.getScratch());
        if (op.getMultiClusterStrategyAttr() != nullptr) {
            chunk.setMultiClusterStrategyAttr(op.getMultiClusterStrategyAttr());
        }
        outputs.push_back(chunk.getOutput());
        state = chunk.getOutputState();
        start += len;
    }

    auto concat = rewriter.create<VPU::ConcatOp>(appendLoc(loc, "gdn_split_concat"), outputs, Dim(SEQ_DIM));
    rewriter.replaceOp(op, {concat.getOutput(), state});
}

mlir::LogicalResult SplitGatedDeltaNet::matchAndRewrite(VPU::GatedDeltaNetOp op,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", op->getName(), op->getLoc());

    // Only the static 4D activation layout is supported; the sequence hand-off cannot be expressed for dynamic shapes.
    if (getShape(op.getQuery()).size() != 4 || IE::hasDynamicTensors(op)) {
        return mlir::failure();
    }

    const auto numSplits = getNumSplits(op);
    if (mlir::failed(numSplits) || numSplits.value() <= 1) {
        return mlir::failure();
    }

    _log.nest().trace("Splitting GatedDeltaNet over sequence into {0} chunks", numSplits.value());
    splitOverSeq(op, rewriter, numSplits.value());
    return mlir::success();
}

//
// SplitGatedDeltaNetPass
//

class SplitGatedDeltaNetPass final : public VPU::impl::SplitGatedDeltaNetBase<SplitGatedDeltaNetPass> {
public:
    explicit SplitGatedDeltaNetPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitGatedDeltaNetPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitGatedDeltaNet>(&ctx, _log);
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createSplitGatedDeltaNetPass(Logger log) {
    return std::make_unique<SplitGatedDeltaNetPass>(std::move(log));
}
