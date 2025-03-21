//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_BATCHMATMULTOMATMUL
#define GEN_PASS_DEF_BATCHMATMULTOMATMUL
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

mlir::Type typeTo4D(const mlir::Type type) {
    auto origType = type.cast<NDTypeInterface>();
    const ShapeRef origShape = origType.getShape();
    VPUX_THROW_WHEN(origShape.size() != DimsGroups5D::Act::numDims, "typeTo4D expects only 5-d shapes as inputs.");

    const Shape newShape(origShape.begin() + 1, origShape.end());
    const Strides origStrides = origType.getStrides();
    const Strides newStrides(origStrides.begin() + 1, origStrides.end());

    // Remove the first dimension from the permutation, then shift all remaining dimensions.
    // [d0, d1, d2, d3, d4] -> erase -> [d1, d2, d3, d4] -> shift -> [d0, d1, d2, d3]
    // [d0, d1, d3, d4, d2] -> erase -> [d1, d3, d4, d2] -> shift -> [d0, d2, d3, d1]
    auto permutation = origType.getDimsOrder().toPermutation();
    VPUX_THROW_WHEN(permutation.front() != DimsGroups5D::Act::G, "Groups must be the outermost dimension.");
    permutation.erase(permutation.begin());
    for (auto& dim : permutation) {
        dim = Dim(dim.ind() - 1);
    }

    const auto newOrder = DimsOrder::fromPermutation(permutation);

    // Update axis in per-axis quantization
    auto elemType = origType.getElementType();
    if (auto perAxisType = elemType.dyn_cast<mlir::quant::QuantileQuantizedPerAxisType>()) {
        const auto origQuantAxis = perAxisType.getQuantizedDimension();
        VPUX_THROW_WHEN(origQuantAxis <= 0, "Quantization over batch is not supported.");
        const auto newQuantAxis = origQuantAxis - 1;

        elemType = mlir::quant::QuantileQuantizedPerAxisType::get(
                perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getQuantileType(),
                perAxisType.getExpressedType(), perAxisType.getQuantiles(), perAxisType.getScales(),
                perAxisType.getZeroPoints(), newQuantAxis, perAxisType.getStorageTypeMin(),
                perAxisType.getStorageTypeMax());
    } else if (auto perAxisType = elemType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto origQuantAxis = perAxisType.getQuantizedDimension();
        VPUX_THROW_WHEN(origQuantAxis <= 0, "Quantization over batch is not supported.");
        const auto newQuantAxis = origQuantAxis - 1;

        elemType = mlir::quant::UniformQuantizedPerAxisType::get(
                perAxisType.getFlags(), perAxisType.getStorageType(), perAxisType.getExpressedType(),
                perAxisType.getScales(), perAxisType.getZeroPoints(), newQuantAxis, perAxisType.getStorageTypeMin(),
                perAxisType.getStorageTypeMax());
    }

    const auto newTypeComponents =
            TypeComponents().setShape(newShape).setDimsOrder(newOrder).setStrides(newStrides).setElementType(elemType);
    return origType.changeTypeComponents(newTypeComponents);
}

VPURT::DeclareBufferOp createNewValue(const mlir::Value val, const int64_t step, mlir::PatternRewriter& rewriter) {
    auto inputProducer = val.getDefiningOp<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(inputProducer == nullptr, "MemRef producer must be a VPURT::DeclareBufferOp");

    rewriter.setInsertionPointAfter(inputProducer);

    auto input4d = typeTo4D(inputProducer.getBuffer().getType());
    const auto byteOffset = inputProducer.getByteOffset() + step;

    auto inputProducer4d = mlir::cast<VPURT::DeclareBufferOp>(rewriter.clone(*inputProducer));
    inputProducer4d.setByteOffset(byteOffset);
    inputProducer4d.getBuffer().setType(input4d);

    return inputProducer4d;
}

int64_t getStep(const mlir::Type type) {
    auto origType = type.cast<NDTypeInterface>();
    auto batchSize = origType.getShape()[DimsGroups5D::Act::G];
    return origType.getTotalAllocSize().count() / batchSize;
}

class MatMulRewriter final : public mlir::OpRewritePattern<VPURT::TaskOp> {
public:
    MatMulRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPURT::TaskOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::TaskOp vpurtTask, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MatMulRewriter::matchAndRewrite(VPURT::TaskOp vpurtTask, mlir::PatternRewriter& rewriter) const {
    auto nceOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
    if (nceOp == nullptr) {
        return mlir::failure();
    }

    auto origType = nceOp.getOutput().getType().cast<NDTypeInterface>();
    const ShapeRef origShape = origType.getShape();
    if (origShape.size() != DimsGroups5D::Act::numDims) {
        return mlir::failure();
    }

    const auto batchSize = origShape[DimsGroups5D::Act::G];

    const auto inputStep = getStep(nceOp.getInput().getType());
    const auto weightsStep = getStep(nceOp.getWeights().getType());
    const auto weightTableStep = getStep(nceOp.getWeightTable().getType());
    const auto parentInputStep = getStep(nceOp.getParentInput().getType());
    const auto parentOutputStep = getStep(nceOp.getParentOutput().getType());
    const auto outputProducerStep = getStep(nceOp.getOutputBuff().getType());

    for (const auto& idx : irange(batchSize)) {
        rewriter.setInsertionPointAfter(vpurtTask);
        auto newVpurtTask = mlir::cast<VPURT::TaskOp>(rewriter.clone(*vpurtTask));
        auto innerTask = newVpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
        VPUX_THROW_WHEN(innerTask == nullptr, "Cloned operation must have inner NCEClusterTaskOp");

        // Reverse the order of operations.
        const auto pos = batchSize - idx - 1;
        auto inputProducer4d = createNewValue(innerTask.getInput(), inputStep * pos, rewriter);
        auto weightProducer4d = createNewValue(innerTask.getWeights(), weightsStep * pos, rewriter);
        auto weightTableProducer4d = createNewValue(innerTask.getWeightTable(), weightTableStep * pos, rewriter);
        auto parentInputProducer4d = createNewValue(innerTask.getParentInput(), parentInputStep * pos, rewriter);
        auto parentOutputProducer4d = createNewValue(innerTask.getParentOutput(), parentOutputStep * pos, rewriter);
        auto outputProducer4d = createNewValue(innerTask.getOutputBuff(), outputProducerStep * pos, rewriter);

        mlir::IRMapping mapper;
        mapper.map(innerTask.getInput(), inputProducer4d);
        mapper.map(innerTask.getWeights(), weightProducer4d);
        mapper.map(innerTask.getWeightTable(), weightTableProducer4d);
        mapper.map(innerTask.getParentInput(), parentInputProducer4d);
        mapper.map(innerTask.getParentOutput(), parentOutputProducer4d);
        mapper.map(innerTask.getOutputBuff(), outputProducer4d);

        rewriter.setInsertionPointAfter(innerTask);

        auto newOp = mlir::cast<VPUIP::NCEClusterTaskOp>(rewriter.clone(*innerTask, mapper));
        newOp.getOutput().setType(typeTo4D(newOp.getOutput().getType()));
        rewriter.replaceOp(innerTask, newOp.getOutputs());
    }

    // Erase original VPURT task
    auto inputProducer = nceOp.getInput().getDefiningOp();
    auto weightProducer = nceOp.getWeights().getDefiningOp();
    auto weightTableProducer = nceOp.getWeightTable().getDefiningOp();
    auto parentInputProducer = nceOp.getParentInput().getDefiningOp();
    auto parentOutputProducer = nceOp.getParentOutput().getDefiningOp();
    auto outputProducer = nceOp.getOutputBuff().getDefiningOp();

    const auto sameInput = parentInputProducer == inputProducer;
    const auto sameOutput = parentOutputProducer == outputProducer;

    if (nceOp->use_empty()) {
        rewriter.eraseOp(nceOp);
    }
    if (inputProducer->use_empty()) {
        rewriter.eraseOp(inputProducer);
    }
    if (weightProducer->use_empty()) {
        rewriter.eraseOp(weightProducer);
    }
    if (weightTableProducer->use_empty()) {
        rewriter.eraseOp(weightTableProducer);
    }

    if (!sameInput && parentInputProducer->use_empty()) {
        rewriter.eraseOp(parentInputProducer);
    }

    if (!sameOutput && parentOutputProducer->use_empty()) {
        rewriter.eraseOp(parentOutputProducer);
    }

    if (outputProducer->use_empty()) {
        rewriter.eraseOp(outputProducer);
    }
    if (vpurtTask->use_empty()) {
        rewriter.eraseOp(vpurtTask);
    }
    return mlir::success();
}

class BatchMatMulToMatMul final : public VPUIP::impl::BatchMatMulToMatMulBase<BatchMatMulToMatMul> {
public:
    explicit BatchMatMulToMatMul(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BatchMatMulToMatMul::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto vpurtTasks = func.getOps<VPURT::TaskOp>();
    const auto is5D = [](VPURT::TaskOp vpurtTask) -> bool {
        auto nceOp = vpurtTask.getInnerTaskOpOfType<VPUIP::NCEClusterTaskOp>();
        if (nceOp == nullptr) {
            return false;
        }
        return getShape(nceOp.getOutput()).size() == DimsGroups5D::Act::numDims;
    };
    if (std::none_of(vpurtTasks.begin(), vpurtTasks.end(), is5D)) {
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MatMulRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createBatchMatMulToMatMulPass(Logger log) {
    return std::make_unique<BatchMatMulToMatMul>(log);
}
