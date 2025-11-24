//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEL2OPS
#define GEN_PASS_DEF_DECOMPOSEL2OPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// NormalizeL2 decomposition into element-wise operations (without Eps)
//
// Input ---> Multiply -> ReduceSum -> Sqrt -> Divide -> Output
//   |           ^                               ^
//   |           |                               |
//   ---------------------------------------------
//

void decomposeNormalizeL2(IE::NormalizeL2Op origOp, Logger log) {
    log.trace("Got NormalizeL2Op for decomposition into eltwise operations - '{0}'", origOp->getLoc());

    mlir::OpBuilder builder(origOp);
    const auto loc = origOp.getLoc();
    const auto data = origOp.getData();
    const auto axesValueAttr = origOp.getAxesValueAttr();

    // Don't decompose if the axes tensor doesn't contain all available dimensions
    const int64_t axesSize = static_cast<int64_t>(axesValueAttr.getValue().size());
    const int64_t dataRank = mlir::cast<vpux::NDTypeInterface>(data.getType()).getRank();
    if (axesSize != dataRank) {
        log.debug("NormalizeL2Op axes tensor:'{0}' doesn't contain all dimensions - '{1}'", axesValueAttr, loc);
        return;
    }

    // --- Calculate sum of squared input data
    auto multiplyOp = builder.create<IE::MultiplyOp>(appendLoc(loc, "_mul"), data, data, IE::AutoBroadcastType::NUMPY,
                                                     nullptr, nullptr, nullptr, nullptr);
    auto reduceSumOp = builder.create<IE::ReduceSumOp>(appendLoc(loc, "_reduceSum"), multiplyOp.getOutput(), nullptr,
                                                       axesValueAttr, false, nullptr, nullptr);

    auto sqrtOp = builder.create<IE::SqrtOp>(appendLoc(loc, "_sqrt"), reduceSumOp.getOutput());

    // --- Divide all input data by the calculated value
    auto divOp = builder.create<IE::DivideOp>(appendLoc(loc, "_div"), data, sqrtOp.getOutput(),
                                              IE::AutoBroadcastType::NUMPY);

    origOp.getOutput().replaceAllUsesWith(divOp.getOutput());
    origOp.erase();
}

//
// ReduceL2 decomposition
//
// Input ---> Multiply -> ReduceSum -> Sqrt -> Output
//

// ReduceL2 decomposition forces ReduceSum to have FP16 output, and there is a risk of getting out-of-range
// values during accumulation. To avoid this, we only decompose ReduceL2 ops that reduce a small number of
// elements. This threshold is conservatively set to 64 to minimize the risk of accuracy loss.
constexpr int64_t REDUCED_ELEMENTS_THRESHOLD = 64;

void decomposeReduceL2(IE::ReduceL2Op origOp, Logger log) {
    log.trace("Got ReduceL2Op for decomposition into eltwise operations - '{0}'", origOp->getLoc());

    const auto inputShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getShape();

    if (inputShape.totalSize() < VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT) {
        log.trace("ReduceL2 shape size is too small ({0}) for decomposition", inputShape.totalSize());
        return;
    }

    const auto loc = origOp.getLoc();

    const auto outputElementType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    if (outputElementType.isF16()) {
        auto axesValue = IE::extractAxes(loc, origOp);
        auto numReducedElements = int64_t(1);
        for (auto axis : axesValue) {
            axis += axis < 0 ? static_cast<int64_t>(inputShape.size()) : 0;
            numReducedElements *= inputShape[Dim(axis)];
        }
        if (numReducedElements > REDUCED_ELEMENTS_THRESHOLD) {
            log.trace("ReduceL2 reduces too many elements ({0}), which may lead to accuracy loss", numReducedElements);
            return;
        }
    }

    mlir::OpBuilder builder(origOp);

    auto input = origOp.getInput();
    auto multiplyOp = builder.create<IE::MultiplyOp>(appendLoc(loc, "square"), input, input,
                                                     IE::AutoBroadcastType::NUMPY, nullptr, nullptr, nullptr, nullptr);
    auto reduceSumOp =
            builder.create<IE::ReduceSumOp>(appendLoc(loc, "reduceSum"), multiplyOp.getOutput(), origOp.getAxes(),
                                            origOp.getAxesValueAttr(), origOp.getKeepDimsAttr());
    auto sqrtOp = builder.create<IE::SqrtOp>(appendLoc(loc, "sqrt"), reduceSumOp.getOutput());

    origOp.getOutput().replaceAllUsesWith(sqrtOp.getOutput());
    origOp.erase();
}

//
// DecomposeL2OpsPass
//

class DecomposeL2OpsPass final : public IE::impl::DecomposeL2OpsBase<DecomposeL2OpsPass> {
public:
    explicit DecomposeL2OpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeL2OpsPass::safeRunOnFunc() {
    auto func = getOperation();
    func.walk([&](IE::NormalizeL2Op origOp) {
        decomposeNormalizeL2(origOp, _log);
    });
    func.walk([&](IE::ReduceL2Op origOp) {
        decomposeReduceL2(origOp, _log);
    });
}

}  // namespace

//
// createDecomposeL2OpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeL2OpsPass(Logger log) {
    return std::make_unique<DecomposeL2OpsPass>(log);
}
