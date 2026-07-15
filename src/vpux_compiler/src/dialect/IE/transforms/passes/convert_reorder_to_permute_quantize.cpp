//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_to_pool_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTREORDERTOPERMUTEQUANTIZE
#define GEN_PASS_DEF_CONVERTREORDERTOPERMUTEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool hasQuantizedAvgPoolUserToPropagate(IE::ReorderOp reorder);
bool canConvertEltwisePatternToMaxPool(IE::ReorderOp reorder, config::ArchKind arch, int64_t numClusters, Logger log);

bool shouldConvertReorderOp(IE::ReorderOp reorder, config::ArchKind arch, int64_t numClusters, Logger log) {
    auto inType = mlir::cast<vpux::NDTypeInterface>(reorder.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(reorder.getOutput().getType());
    const auto inOrder = inType.getDimsOrder();
    const auto outOrder = outType.getDimsOrder();

    if (isTrivialReorder(reorder)) {
        log.trace("Skip trivial reorder");
        return false;
    }

    const auto origMemPerm = vpux::getPermutationFromOrders(inOrder, outOrder, reorder->getContext());
    if (IE::canConvertToNCHWInOrderWithPermuteCast(inType, origMemPerm) && outOrder == DimsOrder::NHWC) {
        // There is a chance to convert reorderOp to permuteQuantizeOp after inserting a permuteCastOp for input
        inType = inType.changeDimsOrder(DimsOrder::NCHW);
    }

    if (!IE::isLegalReorderLikeToPermuteQuantize(inType, outType, log)) {
        log.trace("Can not convert to PermuteQuantize");
        return false;
    }
    if (hasQuantizedAvgPoolUserToPropagate(reorder)) {
        log.trace("PermuteQuantize can not be propagated through avgpool");
        return false;
    }

    // Check pattern: EltwiseOp -> PermuteCast -> Reorder
    // If this pattern can be converted to EltwiseOp -> MaxPool by later passes,
    // block Reorder to PermuteQuantize conversion to enable data spilling optimization between EltwiseOp and MaxPool
    if (canConvertEltwisePatternToMaxPool(reorder, arch, numClusters, log)) {
        log.trace("Block conversion: Eltwise->PermuteCast->Reorder can be converted to MaxPool at {0}",
                  reorder->getLoc());
        return false;
    }

    return true;
}

class FusePermuteRewrite final : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    FusePermuteRewrite(mlir::MLIRContext* ctx, config::ArchKind arch, int64_t numClusters, Logger log)
            : mlir::OpRewritePattern<IE::ReorderOp>(ctx), _arch(arch), _numClusters(numClusters), _log(log) {
        setDebugName("FusePermuteRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    config::ArchKind _arch;
    int64_t _numClusters;
    Logger _log;
};

mlir::LogicalResult FusePermuteRewrite::matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertReorderOp(origOp, _arch, _numClusters, _log)) {
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    auto curInput = origOp.getInput();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto origMemPerm = vpux::getPermutationFromOrders(inOrder, outOrder, origOp->getContext());
    if (IE::canConvertToNCHWInOrderWithPermuteCast(inType, origMemPerm) && outOrder == DimsOrder::NHWC) {
        // There is a chance to convert reorderOp to permuteQuantizeOp after inserting a permuteCastOp for input
        const auto inMemPerm = vpux::getPermutationFromOrders(inOrder, DimsOrder::NCHW, origOp->getContext());
        auto inPermuteCastOp =
                rewriter.create<IE::PermuteCastOp>(appendLoc(origOp->getLoc(), "PermuteCast"), origOp.getInput(),
                                                   DimsOrder::NCHW.toAffineMap(origOp->getContext()), inMemPerm);
        curInput = inPermuteCastOp.getResult();
        inOrder = DimsOrder::NCHW;
    }

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();
    const auto permQuantOutType = origOp.getOutput().getType();
    const auto permQuantElemType = mlir::cast<vpux::NDTypeInterface>(permQuantOutType).getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = takeOpLoc(origOp, "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, curInput, origOp.getDstOrderAttr(), memPermAttr, dstElemTypeAttr,
            getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    rewriter.replaceOp(origOp, permuteQuantizeOp.getOutput());

    return mlir::success();
}

//
// ConvertReorderToPermuteQuantizePass
//

class ConvertReorderToPermuteQuantizePass final :
        public IE::impl::ConvertReorderToPermuteQuantizeBase<ConvertReorderToPermuteQuantizePass> {
public:
    explicit ConvertReorderToPermuteQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool hasQuantizedAvgPoolUserToPropagate(IE::ReorderOp reorder) {
    // For pattern Reorder -> Avgpool(quantize-dequantize or just quantize) -> Eltwise
    // if the Avgpool is quantized, the reorder can't be propagated through eltwise
    // after converting to PermuteQuantize because of the quantized type
    // Don't convert to PermuteQuantize to enable the propagation and fusion
    // Reorder -> Avgpool -> Eltwise will eventually become
    // PermuteCast -> Avgpool -> Eltwise -> PermuteCast with propagation and fusion
    // But if converted to PermuteQuantize
    // PermuteQuantize (f16 to f16) -> Avgpool (f16 to quant)
    // cannot be transformed into
    // Avgpool (f16 to quant) -> PermuteQuantize (quant to quant)
    // because PermuteQuantize expects f16 input

    if (!reorder->hasOneUse()) {
        return false;
    }
    // condition 1, quantize avgpool user
    auto avgpoolOp = mlir::dyn_cast_or_null<IE::AvgPoolOp>(*reorder.getOutput().getUsers().begin());
    if (avgpoolOp == nullptr || !avgpoolOp->hasOneUse() || !isQuantizedPurposeAvgPool(avgpoolOp)) {
        return false;
    }
    // condition 2, optional dequantize avgpool and eltwise after the avgpool(s)
    if (auto nextOp = mlir::dyn_cast_or_null<IE::AvgPoolOp>(*avgpoolOp.getOutput().getUsers().begin())) {
        avgpoolOp = nextOp;
    }
    return (*avgpoolOp.getOutput().getUsers().begin())->hasTrait<IE::EltwiseOp>();
}

bool canConvertEltwisePatternToMaxPool(IE::ReorderOp reorder, config::ArchKind arch, int64_t numClusters, Logger log) {
    auto parentPermuteCast = reorder.getInput().getDefiningOp<IE::PermuteCastOp>();
    if (parentPermuteCast == nullptr || !parentPermuteCast->hasOneUse()) {
        return false;
    }

    auto eltwiseParent = parentPermuteCast.getInput().getDefiningOp();
    if (eltwiseParent == nullptr || !eltwiseParent->hasOneUse() || !eltwiseParent->hasTrait<IE::EltwiseOp>()) {
        return false;
    }

    // Compose PermuteCast and Reorder permutations to get equivalent MemPermute
    auto reorderInType = mlir::cast<vpux::NDTypeInterface>(reorder.getInput().getType());
    auto reorderOutType = mlir::cast<vpux::NDTypeInterface>(reorder.getOutput().getType());
    auto perm1 = parentPermuteCast.getMemPerm();
    auto perm2 = vpux::getPermutationFromOrders(reorderInType.getDimsOrder(), reorderOutType.getDimsOrder(),
                                                reorder->getContext());
    auto composedPerm = perm2.compose(perm1);

    // Check if the composed permutation can be legally converted to MaxPool
    const auto inputType = mlir::cast<NDTypeInterface>(parentPermuteCast.getInput().getType());

    return vpux::isLegalConvertToPool(inputType, reorderOutType, eltwiseParent, composedPerm, reorder->getContext(),
                                      numClusters, "ConvertReorderToPermuteQuantize", arch, log.nest());
}

void ConvertReorderToPermuteQuantizePass::safeRunOnFunc() {
    auto func = getOperation();
    const auto arch = config::getArch(func);
    const auto numClusters = config::getTileExecutor(func).getCount();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusePermuteRewrite>(&ctx, arch, numClusters, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertReorderToPermuteQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertReorderToPermuteQuantizePass(Logger log) {
    return std::make_unique<ConvertReorderToPermuteQuantizePass>(log);
}
