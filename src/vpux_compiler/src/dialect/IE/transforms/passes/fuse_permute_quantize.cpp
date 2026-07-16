//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEPERMUTEQUANTIZE
#define GEN_PASS_DEF_FUSEPERMUTEQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
class FusePermuteQuantizeBase : public mlir::OpRewritePattern<IE::ReorderOp> {
public:
    FusePermuteQuantizeBase(mlir::MLIRContext* ctx, const bool dpuOnly, Logger log)
            : mlir::OpRewritePattern<IE::ReorderOp>(ctx, benefitHigh), _dpuOnly(dpuOnly), _log(log) {
        setDebugName("FusePermuteQuantizeRewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ReorderOp origOp, mlir::PatternRewriter& rewriter) const final;
    virtual bool isLegalPattern(IE::ReorderOp origOp) const = 0;
    virtual void replaceByNewOp(mlir::Operation* opNce, mlir::Value input, mlir::PatternRewriter& rewriter) const = 0;
    virtual mlir::Type getNceOutType(mlir::Operation* opNce) const = 0;

private:
    bool isCompatibleWithDPU(mlir::Type addInput, mlir::Type addOutput) const;
    const bool _dpuOnly;
    Logger _log;
};

// Check if an operation has overlapped tiled inputs when tiling the output
// Example: For a Conv with kernel=3x3, stride=1x1, no padding, input=1x16x8x8, output=1x16x6x6.
// When splitting output into two tiles along height (1x16x3x6 each), the input tiles become:
//   - Tile 1: 1x16x0:5x8 (rows 0-4, needs 5 rows to produce 3 output rows)
//   - Tile 2: 1x16x3:8x8 (rows 3-7, needs 5 rows to produce 3 output rows)
// Rows 3-4 are overlapped between tiles (HALO region) because kernel_size > stride.
bool hasOverlappedInput(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }

    mlir::Value filter;
    mlir::ArrayAttr stridesAttr;

    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op)) {
        filter = convOp.getFilter();
        stridesAttr = convOp.getStrides();
    } else if (auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
        filter = groupConvOp.getFilter();
        stridesAttr = groupConvOp.getStrides();
    } else if (auto transposedConvOp = mlir::dyn_cast<IE::TransposedConvolutionOp>(op)) {
        filter = transposedConvOp.getFilter();
        stridesAttr = transposedConvOp.getStrides();
    } else {
        // Not a supported convolution operation
        return false;
    }

    const auto filterShape = getShape(filter);
    const auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto SY = strides[Dims4D::Strides::Y.ind()];
    const auto SX = strides[Dims4D::Strides::X.ind()];

    return (KY > SY) || (KX > SX);
}

mlir::LogicalResult FusePermuteQuantizeBase::matchAndRewrite(IE::ReorderOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    if (origOp.getOutput().use_empty()) {
        return mlir::failure();
    }

    // check reorder and nce pattern
    if (!isLegalPattern(origOp)) {
        return mlir::failure();
    }

    auto opNce = *origOp.getOutput().getUsers().begin();
    const auto inType = mlir::cast<vpux::NDTypeInterface>(opNce->getOperand(0).getType()).getElementType();
    const auto outType = mlir::cast<vpux::NDTypeInterface>(opNce->getResult(0).getType()).getElementType();
    if (!(inType.isF16() && mlir::isa<mlir::quant::QuantizedType>(outType))) {
        return mlir::failure();
    }

    // check uniform quantize
    const auto qType = mlir::cast<mlir::quant::QuantizedType>(outType);
    if (!mlir::isa<mlir::quant::UniformQuantizedType>(qType)) {
        return mlir::failure();
    }

    // check if a legal PermuteQuantize pattern starts with a trivial Reorder and not beneficial to convert
    auto isTrivialReorderAndNotBeneficialConvertToPermuteQuantize = [](IE::ReorderOp reorderOp) -> bool {
        if (!isTrivialReorder(reorderOp)) {
            return false;
        }

        // TODO: E#200373 Fix it to remove the following beneficial ConvertToPermuteQuantize case

        // At this point, we got a legal PermuteQuantize pattern starting with a trivial Reorder:
        //     TrivialReorder -> Add / AvgPool -> (QuantizeCast) -> User
        // If we convert it, we will later get:
        //     NCE.Permute -> (QuantizeCast) -> User
        // Otherwise, we will later get:
        //     PermuteCast -> ShapeCast -> Add / AvgPool -> ShapeCast -> (QuantizeCast) -> User
        // The added ShapeCast will speed up the quantization process by Add/AvgPool,
        // However, it will also break the sibling connection between the quantization op and its potential NCE user.
        // If the user has overlapped inputs (HALO), the quantization op (Add/AvgPool) will not consider it,
        // and thus introduces additional copies which may outweigh the benefit of faster quantization process.

        // Traverse through Add or AvgPool
        auto quantizeOp = *reorderOp.getOutput().getUsers().begin();
        VPUX_THROW_UNLESS(mlir::isa<IE::AddOp>(quantizeOp) || mlir::isa<IE::AvgPoolOp>(quantizeOp),
                          "Expected quantizeOp to be IE::AddOp or IE::AvgPoolOp");
        mlir::Value output = quantizeOp->getResult(0);

        // Traverse through single user QuantizeCastOps
        while (output) {
            if (!output.hasOneUse()) {
                return true;
            }
            auto user = *output.getUsers().begin();

            if (mlir::isa<IE::QuantizeCastOp>(user)) {
                output = user->getResult(0);
            } else if (hasOverlappedInput(user)) {
                // Found a user has overlapped inputs (HALO)
                return false;
            } else {
                return true;
            }
        }
        return true;
    };
    if (isTrivialReorderAndNotBeneficialConvertToPermuteQuantize(origOp)) {
        return mlir::failure();
    }

    // check and add pass for verified orders and scenarios
    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    auto outOrder = DimsOrder::fromValue(origOp.getOutput());
    if (!((inOrder == DimsOrder::NCHW) && (outOrder == DimsOrder::NHWC))) {
        return mlir::failure();
    }

    const auto iExpType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto oExpType = mlir::cast<vpux::NDTypeInterface>(opNce->getResult(0).getType());
    if (!((iExpType.getRank() == 4) && (oExpType.getRank() == 4))) {
        return mlir::failure();
    }

    const ShapeRef inShape = iExpType.getShape();
    if (!IE::isBeneficialConvertToPermuteQuantize(inShape)) {
        return mlir::failure();
    }

    // If subgraph is SpaceToDepth->Reorder->Add(Quantize),
    // conversion to SpaceToDepthDMA->Swkernel(PermuteQuantize) is much slower than
    // conversion to SpaceToDepth->MemPermute(Reorder)->NCEEltwise(Quantize) which
    // later will be fused as SpaceToDepthDMA->NCEEltwise(Quantize).
    // In this case, disable fuse of Reorder and Add as PermuteQuantize here.
    if (auto s2dOp = origOp.getInput().getDefiningOp<IE::SpaceToDepthOp>()) {
        return mlir::failure();
    }

    if (_dpuOnly && !isCompatibleWithDPU(opNce->getOperand(0).getType(), opNce->getResult(0).getType())) {
        return mlir::failure();
    }

    auto memPermAttr = mlir::AffineMapAttr::get(getPermutationFromOrders(inOrder, outOrder, origOp->getContext()));
    SmallVector<int64_t> noPadBeginEnd(inOrder.numDims(), 0);
    const auto& ctx = origOp.getContext();

    auto permQuantOutType = getNceOutType(opNce);
    const auto permQuantElemType = mlir::cast<vpux::NDTypeInterface>(permQuantOutType).getElementType();
    const auto dstElemTypeAttr = mlir::TypeAttr::get(permQuantElemType);
    const auto permQuantLoc = takeOpLoc(origOp, "PermuteQuantize");
    auto permuteQuantizeOp = rewriter.create<IE::PermuteQuantizeOp>(
            permQuantLoc, permQuantOutType, origOp.getInput(), origOp.getDstOrderAttr(), memPermAttr, dstElemTypeAttr,
            getIntArrayAttr(ctx, noPadBeginEnd), getIntArrayAttr(ctx, noPadBeginEnd));

    replaceByNewOp(opNce, permuteQuantizeOp.getOutput(), rewriter);

    return mlir::success();
}

// ======================================================================================
// FusePermuteQuantizeForAdd
//   FusePermuteQuantizeForAdd -> [Reorder -> Add -> QuantizeCastOp] -> [PermuteQuantize
//   -> QuantizeCastOp]

class FusePermuteQuantizeForAdd final : public FusePermuteQuantizeBase {
public:
    FusePermuteQuantizeForAdd(mlir::MLIRContext* ctx, const bool dpuOnly, Logger log)
            : FusePermuteQuantizeBase(ctx, dpuOnly, log) {
    }

public:
    bool isLegalPattern(IE::ReorderOp origOp) const override;
    void replaceByNewOp(mlir::Operation* opNce, mlir::Value input, mlir::PatternRewriter& rewriter) const override;
    mlir::Type getNceOutType(mlir::Operation* opNce) const override;
};

bool FusePermuteQuantizeForAdd::isLegalPattern(IE::ReorderOp origOp) const {
    return IE::isLegalReorderAddPattern(origOp);
}

mlir::Type FusePermuteQuantizeForAdd::getNceOutType(mlir::Operation* opNce) const {
    // QuantizeToAddRewriter multiplies output scale by 2. It is necessary to cancel out this factor.
    return IE::rescaleUniformQuantizedType(opNce->getResult(0).getType(), 0.5);
}

void FusePermuteQuantizeForAdd::replaceByNewOp(mlir::Operation* opNce, mlir::Value input,
                                               mlir::PatternRewriter& rewriter) const {
    // IE.PermuteQuantize must have quantization parameters from the original IE.Quantize operation.
    // In some cases IE.QuantizeCast which follows IE.Add can contain dstElemType which differs from that
    // IE.Quantize.
    // For example, one IE.QuantizeCast may appear after IE.FakeQuantize gets split into:
    // IE.Quantize qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // Another IE.QuantizeCast will be inserted into graph after QuantizeToAddRewriter:
    // IE.Add qType0 -> IE.QuantizeCast qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // Such chain of two consecutive IE.QuantizeCast will be fused into one:
    // IE.Add qType0 -> IE.QuantizeCast qType2 -> IE.Dequantize
    // In that case, qType1 must be set for IE.PermuteQuantize.
    // IE.QuantizeCast to qType2 must remain in the graph to maintain the integrity:
    // IE.PermuteQuantize qType1 -> IE.QuantizeCast qType2 -> IE.Dequantize
    for (auto user : make_early_inc_range(opNce->getResult(0).getUsers())) {
        auto originalQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(user);
        auto quantCast = rewriter.createOrFold<IE::QuantizeCastOp>(opNce->getLoc(), input,
                                                                   originalQuantizeCast.getDstElemTypeAttr());
        rewriter.replaceOp(originalQuantizeCast, quantCast);
    }
}

// ======================================================================================
// FusePermuteQuantizeForAvgPool
//   FusePermuteQuantizeForAvgPool -> [Expand -> Reorder -> AvgPool] -> [PermuteQuantize]

class FusePermuteQuantizeForAvgPool final : public FusePermuteQuantizeBase {
public:
    FusePermuteQuantizeForAvgPool(mlir::MLIRContext* ctx, const bool dpuOnly, Logger log)
            : FusePermuteQuantizeBase(ctx, dpuOnly, log) {
    }

public:
    bool isLegalPattern(IE::ReorderOp origOp) const override;
    void replaceByNewOp(mlir::Operation* opNce, mlir::Value input, mlir::PatternRewriter& rewriter) const override;
    mlir::Type getNceOutType(mlir::Operation* opNce) const override;
};

bool FusePermuteQuantizeForAvgPool::isLegalPattern(IE::ReorderOp origOp) const {
    return IE::isLegalReorderAvgPoolPattern(origOp);
}

mlir::Type FusePermuteQuantizeForAvgPool::getNceOutType(mlir::Operation* opNce) const {
    return opNce->getResult(0).getType();
}

void FusePermuteQuantizeForAvgPool::replaceByNewOp(mlir::Operation* opNce, mlir::Value input,
                                                   mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOp(opNce, input);
}

bool FusePermuteQuantizeBase::isCompatibleWithDPU(mlir::Type addInput, mlir::Type addOutput) const {
    auto inType = mlir::cast<vpux::NDTypeInterface>(addInput);
    auto outType = mlir::cast<vpux::NDTypeInterface>(addOutput);
    const auto inElemType = inType.getElementType();
    if (!inElemType.isF16()) {
        return false;
    }
    const auto outElemType = outType.getElementType();
    if (!outElemType.isF16() && !mlir::isa<mlir::quant::UniformQuantizedType>(outElemType)) {
        return false;
    }
    const ShapeRef inShape = inType.getShape();
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inElemType);
    if (!IE::isODUPermuteEffectiveForShape(inShape, inAlignment)) {
        return false;
    }
    const ShapeRef outShape = outType.getShape();
    const auto outAlignment = VPU::NCEInvariant::getAlignment(outElemType);
    if (!IE::isODUPermuteEffectiveForShape(outShape, outAlignment)) {
        return false;
    }

    return true;
}

//
// FusePermuteQuantizePass
//

class FusePermuteQuantizePass final : public IE::impl::FusePermuteQuantizeBase<FusePermuteQuantizePass> {
public:
    explicit FusePermuteQuantizePass(const bool dpuOnly, Logger log): _dpuOnly(dpuOnly) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _dpuOnly;
};

mlir::LogicalResult FusePermuteQuantizePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (dpuOnly.hasValue()) {
        _dpuOnly = dpuOnly.getValue();
    }

    return mlir::success();
}

void FusePermuteQuantizePass::safeRunOnFunc() {
    // TODO: #70647

    auto& ctx = getContext();
    auto func = getOperation();

    // dpuOnly flag means that target platform supports only DPU implementation of PermuteQuantize.
    // In that case PermuteQuantize fusion has some limitations:
    // 1. Only NCHW to NHWC permutation is supported
    // 2. Only float16 inputs and quantized outputs are supported.
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FusePermuteQuantizeForAvgPool>(&ctx, _dpuOnly, _log);
    patterns.add<FusePermuteQuantizeForAdd>(&ctx, _dpuOnly, _log);

    mlir::ConversionTarget target(ctx);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFusePermuteQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createFusePermuteQuantizePass(const bool dpuOnly, Logger log) {
    return std::make_unique<FusePermuteQuantizePass>(dpuOnly, log);
}
