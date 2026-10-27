//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/eltwise_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <functional>

namespace vpux::IE {
#define GEN_PASS_DECL_ELTWISEFAKEQUANTIZEFUSION
#define GEN_PASS_DEF_ELTWISEFAKEQUANTIZEFUSION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// EltwiseFakeQuantizeFusion
//

template <typename ConcreteOp>
class EltwiseFakeQuantizeFusion final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    EltwiseFakeQuantizeFusion(mlir::MLIRContext* ctx, const FuncRef<float(float, float)> compute, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _compute(compute), _log(log) {
        this->setDebugName("EltwiseFakeQuantizeFusion");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    FuncRef<float(float, float)> _compute;
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult EltwiseFakeQuantizeFusion<ConcreteOp>::matchAndRewrite(IE::FakeQuantizeOp fakeQuantizeOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    //
    //  Pattern matched:
    //
    //                                                +------------+
    //                                                |Scalar Const|
    //                                                +------------+
    //                                                       |
    //     +-----------------------------+   +--------------------------------+
    //     | non LayerWithPostOpInterface|   |      optional per tensor       |
    //     |     input producer          |   | FakeQuantizeOp or DequantizeOp |
    //     +-----------------------------+   +--------------------------------+
    //           |                                     |
    //     +------------+                              |
    //     | ConcreteOp |------------------------------+
    //     +------------+
    //       |  in_L in_H out_L out_H
    //       |    |    |     |     |
    //     +---------------------------+
    //     | per tensor FakeQuantizeOp |
    //     +---------------------------+
    //
    //  Replace with a single FakeQuantize with input low and high adjusted
    //
    //   +-----------------------------+
    //   | non LayerWithPostOpInterface|
    //   |     input producer          |            out_H
    //   +-----------------------------+       out_L  |
    //          |           in_H -,+,*,/ scalar  |    |
    //          | in_L -,+,*,/ scalar  |         |    |
    //          |            |         |         |    |
    //     +-------------------------------------------+
    //     |        per tensor FakeQuantizeOp          |
    //     +-------------------------------------------+
    //

    auto inLowConst = fakeQuantizeOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fakeQuantizeOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fakeQuantizeOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fakeQuantizeOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        _log.nest().trace("Got non constant parameters for FakeQuantize '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    const auto& inLowContentAttr = inLowConst.getContentAttr();
    const auto& inHighContentAttr = inHighConst.getContentAttr();
    const auto& outLowContentAttr = outLowConst.getContentAttr();
    const auto& outHighContentAttr = outHighConst.getContentAttr();
    if (!inLowContentAttr.isSplat() || !inHighContentAttr.isSplat() || !outLowContentAttr.isSplat() ||
        !outHighContentAttr.isSplat()) {
        _log.nest().trace("Got non scalar FakeQuantize parameters at '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    auto concreteParentOp = fakeQuantizeOp.getInput().getDefiningOp<ConcreteOp>();
    if (concreteParentOp == nullptr) {
        _log.nest().trace("The FakeQuantize input must be ConcreteOp at '{0}'", fakeQuantizeOp.getLoc());
        return mlir::failure();
    }

    bool isRhsScalar = vpux::VPU::isEltwiseLhsActivation<ConcreteOp>(concreteParentOp);
    auto scalarInput = isRhsScalar ? concreteParentOp.getInput2() : concreteParentOp.getInput1();
    auto nonScalarInput =
            scalarInput == concreteParentOp.getInput1() ? concreteParentOp.getInput2() : concreteParentOp.getInput1();
    auto nonScalarProducerOp = nonScalarInput.getDefiningOp();

    if (auto layerWithPostOpInterface = mlir::dyn_cast_if_present<IE::LayerWithPostOpInterface>(nonScalarProducerOp)) {
        // The ConcreteOp is later fused as bias if producer op is executed on DPU
        if (layerWithPostOpInterface.supportsFuseBiasScale()) {
            _log.nest().trace("The ConcreteOp input must not inherit the LayerWithPostOpInterface with supported bias "
                              "and static scale at '{0}'",
                              fakeQuantizeOp.getLoc());
            return mlir::failure();
        }
    }

    // Subtract/Divide Op can only be fused if constant is on the 2nd input
    if (mlir::isa<IE::SubtractOp, IE::DivideOp>(concreteParentOp.getOperation()) && !isRhsScalar) {
        _log.nest().trace("The SubtractOp/DivideOp activation needs to be LHS");
        return mlir::failure();
    }

    auto scalarProducerOp = scalarInput.getDefiningOp();
    mlir::Operation* maybeDequantOrFQ = nullptr;
    if (mlir::isa_and_present<IE::FakeQuantizeOp, IE::DequantizeOp, IE::DynamicDequantizeOp>(scalarProducerOp)) {
        maybeDequantOrFQ = scalarProducerOp;
        scalarProducerOp = scalarProducerOp->getOperand(0).getDefiningOp();
    }
    auto scalarConstantOp = mlir::dyn_cast_if_present<Const::DeclareOp>(scalarProducerOp);

    if (scalarConstantOp == nullptr) {
        _log.nest().trace("Second input of Concrete is not a DeclareOp at '{0}'", fakeQuantizeOp->getLoc());
        return mlir::failure();
    }

    const auto& scalarConstantContentAttr = scalarConstantOp.getContentAttr();
    if (!scalarConstantContentAttr.isSplat()) {
        _log.nest().trace("Constant Concrete input must be scalar at '{0}'", fakeQuantizeOp.getLoc());
        return mlir::failure();
    }

    auto scalarConstantValue = scalarConstantContentAttr.fold().template getSplatValue<float>();
    if (auto scalarInputFqOp = mlir::dyn_cast_if_present<IE::FakeQuantizeOp>(maybeDequantOrFQ)) {
        auto concreteFQInLowConst = scalarInputFqOp.getInputLow().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQInHighConst = scalarInputFqOp.getInputHigh().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQOutLowConst = scalarInputFqOp.getOutputLow().template getDefiningOp<Const::DeclareOp>();
        auto concreteFQOutHighConst = scalarInputFqOp.getOutputHigh().template getDefiningOp<Const::DeclareOp>();
        if (concreteFQInLowConst == nullptr || concreteFQInHighConst == nullptr || concreteFQOutLowConst == nullptr ||
            concreteFQOutHighConst == nullptr) {
            _log.nest().trace("Got non constant parameters of FakeQuantize '{0}'", scalarInputFqOp->getLoc());
            return mlir::failure();
        }
        const auto& concreteFQInLowContentAttr = concreteFQInLowConst.getContentAttr();
        const auto& concreteFQInHighContentAttr = concreteFQInHighConst.getContentAttr();
        const auto& concreteFQOutLowContentAttr = concreteFQOutLowConst.getContentAttr();
        const auto& concreteFQOutHighContentAttr = concreteFQOutHighConst.getContentAttr();
        if (!concreteFQInLowContentAttr.isSplat() || !concreteFQInHighContentAttr.isSplat() ||
            !concreteFQOutLowContentAttr.isSplat() || !concreteFQOutHighContentAttr.isSplat()) {
            _log.nest().trace("Got non scalar fake quantize range '{0}'", scalarInputFqOp->getLoc());
            return mlir::failure();
        }
        auto concreteInLowValue = concreteFQInLowContentAttr.fold().template getSplatValue<float>();
        auto concreteInHighValue = concreteFQInHighContentAttr.fold().template getSplatValue<float>();
        auto concreteOutLowValue = concreteFQOutLowContentAttr.fold().template getSplatValue<float>();
        auto concreteOutHighValue = concreteFQOutHighContentAttr.fold().template getSplatValue<float>();

        if (const auto levels = scalarInputFqOp.getLevels()) {
            scalarConstantValue = fakeQuantize(scalarConstantValue, concreteInLowValue, concreteInHighValue,
                                               concreteOutLowValue, concreteOutHighValue, *levels);
        } else {
            _log.nest().trace("FakeQuantize without levels at '{0}'", scalarInputFqOp->getLoc());
            return mlir::failure();
        }
    } else if (auto scalarInputDqOp = mlir::dyn_cast_if_present<IE::DequantizeOp>(maybeDequantOrFQ)) {
        if (scalarInputDqOp == nullptr) {
            _log.nest().trace("Dequantize input is not a Const::DeclareOp at '{0}'", scalarInputDqOp->getLoc());
            return mlir::failure();
        }
        auto constType = scalarConstantOp.getContentAttr().getType();
        auto elemType = constType.getElementType();
        auto quantElemType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);

        if (quantElemType == nullptr) {
            _log.nest().trace("Dequantize input is not UniformQuantizedType at '{0}'", scalarInputDqOp->getLoc());
            return mlir::failure();
        }
        int64_t concreteZeroPointValue = 0;
        const auto concreteScaleValue = quantElemType.getScale();
        if (quantElemType.getZeroPoint()) {
            concreteZeroPointValue = quantElemType.getZeroPoint();
        }

        scalarConstantValue = dequantize(scalarConstantValue, concreteScaleValue, concreteZeroPointValue);
    } else if (auto scalarInputDynDeqOp = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(maybeDequantOrFQ)) {
        auto scaleConst = scalarInputDynDeqOp.getScale().getDefiningOp<Const::DeclareOp>();
        if (scaleConst == nullptr || !scaleConst.getContentAttr().isSplat()) {
            _log.nest().trace("DynamicDequantize scale is not a scalar constant at '{0}'",
                              scalarInputDynDeqOp->getLoc());
            return mlir::failure();
        }
        const auto concreteScaleValue = scaleConst.getContentAttr().fold().getSplatValue<float>();

        float concreteZeroPointValue = 0.0f;
        if (auto zp = scalarInputDynDeqOp.getZp()) {
            auto zpConst = zp.getDefiningOp<Const::DeclareOp>();
            if (zpConst == nullptr || !zpConst.getContentAttr().isSplat()) {
                _log.nest().trace("DynamicDequantize zero-point is not a scalar constant at '{0}'",
                                  scalarInputDynDeqOp->getLoc());
                return mlir::failure();
            }
            concreteZeroPointValue = zpConst.getContentAttr().fold().getSplatValue<float>();
        }

        scalarConstantValue = (scalarConstantValue - concreteZeroPointValue) * concreteScaleValue;
    }

    // NOTE: Fallback for when adaptive stripping is disabled
    // In case of cst-FQ-Mul, fusing Mul-FQ will generate redundant DQ-Q pair.
    // This pair can be optimized by adaptive-stripping.
    //   (FuseOutstandingDequant removes DQ, and ConvertToMixedPrecision removes Q)
    // But if adaptive-stripping is disabled, it's better to not fuse Mul,
    //   and Mul will convert into GroupConv and fuse any redundant DQ and Q.

    auto moduleOp = getModuleOp(fakeQuantizeOp);
    auto isAdaptiveStrippingEnabled = config::hasEnableAdaptiveStripping(moduleOp);
    if (mlir::isa<IE::MultiplyOp>(concreteParentOp) && maybeDequantOrFQ != nullptr && !isAdaptiveStrippingEnabled) {
        _log.nest().trace("Do not fuse Mul-FQ/DQ when adaptive-stripping disabled "
                          "and Multiply's constant input has FakeQuantize or Dequantize '{0}'",
                          maybeDequantOrFQ->getLoc());
        return mlir::failure();
    }

    auto oldInLowValue = inLowContentAttr.fold().getSplatValue<float>();
    auto oldInHighValue = inHighContentAttr.fold().getSplatValue<float>();
    auto newInLowValue = _compute(oldInLowValue, scalarConstantValue);
    auto newInHighValue = _compute(oldInHighValue, scalarConstantValue);
    auto newInLowConst = Const::createFloatConst(
            rewriter, fakeQuantizeOp.getLoc(), mlir::cast<mlir::RankedTensorType>(inLowConst.getType()), newInLowValue);
    auto newInHighConst =
            Const::createFloatConst(rewriter, fakeQuantizeOp.getLoc(),
                                    mlir::cast<mlir::RankedTensorType>(inHighConst.getType()), newInHighValue);

    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(fakeQuantizeOp, nonScalarInput, newInLowConst, newInHighConst,
                                                    fakeQuantizeOp.getOutputLow(), fakeQuantizeOp.getOutputHigh(),
                                                    fakeQuantizeOp.getLevelsAttr(), fakeQuantizeOp.getLowFpTypeAttr(),
                                                    fakeQuantizeOp.getAutoBroadcastAttr());
    return mlir::success();
}

//
// EltwiseFakeQuantizeFusionPass
//

class EltwiseFakeQuantizeFusionPass final :
        public IE::impl::EltwiseFakeQuantizeFusionBase<EltwiseFakeQuantizeFusionPass> {
public:
    explicit EltwiseFakeQuantizeFusionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void EltwiseFakeQuantizeFusionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    // Because the Eltwise operation with one scalar input is the producer op for the FakeQuantize below the
    // mathematical operation that will fuse the Eltwise operation in FakeQuantize input range will be exactly the
    // opposite one. Example: Subtract(scalar input = 3) - > FakeQuantize(inLow = -5, inHigh = 7, ...) will convert to
    // FakeQuantize(inLow = -2, inHigh = 10, ...), the effect if Subtract scalar being incorporated in the FakeQuantize
    // input range
    patterns.add<EltwiseFakeQuantizeFusion<IE::AddOp>>(&ctx, std::minus<float>(), _log);
    patterns.add<EltwiseFakeQuantizeFusion<IE::SubtractOp>>(&ctx, std::plus<float>(), _log);
    // TODO: E#129083
    patterns.add<EltwiseFakeQuantizeFusion<IE::MultiplyOp>>(&ctx, std::divides<float>(), _log);
    patterns.add<EltwiseFakeQuantizeFusion<IE::DivideOp>>(&ctx, std::multiplies<float>(), _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createEltwiseFakeQuantizeFusionPass(Logger log) {
    return std::make_unique<EltwiseFakeQuantizeFusionPass>(log);
}
