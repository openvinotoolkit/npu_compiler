//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSCALARTOTENSOR
#define GEN_PASS_DEF_CONVERTSCALARTOTENSOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertScalarToTensorPass
//

class ConvertScalarToTensorPass final : public IE::impl::ConvertScalarToTensorBase<ConvertScalarToTensorPass> {
public:
    explicit ConvertScalarToTensorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GatherScalarConverter;

private:
    void safeRunOnFunc() final;
};

//
// GenericConverter
//

mlir::LogicalResult convertGeneric(mlir::Operation* origOp, mlir::ValueRange operands,
                                   mlir::ConversionPatternRewriter& rewriter, const mlir::TypeConverter& typeConverter,
                                   Logger log) {
    log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());
    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::IRMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter.convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

class GenericConverter final : public mlir::OpInterfaceConversionPattern<IE::LayerOpInterface> {
public:
    GenericConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceConversionPattern<IE::LayerOpInterface>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, ArrayRef<mlir::Value> newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto* typeConverter = this->getTypeConverter();
        VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

        return convertGeneric(origOp, newArgs, rewriter, *typeConverter, _log);
    }

private:
    Logger _log;
};

//
// GatherScalarConverter
//

class ConvertScalarToTensorPass::GatherScalarConverter final : public mlir::OpConversionPattern<IE::GatherOp> {
public:
    GatherScalarConverter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::GatherOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertScalarToTensorPass::GatherScalarConverter::matchAndRewrite(
        IE::GatherOp origOp, OpAdaptor newArgs, mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    vpux::IE::GatherOp newOp;
    if ((origOp.getAxis() != nullptr) && (origOp.getAxis().getType().cast<vpux::NDTypeInterface>().getRank() == 0)) {
        const std::array<int64_t, 1> tensorShape = {1};
        const auto shapeAttr = getIntArrayAttr(getContext(), tensorShape);

        _log.nest().trace("New axis type '{0}'", newArgs.getAxis().getType());

        auto axisReshape = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "in_axis"), newArgs.getAxis().getType(),
                                                          origOp.getAxis(), nullptr, true, shapeAttr);
        newOp = rewriter.create<IE::GatherOp>(takeOpLoc(origOp, "gather_axis"), origOp.getInput(), newArgs.getIndices(),
                                              axisReshape.getOutput(), origOp.getAxisValueAttr(), origOp.getBatchDims(),
                                              origOp.getIndicesRankAttr());
    } else {
        newOp = rewriter.create<IE::GatherOp>(takeOpLoc(origOp, "gather_axis"), origOp.getInput(), newArgs.getIndices(),
                                              nullptr, origOp.getAxisValueAttr(), origOp.getBatchDims(),
                                              origOp.getIndicesRankAttr());
    }
    _log.nest().trace("New indices type '{0}'", newArgs.getIndices().getType());

    const auto origShapeAttr = getIntArrayAttr(origOp->getContext(), getShape(origOp.getOutput()));
    auto reshapeOp = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "out_reshape"), origOp.getType(),
                                                    newOp.getOutput(), nullptr, nullptr, origShapeAttr);
    rewriter.replaceOp(origOp, reshapeOp.getOutput());

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertScalarToTensorPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto reshape = [](mlir::OpBuilder& builder, mlir::RankedTensorType dstType, mlir::ValueRange inputs,
                            mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());

        const auto outShapeAttr = builder.getI64ArrayAttr(dstType.getShape());
        return builder.createOrFold<IE::ReshapeOp>(loc, inputs.front(), nullptr, false, outShapeAttr);
    };

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface type) {
        if (type.getRank() == 0) {
            const std::array<int64_t, 1> tensorShape = {1};
            const auto newTensorType = type.changeShape(ShapeRef(tensorShape));
            return newTensorType;
        }
        return type;
    });
    typeConverter.addSourceMaterialization(reshape);
    typeConverter.addTargetMaterialization(reshape);
    typeConverter.addArgumentMaterialization(reshape);

    mlir::ConversionTarget target(ctx);

    target.addLegalOp<IE::ReshapeOp, IE::AffineReshapeOp>();
    target.addLegalOp<IE::LoopOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<IE::LayerOpInterface>(op) || mlir::isa<IE::GatherOp>(op)) {
            return true;
        }
        for (auto operand : op->getOperands()) {
            if (operand.getType().cast<vpux::NDTypeInterface>().getRank() == 0) {
                return false;
            }
        }
        for (auto result : op->getResults()) {
            if (result.getType().cast<vpux::NDTypeInterface>().getRank() == 0) {
                return false;
            }
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.debug("Failed to replace operand from scalar to tensor");
        signalPassFailure();
    }

    mlir::ConversionTarget targetGather(ctx);

    targetGather.addLegalOp<IE::ReshapeOp, IE::AffineReshapeOp>();

    targetGather.addDynamicallyLegalOp<IE::GatherOp>([&](IE::GatherOp op) {
        return typeConverter.isLegal(op);
    });

    mlir::RewritePatternSet patternsGather(&ctx);
    patternsGather.add<GatherScalarConverter>(typeConverter, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, targetGather, std::move(patternsGather)))) {
        _log.debug("Failed to replace indices from scalar to tensor");
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertScalarToTensorPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertScalarToTensorPass(Logger log) {
    return std::make_unique<ConvertScalarToTensorPass>(log);
}
