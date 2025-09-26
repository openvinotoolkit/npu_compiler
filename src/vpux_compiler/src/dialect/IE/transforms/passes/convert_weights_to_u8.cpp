//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTWEIGHTSTOU8
#define GEN_PASS_DEF_CONVERTWEIGHTSTOU8
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// changeStorageTypeToU8
//

// change storage type to U8 and shift zp, min, max attributes by the value of storage type min
mlir::quant::QuantizedType changeStorageTypeToU8(mlir::quant::QuantizedType originQType) {
    if (const auto uniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(originQType)) {
        const auto low = uniformType.getStorageTypeMin();

        return mlir::quant::UniformQuantizedType::get(
                0, getUInt8Type(uniformType.getContext()), uniformType.getExpressedType(), uniformType.getScale(),
                uniformType.getZeroPoint() - low, uniformType.getStorageTypeMin() - low,
                uniformType.getStorageTypeMax() - low);
    } else if (const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(originQType)) {
        const auto low = perAxisType.getStorageTypeMin();
        const auto zeroPoints = perAxisType.getZeroPoints();

        SmallVector<int64_t> newZeroPoints(zeroPoints.size());
        std::transform(zeroPoints.begin(), zeroPoints.end(), newZeroPoints.begin(), [low](int64_t zp) {
            return zp - low;
        });

        return mlir::quant::UniformQuantizedPerAxisType::get(
                0, getUInt8Type(perAxisType.getContext()), perAxisType.getExpressedType(), perAxisType.getScales(),
                newZeroPoints, perAxisType.getQuantizedDimension(), perAxisType.getStorageTypeMin() - low,
                perAxisType.getStorageTypeMax() - low);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}

//
// QuantizeCastRewriter
//

class QuantizeCastRewriter final : public mlir::OpConversionPattern<IE::QuantizeCastOp> {
public:
    QuantizeCastRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::QuantizeCastOp>(typeConverter, ctx), _log(log) {
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeCastRewriter::matchAndRewrite(IE::QuantizeCastOp origOp, OpAdaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto dstElemType = origOp.getDstElemType();
    auto outQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(dstElemType);
    VPUX_THROW_WHEN(outQuantizedType == nullptr, "Type must be quantized, but provided {0}", dstElemType);

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, origOp.getInput(), changeStorageTypeToU8(outQuantizedType));

    return mlir::success();
}

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpInterfaceConversionPattern<IE::LayerOpInterface> {
public:
    LayerRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceConversionPattern<IE::LayerOpInterface>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayerOpInterface origOp, ArrayRef<mlir::Value> newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::LayerOpInterface origOp, ArrayRef<mlir::Value> newOperands,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    // The result type of QuantizeCast is deduced from its attribute,
    // so we can not change its operands and results directly
    if (mlir::isa<IE::QuantizeCastOp>(origOp)) {
        return mlir::failure();
    }

    const auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == newOperands.size(), "Wrong operands size : {0}", newOperands.size());

    mlir::IRMapping mapper;
    mapper.map(origOperands, newOperands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(typeConverter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());
    return mlir::success();
}

inline bool isFloatInputQuantWeightsMixedPrecisionOperation(mlir::Operation* op) {
    // If a TransposeOp or an op with ViewLikeOpInterface can match the pattern
    // ViewLikeOp/TransposeOp -> ViewLikeOp/TransposeOp -> ... -> Conv
    // The op should also be checked here.

    //
    // [E#124175] Also a temporary solution is to match the pattern for:
    // Const -> Split -> AffineReshape -> Concat -> AffineReshape -> Conv
    // The above logic is doing a Reorder on the original Constant
    // After solving [E#124175] we can delete SplitOp and ConcatOp from the above check.

    while (mlir::isa_and_nonnull<IE::ViewLikeOpInterface, IE::TransposeOp, IE::SplitOp, IE::ConcatOp, IE::SliceOp>(
            op)) {
        if (!op->getResult(0).hasOneUse()) {
            if (llvm::any_of(op->getUsers(), [](mlir::Operation* userOp) {
                    return !isFloatInputQuantWeightsMixedPrecisionOperation(userOp);
                })) {
                return false;
            }
        }

        const auto quantizationType = mlir::dyn_cast<mlir::quant::QuantizedType>(
                mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType());
        if (quantizationType == nullptr) {
            return false;
        }

        // Check if there are any users before accessing them
        auto users = op->getUsers();
        if (users.empty()) {
            break;
        }
        op = *users.begin();
    }

    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(op)) {
        return false;
    }
    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
    const auto filterElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType()).getElementType();
    // cases of non quant input and quant wt must remain as SI8
    return !mlir::isa<mlir::quant::QuantizedType>(inputElemType) &&
           mlir::isa<mlir::quant::QuantizedType>(filterElemType);
};

//
// ConstRewriter
//

class ConstRewriter final : public mlir::OpConversionPattern<Const::DeclareOp> {
public:
    ConstRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<Const::DeclareOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConstRewriter::matchAndRewrite(Const::DeclareOp origOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    const auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    const auto origQuantType = mlir::cast<mlir::quant::QuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(origOp.getType()).getElementType());

    const auto newType = mlir::cast<vpux::NDTypeInterface>(typeConverter->convertType(origOp.getType()));
    const auto newQuantType = mlir::cast<mlir::quant::QuantizedType>(newType.getElementType());

    _log.nest().trace("Convert content from '{0}' to '{1}'", origQuantType, newQuantType);

    auto newContentAttr = origOp.getContentAttr().transform().convertElemType(newQuantType).get();
    const auto constTensor = origOp.getResult();
    const auto constUsers = constTensor.getUsers();
    const auto mixedPrecisionUsers = llvm::count_if(constUsers, [&](mlir::Operation* user) {
        return isFloatInputQuantWeightsMixedPrecisionOperation(user) && user->getOperand(1) == constTensor;
    });
    if (mixedPrecisionUsers > 0) {
        auto i8ConstOp = rewriter.create<Const::DeclareOp>(origOp.getLoc(), origOp.getType(), origOp.getContentAttr());
        for (auto* user : llvm::make_early_inc_range(constUsers)) {
            if (isFloatInputQuantWeightsMixedPrecisionOperation(user)) {
                user->setOperand(1, i8ConstOp);
            }
        }
    }
    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, newType, std::move(newContentAttr));
    return mlir::success();
}

//
// ConvertWeightsToU8Pass
//

class ConvertWeightsToU8Pass final : public IE::impl::ConvertWeightsToU8Base<ConvertWeightsToU8Pass> {
public:
    explicit ConvertWeightsToU8Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertWeightsToU8Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](vpux::NDTypeInterface tensor) {
        if (const auto quantType = mlir::dyn_cast_or_null<mlir::quant::QuantizedType>(tensor.getElementType())) {
            // Handle I8 only storage type
            if (llvm::dyn_cast<mlir::IntegerType>(quantType.getStorageType()) && quantType.isSigned() &&
                quantType.getStorageTypeIntegralWidth() == 8) {
                const auto newElemType = changeStorageTypeToU8(quantType);
                return tensor.changeElemType(newElemType);
            }
        }

        return tensor;
    });
    typeConverter.addSourceMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addTargetMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::RankedTensorType>);

    const auto isLegalConstDeclareOp = [&](Const::DeclareOp constOp) {
        // handle mixed precision of FP input and I8 weights
        const auto constTensor = constOp.getResult();
        const auto constUsers = constTensor.getUsers();

        const auto mixedPrecisionUsers = llvm::count_if(constUsers, [&](mlir::Operation* user) {
            return isFloatInputQuantWeightsMixedPrecisionOperation(user);
        });
        if (mixedPrecisionUsers == std::distance(constUsers.begin(), constUsers.end())) {
            return true;
        }
        return typeConverter.isLegal(constOp);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<Const::DeclareOp>(isLegalConstDeclareOp);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (mlir::isa<IE::LayerOpInterface>(op)) {
            if (IE::keepIntTypeForSIWeightsAsInput(op)) {
                return true;
            }

            if (!isFloatInputQuantWeightsMixedPrecisionOperation(op)) {
                return typeConverter.isLegal(op);
            }
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewriter>(typeConverter, &ctx, _log);
    patterns.add<ConstRewriter>(typeConverter, &ctx, _log);
    patterns.add<QuantizeCastRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertWeightsToU8Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertWeightsToU8Pass(Logger log) {
    return std::make_unique<ConvertWeightsToU8Pass>(log);
}
