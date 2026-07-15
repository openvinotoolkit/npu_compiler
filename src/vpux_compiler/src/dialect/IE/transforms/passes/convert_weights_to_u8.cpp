//
// Copyright (C) 2022-2026 Intel Corporation
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

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
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

mlir::LogicalResult QuantizeCastRewriter::matchAndRewrite(IE::QuantizeCastOp origOp, OpAdaptor adaptor,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto dstElemType = origOp.getDstElemType();
    auto outQuantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(dstElemType);
    VPUX_THROW_WHEN(outQuantizedType == nullptr, "Type must be quantized, but provided {0}", dstElemType);

    // Use the adapted (type-converted) input instead of origOp.getInput() to avoid
    // residual unrealized_conversion_cast when QuantizeCast ops form a chain and the
    // upstream QuantizeCast has already been converted from i8 to u8
    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, adaptor.getInput(),
                                                    changeStorageTypeToU8(outQuantizedType));

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

inline bool keepIntTypeQuantization(mlir::Operation* op) {
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
                    return keepIntTypeQuantization(userOp);
                })) {
                return true;
            }
        }
        const auto quantizationType = mlir::dyn_cast<mlir::quant::QuantizedType>(
                mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType());
        if (quantizationType == nullptr) {
            return true;
        }

        // Check if there are any users before accessing them and if there are no users there is not the case of a mixed
        // precision usecase
        auto users = op->getUsers();
        if (users.empty()) {
            return true;
        }
        op = *users.begin();
    }

    auto isValidSignedUsecase = [](mlir::Operation* op) -> bool {
        // Consider Convolution, GroupConvolution and MatMul as valid usecase for mixed precision
        if (mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::MatMulOp>(op)) {
            // Check if the quantization is compatible with mixed precision usecase
            auto isQuantizationFusable = [&op](const mlir::quant::QuantizedType filterType) -> bool {
                const auto isPerChannelQuantType = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(filterType);
                const auto isPerTensorQuantType = mlir::isa<mlir::quant::UniformQuantizedType>(filterType);
                const auto isSymmetricQuant = IE::isSymmetricQuantType(filterType);
                auto moduleOp = getModuleOp(op);
                const auto isAsymmetricPerChannelSupported = config::asymmetricPerChannelZeroPointSupported(moduleOp);
                const auto isAsymmetricPerTensorSupported = config::asymmetricPerTensorZeroPointSupported(moduleOp);
                return (isPerChannelQuantType && (isAsymmetricPerChannelSupported || isSymmetricQuant)) ||
                       (isPerTensorQuantType && (isAsymmetricPerTensorSupported || isSymmetricQuant));
            };
            // Cases of non quant input and quant wt must remain as Signed
            const auto activationProducerOp = op->getOperand(0).getDefiningOp();
            const auto filterProducerOp = op->getOperand(1).getDefiningOp();
            if ((activationProducerOp == nullptr || !mlir::isa<IE::DequantizeOp>(activationProducerOp)) &&
                filterProducerOp != nullptr && mlir::isa<IE::DequantizeOp, IE::DynamicDequantizeOp>(filterProducerOp)) {
                const auto filterElemType =
                        mlir::cast<vpux::NDTypeInterface>(filterProducerOp->getOperand(0).getType()).getElementType();
                const auto quantFilterElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(filterElemType);
                return quantFilterElemType != nullptr && isQuantizationFusable(quantFilterElemType);
            }
            const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
            const auto filterElemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(1).getType()).getElementType();
            const auto quantFilterElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(filterElemType);
            return !mlir::isa<mlir::quant::QuantizedType>(inputElemType) && quantFilterElemType != nullptr &&
                   isQuantizationFusable(quantFilterElemType);
        }

        // GatherOp uses its first operand as an embedding table: keep the I8 storage type.
        // DynamicDequantize’s SHAVE kernel does not support U8 quantized inputs.
        if (mlir::isa<IE::GatherOp, IE::DynamicDequantizeOp>(op)) {
            return true;
        }

        return op->getNumOperands() == 1;
    };

    if (mlir::isa<IE::DequantizeOp>(op)) {
        return llvm::all_of(op->getUsers(), isValidSignedUsecase);
    }

    return isValidSignedUsecase(op);
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
        // Check both cases: Const->(*Group)ConvOp and Const->DequantizeOp->(*Group)ConvOp
        if (auto dequantizeOpUser = mlir::dyn_cast_if_present<IE::DequantizeOp>(user)) {
            const auto dequantizeResult = dequantizeOpUser.getResult();
            const auto dequantizeUsers = dequantizeResult.getUsers();
            return llvm::count_if(dequantizeUsers, [&](mlir::Operation* dqUser) {
                       return keepIntTypeQuantization(dqUser) && dqUser->getNumOperands() > 1 &&
                              dqUser->getOperand(1) == dequantizeResult;
                   }) > 0;
        }
        return keepIntTypeQuantization(user) && user->getNumOperands() > 1 && user->getOperand(1) == constTensor;
    });
    if (mixedPrecisionUsers > 0) {
        auto i8ConstOp = rewriter.create<Const::DeclareOp>(origOp.getLoc(), origOp.getType(), origOp.getContentAttr());
        for (auto* user : llvm::make_early_inc_range(constUsers)) {
            if (keepIntTypeQuantization(user)) {
                // If user is Dequantize set operand 0 with I8 constant else set directly operand 1 of the
                // (*Group)ConvOp
                if (mlir::isa_and_present<IE::DequantizeOp>(user)) {
                    user->setOperand(0, i8ConstOp);
                } else {
                    user->setOperand(1, i8ConstOp);
                }
            }
        }
    }
    rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, newType, std::move(newContentAttr));
    return mlir::success();
}

bool keepIntTypeForSIResultOrConcatInput(mlir::Operation* op) {
    std::function<bool(mlir::Operation*)> searchForReturn = [&](mlir::Operation* currentOp) -> bool {
        for (auto user : currentOp->getUsers()) {
            if (mlir::isa<mlir::func::ReturnOp>(user)) {
                const auto resultType = mlir::cast<vpux::NDTypeInterface>(currentOp->getResult(0).getType());
                if (resultType.getElementType().isSignedInteger()) {
                    // Found int result
                    return true;
                }
            } else if (IE::isPureViewOp(user) ||
                       mlir::isa<IE::QuantizeCastOp, IE::ConcatOp, IE::SliceOp, IE::TransposeOp>(user)) {
                // If a Concat has a signed-integer block argument operand,
                // all its inputs must stay as signed integers to match.
                if (auto concatOp = mlir::dyn_cast<IE::ConcatOp>(user)) {
                    const auto hasIntBlockArg = llvm::any_of(concatOp->getOperands(), [](mlir::Value operand) {
                        if (!mlir::isa<mlir::BlockArgument>(operand)) {
                            return false;
                        }
                        const auto elemType = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getElementType();
                        return elemType.isSignedInteger();
                    });
                    if (hasIntBlockArg) {
                        return true;
                    }
                }
                if (searchForReturn(user)) {
                    return true;
                }
            }
        }

        return false;
    };

    return searchForReturn(op);
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

    const auto isLegalConstDeclareOp = [&](Const::DeclareOp constOp) {
        // handle mixed precision of FP input and I8 weights
        const auto constTensor = constOp.getResult();
        const auto constUsers = constTensor.getUsers();

        const auto mixedPrecisionUsers = llvm::count_if(constUsers, [&](mlir::Operation* user) {
            return keepIntTypeQuantization(user);
        });
        if (mixedPrecisionUsers == std::distance(constUsers.begin(), constUsers.end())) {
            return true;
        }
        return typeConverter.isLegal(constOp.getOperation());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<Const::DeclareOp>(isLegalConstDeclareOp);
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (mlir::isa<IE::LayerOpInterface>(op)) {
            if (keepIntTypeForSIResultOrConcatInput(op) || IE::keepIntTypeForSIWeightsAsInputOrConst(op)) {
                return true;
            }

            if (!keepIntTypeQuantization(op)) {
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
