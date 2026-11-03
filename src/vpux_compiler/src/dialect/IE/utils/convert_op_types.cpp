//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/locations.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace IE;

namespace {

//
// ConvertOpTypes
//

class ConvertOpTypes final : public mlir::ConversionPattern {
public:
    ConvertOpTypes(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, vpux::Logger log,
                   vpux::IE::PreserveOperandCb shouldPreserveOperand)
            : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, vpux::benefitHigh, ctx),
              _log(log),
              _preserveOperand(std::move(shouldPreserveOperand)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    vpux::Logger _log;
    vpux::IE::PreserveOperandCb _preserveOperand;
};

mlir::LogicalResult ConvertOpTypes::matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp->getLoc());

    const auto* converter = getTypeConverter();
    VPUX_THROW_WHEN(converter == nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    SmallVector<mlir::Value> mappedOperands;
    mappedOperands.reserve(origOperands.size());
    for (const auto& operandItem : llvm::enumerate(origOperands)) {
        if (_preserveOperand != nullptr && _preserveOperand(origOp, static_cast<unsigned>(operandItem.index()))) {
            mappedOperands.push_back(operandItem.value());
        } else {
            mappedOperands.push_back(operands[operandItem.index()]);
        }
    }

    mlir::IRMapping mapper;
    mapper.map(origOperands, mappedOperands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// ConvertFuncSignature
//
// Custom func::FuncOp signature conversion that lets selected arguments keep a
// forced (original-precision) type, while every other argument is converted by
// the TypeConverter. Used when a caller needs to keep some inputs at higher
// precision (e.g. f32 Interpolate scales).
class ConvertFuncSignature final : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
    ConvertFuncSignature(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx,
                         vpux::IE::PreserveArgTypeCb getPreserveArgType, vpux::Logger log)
            : mlir::OpConversionPattern<mlir::func::FuncOp>(typeConverter, ctx),
              _preserveArg(std::move(getPreserveArgType)),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final {
        const auto* converter = getTypeConverter();
        VPUX_THROW_WHEN(converter == nullptr, "TypeConverter was not set");

        const auto funcType = funcOp.getFunctionType();

        mlir::TypeConverter::SignatureConversion sigConversion(funcType.getNumInputs());
        for (const auto& funcInputType : llvm::enumerate(funcType.getInputs())) {
            const auto inputIdx = static_cast<unsigned>(funcInputType.index());
            if (auto forcedType = _preserveArg(funcOp, inputIdx)) {
                // Keep this argument at the forced/original type.
                sigConversion.addInputs(inputIdx, forcedType);
                continue;
            }
            const auto converted = converter->convertType(funcInputType.value());
            VPUX_THROW_WHEN(converted == nullptr, "Failed to convert argument {0} type {1}", inputIdx,
                            funcInputType.value());
            sigConversion.addInputs(inputIdx, converted);
        }

        SmallVector<mlir::Type> newResults;
        if (mlir::failed(converter->convertTypes(funcType.getResults(), newResults))) {
            return mlir::failure();
        }

        const auto newFuncType = mlir::FunctionType::get(getContext(), sigConversion.getConvertedTypes(), newResults);

        if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), *converter, &sigConversion))) {
            return mlir::failure();
        }
        rewriter.modifyOpInPlace(funcOp, [&] {
            funcOp.setType(newFuncType);
        });

        return mlir::success();
    }

private:
    vpux::IE::PreserveArgTypeCb _preserveArg;
    vpux::Logger _log;
};

class RemoveDuplicatingConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    using mlir::OpRewritePattern<IE::ConvertOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto firstUser = origOp;
        SmallVector<IE::ConvertOp> matchingConverts;
        for (auto* user : origOp.getInput().getUsers()) {
            auto convertOp = mlir::dyn_cast<IE::ConvertOp>(user);
            if (convertOp == nullptr || convertOp->getBlock() != origOp->getBlock() ||
                convertOp.getOutput().getType() != origOp.getOutput().getType()) {
                continue;
            }
            matchingConverts.push_back(convertOp);
            if (convertOp->isBeforeInBlock(firstUser)) {
                firstUser = convertOp;
            }
        }

        // If there are less than two matching convert operations, there is nothing to deduplicate.
        if (matchingConverts.size() < 2) {
            return mlir::failure();
        }

        for (auto convertOp : matchingConverts) {
            if (convertOp != firstUser) {
                rewriter.replaceOp(convertOp, firstUser.getOutput());
            }
        }
        return mlir::success();
    }
};

}  // namespace

void vpux::IE::setupConvertPrecision(mlir::TypeConverter& typeConverter,
                                     FuncRef<mlir::Type(mlir::Type)> elemTypeConversionCb) {
    typeConverter.addConversion([elemTypeConversionCb](vpux::NDTypeInterface tensor) {
        return tensor.changeElemType(elemTypeConversionCb(tensor.getElementType()));
    });

    const auto convert = [](mlir::OpBuilder& builder, mlir::RankedTensorType type, mlir::ValueRange inputs,
                            mlir::Location) -> mlir::Value {
        // Ignore location of original operation, because this function is responsible for input/output network
        // precision and location of source is more useful
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        const auto dstType = mlir::TypeAttr::get(type.getElementType());
        const auto baseLoc = getValueLocation(inputs[0]);
        const auto newLocation = appendLoc(baseLoc, "converted_to_{0}", dstType);
        return builder.createOrFold<IE::ConvertOp>(newLocation, inputs[0], dstType);
    };

    typeConverter.addSourceMaterialization(convert);
    typeConverter.addTargetMaterialization(convert);
}

mlir::LogicalResult vpux::IE::runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                  mlir::ConversionTarget& target, Logger& log,
                                                  PreserveArgTypeCb getPreserveArgType,
                                                  PreserveOperandCb shouldPreserveOperand) {
    target.addLegalOp<IE::ConvertOp>();
    return runConvertOpTypes(module, typeConverter, target, log, std::move(getPreserveArgType),
                             std::move(shouldPreserveOperand));
}

mlir::LogicalResult vpux::IE::runConvertOpTypes(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                mlir::ConversionTarget& target, Logger& log,
                                                PreserveArgTypeCb getPreserveArgType,
                                                PreserveOperandCb shouldPreserveOperand) {
    mlir::RewritePatternSet patterns(module.getContext());
    if (getPreserveArgType != nullptr) {
        // Custom signature conversion that keeps selected arguments at their forced precision.
        patterns.add<ConvertFuncSignature>(typeConverter, module.getContext(), getPreserveArgType, log);
    } else {
        mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    }
    patterns.add<ConvertOpTypes>(typeConverter, module.getContext(), log, std::move(shouldPreserveOperand));

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        return mlir::failure();
    }

    // Distinct materialization requests (e.g. from different uses of the same value, or intermediate
    // placeholder values resolved separately by the conversion driver) can each produce their own
    // IE.Convert, ending up with an identical (input, dstType) pair (and thus a colliding location).
    // Merge any such duplicates here, once the final operands are known.
    mlir::RewritePatternSet cleanupPatterns(module.getContext());
    cleanupPatterns.add<RemoveDuplicatingConvert>(module.getContext());

    return mlir::applyPatternsGreedily(module, std::move(cleanupPatterns), getDefaultGreedyRewriteConfig());
}
