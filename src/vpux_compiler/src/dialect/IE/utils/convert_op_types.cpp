//
// Copyright (C) 2024-2025 Intel Corporation.
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
    ConvertOpTypes(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, vpux::Logger log)
            : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, vpux::benefitHigh, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    vpux::Logger _log;
};

mlir::LogicalResult ConvertOpTypes::matchAndRewrite(mlir::Operation* origOp, vpux::ArrayRef<mlir::Value> operands,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}'", origOp->getLoc());

    const auto* converter = getTypeConverter();
    VPUX_THROW_UNLESS(converter != nullptr, "TypeConverter was not set");

    const auto origOperands = origOp->getOperands();
    VPUX_THROW_UNLESS(origOperands.size() == operands.size(), "Wrong operands size : {0}", operands.size());

    mlir::IRMapping mapper;
    mapper.map(origOperands, operands);

    auto* newOp = rewriter.clone(*origOp, mapper);
    for (auto result : newOp->getResults()) {
        result.setType(converter->convertType(result.getType()));
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

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
    typeConverter.addArgumentMaterialization(convert);
}

mlir::LogicalResult vpux::IE::runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                  mlir::ConversionTarget& target, Logger& log) {
    target.addLegalOp<IE::ConvertOp>();
    return runConvertOpTypes(module, typeConverter, target, log);
}

mlir::LogicalResult vpux::IE::runConvertOpTypes(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                mlir::ConversionTarget& target, Logger& log) {
    mlir::RewritePatternSet patterns(module.getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    patterns.add<ConvertOpTypes>(typeConverter, module.getContext(), log);

    return mlir::applyPartialConversion(module, target, std::move(patterns));
}
