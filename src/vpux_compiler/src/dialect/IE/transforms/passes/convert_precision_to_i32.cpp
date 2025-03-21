//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPRECISIONTOI32
#define GEN_PASS_DEF_CONVERTPRECISIONTOI32
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

//
// ConvertPrecisionToI32Pass
//

class ConvertPrecisionToI32Pass final : public IE::impl::ConvertPrecisionToI32Base<ConvertPrecisionToI32Pass> {
public:
    explicit ConvertPrecisionToI32Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertPrecisionToI32Pass::safeRunOnModule() {
    auto& ctx = getContext();

    mlir::TypeConverter typeConverter;
    setupConvertPrecision(typeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isSignedInteger(64)) {
            return mlir::IntegerType::get(elemType.getContext(), 32, mlir::IntegerType::Signed);
        } else if (elemType.isUnsignedInteger(64)) {
            return mlir::IntegerType::get(elemType.getContext(), 32, mlir::IntegerType::Unsigned);
        } else {
            return elemType;
        }
    });

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GatherOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BroadcastOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::RollOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceProdOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceL1Op>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceL2Op>(isLegalOp);
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AdaptiveAvgPoolOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AdaptiveMaxPoolOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MaxPool8Op>(isLegalOp);
    target.addDynamicallyLegalOp<IE::EqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::NotEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::PowerOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::DivideOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::FloorModOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::AddOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SubtractOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::MultiplyOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::SelectOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LessOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::NegativeOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::GreaterEqualOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalOrOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::LogicalXorOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BitwiseAndOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BitwiseOrOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BitwiseXorOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::BitwiseNotOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::OneHotOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    // Convert TopK, AdaptiveMaxPool and OneHot element type attribute to avoid failures in infer return type checking.
    auto module = getOperation();
    module.walk([&](IE::TopKOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("element_type", mlir::TypeAttr::get(sInt32Type));
    });
    module.walk([&](IE::AdaptiveMaxPoolOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("index_element_type", mlir::TypeAttr::get(sInt32Type));
    });
    module.walk([&](IE::MaxPool8Op op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr("index_element_type", mlir::TypeAttr::get(sInt32Type));
    });
    module.walk([&](IE::OneHotOp op) {
        if (op.getOutputType().isSignedInteger(64)) {
            mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
            op->setAttr("outputType", mlir::TypeAttr::get(sInt32Type));
        }
    });
    module.walk([&](IE::ShapeOfOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr(op.getDstElemTypeAttrName(), mlir::TypeAttr::get(sInt32Type));
    });
    module.walk([&](IE::NonZeroOp op) {
        mlir::Type sInt32Type = mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
        op->setAttr(op.getDstElemTypeAttrName(), mlir::TypeAttr::get(sInt32Type));
    });
    if (mlir::failed(runConvertPrecision(module, typeConverter, target, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertPrecisionToI32Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToI32Pass(Logger log) {
    return std::make_unique<ConvertPrecisionToI32Pass>(log);
}
