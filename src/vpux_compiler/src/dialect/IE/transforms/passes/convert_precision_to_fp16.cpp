//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/convert_op_types.hpp"
#include "vpux/compiler/dialect/IE/utils/power_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTPRECISIONTOFP16
#define GEN_PASS_DEF_CONVERTPRECISIONTOFP16
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

// E#160872: The compiler must have a much more general solution: either all
// fp16 values must come out as HALF_MAX / HALF_MIN or none. the preferred way
// seems to be "all". However, clamping non-splats produced inaccurate results
// according to tests. This needs to be debugged properly to understand why. The
// logic here is a *workaround* to have sustainable development in the meantime.
Const::ContentAttr clampF16Splat(const Const::ContentAttr& input) {
    if (!input.isSplat()) {
        // Note: clamping values in the pass here is OK, because this logic only
        // works for *splats* (aka single-value constants) and calculating
        // splats is *fast*. If there is ever a need for a proper solution, this
        // has to be migrated to a constant transformation (likely, to
        // #const.ConvertElemType<f16>).
        return nullptr;
    }

    // Note: fp16 internal ctor would convert any type to float thus get float
    // value from the input directly.
    const auto splatValue = input.fold().getSplatValue<float>();
    const auto splatValueFp16 = vpux::type::float16(splatValue);
    if (!std::isinf(splatValueFp16)) {
        // no need to clamp - can use the existing content attr
        return nullptr;
    }
    const auto clampedFp16 = std::numeric_limits<vpux::type::float16>::clamp(splatValueFp16);

    const auto dataType = input.getType().changeElemType(mlir::Float16Type::get(input.getContext()));
    const auto content = Const::createConstContent(mlir::cast<mlir::ShapedType>(dataType), ArrayRef{clampedFp16});

    // Note: Cast to current input type to ensure compatibility with existing
    // constant operation. This cast costs nothing and is likely to be fused
    // with other casts.
    return Const::ContentAttr::get(content, {Const::CastElemTypeAttr::get(input.getType().getElementType())});
}

void insertEpsilonClamp(mlir::OpBuilder& builder, mlir::Operation* divLikeOp, mlir::Operation* op) {
    // Return if there is a subsequent Select designed to screen out Inf/NaN
    if (divLikeOp != nullptr && divLikeOp->hasOneUse()) {
        auto& dataOperand = *divLikeOp->getUses().begin();
        if (auto selOp = mlir::dyn_cast_or_null<IE::SelectOp>(dataOperand.getOwner())) {
            auto dataIndex = dataOperand.getOperandNumber();
            auto maybeZeros = dataIndex == 1 ? selOp.getInput3().getDefiningOp<Const::DeclareOp>()
                                             : selOp.getInput2().getDefiningOp<Const::DeclareOp>();
            if (dataIndex > 0 && maybeZeros != nullptr && Const::hasAllZeroValues(maybeZeros.getContent())) {
                return;
            }
        }
    }

    if (auto sqrtOp = mlir::dyn_cast_or_null<IE::SqrtOp>(op)) {
        // Skip over Sqrt Op since epsilon Add/Max/Clamp Op is usually located before it.
        op = sqrtOp.getInput().getDefiningOp();
    }

    if (op == nullptr || !op->hasOneUse() || mlir::isa_and_nonnull<Const::DeclareOp>(op)) {
        return;
    }

    if (auto inputVal = op->getOperands().front()) {
        if (inputVal.getDefiningOp<Const::DeclareOp>() != nullptr) {
            return;
        }
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    if (!outputType.getElementType().isF16()) {
        return;
    }

    // Insert a new Clamp op to prevent float16 precision divide-by-zero in subsequent Op.
    builder.setInsertionPointAfter(op);
    auto& useResult = *op->getUses().begin();

    auto ctx = op->getContext();
    auto clampMin = getFPAttr(ctx, std::numeric_limits<type::float16>::nearest_non_zero_positive_value);
    auto clampMax = getFPAttr(ctx, static_cast<double>(std::numeric_limits<type::float16>::max()));
    auto newClampOp = builder.create<IE::ClampOp>(takeOpLoc(op, "epsilon"), op->getResult(0), clampMin, clampMax);

    useResult.assign(newClampOp.getResult());
}

//
// ConvertPrecisionToFP16Pass
//

class ConvertPrecisionToFP16Pass final : public IE::impl::ConvertPrecisionToFP16Base<ConvertPrecisionToFP16Pass> {
public:
    explicit ConvertPrecisionToFP16Pass(Logger log, StringRef computeLayersWithHigherPrecision)
            : _computeLayersWithHigherPrecision(computeLayersWithHigherPrecision.str()) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    std::string _computeLayersWithHigherPrecision;
};

mlir::LogicalResult ConvertPrecisionToFP16Pass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (computeLayersWithHigherPrecision.hasValue()) {
        _computeLayersWithHigherPrecision = computeLayersWithHigherPrecision.getValue();
    }

    return mlir::success();
}

void ConvertPrecisionToFP16Pass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto convertElemType = [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isF64() || elemType.isSignlessInteger(CHAR_BIT)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else if (const auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
                   qType != nullptr && qType.getExpressedType().isF32()) {
            return changeExpressedType(qType, mlir::Float16Type::get(qType.getContext()));
        } else {
            return elemType;
        }
    };

    mlir::TypeConverter typeConverter;
    setupConvertPrecision(typeConverter, convertElemType);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    const auto hasDynamicDequantizeUser = [](mlir::Operation* op) {
        return llvm::any_of(op->getUsers(), [](const auto user) {
            return mlir::isa<IE::DynamicDequantizeOp>(user);
        });
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalDialect<IE::IEDialect>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::OneHotOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<IE::DynamicQuantizeOp>();
    target.addLegalOp<IE::DynamicDequantizeOp>();
    target.addDynamicallyLegalOp<IE::QuantizeCastOp>([&](mlir::Operation* op) {
        return isLegalOp(op) || hasDynamicDequantizeUser(op);
    });
    target.addDynamicallyLegalOp<IE::QuantizeOp>([&](mlir::Operation* op) {
        return isLegalOp(op) || hasDynamicDequantizeUser(op);
    });
    target.addLegalOp<IE::IfOp>();
    target.addLegalOp<IE::YieldOp>();
    target.addLegalOp<IE::LoopSelectOp>();
    target.addLegalOp<IE::EqualOp>();
    target.addLegalOp<IE::LessOp>();
    target.addLegalOp<IE::LessEqualOp>();
    target.addLegalOp<IE::GreaterOp>();
    target.addLegalOp<IE::NotEqualOp>();
    // AssignOp & ReadValueOp represent inputs/outputs. Cannot convert their type internally.
    target.addLegalOp<IE::AssignOp>();
    target.addLegalOp<IE::ReadValueOp>();
    target.addLegalOp<IE::BitwiseAndOp>();
    target.addLegalOp<IE::BitwiseOrOp>();
    target.addLegalOp<IE::BitwiseXorOp>();
    target.addLegalOp<IE::BitwiseNotOp>();
    target.addLegalOp<IE::RangeOp>();
    target.addLegalOp<IE::ReduceL2Op>();
    target.addLegalOp<IE::InverseOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    const auto internalHintPrefix = "internal_";
    auto internalMvnHighNorm = false;

    if (!_computeLayersWithHigherPrecision.empty()) {
        std::istringstream optionsStream(_computeLayersWithHigherPrecision);
        std::string dialectNamespace = IE::IEDialect::getDialectNamespace().str() + ".";
        std::string option;
        while (std::getline(optionsStream, option, ',')) {
            bool isAddRMSNorm = option == std::string("Add_RMSNorm");
            if (isAddRMSNorm) {
                option = std::string("Add");
            }

            if (option.find(internalHintPrefix, 0) != std::string::npos) {
                const auto internalMvnOption = std::string(internalHintPrefix) + "MvnNormalize";
                if (option.find(internalMvnOption, 0) != std::string::npos) {
                    internalMvnHighNorm = true;
                }
                continue;
            }

            std::string fullOption = dialectNamespace + option;
            StringRef opnameRef(fullOption);
            auto opname = mlir::OperationName(opnameRef, &ctx);
            VPUX_THROW_UNLESS(opname.isRegistered(), "Invalid input layer '{0}'", opname);
            // Keep the original precision for all instances of specified layer name(s) during the conversion to FP16

            // If AddOp is listed into computeLayersWithHigherPrecision list,
            // keep precision only in RMSNorm pattern
            if (isAddRMSNorm) {
                target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
                    // Try to find RMSNorm pattern
                    // ReaduceMeanOp -> AddOp -> SqrtOp
                    if ((op.getInput1().getDefiningOp<IE::ReduceMeanOp>() != nullptr ||
                         op.getInput2().getDefiningOp<IE::ReduceMeanOp>() != nullptr) &&
                        mlir::isa_and_nonnull<IE::SqrtOp>(*op.getOutput().getUsers().begin())) {
                        return true;
                    }
                    return isLegalOp(op);
                });
            } else {
                target.addLegalOp(opname);
            }
        }
    }

    auto module = getOperation();

    // E#160869: Splat floating-point constants must be clamped, not converted,
    // to ensure accurate results (e.g. when comparing to CPU inference).
    module.walk([&](Const::DeclareOp op) {
        const auto inElemType = op.getContentAttr().getType().getElementType();
        const auto precisionLoweringToF16 =
                mlir::isa<mlir::FloatType>(inElemType) && inElemType.getIntOrFloatBitWidth() >= sizeof(float);
        const auto newContentAttr = precisionLoweringToF16 ? clampF16Splat(op.getContentAttr()) : nullptr;
        if (newContentAttr != nullptr) {
            VPUX_THROW_WHEN(op.getType() != newContentAttr.getType(),
                            "Const op type {0} must match new content attribute type {1}", op.getType(),
                            newContentAttr.getType());
            op.setContentAttr(newContentAttr);
        }
    });

    // Some ops infer their output type based on a member type attribute, which should also be converted.
    module.walk([&](mlir::Operation* op) {
        if (!target.isIllegal(op)) {
            return;
        }

        mlir::TypeSwitch<mlir::Operation*, void>(op)
                .Case<IE::DequantizeOp, IE::QuantizeCastOp, IE::QuantizeOp>([&](auto op) {
                    op.setDstElemType(convertElemType(op.getDstElemType()));
                })
                .Case<IE::OneHotOp, IE::RandomUniformOp, IE::EyeOp>([&](auto op) {
                    op.setOutputType(convertElemType(op.getOutputType()));
                })
                .Case<IE::MVNOp>([&](auto op) {
                    if (internalMvnHighNorm) {
                        // fp16 buffs with f32 internal processing for normalization
                        op.setHighPrecisionNormalize(true);
                    }
                })
                .Case<IE::DynamicDataMaskOp>([&](auto op) {
                    auto outTensorType = mlir::cast<NDTypeInterface>(op.getOutputTensorType());
                    op.setOutputTensorType(
                            outTensorType.changeElemType(convertElemType(outTensorType.getElementType())));
                });
    });

    if (mlir::failed(runConvertPrecision(module, typeConverter, target, _log))) {
        signalPassFailure();
    }

    mlir::TypeConverter uniquifyPrecisionTypeConverter;
    setupConvertPrecision(uniquifyPrecisionTypeConverter, [](mlir::Type elemType) -> mlir::Type {
        return mlir::Float16Type::get(elemType.getContext());
    });
    const auto isPrecisionUniquifiedOp = [](mlir::Operation* op) {
        auto operandTypes = op->getOperandTypes();
        return std::all_of(operandTypes.begin() + 1, operandTypes.end(), [&](auto operandType) {
            return mlir::cast<NDTypeInterface>(operandType).getElementType() ==
                   mlir::cast<NDTypeInterface>(operandTypes.front()).getElementType();
        });
    };
    mlir::ConversionTarget uniquifyPrecisionTarget(ctx);
    uniquifyPrecisionTarget.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });
    uniquifyPrecisionTarget.addDynamicallyLegalOp<IE::AndOp>(isPrecisionUniquifiedOp);
    if (mlir::failed(runConvertPrecision(module, uniquifyPrecisionTypeConverter, uniquifyPrecisionTarget, _log))) {
        signalPassFailure();
    }

    mlir::TypeConverter additionalTypeConverter;
    setupConvertPrecision(additionalTypeConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isF64() || elemType.isSignedInteger(64)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else {
            return elemType;
        }
    });

    const auto isLegalAdditionalOp = [&](mlir::Operation* op) {
        return additionalTypeConverter.isLegal(op);
    };

    mlir::ConversionTarget additionalTarget(ctx);
    additionalTarget.addDynamicallyLegalOp<IE::LessOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::LessEqualOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::GreaterOp>(isLegalAdditionalOp);
    additionalTarget.addDynamicallyLegalOp<IE::ClampOp>(isLegalAdditionalOp);
    additionalTarget.addLegalOp<mlir::ModuleOp>();
    if (mlir::failed(runConvertPrecision(module, additionalTypeConverter, additionalTarget, _log))) {
        signalPassFailure();
    }

    // SelectOp
    mlir::TypeConverter selectOpConverter;
    setupConvertPrecision(selectOpConverter, [](mlir::Type elemType) -> mlir::Type {
        if (elemType.isF32() || elemType.isF64() || elemType.isSignlessInteger(CHAR_BIT) ||
            elemType.isSignedInteger(32) || elemType.isSignedInteger(64)) {
            return mlir::Float16Type::get(elemType.getContext());
        } else {
            return elemType;
        }
    });

    const auto isLegalSelectOp = [&](mlir::Operation* op) {
        return selectOpConverter.isLegal(op);
    };

    mlir::ConversionTarget selectOpTarget(ctx);
    selectOpTarget.addDynamicallyLegalOp<IE::SelectOp>(isLegalSelectOp);
    selectOpTarget.addLegalOp<mlir::ModuleOp>();
    if (mlir::failed(runConvertPrecision(module, selectOpConverter, selectOpTarget, _log))) {
        signalPassFailure();
    }

    mlir::OpBuilder builder(module);

    // Insert Clamp Ops to prevent divide-by-zero in low precision Divide and Divide-like Power Ops
    // to ensure accurate results (e.g. when comparing to CPU inference).
    module.walk([&](IE::DivideOp divideOp) {
        insertEpsilonClamp(builder, divideOp, divideOp.getInput2().getDefiningOp());
    });

    module.walk([&](IE::PowerOp powerOp) {
        auto exponent = IE::getExponentSplatVal(powerOp);
        const bool powerIsDivisionLike = (exponent.has_value() && exponent.value() < 0.f);
        if (!powerIsDivisionLike) {
            return;
        }

        insertEpsilonClamp(builder, powerOp, powerOp.getInput1().getDefiningOp());
    });
}

}  // namespace

//
// createConvertPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertPrecisionToFP16Pass(Logger log,
                                                                       StringRef computeLayersWithHigherPrecision) {
    return std::make_unique<ConvertPrecisionToFP16Pass>(log, computeLayersWithHigherPrecision);
}
