//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FINALIZECOMPUTEFUNCTIONBOUNDARIES
#define GEN_PASS_DEF_FINALIZECOMPUTEFUNCTIONBOUNDARIES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

template <typename TensorSliceOp>
void updateOffsetAndSizeToMemoryOrder(mlir::Operation* origOp, mlir::Operation* newOp) {
    TensorSliceOp origSliceOp = mlir::cast<TensorSliceOp>(origOp);
    auto origResultType = mlir::cast<mlir::RankedTensorType>(origOp->getResult(0).getType());

    const auto order = vpux::getOrder(origResultType);
    auto permutation = DimsOrder::fromAffineMap(order);

    SmallVector<int64_t> newOffsets = permutation.toMemoryOrder(Shape(origSliceOp.getStaticOffsets())).raw();
    SmallVector<int64_t> newSizes = permutation.toMemoryOrder(Shape(origSliceOp.getStaticSizes())).raw();

    TensorSliceOp newExtractSliceOp = mlir::cast<TensorSliceOp>(newOp);
    newExtractSliceOp.setStaticOffsets(newOffsets);
    newExtractSliceOp.setStaticSizes(newSizes);
}

void updateTensorDimIndexToMemoryOrder(mlir::Operation* origOp, mlir::Operation* newOp,
                                       mlir::PatternRewriter& rewriter) {
    auto origDimOp = mlir::cast<mlir::tensor::DimOp>(origOp);
    auto origResultType = mlir::cast<mlir::RankedTensorType>(origDimOp.getSource().getType());
    const auto order = vpux::getOrder(origResultType);
    auto permutation = DimsOrder::fromAffineMap(order).toPermutation();

    assert(origDimOp.getConstantIndex().has_value() && "Expected constant index for tensor::DimOp");
    const auto origDimIndex = origDimOp.getConstantIndex().value();
    // nothing to do
    if (origDimIndex == permutation[origDimIndex].ind()) {
        return;
    }
    const auto newDimIndex = std::find(permutation.begin(), permutation.end(), Dim(origDimIndex)) - permutation.begin();

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(newOp);
    auto newDimIndexOp = rewriter.create<mlir::arith::ConstantIndexOp>(origDimOp.getLoc(), newDimIndex);

    auto newDimOp = mlir::cast<mlir::tensor::DimOp>(newOp);
    newDimOp.getIndexMutable().set(newDimIndexOp.getResult());
}

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
    _log.trace("Process Operation '{0}' with name {1}", origOp->getLoc(), origOp->getName());

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

    if (mlir::isa<mlir::tensor::ExtractSliceOp>(origOp)) {
        updateOffsetAndSizeToMemoryOrder<mlir::tensor::ExtractSliceOp>(origOp, newOp);
    } else if (mlir::isa<mlir::tensor::InsertSliceOp>(origOp)) {
        updateOffsetAndSizeToMemoryOrder<mlir::tensor::InsertSliceOp>(origOp, newOp);
    } else if (mlir::isa<mlir::tensor::DimOp>(origOp)) {
        updateTensorDimIndexToMemoryOrder(origOp, newOp, rewriter);
    }

    rewriter.replaceOp(origOp, newOp->getResults());

    return mlir::success();
}

//
// FinalizeComputeFunctionBoundariesPass
//

class FinalizeComputeFunctionBoundariesPass final :
        public VPU::impl::FinalizeComputeFunctionBoundariesBase<FinalizeComputeFunctionBoundariesPass> {
public:
    explicit FinalizeComputeFunctionBoundariesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void FinalizeComputeFunctionBoundariesPass::safeRunOnModule() {
    auto module = getOperation();
    mlir::MLIRContext* ctx = module->getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
        const auto order = vpux::getOrder(type);
        auto ndType = mlir::cast<NDTypeInterface>(type);
        auto permutation = DimsOrder::fromAffineMap(order);
        SmallVector<int64_t> newShape = permutation.toMemoryOrder(ndType.getShape()).raw();

        return mlir::RankedTensorType::get(newShape, type.getElementType());
    });
    typeConverter.addConversion([](mlir::IndexType type) {
        return type;
    });

    const auto convert = [&ctx, this](mlir::OpBuilder& builder, mlir::RankedTensorType destType,
                                      mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        auto newLocation = appendLoc(loc, "casted");
        auto isOutputValue = [](mlir::Value value) {
            return llvm::any_of(value.getUsers(), [](mlir::Operation* user) {
                return mlir::isa<mlir::func::ReturnOp>(user);
            });
        };
        auto isInputValue = [](mlir::Value value) {
            return mlir::isa<mlir::BlockArgument>(value);
        };

        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        auto input = inputs[0];

        _log.trace("Materialization called: input={0}, destType={1}", input.getType(), destType);

        auto inputNdType = mlir::cast<NDTypeInterface>(input.getType());
        auto inputOrder = inputNdType.getDimsOrder();
        auto getPermutation = [&](auto getTypeFunc) {
            auto castOpIt = llvm::find_if(input.getUsers(), [&](mlir::Operation* user) {
                return mlir::isa<mlir::UnrealizedConversionCastOp>(user) &&
                       vpux::getTensorAttr(mlir::cast<mlir::RankedTensorType>(getTypeFunc(user))) != nullptr;
            });
            if (castOpIt != input.getUsers().end()) {
                auto castOp = mlir::cast<mlir::UnrealizedConversionCastOp>(*castOpIt);
                auto castNdType = mlir::cast<vpux::NDTypeInterface>(getTypeFunc(castOp));
                return std::optional(getPermutationFromOrders(inputOrder, castNdType.getDimsOrder(), ctx));
            }
            return std::optional<mlir::AffineMap>();
        };
        auto dstOrder = [&] {
            if (isOutputValue(input)) {
                return getPermutation([](mlir::Operation* op) {
                    return op->getOperand(0).getType();
                });
            } else if (isInputValue(input)) {
                return getPermutation([](mlir::Operation* op) {
                    return op->getResult(0).getType();
                });
            }
            return std::optional<mlir::AffineMap>();
        }();
        VPUX_THROW_UNLESS(dstOrder.has_value(), "Cannot detect destination order for input '{0}'",
                          input.getDefiningOp()->getName());

        if (isInputValue(input) || isOutputValue(input)) {
            return builder.createOrFold<Core::ReinterpretCastOp>(newLocation, destType, input);
        }

        return input;
    };
    typeConverter.addSourceMaterialization(convert);
    typeConverter.addTargetMaterialization(convert);
    typeConverter.addArgumentMaterialization(convert);

    const auto isLegalOp = [&](mlir::Operation* op) {
        return typeConverter.isLegal(op);
    };

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::affine::AffineDialect>();
    target.addLegalDialect<vpux::VPU::VPUDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    target.addDynamicallyLegalDialect<mlir::scf::SCFDialect>(isLegalOp);

    target.addDynamicallyLegalOp<mlir::tensor::ExtractSliceOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::tensor::InsertSliceOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::tensor::EmptyOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::tensor::DimOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::tensor::CastOp>(isLegalOp);

    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(isLegalOp);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp funcOp) {
        return typeConverter.isSignatureLegal(funcOp.getFunctionType());
    });

    mlir::RewritePatternSet patterns(module.getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    patterns.add<ConvertOpTypes>(typeConverter, module.getContext(), _log);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFinalizeComputeFunctionBoundariesPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFinalizeComputeFunctionBoundariesPass(Logger log) {
    return std::make_unique<FinalizeComputeFunctionBoundariesPass>(log);
}
