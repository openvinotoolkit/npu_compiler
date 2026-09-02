//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/allocate_buffers.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <memory>
#include <utility>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_CONVERTDYNAMICMEMREFSTOBOUNDEDBUFFERS
#define GEN_PASS_DEF_CONVERTDYNAMICMEMREFSTOBOUNDEDBUFFERS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
class ConvertDynamicMemrefsToBoundedBuffers final :
        public VPUIP::impl::ConvertDynamicMemrefsToBoundedBuffersBase<ConvertDynamicMemrefsToBoundedBuffers> {
public:
    explicit ConvertDynamicMemrefsToBoundedBuffers(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

/// Returns whether the type is a dynamic memref, i.e. a memref with non-static
/// shape and bounds.
bool isDynamicMemref(mlir::Type type) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memrefType || memrefType.hasStaticShape()) {
        return false;
    }
    if (auto memrefAttr = mlir::dyn_cast_or_null<vpux::MemRefAttr>(memrefType.getLayout())) {
        return !memrefAttr.bounds().empty();
    }
    return false;
}

/// Returns a bounded buffer type created from a dynamic memref.
VPUIP::BoundedBufferType createBoundedBufferFromDynamicMemref(mlir::MemRefType type) {
    const auto bounds = getBounds(type);
    assert(!bounds.empty() && "Dynamic memref must have bounds");
    const auto ndType = mlir::cast<NDTypeInterface>(type);
    const auto strides = ndType.getStrides();
    // Note: shape change removes strides, but it makes sense to preserve them
    const auto dataMemref = ndType.changeShape(ShapeRef(bounds.raw())).changeStrides(strides);
    assert(mlir::cast<mlir::MemRefType>(dataMemref).hasStaticShape() && getBounds(dataMemref).empty() &&
           "Data memref must be static");
    const auto shapeMemref = getMemRefType({static_cast<int32_t>(bounds.size())}, getSInt32Type(type.getContext()),
                                           DimsOrder::C, ndType.getMemSpace());
    return VPUIP::BoundedBufferType::get(dataMemref, shapeMemref);
}

/// Legalizes function signature to operate on bounded buffers instead of
/// dynamic memrefs.
struct FuncOpRewrite final : public mlir::OpConversionPattern<mlir::func::FuncOp> {
    FuncOpRewrite(const mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::func::FuncOp>(typeConverter, ctx), _log(std::move(log)) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op, OpAdaptor opAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuncOpRewrite::matchAndRewrite(mlir::func::FuncOp funcOp, OpAdaptor,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto log = _log;
    log.trace("Legalizing arguments of '{0}' at '{1}'", funcOp.getSymName(), funcOp->getLoc());
    log = log.nest();

    const auto funcType = funcOp.getFunctionType();
    mlir::TypeConverter::SignatureConversion sigConversion(funcType.getNumInputs());
    for (size_t i = 0; i < funcType.getNumInputs(); ++i) {
        const auto inputType = funcType.getInput(i);
        sigConversion.addInputs(i, getTypeConverter()->convertType(inputType));
    }
    SmallVector<mlir::Type> newResults;
    if (mlir::failed(getTypeConverter()->convertTypes(funcType.getResults(), newResults))) {
        return mlir::failure();
    }

    if (mlir::failed(rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(), &sigConversion))) {
        return mlir::failure();
    }
    rewriter.modifyOpInPlace(funcOp, [&] {
        funcOp.setType(mlir::FunctionType::get(getContext(), sigConversion.getConvertedTypes(), newResults));
    });

    return mlir::success();
}

/// Replaces dynamic memref allocation with bounded buffer allocation.
struct AllocOpRewrite final : public mlir::OpConversionPattern<mlir::memref::AllocOp> {
    AllocOpRewrite(const mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<mlir::memref::AllocOp>(typeConverter, ctx), _log(std::move(log)) {
    }

    mlir::LogicalResult matchAndRewrite(mlir::memref::AllocOp op, OpAdaptor opAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AllocOpRewrite::matchAndRewrite(mlir::memref::AllocOp op, OpAdaptor,
                                                    mlir::ConversionPatternRewriter& rewriter) const {
    auto log = _log;
    log.trace("Legalizing AllocOp at '{0}'", op->getLoc());
    log = log.nest();

    const auto newType = getTypeConverter()->convertType(op.getType());
    auto buffers = VPUIP::allocateBuffersOfType(log, op->getLoc(), rewriter, newType,
                                                /*individualBuffers=*/false);
    assert(buffers.size() == 1 && "Bounded buffer allocation returns one grouped buffer");
    rewriter.replaceOp(op, buffers.front());
    return mlir::success();
}

struct GenericOpRewrite final : public mlir::ConversionPattern {
    GenericOpRewrite(const mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{}, /*benefit=*/1, ctx),
              _log(std::move(log)) {
    }
    mlir::LogicalResult matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GenericOpRewrite::matchAndRewrite(mlir::Operation* op, ArrayRef<mlir::Value> newArgs,
                                                      mlir::ConversionPatternRewriter& rewriter) const {
    assert((newArgs.size() == op->getNumOperands()) && "New operands size must match the original");

    auto log = _log;
    log.trace("Legalizing operation '{0}' at '{1}'", op->getName(), op->getLoc());
    log = log.nest();

    rewriter.modifyOpInPlace(op, [&] {
        for (size_t i = 0; i < newArgs.size(); ++i) {
            op->setOperand(i, newArgs[i]);
        }

        for (auto result : op->getResults()) {
            result.setType(getTypeConverter()->convertType(result.getType()));
        }
    });

    mlir::TypeConverter::SignatureConversion sigConversion(newArgs.size());
    for (size_t i = 0; i < newArgs.size(); ++i) {
        sigConversion.addInputs(i, newArgs[i].getType());
    }
    for (auto& region : op->getRegions()) {
        auto result = rewriter.convertRegionTypes(&region, *getTypeConverter(), &sigConversion);
        VPUX_THROW_WHEN(mlir::failed(result), "Failed to convert region types for operation '{0}' at '{1}'",
                        op->getName(), op->getLoc());
    }

    return mlir::success();
}

struct ReinterpretCastOpRewrite final : public mlir::OpConversionPattern<Core::ReinterpretCastOp> {
    ReinterpretCastOpRewrite(const mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<Core::ReinterpretCastOp>(typeConverter, ctx), _log(std::move(log)) {
    }
    mlir::LogicalResult matchAndRewrite(Core::ReinterpretCastOp op, OpAdaptor opAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReinterpretCastOpRewrite::matchAndRewrite(Core::ReinterpretCastOp op, OpAdaptor opAdaptor,
                                                              mlir::ConversionPatternRewriter& rewriter) const {
    assert((opAdaptor.getOperands().size() == 1) && "ReinterpretCastOp should have exactly one operand");

    auto log = _log;
    log.trace("Legalizing ReinterpretCastOp at '{0}'", op->getLoc());
    log = log.nest();

    // Use the converted operand/result types (via adaptor/type converter), not the op's own
    // possibly-stale types: a producer (e.g. VPUIP.SW.Kernel) may already have been legalized
    // to a bounded buffer in-place
    const bool inputIsBounded = mlir::isa<VPUIP::BoundedBufferType>(opAdaptor.getInput().getType());
    const auto convertedResultType = getTypeConverter()->convertType(op.getResult().getType());
    const bool resultIsBounded = mlir::isa<VPUIP::BoundedBufferType>(convertedResultType);
    assert(!(inputIsBounded && resultIsBounded) &&
           "ReinterpretCastOp cannot have both input and result as bounded buffers");

    if (resultIsBounded) {
        log.trace("Legalizing result that must become a bounded buffer");
        // ReinterpretCast is special, that is, it cannot cast anything to a
        // bounded buffer, so an explicit grouping is necessary
        const auto boundedBufferType = mlir::cast<VPUIP::BoundedBufferType>(convertedResultType);
        auto dataCast = rewriter.create<Core::ReinterpretCastOp>(appendLoc(op->getLoc(), "data_cast"),
                                                                 boundedBufferType.getData(), opAdaptor.getInput());
        auto shapeAlloc = rewriter.create<mlir::memref::AllocOp>(
                appendLoc(op->getLoc(), "shape_alloc"),
                mlir::cast<mlir::MemRefType>(boundedBufferType.getDynamicShape()));
        auto grouped = rewriter.create<VPUIP::GroupBoundedBufferOp>(appendLoc(op->getLoc(), "group"),
                                                                    dataCast.getResult(), shapeAlloc);
        rewriter.replaceOp(op, grouped.getResult());
        return mlir::success();
    }

    if (inputIsBounded) {
        log.trace("Legalizing input that is a bounded buffer");
        // ReinterpretCast is special, that is, it cannot cast a bounded buffer
        // to anything, so an explicit ungrouping is necessary
        auto ungroupOp = rewriter.create<VPUIP::UngroupBoundedBufferOp>(appendLoc(op->getLoc(), "ungroup"),
                                                                        opAdaptor.getInput());
        auto newCastOp = rewriter.create<Core::ReinterpretCastOp>(appendLoc(op->getLoc(), "cast"), op.getType(),
                                                                  ungroupOp.getData());
        rewriter.replaceOp(op, newCastOp.getResult());
        return mlir::success();
    }

    log.trace("ReinterpretCastOp has no dynamic memrefs");
    return mlir::failure();
}

struct SwKernelOpRewrite final : public mlir::OpConversionPattern<VPUIP::SwKernelOp> {
    SwKernelOpRewrite(const mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<VPUIP::SwKernelOp>(typeConverter, ctx), _log(std::move(log)) {
    }
    mlir::LogicalResult matchAndRewrite(VPUIP::SwKernelOp op, OpAdaptor opAdaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;

private:
    Logger _log;
    /// A helper cache to track already-updated SW kernels in case where there
    /// are multiple SW operations pointing to the same builtin function.
    mutable mlir::DenseSet<mlir::func::FuncOp> _updatedBuiltinSwFuncs;

    void rewriteShapeOfSwKernel(Logger log, VPUIP::SwKernelOp op, ArrayRef<mlir::Value> newArgs,
                                mlir::ConversionPatternRewriter& rewriter) const;
    void rewriteBuiltinFuncSignature(Logger log, VPUIP::SwKernelOp op, ArrayRef<mlir::Value> newArgs,
                                     FuncRef<void(std::vector<mlir::Type>&, size_t, mlir::Type)> modifyType) const;
};

mlir::LogicalResult SwKernelOpRewrite::matchAndRewrite(VPUIP::SwKernelOp op, OpAdaptor opAdaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto log = _log.nest();
    log.trace("Legalizing the SwKernelOp '{0}' at '{1}'", op->getName(), op->getLoc());
    log = log.nest();

    // Note: ShapeOf is a special kind of SW kernel. Unlike other SW kernels, it
    // completely disregards the data buffer and only uses shape buffer from the
    // bounded buffer. Thus, its legalization is different from a normal SW
    // kernel.
    const bool isShapeOf = VPUIP::getSwKernelEntryName(op) == "shape_of";

    const auto newArgs = to_small_vector(opAdaptor.getOperands());

    // first, legalize the SW kernel op itself (operands, results, signature)
    if (isShapeOf) {
        rewriteShapeOfSwKernel(log, op, newArgs, rewriter);
    } else if (mlir::failed(GenericOpRewrite(*getTypeConverter(), getContext(), log)
                                    .matchAndRewrite(op, newArgs, rewriter))) {
        return mlir::failure();
    }

    // once the op itself is "legal", SwKernelRun must be updated along with the
    // builtin functions

    // Note: SW kernel runs only have operands and no results; their operands
    // also include output buffers
    rewriter.modifyOpInPlace(op, [&]() {
        for (auto swKernelRun : op.getOps<VPUIP::SwKernelRun>()) {
            log.trace("Changing types of SwKernelRun '{0}' at '{1}'", swKernelRun->getName(), swKernelRun->getLoc());
            for (size_t i = 0; i < newArgs.size(); ++i) {
                swKernelRun.getOperand(i).setType(op->getOperand(i).getType());
            }
        }
    });

    const auto modifyType = [&](std::vector<mlir::Type>& argumentTypes, size_t operandIndex, mlir::Type newType) {
        if (isShapeOf) {
            // ShapeOf replaces dynamic memref with shape
            argumentTypes[operandIndex] = newType;
            return;
        }

        // bounded buffer becomes two SW kernel entries: one for the data, one
        // for the shape
        argumentTypes.insert(argumentTypes.begin() + operandIndex + 1, newType);
    };
    rewriteBuiltinFuncSignature(log, op, newArgs, modifyType);

    return mlir::success();
}

void SwKernelOpRewrite::rewriteShapeOfSwKernel(Logger log, VPUIP::SwKernelOp op, ArrayRef<mlir::Value> newArgs,
                                               mlir::ConversionPatternRewriter& rewriter) const {
    assert((newArgs.size() == op->getNumOperands()) && "New operands size must match the original");
    rewriter.modifyOpInPlace(op, [&]() {
        for (size_t i = 0; i < newArgs.size(); ++i) {
            auto newOperand = newArgs[i];
            if (!mlir::isa<VPUIP::BoundedBufferType>(newOperand.getType())) {
                continue;
            }

            log.trace("Legalizing operand '{0}' that is a dynamic memref", i);
            auto ungroup =
                    rewriter.create<VPUIP::UngroupBoundedBufferOp>(appendLoc(op->getLoc(), "ungroup"), newOperand);
            op->setOperand(i, ungroup.getDynamicShape());
        }
    });
}

void SwKernelOpRewrite::rewriteBuiltinFuncSignature(
        Logger log, VPUIP::SwKernelOp op, ArrayRef<mlir::Value> newArgs,
        FuncRef<void(std::vector<mlir::Type>&, size_t, mlir::Type)> modifyType) const {
    auto swKernelSymbol = op.getKernelFunction();
    auto moduleOfThisIr = op->getParentOfType<mlir::ModuleOp>();
    auto builtinFunc = moduleOfThisIr.lookupSymbol<mlir::func::FuncOp>(swKernelSymbol);
    assert(builtinFunc != nullptr && "Builtin function for the SW kernel must always exist here");
    const bool firstOccurrence = _updatedBuiltinSwFuncs.insert(builtinFunc).second;
    if (!firstOccurrence) {
        log.trace("Builtin function '{0}' already has the correct signature, no changes needed", swKernelSymbol);
        return;
    }

    SmallVector<size_t> dynamicOperands;
    for (size_t i = 0; i < newArgs.size(); ++i) {
        if (mlir::isa<VPUIP::BoundedBufferType>(newArgs[i].getType())) {
            dynamicOperands.push_back(i);
        }
    }

    log.trace("Changing the signature of the builtin function '{0}'", swKernelSymbol);
    const auto builtinFuncType = builtinFunc.getFunctionType();
    const auto builtinFuncInputTypes = builtinFuncType.getInputs();
    auto argumentTypes = to_std_vector(builtinFuncInputTypes);
    argumentTypes.reserve(argumentTypes.size() + dynamicOperands.size());
    for (size_t operandIndex : dynamicOperands | reversed) {
        const auto operandType = mlir::cast<NDTypeInterface>(builtinFuncInputTypes[operandIndex]);
        const auto dynamicShapeType =
                mlir::UnrankedMemRefType::get(getSInt32Type(op.getContext()), operandType.getMemSpace());
        modifyType(argumentTypes, operandIndex, dynamicShapeType);
    }
    const auto newBuiltinFuncType =
            mlir::FunctionType::get(op.getContext(), argumentTypes, builtinFuncType.getResults());
    log.trace("Old builtin func signature is '{0}', new signature is '{1}'", builtinFuncType, newBuiltinFuncType);
    builtinFunc.setType(newBuiltinFuncType);
}

void ConvertDynamicMemrefsToBoundedBuffers::safeRunOnModule() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([](mlir::func::FuncOp funcOp) {
        const auto funcType = funcOp.getFunctionType();
        return llvm::none_of(funcType.getInputs(), isDynamicMemref) &&
               llvm::none_of(funcType.getResults(), isDynamicMemref);
    });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
        if (llvm::any_of(op->getOperandTypes(), isDynamicMemref)) {
            return false;
        }
        if (llvm::any_of(op->getResultTypes(), isDynamicMemref)) {
            return false;
        }
        for (auto& region : op->getRegions()) {
            if (llvm::any_of(region.getArgumentTypes(), isDynamicMemref)) {
                return false;
            }
        }
        return true;
    });
    // Core.ReinterpretCast cannot accept or produce bounded buffers, so we introduce
    // a stricter check, than for the general case
    target.addDynamicallyLegalOp<Core::ReinterpretCastOp>([](Core::ReinterpretCastOp op) {
        return !isDynamicMemref(op.getInput().getType()) && !isDynamicMemref(op.getResult().getType()) &&
               !mlir::isa<VPUIP::BoundedBufferType>(op.getInput().getType()) &&
               !mlir::isa<VPUIP::BoundedBufferType>(op.getResult().getType());
    });
    target.addLegalOp<VPUIP::SwKernelRun>();  // handled as part of VPUIP::SwKernelOp

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) -> mlir::Type {
        if (isDynamicMemref(type)) {
            return createBoundedBufferFromDynamicMemref(mlir::cast<mlir::MemRefType>(type));
        }
        return type;
    });
    const auto legalize = [](mlir::OpBuilder& builder, mlir::Type dstType, mlir::ValueRange inputs,
                             mlir::Location loc) -> mlir::Value {
        VPUX_THROW_UNLESS(inputs.size() == 1, "Got wrong number of inputs : {0}", inputs.size());
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, dstType, inputs).getResult(0);
    };
    typeConverter.addSourceMaterialization(legalize);
    typeConverter.addTargetMaterialization(legalize);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuncOpRewrite>(typeConverter, &ctx, _log);
    patterns.add<AllocOpRewrite>(typeConverter, &ctx, _log);
    patterns.add<ReinterpretCastOpRewrite>(typeConverter, &ctx, _log);
    patterns.add<SwKernelOpRewrite>(typeConverter, &ctx, _log);
    patterns.add<GenericOpRewrite>(typeConverter, &ctx, _log);

    auto moduleOp = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(moduleOp, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createConvertDynamicMemrefsToBoundedBuffersPass(Logger log) {
    return std::make_unique<ConvertDynamicMemrefsToBoundedBuffers>(log);
}
