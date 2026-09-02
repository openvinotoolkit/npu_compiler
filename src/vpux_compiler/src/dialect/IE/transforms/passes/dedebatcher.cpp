//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/Transforms/Inliner.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mutex>
#include "vpux/compiler/dialect/HostExec/IR/attributes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/wrap_func_attr.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace vpux::IE {
#define GEN_PASS_DECL_DEDEBATCHER
#define GEN_PASS_DEF_DEDEBATCHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

namespace detail {

constexpr StringRef inliningCallAttrName{"dedebatcher_inlined_call"};
void assignCallForInlining(mlir::func::CallOp op) {
    VPUX_THROW_WHEN(op == nullptr, "CallOp must not be null");
    op->setAttr(inliningCallAttrName, mlir::UnitAttr::get(op.getContext()));
}

bool hasAssignedForInlining(mlir::func::CallOp op) {
    VPUX_THROW_WHEN(op == nullptr, "CallOp must not be null");
    return op->hasAttr(inliningCallAttrName);
}

using CoeffExtractor = std::function<std::tuple<int64_t, int64_t>(int64_t, int64_t)>;
std::tuple<std::optional<DebatchCoeffDescription>, int64_t> tryExtractDebatchCoefficients(mlir::Value operand,
                                                                                          CoeffExtractor extract) {
    if (auto convertCast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        // determine a batch dimension index & debatched value
        auto batchedShape = getShape(convertCast.getOperand(0));
        auto debatchedShape = getShape(convertCast.getResult(0));
        for (size_t i = 0; i < batchedShape.size(); i++) {
            const auto batchedInput = batchedShape[Dim(i)];
            const auto debatchedOutput = debatchedShape[Dim(i)];
            // The DebatcherPass executing previously, downgrades the N dimension or func arguments only,
            // thus if dimension values before and after UnrealizedConversionCastOp are different,
            // then this dimension is a batch dimension
            if (debatchedOutput == batchedInput) {
                continue;
            }

            // Remember the batch index and other properties
            DebatchCoeffDescription coeff;
            coeff.batchPositionIndex = Dim(i);
            int64_t ratio = 0;
            std::tie(coeff.desiredBatchValue, ratio) = extract(batchedInput, debatchedOutput);
            return {std::make_optional<DebatchCoeffDescription>(std::move(coeff)), ratio};
        }
    }
    return {std::nullopt, 0};
}
}  // namespace detail

std::tuple<std::optional<DebatchCoeffDescription>, int64_t> tryExtractDebatchCoefficientsFromInput(
        mlir::Value inputArgument) {
    auto extractBatchAttrsFromInputArg = [](int64_t batchFromInputOperand, int64_t batchFromResult) {
        VPUX_THROW_WHEN(batchFromInputOperand % batchFromResult,
                        "batchFromResult: {0} in extractBatchAttrsFromInputArg must divide batchFromInputOperand: {1} "
                        "without a remnant",
                        batchFromResult, batchFromInputOperand);
        return std::make_tuple(batchFromResult, batchFromInputOperand / batchFromResult);
    };
    return detail::tryExtractDebatchCoefficients(inputArgument, extractBatchAttrsFromInputArg);
}

std::tuple<std::optional<DebatchCoeffDescription>, int64_t> tryExtractDebatchCoefficientsFromResult(
        mlir::Value inputArgument) {
    auto extractBatchAttrsFromResult = [](int64_t batchFromInputOperand, int64_t batchFromResult) {
        VPUX_THROW_WHEN(batchFromResult % batchFromInputOperand,
                        "batchFromInputOperand: {0} in extractBatchAttrsFromResult must divide batchFromResult: {1} "
                        "without a remnant",
                        batchFromInputOperand, batchFromResult);
        return std::make_tuple(batchFromInputOperand, batchFromResult / batchFromInputOperand);
    };
    return detail::tryExtractDebatchCoefficients(inputArgument, extractBatchAttrsFromResult);
}

mlir::func::FuncOp cloneFunction(mlir::OpBuilder& moduleBuilder, mlir::func::FuncOp origFunc) {
    VPUX_THROW_WHEN(origFunc == nullptr, "Cannot clone a non-existent function");
    auto funcName = vpux::formatv("{0}_{1}", origFunc.getName(), detail::inliningCallAttrName);
    const auto funcLoc = appendLoc(origFunc.getLoc(), funcName);
    auto newFunc = moduleBuilder.create<mlir::func::FuncOp>(funcLoc, funcName.str(), origFunc.getFunctionType());
    mlir::IRMapping mapper;
    origFunc.getBody().cloneInto(&newFunc.getBody(), mapper);
    newFunc.setNested();
    return newFunc;
}

mlir::ValueRange getOutputShapes(mlir::func::FuncOp caller, mlir::OpBuilder& builder) {
    auto module = vpux::getModuleOp(caller);
    constexpr StringRef shapeCalculationFuncName{"output_shape"};
    auto outputShapeFuncOp = module.lookupSymbol<mlir::func::FuncOp>(shapeCalculationFuncName);
    if (outputShapeFuncOp == nullptr) {
        auto parentModule = module->getParentOfType<mlir::ModuleOp>();
        if (parentModule != nullptr) {
            mlir::OpBuilder moduleBuilder(module);
            moduleBuilder.setInsertionPointAfter(caller);
            auto origFunc = parentModule.lookupSymbol<mlir::func::FuncOp>(shapeCalculationFuncName);
            outputShapeFuncOp = cloneFunction(moduleBuilder, origFunc);
        }
    }

    VPUX_THROW_WHEN(outputShapeFuncOp == nullptr, "HostCompile pipeline must provide the \"{0}\" function",
                    shapeCalculationFuncName);
    auto outputShapeCallOp = builder.create<mlir::func::CallOp>(
            appendLoc(caller.getLoc(), "output_shape"), outputShapeFuncOp, mlir::ValueRange{caller.getArguments()});
    detail::assignCallForInlining(outputShapeCallOp);
    return outputShapeCallOp.getResults();
}

/*
 * @DebatcherInlinerInterface
 * The interface allows all operations (without exceptions) to be inlined in a context
 * of calling of the DeDebatcherPass.
 * Inlining is applied to embed auxiliary functions (e.g., output_shape) directly into the body of the main function.
 * This helps avoid issues caused by introduction of additional pure host-compiled functions when the entire module
 * context is packed into nested modules later in the HostCompile pipeline. By embedding these helpers into main, nested
 * modules remain clean and we prevent mixing different dialects.
 */
struct DebatcherInlinerInterface : public mlir::InlinerInterface {
    using InlinerInterface::InlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

void inlineAuxiliaryCallOps(mlir::func::FuncOp caller, const Logger& log) {
    auto module = vpux::getModuleOp(caller);
    mlir::DenseMap<mlir::func::FuncOp, std::vector<mlir::func::CallOp>> inliningCallOpsPerFunc;
    caller->walk([&](mlir::func::CallOp op) {
        if (detail::hasAssignedForInlining(op)) {
            auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
            VPUX_THROW_WHEN(
                    funcOp == nullptr,
                    "DeDebatcherPass doesn't allow orphan calls, found callOp: {0} doesn't relate to any function",
                    op.getCallee());
            inliningCallOpsPerFunc[funcOp].push_back(op);
            log.trace("CallOps of function: {0} to inline count: {1}", funcOp.getName(),
                      inliningCallOpsPerFunc[funcOp].size());
        }
    });

    log.debug("Found functions to inline count: {0}", inliningCallOpsPerFunc.size());
    size_t inlinedCallOpsCount = 0;
    mlir::InlinerConfig config;
    DebatcherInlinerInterface interface(caller.getContext());
    for (auto [funcOp, callOps] : inliningCallOpsPerFunc) {
        log.trace("Inline function's: {0} calls: {1}", funcOp.getName(), callOps.size());
        for (auto&& callOp : callOps) {
            if (mlir::failed(mlir::inlineCall(interface, config.getCloneCallback(), callOp, funcOp, &funcOp.getBody(),
                                              true))) {
                VPUX_THROW("Cannot inline the function: {0} required for calculation of sizes of output shapes. "
                           "DeDebatching with the host-compile has been failed",
                           funcOp.getName());
            }
            callOp.erase();
            inlinedCallOpsCount++;
        }
        if (funcOp.use_empty() && funcOp.getName().find(detail::inliningCallAttrName) != std::string::npos) {
            log.trace("Remove the intrinsic function: {0}, as it has no uses anymore", funcOp.getName());
            funcOp.erase();
        }
    }
    log.debug("Inlined calls count: {0}", inlinedCallOpsCount);
}

SmallVector<mlir::Value> createLoopCapturedVariablesBoundToCallerOutputs(mlir::func::FuncOp caller,
                                                                         mlir::OpBuilder& builder,
                                                                         mlir::ValueRange outputShapes,
                                                                         const Logger& log) {
    // initialize function outputs, which the loop will use as its results storage,
    // aka captured loop-carried variables
    SmallVector<mlir::Value> loopResults;
    if (caller.isExternal()) {
        // Skip function declarations
        return loopResults;
    }
    auto loc = caller.getLoc();
    for (auto& block : caller.getBody()) {
        for (auto& op : llvm::make_early_inc_range(block)) {
            if (mlir::isa<mlir::func::ReturnOp>(op)) {
                log.debug("Capture caller results variables, count: {0}", op.getOperands().size());
                for (auto const& [resIndex, resOutputShape] : llvm::zip(op.getOperands(), outputShapes) | indexed) {
                    auto [res, pivotArg] = resOutputShape;
                    // for each result operand of `main` we create a corresponding
                    // loop-carried variable, which will accumulate every result
                    // of batched function execution.
                    // Those variables must have been debatched previously,
                    // which means that we can determine DebatchCoeffDescription for them
                    auto [debatchCoefficient, ratio] = tryExtractDebatchCoefficientsFromResult(res);
                    if (!debatchCoefficient.has_value()) {
                        continue;
                    }
                    SmallVector<mlir::Value> outShape;
                    auto argShape = Shape{vpux::getShape(res).raw()};
                    for (size_t i = 0; i < argShape.size(); i++) {
                        if (argShape[Dim{i}] == mlir::ShapedType::kDynamic) {
                            mlir::Location dimLoc = loc;
                            if (debatchCoefficient.value().batchPositionIndex == Dim{i}) {
                                dimLoc = appendLoc(dimLoc, "batchFromOutTensor_{0}_{1}", resIndex, i);
                            } else {
                                dimLoc = appendLoc(dimLoc, "dynOutTensor_{0}_{1}", resIndex, i);
                            }
                            mlir::Value dimIdxCnst = builder.create<mlir::arith::ConstantIndexOp>(dimLoc, i);
                            auto dimValByIdx = builder.create<mlir::tensor::ExtractOp>(dimLoc, pivotArg, dimIdxCnst);
                            auto dynamicDim = builder.create<mlir::arith::IndexCastOp>(dimLoc, builder.getIndexType(),
                                                                                       dimValByIdx.getResult());
                            outShape.push_back(dynamicDim);
                        }
                    }
                    auto outType = mlir::cast<mlir::ShapedType>(res.getType());
                    log.debug("Capture a result: {0} of type: {1} with the debatch coefficient: {2}, output shape: {3}",
                              resIndex, outType, debatchCoefficient.value().to_string(), outShape);
                    loopResults.push_back(builder.create<mlir::tensor::EmptyOp>(appendLoc(loc, "output_{0}", resIndex),
                                                                                outType, outShape));
                }
            }
        }
    }
    return loopResults;
}

SmallVector<mlir::OpFoldResult> collectMixedSizesForSliceOp(mlir::OpBuilder& forBodyBuilder, mlir::Location loc,
                                                            int64_t forStep, mlir::Value arg, int64_t argIndex) {
    SmallVector<mlir::OpFoldResult> sizes{forBodyBuilder.getIndexAttr(forStep)};
    auto argType = mlir::cast<mlir::ShapedType>(arg.getType());
    for (int64_t i = 1; i < argType.getRank(); ++i) {
        if (argType.getDimSize(i) == mlir::ShapedType::kDynamic) {
            // If the dimension value is dynamic, we cannot just insert kDynamic
            // as an opFoldResult into a static sizes array as a validation of extracted slice
            // will return the assert error that the index kDynamic is out of bound.
            // Instead we must create DimOp to get the dynamic dimension value,
            // fold it and use that value as a dynamic size representation
            auto dynamicOpRes = mlir::getAsOpFoldResult(forBodyBuilder.create<mlir::tensor::DimOp>(
                    appendLoc(loc, "dynExtractSliceDim_{0}_{1}", argIndex, i), arg, i));
            sizes.push_back(dynamicOpRes);
        } else {
            sizes.push_back(forBodyBuilder.getIndexAttr(argType.getDimSize(i)));
        }
    }
    return sizes;
}

SmallVector<mlir::Value> generateCalleeInputSlicesFromCallerInputs(mlir::func::FuncOp caller,
                                                                   mlir::OpBuilder& forBodyBuilder,
                                                                   mlir::scf::ForOp forOp, mlir::func::FuncOp callee,
                                                                   const Logger& log) {
    auto forInductionVariable = forOp.getInductionVar();
    auto loc = caller.getLoc();
    auto forStep = getConstantIntValue(forOp.getStep());
    VPUX_THROW_UNLESS(forStep.has_value(), "forOp step must exist and has a constant value");

    // insert input argument slices
    SmallVector<mlir::Value> inputArgSlices;
    log.debug("Extract slice attrs for caller arguments: {0}", caller.getArguments().size());
    for (auto const& [argIndex, arg] : caller.getArguments() | indexed) {
        auto argType = mlir::cast<mlir::ShapedType>(arg.getType());
        SmallVector<mlir::OpFoldResult> offsets{forInductionVariable};
        for (int64_t i = 1; i < argType.getRank(); ++i) {
            offsets.push_back(forBodyBuilder.getIndexAttr(0));
        }

        SmallVector<mlir::OpFoldResult> sizes =
                collectMixedSizesForSliceOp(forBodyBuilder, loc, forStep.value(), arg, argIndex);
        SmallVector<mlir::OpFoldResult> strides(argType.getRank(), forBodyBuilder.getIndexAttr(1));
        auto slice = forBodyBuilder.create<mlir::tensor::ExtractSliceOp>(appendLoc(loc, "slice_{0}", argIndex), arg,
                                                                         offsets, sizes, strides);
        log.debug("Slice: {0} created, sizes: {1}", argIndex, sizes);
        slice.getResult().setType(mlir::cast<mlir::RankedTensorType>(callee.getArgumentTypes()[argIndex]));
        inputArgSlices.push_back(slice);
    }
    return inputArgSlices;
}

void generateCalleeResultSlicesInForCtx(mlir::scf::ForOp forCtx, mlir::OpBuilder& forBodyBuilder,
                                        mlir::func::CallOp calleeOp, const Logger& log) {
    auto loc = forCtx.getLoc();
    auto forInductionVariable = forCtx.getInductionVar();
    auto forStep = getConstantIntValue(forCtx.getStep());
    VPUX_THROW_UNLESS(forStep.has_value(), "forOp step must exist and has a constant value");

    SmallVector<mlir::Value> resultSlices;
    SmallVector<mlir::Value> yieldResults;
    log.debug("Extract slice attrs for callee results: {0}", calleeOp.getResults().size());
    for (auto const& [argIndex, result] : calleeOp.getResults() | indexed) {
        auto outType = mlir::cast<mlir::ShapedType>(result.getType());

        // Insert result gathering slices
        SmallVector<mlir::OpFoldResult> outOffsets{forInductionVariable};
        for (int64_t i = 1; i < outType.getRank(); ++i) {
            outOffsets.push_back(forBodyBuilder.getIndexAttr(0));
        }
        SmallVector<mlir::OpFoldResult> outSizes = collectMixedSizesForSliceOp(
                forBodyBuilder, loc, forStep.value(), forCtx.getRegionIterArg(argIndex), argIndex);
        SmallVector<mlir::OpFoldResult> outStrides(outType.getRank(), forBodyBuilder.getIndexAttr(1));
        auto inserted = forBodyBuilder.create<mlir::tensor::InsertSliceOp>(appendLoc(loc, "insert_{0}", argIndex),
                                                                           result, forCtx.getRegionIterArg(argIndex),
                                                                           outOffsets, outSizes, outStrides);
        log.debug("Slice: {0} created, sizes: {1}", argIndex, outSizes);
        resultSlices.push_back(inserted);
        yieldResults.push_back(inserted.getResult());
    }

    log.debug("Finalize loop creation by yielding given slices count: {0}", yieldResults.size());
    forBodyBuilder.create<mlir::scf::YieldOp>(appendLoc(loc, "yield"), yieldResults);
}

mlir::func::FuncOp getAppropriateDebatchingFunction(mlir::func::CallOp op) {
    auto realDebatchingFunction = getCalledFunction(op);
    const auto funcLoc = appendLoc(realDebatchingFunction.getLoc(), "_unresolved");
    auto funcType = realDebatchingFunction.getFunctionType();
    auto moduleOp = vpux::getModuleOp(realDebatchingFunction);
    mlir::OpBuilder builder(moduleOp.getBodyRegion());
    auto funcCloneStubDeclaration = builder.create<mlir::func::FuncOp>(
            funcLoc, vpux::formatv("{0}_unresolved", realDebatchingFunction.getName()).str(), funcType);
    funcCloneStubDeclaration.setNested();
    config::setFunctionToPackAttribute(realDebatchingFunction, "NPUModule");
    config::setFunctionToPackEntryPointAttribute(realDebatchingFunction);
    vpux::HostExec::setHostCompileInferenceExpectedCommandListsNumber(realDebatchingFunction, 1);
    WrapFuncDataAttributeView::inject(funcCloneStubDeclaration, realDebatchingFunction.getName(), true);
    return funcCloneStubDeclaration;
}

void injectHostPipelineStage(mlir::func::FuncOp main, mlir::func::CallOp callOp, int64_t ratio,
                             ArrayRef<DebatchCoeffDescription> debatchCoeff, mlir::OpBuilder& builder,
                             const Logger& log) {
    // TODO Mutually exclusive passes: Dedebatcher & (createSCFVerticalFusionPass + createApplyTilingPass)
    auto debatchedAttr = DebatchedCallOpAttributeView::extract(callOp);
    if (!debatchedAttr.has_value()) {
        return;
    }

    auto loc = main.getLoc();
    auto mainArgs = main.getArguments();
    builder.setInsertionPointToStart(&main.getBody().front());
    VPUX_THROW_UNLESS(
            debatchCoeff.size() == mainArgs.size(),
            "injectHostPipelineStage params must be consistent, mainArgs count: {0}, debatched coefficients count: {1}",
            mainArgs.size(), debatchCoeff.size());

    // Total loop iteration count is a highest number of "ratio" calculated from
    // function arguments regardless of whether they are batched or not.
    // "Ratio" of a particular argument is the proportion
    // between its actual batch value and the debatched value

    // To cover the situation where we have a Const as a first argument of the function,
    // which obviously has no "ratio" at all, let's look through all arguments
    // and find the "ratio" at least for one argument.
    // We don't need to check all arguments to find out discrepancy in their "ratio"
    // because this step has been already done at a previous phase
    int64_t loopBegin = 0;  // always start from 0
    Dim pivotTensorBatchDimIndex;
    int64_t step = 1;
    std::optional<mlir::Value> pivotArg;
    for (auto const& [argIndex, arg] : mainArgs | indexed) {
        const DebatchCoeffDescription& argDebatchCoeff = debatchCoeff[argIndex];
        auto origShape = Shape{vpux::getShape(arg).raw()};
        auto debatchedShape = argDebatchCoeff.apply(origShape);

        // filter non-batched args out
        if (origShape[argDebatchCoeff.batchPositionIndex] / debatchedShape[argDebatchCoeff.batchPositionIndex] !=
            ratio) {
            continue;
        }
        pivotTensorBatchDimIndex = Dim(argDebatchCoeff.batchPositionIndex);
        pivotArg = arg;
        step = debatchedShape[argDebatchCoeff.batchPositionIndex];
        break;
    }
    VPUX_THROW_UNLESS(pivotArg.has_value(), "injectHostPipelineStage failed because there is nothing to iterate");

    // initialize loop arguments
    auto batchIterBegin = builder.create<mlir::arith::ConstantIndexOp>(appendLoc(loc, "batchIterBegin"), loopBegin);
    auto batchDimIndex =
            builder.create<mlir::arith::ConstantIndexOp>(appendLoc(loc, "batchIndex"), pivotTensorBatchDimIndex.ind());
    auto batchFromTensor =
            builder.create<mlir::tensor::DimOp>(appendLoc(loc, "batchFromTensor"), *pivotArg, batchDimIndex);
    auto batchStep = builder.create<mlir::arith::ConstantIndexOp>(appendLoc(loc, "batchStep"), step);

    // determine which output shapes are required for results
    auto outputShapes = getOutputShapes(main, builder);
    auto loopCapturedResults = createLoopCapturedVariablesBoundToCallerOutputs(main, builder, outputShapes, log);

    log.debug("Insert a forOp over batch dimension using captured variables count: {0}", loopCapturedResults.size());
    auto forOp = builder.create<mlir::scf::ForOp>(appendLoc(loc, "for"), batchIterBegin, batchFromTensor, batchStep,
                                                  mlir::ValueRange{loopCapturedResults});
    mlir::func::FuncOp batchingFunc = getAppropriateDebatchingFunction(callOp);
    auto loopBodyBuilder = mlir::OpBuilder(forOp.getBody(), forOp.getBody()->begin());
    auto inputArgSlices = generateCalleeInputSlicesFromCallerInputs(main, loopBodyBuilder, forOp, batchingFunc, log);

    log.debug("Insert a batching function call on given slices count: {0}", inputArgSlices.size());
    auto call = loopBodyBuilder.create<mlir::func::CallOp>(appendLoc(loc, "call"), batchingFunc,
                                                           mlir::ValueRange{inputArgSlices});
    generateCalleeResultSlicesInForCtx(forOp, loopBodyBuilder, call, log);

    // Replace original main function returns with new yielded results
    for (auto& block : main.getBody()) {
        for (auto& op : llvm::make_early_inc_range(block)) {
            if (mlir::isa<mlir::func::ReturnOp>(op)) {
                for (auto const& [returnOperandIdx, retOp] : forOp.getResults() | indexed) {
                    op.setOperand(returnOperandIdx, forOp.getResult(returnOperandIdx));
                }
            }
        }
    }

    // Do not keep utility functions in the module, as it may affect further passes
    inlineAuxiliaryCallOps(main, log);
}

std::tuple<int64_t, int64_t, SmallVector<DebatchCoeffDescription>> getDeDebatchParams(
        mlir::func::CallOp callOp, bool injectDebatchingReorderingAttr) {
    const auto callOperands = callOp.getOperands();
    SmallVector<DebatchCoeffDescription> debatchCoefficients;
    int64_t dedebatchNum = 0, castOpCnt = 0;
    for (auto operand : callOperands) {
        auto [debatchCoefficient, ratio] = tryExtractDebatchCoefficientsFromInput(operand);
        // skip not batched/debatched arguments
        if (!debatchCoefficient.has_value()) {
            continue;
        }

        if (dedebatchNum == 0) {
            dedebatchNum = ratio;
        }
        VPUX_THROW_UNLESS(dedebatchNum == ratio, "De-de-batch number is not matched for various inputs");
        castOpCnt++;
        debatchCoefficients.push_back(std::move(debatchCoefficient.value()));
    }

    if (dedebatchNum != 0) {
        DebatchedCallOpAttributeView::inject(callOp, castOpCnt, dedebatchNum);
        if (injectDebatchingReorderingAttr) {
            DebatchedCallOpAttributeView::setReorderingAttr(callOp);
        }
    }
    return std::make_tuple(dedebatchNum, castOpCnt, debatchCoefficients);
}

mlir::SmallVector<mlir::Operation*> sliceCallsOp(mlir::OpBuilder& builder, mlir::func::CallOp callOp,
                                                 const int64_t dedebatchNum, bool injectDebatchingReorderingAttr) {
    const auto callLoc = callOp.getLoc();
    const auto funcOperands = callOp.getOperands();
    auto newCallOps = SmallVector<mlir::Operation*>();
    for (int i = 0; i < dedebatchNum; i++) {
        // Create sliced function operands
        mlir::SmallVector<mlir::Value> newOperands;
        size_t sliceIdx = 0;
        for (auto operand : funcOperands) {
            if (auto convertCast = operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
                auto batchedInput = convertCast.getOperand(0);
                auto debatchedInput = convertCast.getResult(0);
                builder.setInsertionPoint(callOp);
                // prepare  slice offset: we must create slice offset with shape rank
                // equal to the batched operand rank
                Shape sliceOffset{
                        SmallVector<int64_t>(mlir::cast<vpux::NDTypeInterface>(batchedInput.getType()).getRank(), 0)};
                sliceOffset[Dims4D::Act::N] = getShape(debatchedInput)[Dims4D::Act::N] * i;
                const auto sliceLoc = appendLoc(callLoc, "slice_{0}_op_{1}", sliceIdx + 1, i + 1);
                auto slicedOperand =
                        builder.create<IE::SliceOp>(sliceLoc, batchedInput, sliceOffset, getShape(debatchedInput));
                newOperands.push_back(slicedOperand.getResult());
                sliceIdx++;
            } else {
                newOperands.push_back(operand);
            }
        }

        // Create multi-batched function calls
        auto newCall =
                builder.create<mlir::func::CallOp>(callLoc, callOp.getCallee(), callOp->getResultTypes(), newOperands);
        DebatchedCallOpAttributeView::inject(newCall, i, dedebatchNum);
        if (injectDebatchingReorderingAttr) {
            DebatchedCallOpAttributeView::setReorderingAttr(newCall);
        }
        newCallOps.push_back(newCall);
    }
    return newCallOps;
}

void concatenateCallOps(mlir::OpBuilder& builder, mlir::func::CallOp callOp, ArrayRef<mlir::Operation*> newCallOps) {
    const auto callLoc = callOp.getLoc();
    const auto funcResNum = callOp.getResults().size();
    for (size_t i = 0; i < funcResNum; i++) {
        mlir::SmallVector<mlir::Value> newCallResults;
        for (auto newCall : newCallOps) {
            auto res = newCall->getResult(i);
            newCallResults.push_back(res);
        }

        const auto concatCallLoc = appendLoc(callLoc, "" + std::to_string(i));
        auto newConcatResult = builder.create<IE::ConcatOp>(concatCallLoc, newCallResults, 0);
        auto origCallResUsers = callOp.getResult(i).getUsers();
        for (auto usr : origCallResUsers) {
            usr->getResult(0).replaceAllUsesWith(newConcatResult->getResult(0));
        }
    }
    return;
}

//
// DeDebatcherPass
//

class DeDebatcherPass final : public IE::impl::DeDebatcherBase<DeDebatcherPass> {
public:
    explicit DeDebatcherPass(const DebatcherOptions& options, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        Base::copyOptionValuesFrom(options);
        log.debug("Create {0}", getName());
    }

    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;

private:
    void safeRunOnModule() final;
    mlir::LogicalResult parseFromOptions();
};

mlir::LogicalResult DeDebatcherPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }

    _log.trace("{0}: {1}", debatcherMethod.getArgStr(), debatcherMethod.getValue());
    _log.trace("initializing of {0} succeeded", getName());
    return mlir::success();
}

//
// safeRunOnModule
//

void DeDebatcherPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto mainFuncOp = net::getMainFunc(moduleOp);

    mlir::OpBuilder builder(mainFuncOp);
    auto parsedDebatcherMethod = this->debatcherMethod.getValue();
    bool injectDebatchingReorderingMethodAttr = (parsedDebatcherMethod == "reordering");
    // TODO E#186494 Mutually exclusive passes: Dedebatcher & (createSCFVerticalFusionPass + createApplyTilingPass)
    bool generateHostPipeline = (parsedDebatcherMethod == "host_pipeline");
    VPUX_THROW_WHEN(generateHostPipeline && injectDebatchingReorderingMethodAttr,
                    "generateHostPipeline && injectDebatchingReorderingMethodAttr cannot coexist, the debatcher method "
                    "requested: {0}",
                    parsedDebatcherMethod);
    _log.debug("{0} applying method: {1}", getName(), parsedDebatcherMethod);

    // Check all function calls in main function
    auto callOps = mainFuncOp.getFunctionBody().getOps<mlir::func::CallOp>();
    for (auto callOp : callOps) {
        //  Acquire and validate de-debatch number
        auto [dedebatchNum, castOpCnt, debatchCoefficients] =
                getDeDebatchParams(callOp, injectDebatchingReorderingMethodAttr);
        // Not batched case
        if (castOpCnt == 0) {
            continue;
        }

        if (generateHostPipeline) {
            injectHostPipelineStage(mainFuncOp, callOp, dedebatchNum, debatchCoefficients, builder, _log);
            vpux::runLocalDCE(mainFuncOp);
            config::setPureHostCompileFuncAttribute(mainFuncOp);
            return;
        }
        // Get multi-batch sliced function calls
        auto newCallOps = sliceCallsOp(builder, callOp, dedebatchNum, injectDebatchingReorderingMethodAttr);

        // Create concat for multi-batched function results
        concatenateCallOps(builder, callOp, newCallOps);
    }
}
}  // namespace

//
// createDeDebatcherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDeDebatcherPass(const DebatcherOptions& options, Logger log) {
    return std::make_unique<DeDebatcherPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createDeDebatcherPass(Logger log) {
    return createDeDebatcherPass(DebatcherOptions{}, log);
}
