//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/IE/prefix.hpp"

using namespace vpux;

namespace {

//
// AssignRewriter
//

class AssignRewriter final : public mlir::OpRewritePattern<IE::AssignOp> {
public:
    AssignRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpRewritePattern<IE::AssignOp>(ctx), _topModule(module), _log(log) {
        setDebugName("AssignRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AssignOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::ModuleOp _topModule;
    Logger _log;
};

mlir::LogicalResult AssignRewriter::matchAndRewrite(IE::AssignOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Assign layer at '{1}'", getDebugName(), origOp->getLoc());

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp mainFunc;
    IE::CNNNetworkOp::getFromModule(_topModule, netInfo, mainFunc);

    const auto mainFuncType = mainFunc.getType();
    const auto assignInputType = origOp.input().getType();
    const auto newReturnsTypes = to_small_vector(
            llvm::concat<const mlir::Type>(mainFuncType.getResults(), llvm::makeArrayRef(assignInputType)));
    const auto newMainFuncTypes =
            mlir::FunctionType::get(mainFunc.getContext(), mainFuncType.getInputs(), newReturnsTypes);
    mainFunc.setType(newMainFuncTypes);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&_topModule->getRegion(0).front(), &builderLog);
    auto outputsInfoBuilder = mlir::OpBuilder::atBlockEnd(&netInfo.outputsInfo().front(), builder.getListener());
    auto* ctx = builder.getContext();
    const auto outputTypeAttr = mlir::TypeAttr::get(assignInputType);
    const auto outputNameAttr = mlir::StringAttr::get(ctx, ASSIGN_PREFIX + origOp.name());
    outputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), outputNameAttr, outputTypeAttr);

    rewriter.replaceOp(origOp, origOp.input());

    const auto retOps = to_small_vector(mainFunc.getOps<mlir::ReturnOp>());
    VPUX_THROW_UNLESS(retOps.size() == 1,
                      "Can't have more than one 'mlir::ReturnOp' Operation in main function, got '{0}'", retOps.size());
    auto mainRetOp = retOps.front();
    mainRetOp.operandsMutable().append(origOp.input());

    return mlir::success();
}

//
// ReadValueRewriter
//

class ReadValueRewriter final : public mlir::OpRewritePattern<IE::ReadValueOp> {
public:
    ReadValueRewriter(mlir::MLIRContext* ctx, mlir::ModuleOp module, Logger log)
            : mlir::OpRewritePattern<IE::ReadValueOp>(ctx), _topModule(module), _log(log) {
        setDebugName("ReadValueRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReadValueOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::ModuleOp _topModule;
    Logger _log;
};

mlir::LogicalResult ReadValueRewriter::matchAndRewrite(IE::ReadValueOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReadValue layer at '{1}'", getDebugName(), origOp->getLoc());

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp mainFunc;
    IE::CNNNetworkOp::getFromModule(_topModule, netInfo, mainFunc);

    const auto mainFuncType = mainFunc.getType();
    const auto readValueInputType = origOp.input().getType();
    const auto newInputIndex = mainFunc.getNumArguments();
    const auto newInputsTypes = to_small_vector(
            llvm::concat<const mlir::Type>(mainFuncType.getInputs(), llvm::makeArrayRef(readValueInputType)));
    const auto newMainFuncTypes =
            mlir::FunctionType::get(mainFunc.getContext(), newInputsTypes, mainFuncType.getResults());
    mainFunc.setType(newMainFuncTypes);
    mainFunc.front().addArgument(readValueInputType, mainFunc.getLoc());

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&_topModule->getRegion(0).front(), &builderLog);
    auto inputsInfoBuilder = mlir::OpBuilder::atBlockEnd(&netInfo.inputsInfo().front(), builder.getListener());
    auto* ctx = builder.getContext();
    const auto inputTypeAttr = mlir::TypeAttr::get(readValueInputType);
    const auto inputNameAttr = mlir::StringAttr::get(ctx, READVALUE_PREFIX + origOp.name());
    inputsInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx), inputNameAttr, inputTypeAttr);

    rewriter.replaceOp(origOp, mainFunc.getArgument(newInputIndex));

    return mlir::success();
}

//
// ConvertAssignReadValueToReturnsAndInputs
//

class ConvertAssignReadValueToReturnsAndInputs final :
        public IE::ConvertAssignReadValueToReturnsAndInputsBase<ConvertAssignReadValueToReturnsAndInputs> {
public:
    explicit ConvertAssignReadValueToReturnsAndInputs(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertAssignReadValueToReturnsAndInputs::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::AssignOp>();
    target.addIllegalOp<IE::ReadValueOp>();

    auto function = getOperation();
    auto topModule = getModuleOp(function);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ReadValueRewriter>(&ctx, topModule, _log);
    patterns.insert<AssignRewriter>(&ctx, topModule, _log);

    auto mainFunc = getFunction();

    if (mlir::failed(mlir::applyPartialConversion(mainFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAssignReadValueToReturnsAndInputs
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertAssignReadValueToReturnsAndInputs(Logger log) {
    return std::make_unique<ConvertAssignReadValueToReturnsAndInputs>(log);
}
