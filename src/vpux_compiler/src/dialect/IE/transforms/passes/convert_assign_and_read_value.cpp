//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <intel_npu/prefix.hpp>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTASSIGNREADVALUETORETURNSANDINPUTS
#define GEN_PASS_DEF_CONVERTASSIGNREADVALUETORETURNSANDINPUTS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

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

    auto [netInfo, mainFunc] = net::getFromModule(_topModule);

    const auto mainFuncType = mainFunc.getFunctionType();
    const auto assignInputType = origOp.getInput().getType();
    const auto newReturnsTypes =
            to_small_vector(llvm::concat<const mlir::Type>(mainFuncType.getResults(), llvm::ArrayRef(assignInputType)));
    const auto newMainFuncTypes =
            mlir::FunctionType::get(mainFunc.getContext(), mainFuncType.getInputs(), newReturnsTypes);
    mainFunc.setType(newMainFuncTypes);

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&_topModule->getRegion(0).front(), &builderLog);
    auto outputsInfoBuilder = mlir::OpBuilder::atBlockEnd(&netInfo.getOutputsInfo().front(), builder.getListener());
    const auto outputName = std::string(intel_npu::ASSIGN_PREFIX) + origOp.getName().str();
    net::DataInfoOp::create(outputsInfoBuilder, takeOpLoc(origOp, "assign_{0}", origOp.getName()), outputName,
                            assignInputType);

    rewriter.replaceOp(origOp, origOp.getInput());

    const auto retOps = to_small_vector(mainFunc.getOps<mlir::func::ReturnOp>());
    VPUX_THROW_UNLESS(retOps.size() == 1,
                      "Can't have more than one 'mlir::func::ReturnOp' Operation in main function, got '{0}'",
                      retOps.size());
    auto mainRetOp = retOps.front();
    mainRetOp.getOperandsMutable().append(origOp.getInput());

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

    auto [netInfo, mainFunc] = net::getFromModule(_topModule);

    const auto mainFuncType = mainFunc.getFunctionType();
    const auto readValueInputType = origOp.getInput().getType();
    const auto newInputIndex = mainFunc.getNumArguments();
    const auto newInputsTypes = to_small_vector(
            llvm::concat<const mlir::Type>(mainFuncType.getInputs(), llvm::ArrayRef(readValueInputType)));
    const auto newMainFuncTypes =
            mlir::FunctionType::get(mainFunc.getContext(), newInputsTypes, mainFuncType.getResults());
    mainFunc.setType(newMainFuncTypes);
    mainFunc.front().addArgument(readValueInputType, mainFunc.getLoc());

    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&_topModule->getRegion(0).front(), &builderLog);
    auto inputsInfoBuilder = mlir::OpBuilder::atBlockEnd(&netInfo.getInputsInfo().front(), builder.getListener());
    const auto inputName = std::string(intel_npu::READVALUE_PREFIX) + origOp.getName().str();
    net::DataInfoOp::create(inputsInfoBuilder, takeOpLoc(origOp, "read_{0}", origOp.getName()), inputName,
                            readValueInputType);

    rewriter.replaceOp(origOp, mainFunc.getArgument(newInputIndex));

    return mlir::success();
}

//
// ConvertAssignReadValueToReturnsAndInputs
//

class ConvertAssignReadValueToReturnsAndInputs final :
        public IE::impl::ConvertAssignReadValueToReturnsAndInputsBase<ConvertAssignReadValueToReturnsAndInputs> {
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

    auto mainFunc = getOperation();

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
