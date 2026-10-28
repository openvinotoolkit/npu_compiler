//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEPRECISIONACROSSFUNCTIONCALLS
#define GEN_PASS_DEF_OPTIMIZEPRECISIONACROSSFUNCTIONCALLS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class OptimizeDequantQuantPair final : public mlir::OpRewritePattern<IE::QuantizeOp> {
private:
    Logger _log;

public:
    OptimizeDequantQuantPair(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("OptimizeDequantQuantPair");
    }

    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), quantOp->getName(), quantOp->getLoc());
        const auto dequantOp = quantOp->getOperand(0).getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::DequantizeOp>(dequantOp)) {
            return matchFailed(_log, rewriter, quantOp, "Missing Dequantize parent op");
        }
        const auto origType = dequantOp->getOperand(0).getType();
        if (origType != quantOp->getResult(0).getType()) {
            return matchFailed(_log, rewriter, quantOp,
                               "Input type for Dequantize is different from the Quantize output type");
        }
        _log.nest().trace("Optimizing Dequant->Quant pair at '{0}'", quantOp->getLoc());
        rewriter.replaceAllUsesWith(quantOp->getResult(0), dequantOp->getOperand(0));
        return mlir::success();
    }
};

class OptimizeConvertPair final : public mlir::OpRewritePattern<IE::ConvertOp> {
private:
    Logger _log;

public:
    OptimizeConvertPair(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("OptimizeConvertPair");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), convertOp->getName(), convertOp->getLoc());
        const auto parentConvertOp = convertOp->getOperand(0).getDefiningOp();
        if (!mlir::isa_and_nonnull<IE::ConvertOp>(parentConvertOp)) {
            return matchFailed(_log, rewriter, convertOp, "Missing Convert parent op");
        }
        const auto origType = parentConvertOp->getOperand(0).getType();
        if (origType != convertOp->getResult(0).getType()) {
            return matchFailed(_log, rewriter, convertOp,
                               "Input type for parent Convert is different from the Convert output type");
        }
        _log.nest().trace("Optimizing Convert->Convert pair at '{0}'", convertOp->getLoc());
        rewriter.replaceAllUsesWith(convertOp->getResult(0), parentConvertOp->getOperand(0));
        return mlir::success();
    }
};

class OptimizePrecisionAcrossFunctionCallsPass final :
        public IE::impl::OptimizePrecisionAcrossFunctionCallsBase<OptimizePrecisionAcrossFunctionCallsPass> {
private:
    using FunctionCalls = DenseMap<mlir::func::FuncOp, SmallVector<mlir::func::CallOp>>;
    using CallFunction = DenseMap<mlir::func::CallOp, mlir::func::FuncOp>;
    FunctionCalls _functionCalls;
    CallFunction _callFunction;

public:
    explicit OptimizePrecisionAcrossFunctionCallsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final {
        collectFunctionCalls();
        if (_functionCalls.empty()) {
            _log.trace("No functions to optimize");
        }

        auto moduleOp = getOperation();
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            _log.trace("Handling function {0}", funcOp.getSymName());

            if (_functionCalls[funcOp].empty()) {
                _log.nest().trace("Function has no calls");
                return;
            }

            DenseMap<size_t, SmallVector<mlir::Operation*>> argOperations;
            DenseMap<size_t, mlir::Operation*> resOperations;
            findOptimizableArgumentsAndResults(funcOp, argOperations, resOperations);
            if (argOperations.empty() && resOperations.empty()) {
                _log.nest().trace("Function has no optimizeable arguments or results");
                return;
            }
            _log.nest().trace("Function has {0} optimizable arguments and {1} optimizable results",
                              argOperations.size(), resOperations.size());

            updateCallOperations(funcOp, argOperations, resOperations);
            updateFunction(funcOp, argOperations, resOperations);
        });

        _log.trace("Optimizing pairs of precision conversion operations");

        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<OptimizeDequantQuantPair>(&getContext(), _log);
        patterns.add<OptimizeConvertPair>(&getContext(), _log);
        if (mlir::failed(mlir::applyPatternsGreedily(moduleOp, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
            signalPassFailure();
        }
    }

    // Collect information on what CallOps each function has, as well as what FuncOp each CallOp refers to
    void collectFunctionCalls() {
        auto moduleOp = getOperation();
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            funcOp.walk([&](mlir::func::CallOp callOp) {
                auto calledFuncOp = vpux::getCalledFunction(callOp);
                _functionCalls[calledFuncOp].push_back(callOp);
                _callFunction[callOp] = calledFuncOp;
            });
        });
    }

    // Find the function arguments and results which can be optimized in terms of element type.
    // An argument can be optimized if it is only consumed by Quantize / Convert operations; if there are more than one
    // one such operations, they must all produce the same type. This operation can be moved outside the function, near
    // its call operations, and potentially reduced if a matching Dequantize / Convert is also present or moved there.
    // Optimizable results are treated identified the same way as arguments, in case the producer operation is a
    // Dequantize / Convert one.
    void findOptimizableArgumentsAndResults(mlir::func::FuncOp funcOp,
                                            DenseMap<size_t, SmallVector<mlir::Operation*>>& argOperations,
                                            DenseMap<size_t, mlir::Operation*>& resOperations) {
        const auto findCompatibleUserOps =
                [](mlir::BlockArgument arg) -> mlir::FailureOr<SmallVector<mlir::Operation*>> {
            SmallVector<mlir::Operation*> userOps;
            for (auto userOp : arg.getUsers()) {
                if (!mlir::isa<IE::QuantizeOp, IE::ConvertOp>(userOp)) {
                    return mlir::failure();
                }
                if (!userOps.empty()) {
                    const auto firstOp = userOps.front();
                    if (userOp->getName() != firstOp->getName() ||
                        userOp->getResult(0).getType() != firstOp->getResult(0).getType()) {
                        return mlir::failure();
                    }
                }
                userOps.push_back(userOp);
            }
            return userOps;
        };

        _log.nest().trace("Searching for optimizable arguments and results");

        for (auto arg : funcOp.getArguments()) {
            const auto userOps = findCompatibleUserOps(arg);
            if (mlir::failed(userOps) || userOps.value().empty()) {
                continue;
            }
            const auto numCompatibleProducers =
                    countCompatibleOuterProducerOps(funcOp, arg.getArgNumber(), userOps.value().front());
            if (numCompatibleProducers < 1) {
                continue;
            }
            argOperations[arg.getArgNumber()] = userOps.value();
            _log.nest(2).trace("Argument {0} is optimizable with {1} {2} user(s) and {3} producer(s)",
                               arg.getArgNumber(), userOps.value().size(), userOps.value().front()->getName(),
                               numCompatibleProducers);
        }

        const auto returnOp = findReturnOp(funcOp);
        if (returnOp == nullptr) {
            _log.nest(2).debug("Unable to find return operation for function {0}", funcOp.getSymName());
            return;
        }
        for (auto& operand : returnOp->getOpOperands()) {
            const auto producerOp = operand.get().getDefiningOp();
            if (!mlir::isa_and_nonnull<IE::DequantizeOp, IE::ConvertOp>(producerOp)) {
                continue;
            }
            const auto numCompatibleUsers = countCompatibleOuterUserOps(funcOp, operand.getOperandNumber(), producerOp);
            if (numCompatibleUsers < 1) {
                continue;
            }
            resOperations[operand.getOperandNumber()] = producerOp;
            _log.nest(2).trace("Result {0} is optimizable with {1} producer(s) and {2} user(s)",
                               operand.getOperandNumber(), producerOp->getName(), numCompatibleUsers);
        }
    }

    bool isOperationPairOptimizable(mlir::Operation* producerOp, mlir::Operation* userOp) {
        if (producerOp->getNumOperands() != 1 || userOp->getNumResults() != 1) {
            return false;
        }
        if (producerOp->getOperand(0).getType() != userOp->getResult(0).getType()) {
            return false;
        }
        return (mlir::isa<IE::DequantizeOp>(producerOp) && mlir::isa<IE::QuantizeOp>(userOp)) ||
               (mlir::isa<IE::ConvertOp>(producerOp) && mlir::isa<IE::ConvertOp>(userOp));
    }

    // Count the producer operations which are connected to the given block argument of the function and which can be
    // optimized together with the given user operation. The producers can either be direct parents to the function's
    // call operation, or they can be found inside another function when the parent op is a call operation. Multiple
    // producers can be returned in case the function is called multiple times
    size_t countCompatibleOuterProducerOps(mlir::func::FuncOp funcOp, size_t argNumber, mlir::Operation* userOp) {
        size_t numCompatibleProducerOps = 0;
        for (auto callOp : _functionCalls[funcOp]) {
            auto operand = callOp.getOperand(argNumber);
            auto producerOp = operand.getDefiningOp();
            if (producerOp == nullptr) {
                continue;
            }
            if (isOperationPairOptimizable(producerOp, userOp)) {
                ++numCompatibleProducerOps;
                continue;
            }
            if (auto producerCallOp = mlir::dyn_cast<mlir::func::CallOp>(producerOp)) {
                if (!_callFunction.contains(producerCallOp)) {
                    _log.warning("Call operation {0} at {1} is not tracked in operation cache. Skipping producer",
                                 (producerCallOp != nullptr) ? producerCallOp->getName().getStringRef().str() : nullptr,
                                 (producerCallOp != nullptr) ? producerCallOp->getLoc() : nullptr);
                    continue;
                }
                const auto producerCallResult = mlir::cast<mlir::OpResult>(operand);
                _callFunction[producerCallOp].walk([&](mlir::func::ReturnOp returnOp) {
                    auto operand = returnOp.getOperand(producerCallResult.getResultNumber());
                    auto producerOp = operand.getDefiningOp();
                    if (producerOp == nullptr) {
                        return;
                    }
                    if (isOperationPairOptimizable(producerOp, userOp)) {
                        ++numCompatibleProducerOps;
                    }
                });
            }
        }
        return numCompatibleProducerOps;
    }

    // Count the user operations which use the given result of the function and which can be optimized together with the
    // given producer operation. The users can either be direct users to the function's call operation, or
    // they can be found inside another function when the user op is a call operation
    size_t countCompatibleOuterUserOps(mlir::func::FuncOp funcOp, size_t resultNumber, mlir::Operation* producerOp) {
        size_t numCompatibleUserOps = 0;
        for (auto callOp : _functionCalls[funcOp]) {
            auto result = callOp->getResult(resultNumber);
            for (auto& use : result.getUses()) {
                const auto userOp = use.getOwner();
                if (isOperationPairOptimizable(producerOp, userOp)) {
                    ++numCompatibleUserOps;
                    continue;
                }
                if (auto userCallOp = mlir::dyn_cast<mlir::func::CallOp>(userOp)) {
                    if (!_callFunction.contains(userCallOp)) {
                        _log.debug("Call operation {0} at {1} is not tracked in operation cache. Skipping user",
                                   (userCallOp != nullptr) ? userCallOp->getName().getStringRef().str() : nullptr,
                                   (userCallOp != nullptr) ? userCallOp->getLoc() : nullptr);
                        continue;
                    }
                    const auto argNumber = use.getOperandNumber();
                    const auto funcArg = _callFunction[userCallOp].getArgument(argNumber);
                    for (auto argUserOp : funcArg.getUsers()) {
                        if (isOperationPairOptimizable(producerOp, argUserOp)) {
                            ++numCompatibleUserOps;
                        }
                    }
                }
            }
        }
        return numCompatibleUserOps;
    }

    // Clone the conversion operations near the call operations and connect them to the given arguments and results
    void updateCallOperations(mlir::func::FuncOp funcOp,
                              const DenseMap<size_t, SmallVector<mlir::Operation*>>& argOperations,
                              const DenseMap<size_t, mlir::Operation*>& resOperations) {
        _log.nest().trace("Updating call operations");

        for (auto callOp : _functionCalls[funcOp]) {
            _log.nest(2).trace("Handling call operation '{0}'", callOp->getLoc());

            mlir::OpBuilder builder(callOp);
            for (const auto& [argNum, userOps] : argOperations) {
                if (userOps.empty()) {
                    _log.warning("Missing users for argument {0}", argNum);
                    continue;
                }
                // All users of an argument are expected to be the same, so the first one is used for cloning
                const auto userOp = userOps.front();
                mlir::IRMapping mapper;
                mapper.map(userOp->getOperand(0), callOp->getOperand(argNum));
                const auto newOp = builder.clone(*userOp, mapper);
                callOp.setOperand(argNum, newOp->getResult(0));
                _log.nest(3).trace("Inserted operation of type '{0}' for argument {1}", newOp->getName(), argNum);
            }

            builder.setInsertionPointAfter(callOp);
            for (const auto& [resNum, producerOp] : resOperations) {
                callOp.getResult(resNum).setType(producerOp->getOperand(0).getType());
                mlir::IRMapping mapper;
                mapper.map(producerOp->getOperand(0), callOp->getResult(resNum));
                const auto newOp = builder.clone(*producerOp, mapper);
                callOp.getResult(resNum).replaceAllUsesExcept(newOp->getResult(0), newOp);
                _log.nest(3).trace("Inserted operation of type '{0}' for result {1}", newOp->getName(), resNum);
            }
        }
    }

    // Update the function arguments & results to the new precision and remove the original conversion operations
    void updateFunction(mlir::func::FuncOp funcOp, const DenseMap<size_t, SmallVector<mlir::Operation*>>& argOperations,
                        const DenseMap<size_t, mlir::Operation*>& resOperations) {
        _log.nest().trace("Updating function signature and operations");

        for (const auto& [argNum, userOps] : argOperations) {
            if (userOps.empty()) {
                _log.warning("Missing users for argument {0}", argNum);
                continue;
            }
            const auto origArgType = funcOp.getArgument(argNum).getType();
            const auto newArgType = userOps.front()->getResult(0).getType();
            VPUX_THROW_WHEN(failed(funcOp.insertArgument(argNum, newArgType, /*argAttrs=*/nullptr,
                                                         funcOp.getArgument(argNum).getLoc())),
                            "Failed to insert argument #{0} of type '{1}' into function '{2}'", argNum, newArgType,
                            funcOp.getSymName());

            const auto newArg = funcOp.getArgument(argNum);

            for (auto userOp : llvm::make_early_inc_range(userOps)) {
                for (auto result : userOp->getResults()) {
                    result.replaceAllUsesWith(newArg);
                }
                userOp->erase();
            }

            VPUX_THROW_WHEN(failed(funcOp.eraseArgument(argNum + 1)),
                            "Failed to erase argument at index {0} in function '{1}'", argNum + 1, funcOp.getSymName());

            _log.nest(2).trace("Replaced argument {0} of type '{1}' with argument of type '{2}'", argNum, origArgType,
                               newArgType);
        }

        const auto returnOp = findReturnOp(funcOp);
        for (const auto& [resNum, producerOp] : resOperations) {
            const auto origResType = returnOp->getOperand(resNum).getType();
            returnOp->setOperand(resNum, producerOp->getOperand(0));
            if (producerOp->use_empty()) {
                producerOp->erase();
            }

            SmallVector<mlir::Type> resultTypes(funcOp.getResultTypes());
            resultTypes[resNum] = returnOp->getOperand(resNum).getType();
            funcOp.setType(mlir::FunctionType::get(funcOp->getContext(), funcOp.getArgumentTypes(), resultTypes));

            const auto newResType = returnOp->getOperand(resNum).getType();
            _log.nest(2).trace("Replaced result {0} of type '{1}' with argument of type '{2}'", resNum, origResType,
                               newResType);
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizePrecisionAcrossFunctionCallsPass(const Logger& log) {
    return std::make_unique<OptimizePrecisionAcrossFunctionCallsPass>(log);
}
