//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/function_statistics_instrumentation.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

using namespace vpux;

namespace {

class FunctionStatisticsInstrumentation final : public mlir::PassInstrumentation {
private:
    Logger _logOps;
    Logger _logConst;
    std::map<std::string, size_t> _functionNumOps;

public:
    explicit FunctionStatisticsInstrumentation(const Logger& log): _logOps(log), _logConst(log) {
        _logOps.setName("function-statistics-instrumentation-ops");
        _logConst.setName("function-statistics-instrumentation-const");
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
        if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
            countConstants(pass, funcOp);
            updateState(pass, funcOp);
            return;
        }

        if (auto moduleOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
            moduleOp.walk([&](mlir::func::FuncOp funcOp) {
                countConstants(pass, funcOp);
                updateState(pass, funcOp);
            });

            _logOps.info("Current state of functions:");
            for (const auto& [funcName, numOps] : _functionNumOps) {
                _logOps.nest().info("function '{0}': {1} operations", funcName, numOps);
            }
        }
    }

private:
    void updateState(mlir::Pass* pass, mlir::func::FuncOp funcOp) {
        const auto countNumOps = [&]() -> size_t {
            return std::distance(funcOp.getOps().begin(), funcOp.getOps().end());
        };

        const auto funcName = funcOp.getSymName().str();
        const auto numOps = countNumOps();
        if (_functionNumOps.find(funcName) == _functionNumOps.end()) {
            _functionNumOps[funcName] = numOps;
            return;
        }
        const auto prevNumOps = _functionNumOps[funcName];
        if (numOps != prevNumOps) {
            _logOps.info("[Pass '{0}'] The number of operations for function '{1}' changed from {2} to {3}",
                         pass->getName(), funcName, prevNumOps, numOps);
            _functionNumOps[funcName] = numOps;
        }
    }

    void countConstants(mlir::Pass* pass, mlir::func::FuncOp funcOp) {
        const auto countNumConstants = [&]() -> std::pair<size_t, size_t> {
            size_t numConstants = 0;
            size_t totalConstSize = 0;
            funcOp.walk([&](mlir::Operation* op) {
                if (mlir::isa<Const::DeclareOp>(op)) {
                    ++numConstants;
                    const auto size =
                            mlir::cast<NDTypeInterface>(op->getResult(0).getType()).getTotalAllocSize().count();
                    totalConstSize += size;
                } else if (auto constBufferOp = mlir::dyn_cast<VPUASM::ConstBufferOp>(op)) {
                    ++numConstants;
                    const auto size = mlir::cast<NDTypeInterface>(constBufferOp.getContentAttr().getType())
                                              .getTotalAllocSize()
                                              .count();
                    totalConstSize += size;
                }
            });
            return std::pair{numConstants, totalConstSize};
        };

        const auto funcName = funcOp.getSymName().str();
        const auto [numConstOps, totalConstSize] = countNumConstants();
        _logConst.info("[Pass '{0}'] Function '{1}' has {2} constants - total size {3} bytes", pass->getName(),
                       funcName, numConstOps, totalConstSize);
    }
};

}  // namespace

void vpux::addFunctionStatisticsInstrumentation(mlir::PassManager& pm, const Logger& log) {
    auto instrumentation = std::make_unique<FunctionStatisticsInstrumentation>(log);
    pm.addInstrumentation(std::move(instrumentation));
}
