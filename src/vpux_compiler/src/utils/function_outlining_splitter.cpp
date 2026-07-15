//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/async_dialect_utils.hpp"

using namespace vpux;
void vpux::printOutliningInstances(ArrayRef<OutliningInstance> outliningInstances, Logger& log) {
    if (!log.isActive(LogLevel::Debug)) {
        return;
    }
    log.debug("Functions to outline: {0}", outliningInstances.size());
    for (auto& outliningInstance : outliningInstances) {
        log.nest().debug("Number of instances in IR: {0}", outliningInstance.size());
        for (const auto& p : outliningInstance | indexed) {
            const auto& slice = p.value();
            log.nest().debug("Instance {0}", p.index());
            log.nest(2).debug("Input values: {0}", slice.inputs.size());
            for (auto input : slice.inputs) {
                auto producerOp = input.getDefiningOp();
                if (producerOp != nullptr) {
                    log.nest(3).debug("{0} at {1}", producerOp->getName(), producerOp->getLoc());
                    continue;
                }
                log.nest(3).debug("{0}", input);
            }
            log.nest(2).debug("Output values: {0}", slice.outputs.size());
            for (auto output : slice.outputs) {
                auto producerOp = output.getDefiningOp();
                if (producerOp != nullptr) {
                    log.nest(3).debug("{0} at {1}", producerOp->getName(), producerOp->getLoc());
                    continue;
                }
                log.nest(3).debug("{0}", output);
            }
            log.nest(2).debug("Number of operations in slice: {0}", slice.operations.size());
            for (auto op : slice.operations) {
                log.nest(3).debug("Operation {0} at {1}", op->getName(), op->getLoc());
            }
            if (!slice.inputUserMapping.empty()) {
                log.nest(2).debug("Input user mapping");
                for (const auto& [argIdx, user] : slice.inputUserMapping | indexed) {
                    log.nest(3).debug("Argument {0}, user operation {1}, operand {2}", argIdx, user.first->getName(),
                                      user.second);
                }
            }
        }
    }
}

bool OutlinerBase::outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) {
    auto netFunc = net::getMainFunc(moduleOp);

    auto outlinedTargets = getOutliningTargets(netFunc);
    if (outlinedTargets.empty()) {
        _log.debug("Empty outline targets");
        return false;
    }

    _log.info("Creating {0} functions", outlinedTargets.size());

    SmallVector<SmallVector<FuncInfo>> funcsInfo(outlinedTargets.size());
    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto slice = slices.front();
        SmallVector<mlir::Type> inputTypes;
        SmallVector<mlir::Type> outputTypes;
        for (const auto input : slice.inputs) {
            inputTypes.push_back(getAsyncValueType(input));
        }
        for (const auto output : slice.outputs) {
            outputTypes.push_back(getAsyncValueType(output));
        }
        auto funcName = printToString("{0}_{1}{2}", netFunc.getName(), functionSuffix, targetIdx + 1);
        funcsInfo[targetIdx].push_back({std::move(inputTypes), std::move(outputTypes), std::move(funcName)});
    }

    buildFuncOps(moduleOp, funcsInfo, outlinedTargets);
    buildCallOps(moduleOp, funcsInfo, outlinedTargets);
    updateMainFuncOp(moduleOp, outlinedTargets);
    return true;
}
