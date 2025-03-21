//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

//
// Default number of tasks per block when control graph split is enabled
//
#define CONTROL_GRAPH_SPLIT_BLOCK_SIZE 5000

//
// Default threshold for number of virtual barriers for workload management optimization
//
#define VIRTUAL_BARRIER_THRESHOLD_WLM 10000

//
// Default threshold for number of clustered op for op tiling cache optimization
//
#define CLUSTERED_OP_THRESHOLD_FOR_TILING_CACHE 100

//
// Default file names to dump and read manual strategies from
//

constexpr StringLiteral writeStrategyFileLocation = "strategy_out.json";
constexpr StringLiteral readStrategyFileLocation = "strategy_in.json";

//
// PatternBenefit
//

extern const mlir::PatternBenefit benefitLow;
extern const mlir::PatternBenefit benefitMid;
extern const mlir::PatternBenefit benefitHigh;

SmallVector<mlir::PatternBenefit> getBenefitLevels(uint32_t levels);

//
// FunctionPass
//

class FunctionPass : public mlir::OperationPass<mlir::func::FuncOp> {
public:
    using mlir::OperationPass<mlir::func::FuncOp>::OperationPass;

protected:
    void initLogger(Logger log, StringLiteral passName);

protected:
    virtual void safeRunOnFunc() = 0;

protected:
    Logger _log = Logger::global();

private:
    void runOnOperation() final;
};

//
// ModulePass
//

class ModulePass : public mlir::OperationPass<mlir::ModuleOp> {
public:
    using mlir::OperationPass<mlir::ModuleOp>::OperationPass;

protected:
    void initLogger(Logger log, StringLiteral passName);

protected:
    virtual void safeRunOnModule() = 0;

protected:
    Logger _log = Logger::global();

private:
    void runOnOperation() final;
};

}  // namespace vpux
