//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// PatternBenefit
//

const mlir::PatternBenefit vpux::benefitLow(1);
const mlir::PatternBenefit vpux::benefitMid(2);
const mlir::PatternBenefit vpux::benefitHigh(3);

// Return a pattern benefit vector from large to small
SmallVector<mlir::PatternBenefit> vpux::getBenefitLevels(uint32_t levels) {
    SmallVector<mlir::PatternBenefit> benefitLevels;
    for (const auto level : irange(levels) | reversed) {
        benefitLevels.push_back(mlir::PatternBenefit(level));
    }
    return benefitLevels;
}

//
// FunctionPass
//

void vpux::FunctionPass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::FunctionPass::runOnOperation() {
    auto currentOp = getOperation();
    if (currentOp.isExternal()) {
        return;
    }

    auto passName = getName();
    try {
        vpux::Logger::global().trace("Started {0} pass on function '{1}'", passName, currentOp.getName());

        _log = _log.nest();
        safeRunOnFunc();
        _log = _log.unnest();
    } catch (const std::exception& e) {
        (void)errorAt(currentOp, "{0} Pass failed : {1}", passName, e.what());
        signalPassFailure();
    }
}

//
// ModulePass
//

void vpux::ModulePass::initLogger(Logger log, StringLiteral passName) {
    _log = log;
    _log.setName(passName);
}

void vpux::ModulePass::runOnOperation() {
    auto currentOp = getOperation();
    auto passName = getName();
    try {
        vpux::Logger::global().trace("Started {0} pass on module '{1}'", passName, currentOp.getName());
        safeRunOnModule();
    } catch (const std::exception& e) {
        (void)errorAt(currentOp, "{0} failed : {1}", passName, e.what());
        signalPassFailure();
    }
}
