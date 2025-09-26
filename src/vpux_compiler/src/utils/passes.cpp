//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/options.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Options
//

std::optional<std::string> vpux::convertToOptional(const StrOption& strOption) {
    if (!strOption.getValue().empty()) {
        return strOption.getValue();
    }
    return std::nullopt;
}

bool vpux::isOptionEnabled(const BoolOption& option) {
    return option.getValue();
}

StringLiteral vpux::stringifyEnum(WorkloadManagementBarrierProgrammingMode val) {
    switch (val) {
    case WorkloadManagementBarrierProgrammingMode::LEGACY:
        return "LEGACY";
    case WorkloadManagementBarrierProgrammingMode::NO_BARRIER_DMAS_SCHEDULED:
        return "NO_BARRIER_DMAS_SCHEDULED";
    case WorkloadManagementBarrierProgrammingMode::INITIAL_BARRIER_DMAS_SCHEDULED:
        return "INITIAL_BARRIER_DMAS_SCHEDULED";
    case WorkloadManagementBarrierProgrammingMode::ALL_BARRIER_DMAS_SCHEDULED:
        return "ALL_BARRIER_DMAS_SCHEDULED";
    default:
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(DMAFifoType val) {
    switch (val) {
    case DMAFifoType::SW:
        return "SW";
    case DMAFifoType::HW:
        return "HW";
    default:
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(WeightsTableReuseMode val) {
    switch (val) {
    case WeightsTableReuseMode::ENABLED:
        return "ENABLED";
    case WeightsTableReuseMode::VF_ENABLED:
        return "VF_ENABLED";
    case WeightsTableReuseMode::DISABLED:
        return "DISABLED";
    default:
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(WorkloadManagementMode val) {
    switch (val) {
    case WorkloadManagementMode::PWLM_V0_LCA:
        return "PWLM_V0_LCA";
    case WorkloadManagementMode::PWLM_V1_BARRIER_FIFO:
        return "PWLM_V1_BARRIER_FIFO";
    case WorkloadManagementMode::PWLM_V2_PAGES:
        return "PWLM_V2_PAGES";
    default:
        return "UNKNOWN";
    }
}

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
