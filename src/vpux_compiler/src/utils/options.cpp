//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/options.hpp"

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

std::optional<WorkloadManagementMode> vpux::symbolizeWorkloadManagementMode(llvm::StringRef str) {
    return ::llvm::StringSwitch<::std::optional<WorkloadManagementMode>>(str)
            .Case("PWLM_V0_1_PAGES", WorkloadManagementMode::PWLM_V0_1_PAGES)
            .Case("FWLM_V1_PAGES", WorkloadManagementMode::FWLM_V1_PAGES)
            .Default(::std::nullopt);
}

StringLiteral vpux::stringifyEnum(WorkloadManagementBarrierProgrammingMode val) {
    switch (val) {
    case WorkloadManagementBarrierProgrammingMode::LEGACY:
        return "LEGACY";
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

StringLiteral vpux::stringifyEnum(VFMergeConfiguration val) {
    switch (val) {
    case VFMergeConfiguration::COST_BASED:
        return "COST_BASED";
    case VFMergeConfiguration::GREEDY:
        return "GREEDY";
    default:
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(WorkloadManagementMode val) {
    switch (val) {
    case WorkloadManagementMode::PWLM_V0_1_PAGES:
        return "PWLM_V0_1_PAGES";
    case WorkloadManagementMode::FWLM_V1_PAGES:
        return "FWLM_V1_PAGES";
    default:
        llvm::outs() << "Unknown WorkloadManagementMode value: " << static_cast<int>(val) << "\n";
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(SkipOCMode val) {
    switch (val) {
    case SkipOCMode::SKIP_NONE:
        return "SKIP_NONE";
    case SkipOCMode::SKIP_LARGE_SPATIAL:
        return "SKIP_LARGE_SPATIAL";
    case SkipOCMode::SKIP_ALL:
        return "SKIP_ALL";
    default:
        return "UNKNOWN";
    }
}

StringLiteral vpux::stringifyEnum(AutoUnrollingMode val) {
    switch (val) {
    case AutoUnrollingMode::DISABLED:
        return "disabled";
    case AutoUnrollingMode::INNER:
        return "inner";
    case AutoUnrollingMode::OUTER:
        return "outer";
    case AutoUnrollingMode::ALL:
        return "all";
    case AutoUnrollingMode::BIGGEST:
        return "biggest";
    default:
        return "UNKNOWN";
    }
}
