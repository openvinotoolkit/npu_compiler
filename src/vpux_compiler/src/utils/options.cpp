//
// Copyright (C) 2025 Intel Corporation.
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

StringLiteral vpux::stringifyEnum(WorkloadManagementStatus val) {
    switch (val) {
    case WorkloadManagementStatus::ENABLED:
        return "ENABLED";
    case WorkloadManagementStatus::DISABLED:
        return "DISABLED";
    case WorkloadManagementStatus::FAILED:
        return "FAILED";
    }
    return "";
}

std::optional<WorkloadManagementStatus> vpux::symbolizeWorkloadManagementStatus(llvm::StringRef str) {
    return ::llvm::StringSwitch<::std::optional<WorkloadManagementStatus>>(str)
            .Case("ENABLED", WorkloadManagementStatus::ENABLED)
            .Case("DISABLED", WorkloadManagementStatus::DISABLED)
            .Case("FAILED", WorkloadManagementStatus::FAILED)
            .Default(::std::nullopt);
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
    case WorkloadManagementMode::FWLM_V1_PAGES:
        return "FWLM_V1_PAGES";
    default:
        return "UNKNOWN";
    }
}
