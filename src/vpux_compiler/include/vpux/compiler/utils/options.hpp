//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Pass/PassOptions.h>
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace vpux {

using IntOption = mlir::detail::PassOptions::Option<int>;
using Int64Option = mlir::detail::PassOptions::Option<int64_t>;
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;
using DoubleOption = mlir::detail::PassOptions::Option<double>;

enum class WorkloadManagementMode { PWLM_V0_1_PAGES = 0, FWLM_V1_PAGES = 1 };

enum class AllocateDDRStackFrames { ENABLED = 0, DISABLED = 1 };
enum class WorkloadManagementBarrierProgrammingMode {
    LEGACY = 0,
    ALL_BARRIER_DMAS_SCHEDULED = 3,
    ALL_BARRIER_DMAS_SCHEDULED_4K = 4,
    UNKNOWN = 255
};
enum class DMAFifoType { SW = 0, HW = 1 };

/**
 * @brief This enum is used to specify the mode of weights table reuse.
 *
 * It can be set to ENABLED, VF_ENABLED, or DISABLED.
 * ENABLED means that the weights table can be reused for all operations that support it.
 * VF_ENABLED means that the weights table can be reused for operations in pure-vertical-fusion region, to avoid
 * possible memory fragmentation. DISABLED means that the weights table cannot be reused.
 */
enum class WeightsTableReuseMode { ENABLED = 0, VF_ENABLED = 1, DISABLED = 2 };

enum class SkipOCMode { SKIP_NONE = 0, SKIP_LARGE_SPATIAL = 1, SKIP_ALL = 2 };

enum class AutoUnrollingMode { DISABLED = 0, INNER = 1, OUTER = 2, ALL = 3, BIGGEST = 4 };

enum class VFMergeConfiguration { COST_BASED = 0, GREEDY = 1 };

std::optional<WorkloadManagementMode> symbolizeWorkloadManagementMode(llvm::StringRef str);
StringLiteral stringifyEnum(WorkloadManagementBarrierProgrammingMode val);
StringLiteral stringifyEnum(DMAFifoType val);
StringLiteral stringifyEnum(WeightsTableReuseMode val);
StringLiteral stringifyEnum(VFMergeConfiguration val);
std::optional<std::string> convertToOptional(const StrOption& strOption);
StringLiteral stringifyEnum(WorkloadManagementMode val);
StringLiteral stringifyEnum(SkipOCMode val);
StringLiteral stringifyEnum(AutoUnrollingMode val);
bool isOptionEnabled(const BoolOption& option);
}  // namespace vpux

namespace llvm {
inline ::llvm::raw_ostream& operator<<(::llvm::raw_ostream& p, vpux::WorkloadManagementMode value) {
    auto valueStr = vpux::stringifyEnum(value);
    return p << valueStr;
}

template <>
struct format_provider<vpux::WorkloadManagementMode> {
    static void format(const vpux::WorkloadManagementMode& val, raw_ostream& OS, StringRef /*Options*/) {
        OS << vpux::stringifyEnum(val);
    }
};

inline ::llvm::raw_ostream& operator<<(::llvm::raw_ostream& p, vpux::SkipOCMode value) {
    auto valueStr = vpux::stringifyEnum(value);
    return p << valueStr;
}

template <>
struct format_provider<vpux::SkipOCMode> {
    static void format(const vpux::SkipOCMode& val, raw_ostream& OS, StringRef /*Options*/) {
        OS << vpux::stringifyEnum(val);
    }
};

inline ::llvm::raw_ostream& operator<<(::llvm::raw_ostream& p, vpux::AutoUnrollingMode value) {
    return p << vpux::stringifyEnum(value);
}

template <>
struct format_provider<vpux::AutoUnrollingMode> {
    static void format(const vpux::AutoUnrollingMode& val, raw_ostream& OS, StringRef /*Options*/) {
        OS << vpux::stringifyEnum(val);
    }
};

}  // namespace llvm
