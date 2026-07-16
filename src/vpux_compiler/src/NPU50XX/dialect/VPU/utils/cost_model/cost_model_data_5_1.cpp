//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model_data.hpp"

namespace vpux {
namespace VPU {

#include <vpux/compiler/dialect/VPU/generated/cost_model_cache_data_5_1.hpp.inc>
#include <vpux/compiler/dialect/VPU/generated/cost_model_cache_data_5_2.hpp.inc>
#include <vpux/compiler/dialect/VPU/generated/cost_model_data_5_1.hpp.inc>
#include <vpux/compiler/dialect/VPU/generated/cost_model_data_5_2.hpp.inc>
#ifdef VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_5_1
#include <vpux/compiler/dialect/VPU/generated/precomputed_strategy_table_builtin_5_1.hpp.inc>
#endif  // VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_5_1

}  // namespace VPU
}  // namespace vpux
