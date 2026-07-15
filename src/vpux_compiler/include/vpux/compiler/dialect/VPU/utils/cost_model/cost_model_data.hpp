//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace vpux {
namespace VPU {

extern const char COST_MODEL_2_7[];
extern const size_t COST_MODEL_2_7_SIZE;

extern const char COST_MODEL_2_7_FAST[];
extern const size_t COST_MODEL_2_7_FAST_SIZE;

extern const char COST_MODEL_4_0[];
extern const size_t COST_MODEL_4_0_SIZE;

extern const char COST_MODEL_4_0_FAST[];
extern const size_t COST_MODEL_4_0_FAST_SIZE;

extern const char COST_MODEL_5_1[];
extern const size_t COST_MODEL_5_1_SIZE;

extern const char COST_MODEL_CACHE_5_1[];
extern const size_t COST_MODEL_CACHE_5_1_SIZE;

#ifdef VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_5_1
extern const char PRECOMPUTED_STRATEGY_TABLE_5_1[];
extern const size_t PRECOMPUTED_STRATEGY_TABLE_5_1_SIZE;
#endif  // VPUX_BUILTIN_PRECOMPUTED_STRATEGY_TABLE_5_1

extern const char COST_MODEL_5_2[];
extern const size_t COST_MODEL_5_2_SIZE;

extern const char COST_MODEL_CACHE_5_2[];
extern const size_t COST_MODEL_CACHE_5_2_SIZE;

}  // namespace VPU
}  // namespace vpux
