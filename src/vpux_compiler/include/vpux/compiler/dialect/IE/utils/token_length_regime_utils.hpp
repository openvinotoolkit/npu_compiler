//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {
namespace IE {

// Leading (token) dimension threshold separating the two LLM execution regimes:
//   * <= this value  -> decode / speculative-decoding KV-cache: workload is memory (data) bound
//   * >  this value  -> prefill: workload is compute bound
constexpr int64_t DECODE_MAX_TOKEN_LENGTH = 16;

// True when the leading (token) dimension is static and small enough to be in the data-bound decode regime
// (KV-cache with token length 1, or speculative decoding with token length up to DECODE_MAX_TOKEN_LENGTH).
// A dynamic leading dimension is treated as NOT data bound, so callers keep their default (prefill) behavior
// instead of accidentally triggering decode-only optimizations on unknown token lengths.
bool isDataBoundTokenDim(vpux::ShapeRef outShape);

}  // namespace IE
}  // namespace vpux
