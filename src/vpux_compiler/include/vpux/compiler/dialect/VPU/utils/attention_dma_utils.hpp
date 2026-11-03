//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace vpux::VPU {

// Returns true when {qHeadSize, tSL, sSL, eDim, eVDim} exactly matches one of the static shape
// configs validated for AttentionDMA's non-flash kernel. Shared by IE::FuseAttentionPass (deciding
// whether AttentionOp -> AttentionDMAOp is legal without relying on a dynamic seqLenK/KV-cache)
// and VPU::AttentionDMAOp::fitsAndLegalForNonFlashKernel (deciding non-flash vs flash kernel).
bool isLegalNonFlashAttentionDMAConfig(int64_t qHeadSize, int64_t tSL, int64_t sSL, int64_t eDim, int64_t eVDim);

}  // namespace vpux::VPU
