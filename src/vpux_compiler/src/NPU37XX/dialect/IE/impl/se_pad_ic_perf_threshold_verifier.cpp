//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/se_pad_ic_perf_threshold_verifier.hpp"

using namespace vpux::IE::arch37xx;

//
// SEPadICPerfThresholdVerifier
//

bool SEPadICPerfThresholdVerifier::isBeneficialForPerformance(const Shape& newInputShape, mlir::Operation*) const {
    return newInputShape[Dims4D::Act::C] > SEP_PAD_IC_NUM_PERF_THRESHOLD;
}
