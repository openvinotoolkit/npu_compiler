//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/interfaces/se_pad_ic_perf_threshold_verifier.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

namespace vpux::IE::arch40xx {

/*
    Class for verifying if a Pad operation with a given input shape is beneficial to convert to a SE op based on an
    empirical threshold.
*/
class SEPadICPerfThresholdVerifier : public vpux::IE::SEPadICPerfThresholdVerifierBase {
public:
    bool isBeneficialForPerformance(const Shape& newInputShape, mlir::Operation* op) const override;
};

}  // namespace vpux::IE::arch40xx
