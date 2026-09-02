//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {
namespace IE {

/*
    Class for verifying if a Pad operation with a given input shape is beneficial to convert to a SE op based on an
    empirical threshold.
*/
class SEPadICPerfThresholdVerifierBase {
public:
    virtual ~SEPadICPerfThresholdVerifierBase() = default;

    virtual bool isBeneficialForPerformance(const Shape& newInputShape, mlir::Operation* op) const = 0;

    // Empirical threshold based on profiling traces and based on RTL simulations.
    // VPUNN CostModel does not currently model differences in performance coming from SEP usage.
    static constexpr int64_t SEP_PAD_IC_NUM_PERF_THRESHOLD = 32;
};

}  // namespace IE
}  // namespace vpux
