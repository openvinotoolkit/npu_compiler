//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/config/constraints.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

using namespace vpux;

void config::ConstraintsInitializer40XX::initialize(mlir::MLIRContext* context) {
    NPUConstraints constraints;

    constraints.frequencyTable.base = arch40xx::FREQ_BASE;
    constraints.frequencyTable.step = arch40xx::FREQ_STEP;
    constraints.perfClock.defaultFreq = arch40xx::PERF_CLK_DEFAULT_VALUE_MHZ;

    constraints.mappedInferenceFormat = NPUConstraints::MappedInferenceFormat::MappedInference;

    constraints.baseElfAbiVersion = config::Version(1, 2, 2);
    constraints.dynamicStridesMinElfAbiVersion = config::Version(1, 3, 0);

    constraints.maxKernelSize = arch37xx::DPU_MAX_KERNEL_SIZE;

    setNPUConstraints(context, constraints);
}
