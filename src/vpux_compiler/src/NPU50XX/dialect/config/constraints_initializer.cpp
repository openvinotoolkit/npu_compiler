//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints.hpp"
#include "vpux/compiler/NPU50XX/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

using namespace vpux;

config::ConstraintsInitializer50XX::ConstraintsInitializer50XX(config::Platform platform): _platform(platform) {
}

void config::ConstraintsInitializer50XX::initialize(mlir::MLIRContext* context) {
    NPUConstraints constraints;

    constraints.frequencyTable.base = arch50xx::FREQ_BASE;
    constraints.frequencyTable.step = arch50xx::FREQ_STEP;
    constraints.perfClock.defaultFreq = arch40xx::PERF_CLK_DEFAULT_VALUE_MHZ;

    constraints.mappedInferenceFormat = NPUConstraints::MappedInferenceFormat::MappedInference;
    constraints.baseElfAbiVersion = config::Version(2, 0, 0);
    constraints.dynamicStridesMinElfAbiVersion = config::Version(2, 1, 0);
    if (_platform == config::Platform::NPU5020) {
        constraints.mappedInferenceFormat = NPUConstraints::MappedInferenceFormat::ManagedMappedInference;
        constraints.baseElfAbiVersion = config::Version(2, 2, 0);
    }

    constraints.maxKernelSize = arch50xx::DPU_MAX_KERNEL_SIZE;

    setNPUConstraints(context, constraints);
}
