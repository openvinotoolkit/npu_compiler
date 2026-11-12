//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/config/constraints.hpp"
#include "vpux/compiler/NPU50XX/dialect/config/constraints.hpp"
#include "vpux/compiler/NPU60XX/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"

using namespace vpux;

void config::ConstraintsInitializer60XX::initialize(mlir::MLIRContext* context) {
    NPUConstraints constraints;

    constraints.frequencyTable.base = arch40xx::FREQ_BASE;
    constraints.frequencyTable.step = arch40xx::FREQ_STEP;
    constraints.perfClock.defaultFreq = arch40xx::PERF_CLK_DEFAULT_VALUE_MHZ;

    constraints.baseElfAbiVersion = config::Version(2, 4, 0);
    constraints.mappedInferenceFormat = NPUConstraints::MappedInferenceFormat::ManagedMappedInference;

    constraints.maxKernelSize = arch50xx::DPU_MAX_KERNEL_SIZE;
    constraints.legalRoPEConfigs = getDefaultLegalRoPEConfigs();

    constraints.asymmetricPerTensorZP = arch60xx::ASYMMETRIC_PER_TENSOR_ZP;
    setNPUConstraints(context, constraints);
}
