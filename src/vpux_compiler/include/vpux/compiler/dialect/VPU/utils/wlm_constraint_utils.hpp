//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/barrier_variant_constraint.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef BARR_MAX_VARIANT_SUM = "VPU.BarrierMaxVariantSum";
constexpr StringRef BARR_MAX_VARIANT_COUNT = "VPU.BarrierMaxVariantCount";
constexpr StringRef METADATA_MAX_VARIANT_COUNT = "VPU.MetadataMaxVariantCount";
constexpr StringRef METADATA_MAX_INVARIANT_COUNT = "VPU.MetadataMaxInvariantCount";
constexpr StringRef METADATA_MAX_KERNEL_INVOCATION_COUNT = "VPU.MetadataMaxKernelInvocationCount";
constexpr StringRef METADATA_MAX_KERNEL_RANGE_COUNT = "VPU.MetadataMaxKernelRangeCount";
constexpr StringRef METADATA_MAX_MEDIA_COUNT = "VPU.MetadataMaxMediaCount";

// This function returns the value of the attrName constraint
size_t getConstraint(mlir::Operation* op, StringRef attrName);
uint32_t getDefaultTaskListCount(VPU::TaskType taskType, VPU::ArchKind archKind);

}  // namespace VPU
}  // namespace vpux
