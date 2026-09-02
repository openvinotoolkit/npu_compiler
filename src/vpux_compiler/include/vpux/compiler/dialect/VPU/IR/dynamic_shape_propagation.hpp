//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/BuiltinOps.h>

#include <cstdint>

namespace vpux {
struct TypeComponents;
class Logger;
}  // namespace vpux

namespace mlir {
class TypeConverter;
class ConversionTarget;
}  // namespace mlir

namespace vpux::VPU {
enum class BoundsRepresentation : uint64_t;

void assignDynamicTypeComponents(vpux::TypeComponents& typeComponents, VPU::BoundsRepresentation boundsRepresentation,
                                 ArrayRef<int64_t> shape, ArrayRef<int64_t> bounds);

/// Configures type converter and conversion target for passes that analyze
/// dynamism-based IR (such as IR containing bounds or dynamic_dims_mask
/// tensors).
void configureDynamismConversionPassEnvironment(Logger& log, mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                                mlir::ConversionTarget& target);
}  // namespace vpux::VPU
