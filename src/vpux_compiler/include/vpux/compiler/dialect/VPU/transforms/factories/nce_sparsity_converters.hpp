//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <vpux/utils/core/type/float8_e4m3.hpp>
#include <vpux/utils/core/type/float8_e5m2.hpp>

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPU {
namespace NCESparsity {

using IntOrFloatType = std::variant<int32_t, float, vpux::type::float8_e5m2, vpux::type::float8_e4m3>;
using PPEConverterCb = IntOrFloatType (*)(uint8_t, int16_t, double, mlir::Type);
using BiasConverterCb = IntOrFloatType (*)(double, mlir::Type);

PPEConverterCb getPPEConverterCb(VPU::ArchKind arch, bool isFloatType = false);
BiasConverterCb getBiasConverterCb(VPU::ArchKind arch, bool isFloatType = false);

}  // namespace NCESparsity
}  // namespace VPU
}  // namespace vpux
