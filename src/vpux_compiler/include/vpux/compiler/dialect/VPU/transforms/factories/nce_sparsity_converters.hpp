//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/utils/core/type/float8_e4m3.hpp"
#include "vpux/utils/core/type/float8_e5m2.hpp"

#include <mlir/IR/Types.h>
#include <variant>

namespace vpux {
namespace VPU {
namespace NCESparsity {

using IntOrFloatType = std::variant<int32_t, float, vpux::type::float8_e5m2, vpux::type::float8_e4m3>;
using PPEConverterCb = IntOrFloatType (*)(uint8_t, int16_t, double, mlir::Type);
using BiasConverterCb = IntOrFloatType (*)(double, mlir::Type);
using ScaleRetrieveCb = double (*)(IntOrFloatType, mlir::Type);

PPEConverterCb getPPEConverterCb(config::ArchKind arch, bool isFloatType = false);
BiasConverterCb getBiasConverterCb(config::ArchKind arch, bool isFloatType = false);
ScaleRetrieveCb getScaleRetrieveCb(config::ArchKind arch, bool isNewWeightTableFormat = false);

}  // namespace NCESparsity
}  // namespace VPU
}  // namespace vpux
