//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace VPUIP {

constexpr int64_t INPUT_DDR_CONTIGUOUS_WIDTH_37XX = 64 * 8;
constexpr int64_t INPUT_DDR_CONTIGUOUS_WIDTH_40XX = 512 * 8;

MemDimArr getStridesMemDims(vpux::NDTypeInterface tensorType);
bool isDDRCopyEfficient(vpux::NDTypeInterface tensorType, config::ArchKind arch);
bool isEffectivelyStrided(vpux::NDTypeInterface type);

}  // namespace VPUIP
}  // namespace vpux
