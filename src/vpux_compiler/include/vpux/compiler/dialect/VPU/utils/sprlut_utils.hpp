//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

constexpr auto SPRLUT_ALIGNMENT_REQUIREMENT = 32;

bool hasSprLUTAttribute(PPEAttr ppeAttr);

Byte getSprLUTSize(PPEAttr ppeAttr);

void addSprLutBufferIfPresent(PPEAttr ppeAttr, SmallVector<Byte>& buffers);

}  // namespace VPU
}  // namespace vpux
