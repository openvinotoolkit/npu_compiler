//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/descriptors.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace NPUReg60XX {

namespace DMADescriptorComposer {

Descriptors::DMARegister compose(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap);

}  // namespace DMADescriptorComposer
}  // namespace NPUReg60XX
}  // namespace vpux
