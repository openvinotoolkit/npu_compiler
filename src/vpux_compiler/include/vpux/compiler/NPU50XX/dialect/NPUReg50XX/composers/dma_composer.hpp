//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/descriptors.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::NPUReg50XX::DMADescriptorComposer {

Descriptors::DMARegister compose(VPUASM::NNDMAOp origOp, ELF::SymbolReferenceMap& symRefMap);

}  // namespace vpux::NPUReg50XX::DMADescriptorComposer
