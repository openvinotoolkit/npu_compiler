//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELFNPU37XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPURT/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/properties.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux::VPUIP {
class DistributedBufferType;
}  // namespace vpux::VPUIP

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPURT/ops.hpp.inc>

//
// Template methods
//

namespace vpux {
namespace VPURT {

template <typename T>
T vpux::VPURT::TaskOp::getInnerTaskOpOfType() {
    return mlir::dyn_cast<T>(&getBody().front().front());
}

}  // namespace VPURT
}  // namespace vpux
