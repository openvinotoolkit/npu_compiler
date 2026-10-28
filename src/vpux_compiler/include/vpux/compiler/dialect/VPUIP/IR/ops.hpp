//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIP/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace VPUIP {

constexpr Bit FP16_SIZE = 16_Bit;
constexpr KB SHAVE_LIB_DATA_SIZE = 112_KB;

}  // namespace VPUIP
}  // namespace vpux

//
// Template methods
//

namespace vpux {
namespace VPUIP {

template <typename... Args>
VPUIP::PPETaskOp NCEClusterTaskOp::addPPETask(mlir::OpBuilder& builder, Args&&... args) {
    if (getPpe().empty()) {
        getPpe().emplaceBlock();
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&getPpe().front());

    return VPUIP::PPETaskOp::create(builder, getLoc(), std::forward<Args>(args)...);
}

}  // namespace VPUIP
}  // namespace vpux
