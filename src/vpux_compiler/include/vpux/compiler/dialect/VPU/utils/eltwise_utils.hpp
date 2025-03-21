//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Types.h>

namespace vpux {
namespace VPU {
bool isNCEEltwiseSupported(mlir::Operation* op, vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                           vpux::NDTypeInterface outputType, bool allowDifferentScales, bool allowDifferentZp,
                           bool checkLayout, bool checkChannelAlignment, LogCb logCb);

template <class ConcreteOp>
bool isEltwiseLhsActivation(ConcreteOp op) {
    const auto lhsType = op.getInput1().getType().template cast<mlir::ShapedType>();
    const auto outShapeRes = op.getOutput().getType().template cast<mlir::ShapedType>();

    return (lhsType == outShapeRes);
}

vpux::VPU::EltwiseType decodeNceEltwiseType(mlir::Operation* operation);

}  // namespace VPU
}  // namespace vpux
