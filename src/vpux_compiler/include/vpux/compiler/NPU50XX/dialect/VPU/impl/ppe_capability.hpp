//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/ppe_capability.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPU::arch50xx {

/*!
 * @brief NPU50XX+ PPE capability: bias always stored as float32 bit-pattern (toHex).
 *
 * arch50xx::getBias always uses toHex() regardless of activation type, so no int32
 * range constraint ever applies.
 */
class PPECapability final : public VPU::IPPECapability {
public:
    using VPU::IPPECapability::IPPECapability;

    [[nodiscard]] mlir::Type getBiasStorageType(mlir::Type activationType) const override {
        return mlir::Float32Type::get(activationType.getContext());
    }
};

}  // namespace vpux::VPU::arch50xx
