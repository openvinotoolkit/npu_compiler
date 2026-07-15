//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/ppe_capability.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux::VPU::arch37xx {

/*!
 * @brief NPU37XX/40XX PPE capability: bias storage type depends on activation type.
 *
 * arch37xx::getBias uses checked_cast<int32_t>(round(v)) for quantized activation
 * (int32 range constraint applies) and toHex() for float activation (no constraint).
 */
class PPECapability final : public VPU::IPPECapability {
public:
    using VPU::IPPECapability::IPPECapability;

    [[nodiscard]] mlir::Type getBiasStorageType(mlir::Type activationType) const override {
        auto* ctx = activationType.getContext();
        if (mlir::isa<mlir::quant::QuantizedType>(activationType)) {
            return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
        }
        return mlir::Float32Type::get(ctx);
    }
};

}  // namespace vpux::VPU::arch37xx
