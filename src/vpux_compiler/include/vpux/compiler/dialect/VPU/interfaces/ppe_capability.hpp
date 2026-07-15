//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/Types.h>

namespace vpux::VPU {

/*!
 * @brief DialectInterface that reports hardware capabilities related to PPE bias storage.
 *
 * Registered once per dialect instance by the arch-specific initializer. Different architectures
 * register different concrete subclasses so callers can query capabilities without knowing the arch.
 */
class IPPECapability : public mlir::DialectInterface::Base<IPPECapability> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IPPECapability)

    IPPECapability(mlir::Dialect* dialect): Base(dialect) {
    }

    /*!
     * @brief Returns the MLIR type used to store per-channel bias in the hardware weight table.
     *
     * Returns mlir::IntegerType (i32) when the bias is stored as a true integer requiring an int32
     * range check (e.g. NPU37XX/40XX with quantized activation). Returns mlir::Float32Type when the
     * bias is stored as float32 bit-pattern via toHex() — no int32 overflow constraint applies.
     */
    [[nodiscard]] virtual mlir::Type getBiasStorageType(mlir::Type activationType) const = 0;

    virtual ~IPPECapability() = default;
};

// Asserting accessor — use only in contexts where the arch-specific InterfacesRegistry is guaranteed
// to have been applied (main compiler pipeline, vpux-opt tool).
const IPPECapability& getPPECapability(mlir::MLIRContext* context);

// Safe accessor — returns nullptr when the interface is not registered (hwtest builders, unit tests,
// any context that skips InterfacesRegistry). Callers must handle nullptr by falling back to the
// conservative default (checkInt32Range = true).
const IPPECapability* tryGetPPECapability(mlir::MLIRContext* context);

}  // namespace vpux::VPU
