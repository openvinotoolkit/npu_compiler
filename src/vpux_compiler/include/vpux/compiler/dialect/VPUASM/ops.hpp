//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <cstdint>
#include "vpux/compiler/NPU40XX/dialect/ELF/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/const/attributes/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUASM {
class KernelParamsProperty {
private:
    SmallVector<uint8_t> kernelParams;

public:
    KernelParamsProperty() = default;
    ~KernelParamsProperty() = default;

    KernelParamsProperty(ArrayRef<uint8_t> params) {
        kernelParams = SmallVector<uint8_t>(params);
    }

    void setStorage(ArrayRef<uint8_t> params) {
        kernelParams = SmallVector<uint8_t>(params);
    }

    llvm::MutableArrayRef<uint8_t> getStorage() {
        return kernelParams;
    }

    ArrayRef<uint8_t> getStorage() const {
        return kernelParams;
    }

    friend bool operator==(const KernelParamsProperty& rhs, const KernelParamsProperty& lhs) {
        return rhs.kernelParams == lhs.kernelParams;
    }

    friend llvm::hash_code hash_value(const KernelParamsProperty& kernelParams) {
        using ::llvm::hash_value;
        return llvm::hash_value(kernelParams.getStorage());
    }
};

mlir::LogicalResult convertFromAttribute(KernelParamsProperty&, mlir::Attribute,
                                         llvm::function_ref<mlir::InFlightDiagnostic()>);
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, const KernelParamsProperty& kernelParams);

}  // namespace vpux::VPUASM

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUASM/ops.hpp.inc>
