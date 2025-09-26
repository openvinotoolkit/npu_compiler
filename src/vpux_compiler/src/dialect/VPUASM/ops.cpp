//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <cstdint>

using namespace vpux;

mlir::LogicalResult VPUASM::convertFromAttribute(KernelParamsProperty& kParams, mlir::Attribute attr,
                                                 llvm::function_ref<mlir::InFlightDiagnostic()>) {
    auto arrayAttr = llvm::dyn_cast_if_present<::mlir::ArrayAttr>(attr);
    if (!arrayAttr) {
        return mlir::failure();
    }
    kParams.setStorage(parseIntArrayAttr<uint8_t>(arrayAttr));
    return mlir::success();
}
mlir::Attribute VPUASM::convertToAttribute(mlir::MLIRContext* ctx, const KernelParamsProperty& kernelParams) {
    return getIntArrayAttr(ctx, kernelParams.getStorage());
}

//
// initialize
//

void vpux::VPUASM::VPUASMDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUASM/ops.cpp.inc>
            >();

    registerTypes();
    // registerAttributes();
}

// Note: for some reason, this cpp-only printer method has to be declared in
// vpux::VPUASM namespace.
namespace vpux::VPUASM {
void printContentAttr(mlir::OpAsmPrinter& printer, const ConstBufferOp&, const vpux::Const::ContentAttr& content) {
    vpux::Const::printContentAttr(printer, content);
}
}  // namespace vpux::VPUASM

//
// Generated
//

#include <vpux/compiler/dialect/VPUASM/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUASM/ops.cpp.inc>
