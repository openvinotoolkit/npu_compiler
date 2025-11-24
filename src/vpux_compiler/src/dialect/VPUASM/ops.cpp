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

#include <mlir/Dialect/Quant/QuantOps.h>

using namespace vpux;

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
