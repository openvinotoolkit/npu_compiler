//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/attributes.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/compiler/dialect/bytecode/IR/dialect.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <vpux/utils/core/error.hpp>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/bytecode/attributes.cpp.inc>
#include <vpux/compiler/dialect/bytecode/enums.cpp.inc>

//
// Dialect hooks
//

void bytecode::BytecodeDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/bytecode/attributes.cpp.inc>
            >();
}

mlir::LogicalResult bytecode::BufferTypeAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                                                     mlir::FlatSymbolRefAttr /*elementType*/, int64_t rank,
                                                     mlir::DenseI64ArrayAttr shape, mlir::DenseI64ArrayAttr strides) {
    if (rank < 0 || rank > intel_npu::vm::BufferType::MAX_RANK) {
        return emitError() << "buffer_type rank " << rank << " does not fit in buffer rank type [0, "
                           << intel_npu::vm::BufferType::MAX_RANK << "]";
    }
    if (rank != shape.size()) {
        return emitError() << "buffer_type rank " << rank << " does not match shape size " << shape.size();
    }
    if (rank != strides.size()) {
        return emitError() << "buffer_type rank " << rank << " does not match strides size " << strides.size();
    }
    return mlir::success();
}

intel_npu::vm::FloatTypeFormat bytecode::convertToFloatTypeFormat(bytecode::FloatFormat format) {
    switch (format) {
    case bytecode::FloatFormat::IEEE754: {
        return intel_npu::vm::FloatTypeFormat::IEEE754;
    }
    case bytecode::FloatFormat::BFloat: {
        return intel_npu::vm::FloatTypeFormat::BFloat;
    }
    case bytecode::FloatFormat::TFloat: {
        return intel_npu::vm::FloatTypeFormat::TFloat;
    }
    case bytecode::FloatFormat::E4M3: {
        return intel_npu::vm::FloatTypeFormat::E4M3;
    }
    case bytecode::FloatFormat::E5M2: {
        return intel_npu::vm::FloatTypeFormat::E5M2;
    }
    case bytecode::FloatFormat::E2M1: {
        return intel_npu::vm::FloatTypeFormat::E2M1;
    }
    case bytecode::FloatFormat::E8M0: {
        return intel_npu::vm::FloatTypeFormat::E8M0;
    }
    case bytecode::FloatFormat::NF4: {
        return intel_npu::vm::FloatTypeFormat::NF4;
    }
    default: {
        VPUX_THROW("Unsupported FloatFormat value: {0}", format);
    }
    }
}
