//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace vpux;

namespace {
struct VPUInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

}  // namespace

//
// materializeConstant
//

mlir::Operation* vpux::VPU::VPUDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                            mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize VPU Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!mlir::isa<mlir::RankedTensorType>(type)) {
        (void)errorAt(loc, "Can't materialize VPU Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//
// initialize
//

void vpux::VPU::VPUDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/activation.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/arithmetic.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/bitwise.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/comparison.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/control_flow.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/convolution.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/data_movement.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/data_type.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/dpu.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/eltwise.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/image.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/internal.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/logical.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/m2i.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/normalization.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/pooling.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/recurrent.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/reduce.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/shape_manipulation.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPU/ops/specialized.cpp.inc>
            >();

    registerAttributes();
    registerTypes();

    addInterface<VPUInlinerInterface>();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPU/dialect.cpp.inc>
