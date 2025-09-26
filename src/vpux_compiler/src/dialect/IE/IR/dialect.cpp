//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include <vpux/compiler/dialect/config/IR/dialect.hpp>
#include <vpux/compiler/dialect/core/IR/dialect.hpp>
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/TensorEncoding.h>

using namespace vpux;

namespace {

//
// IEAsmHooks
//

class IEAsmHooks final : public mlir::OpAsmDialectInterface {
public:
    using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

public:
    AliasResult getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const final;
    AliasResult getAlias(mlir::Type type, llvm::raw_ostream& os) const final;
};

IEAsmHooks::AliasResult IEAsmHooks::getAlias(mlir::Attribute attr, llvm::raw_ostream& os) const {
    if (const auto mapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(attr)) {
        const auto map = mapAttr.getValue();

        if (map.isPermutation()) {
            const auto dimsOrder = DimsOrder::fromAffineMap(map);

            if (const auto name = dimsOrder.getCanonicalName(); !name.empty()) {
                os << name;
                return AliasResult::FinalAlias;
            }
        }
    }

    return AliasResult::NoAlias;
}

IEAsmHooks::AliasResult IEAsmHooks::getAlias(mlir::Type type, llvm::raw_ostream& os) const {
    if (mlir::isa<mlir::quant::QuantizedType>(type)) {
        os << "qElemType";
        return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
}

}  // namespace

//
// initialize
//

void vpux::IE::IEDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/activation.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/arithmetic.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/bitwise.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/comparison.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/control_flow.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/convolution.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/data_movement.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/eltwise.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/data_type.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/image.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/logical.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/normalization.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/pooling.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/recurrent.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/reduce.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/shape_manipulation.cpp.inc>
            >();
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/IE/ops/specialized.cpp.inc>
            >();

    addInterfaces<IEAsmHooks>();

    registerAttributes();
}

//
// materializeConstant
//

mlir::Operation* vpux::IE::IEDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                          mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize IE Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!mlir::isa<mlir::RankedTensorType>(type)) {
        (void)errorAt(loc, "Can't materialize IE Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//
// Generated
//

#include <vpux/compiler/dialect/IE/dialect.cpp.inc>
