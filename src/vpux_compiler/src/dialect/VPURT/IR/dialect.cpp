//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/IERT/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/Transforms/InliningUtils.h>

namespace {

//
// VPURTInlinerInterface
//

struct VPURTInlinerInterface : public mlir::DialectInlinerInterface {
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
// initialize
//

void vpux::VPURT::VPURTDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPURT/ops.cpp.inc>
            >();

    registerTypes();
    registerAttributes();

    addInterfaces<VPURTInlinerInterface>();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/dialect.cpp.inc>
