//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FIXDYNAMICOPSLOCATIONS
#define GEN_PASS_DEF_FIXDYNAMICOPSLOCATIONS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class FixDynamicOpsLocationsPass final : public IE::impl::FixDynamicOpsLocationsBase<FixDynamicOpsLocationsPass> {
public:
    explicit FixDynamicOpsLocationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FixDynamicOpsLocationsPass::safeRunOnFunc() {
    std::map<std::string, size_t> counter;
    getOperation()->walk([&](mlir::Operation* op) {
        if (!mlir::isa<mlir::tensor::TensorDialect, mlir::arith::ArithDialect>(op->getDialect())) {
            return mlir::WalkResult::advance();
        }

        auto loc = op->getLoc();
        if (mlir::isa<mlir::UnknownLoc>(loc)) {
            return mlir::WalkResult::advance();
        }

        std::string strLoc = vpux::stringifyPrimaryLocation(loc);
        if (counter.count(strLoc) == 0) {
            counter[strLoc] = 1;
            return mlir::WalkResult::advance();
        }
        _log.trace("Got op '{0}' with duplicate location at '{1}'", op->getName(), loc);
        extendOpLoc(op, StringLiteral("duplicated_{0}"), counter[strLoc]++);
        _log.nest().trace("Extended loc '{0}'", op->getLoc());
        return mlir::WalkResult::advance();
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFixDynamicOpsLocationsPass(Logger log) {
    return std::make_unique<FixDynamicOpsLocationsPass>(log);
}
