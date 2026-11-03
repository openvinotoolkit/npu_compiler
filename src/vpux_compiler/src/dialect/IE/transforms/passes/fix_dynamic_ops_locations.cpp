//
// Copyright (C) 2025-2026 Intel Corporation
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
    const auto isShapeReificationOp = [](mlir::Operation* op) {
        return mlir::isa<mlir::tensor::TensorDialect, mlir::arith::ArithDialect>(op->getDialect());
    };

    std::map<std::string, size_t> counter;

    // Shape reification copies the location of the operation whose shape is computed onto the tensor/arith
    // ops it introduces, so such an op clashes with the original operation and not only with its siblings.
    // Register the locations of all the other operations first, otherwise the clash stays unnoticed here and
    // is reported later by the locations verifier.
    getOperation()->walk([&](mlir::Operation* op) {
        if (isShapeReificationOp(op) || mlir::isa<mlir::UnknownLoc>(op->getLoc())) {
            return;
        }
        counter.emplace(vpux::stringifyPrimaryLocation(op->getLoc()), 1);
    });

    getOperation()->walk([&](mlir::Operation* op) {
        if (!isShapeReificationOp(op)) {
            return;
        }

        auto loc = op->getLoc();
        if (mlir::isa<mlir::UnknownLoc>(loc)) {
            return;
        }

        std::string strLoc = vpux::stringifyPrimaryLocation(loc);
        auto counterIt = counter.find(strLoc);
        if (counterIt == counter.end()) {
            counter[strLoc] = 1;
            return;
        }
        _log.trace("Got op '{0}' with duplicate location at '{1}'", op->getName(), loc);
        extendOpLoc(op, "duplicated_{0}", counterIt->second++);
        _log.nest().trace("Extended loc '{0}'", op->getLoc());
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFixDynamicOpsLocationsPass(Logger log) {
    return std::make_unique<FixDynamicOpsLocationsPass>(log);
}
