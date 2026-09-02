//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/verifiers/memref_format_verifier.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>

using namespace vpux;

namespace {

mlir::LogicalResult verifyMemRefLayout(mlir::Type type, mlir::Operation* ownerOp) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
    if (!memref) {
        return mlir::success();
    }

    auto loc = ownerOp->getLoc();

    // null layout is not possible - it's at least identity
    auto layout = memref.getLayout();

    VPUX_THROW_WHEN(!layout, "Upstream contract changed, align interfaces");

    if (vpux::MemRefAttr::classof(layout)) {
        return mlir::success();
    }

    // Bare AffineMapAttr - check if it's identity
    if (auto affineMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(layout);
        affineMapAttr && !affineMapAttr.getAffineMap().isIdentity()) {
        // Only identity map is allowed (standard MLIR behavior for bare memref<shape>)
        return mlir::emitError(loc) << "MemRef format verification failed at op with name '" << ownerOp->getName()
                                    << "'\n"
                                    << "MemRef type uses bare non-identity affine_map layout: " << memref
                                    << "\nExpected format: memref<shape, {order = ...}> for non-identity layouts\n"
                                    << "Got bare affine_map layout: " << layout;
    }

    return mlir::success();
}

mlir::LogicalResult verifyMemRefFormat(mlir::Operation* rootOp) {
    auto walkResult = rootOp->walk([&](mlir::Operation* op) -> mlir::WalkResult {
        for (auto resultType : op->getResultTypes()) {
            if (mlir::failed(verifyMemRefLayout(resultType, op))) {
                return mlir::WalkResult::interrupt();
            }
        }

        for (auto operand : op->getOperands()) {
            if (mlir::failed(verifyMemRefLayout(operand.getType(), op))) {
                return mlir::WalkResult::interrupt();
            }
        }

        // Check block argument types in regions
        for (auto& region : op->getRegions()) {
            for (auto& block : region.getBlocks()) {
                for (auto arg : block.getArguments()) {
                    if (mlir::failed(verifyMemRefLayout(arg.getType(), op))) {
                        return mlir::WalkResult::interrupt();
                    }
                }
            }
        }

        return mlir::WalkResult::advance();
    });

    return walkResult.wasInterrupted() ? mlir::failure() : mlir::success();
}

class VerifyMemRefFormatInstrumentation final : public mlir::PassInstrumentation {
public:
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) final {
        if (pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
            return;
        }

        if (mlir::failed(verifyMemRefFormat(op))) {
            VPUX_THROW("MemRef format verification failure after '{0}' pass", pass->getName());
        }
    }
};

}  // namespace

void vpux::verifiers::addMemRefFormatVerifier(mlir::PassManager& pm) {
    pm.addInstrumentation(std::make_unique<VerifyMemRefFormatInstrumentation>());
}
