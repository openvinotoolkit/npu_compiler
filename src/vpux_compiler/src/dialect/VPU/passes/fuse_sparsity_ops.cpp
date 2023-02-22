//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// FuseSparsityOpsPass
//

class FuseSparsityOpsPass final : public VPU::FuseSparsityOpsBase<FuseSparsityOpsPass> {
public:
    explicit FuseSparsityOpsPass(Optional<bool> fuseSparsify, Logger log): _fuseSparsify(fuseSparsify) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    Optional<bool> _fuseSparsify;
};

mlir::LogicalResult FuseSparsityOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (!fuseSparsify.hasValue()) {
        return mlir::success();
    }
    if (_fuseSparsify.hasValue()) {
        _log.trace("Overloading C++  createFuseSparsityOpsPass argument by MLIR variable");
    }
    _fuseSparsify = fuseSparsify;
    return mlir::success();
}

// Legend:
// === sparse path
// --- dense path
// Convert SparseProducer => VPU.Desparsify -> SparseCompatibleConsumer (1) pattern
// to SparseProducer => SparseCompatibleConsumer
// By algorithm design we expect to have mostly (1) pattern, but following
// patterns are supported as well(with variadic number of [non]sparse consumers)
// SparseProducer => VPU.Desparsify ---> SparseCompatibleConsumer
//                        ...        +-> SparseCompatibleConsumer
//                                   \-> Consumer
// To
// SparseProducer ===> SparseCompatibleConsumer
//                 +=> SparseCompatibleConsumer
//                 \=> Desparsify -> Consumer
void fuseDesparsify(mlir::FuncOp func, Logger log) {
    func->walk([&](VPU::DesparsifyOp desparsifyOp) {
        mlir::DenseSet<mlir::Operation*> fusibleConsumers;
        for (const auto user : desparsifyOp->getUsers()) {
            if (VPU::supportsSparseInputs(user)) {
                fusibleConsumers.insert(user);
            }
        }
        // TODO: E#53581
        const auto sparseSource = desparsifyOp.input();
        desparsifyOp.output().replaceUsesWithIf(sparseSource, [&](mlir::OpOperand& opOperand) {
            return fusibleConsumers.contains(opOperand.getOwner());
        });
        if (desparsifyOp.output().use_empty()) {
            log.trace("'{0}' was completely fused with consumers", desparsifyOp);
            desparsifyOp.erase();
        }
    });
}

// Convert SparseCompatibleProducer -> Sparsify => Consumers
// To
// SparseProducer => Consumers
void fuseSparsify(mlir::FuncOp func, Logger log) {
    func->walk([&](VPU::SparsifyOp sparsifyOp) {
        // Fusing only VPU.NCE.Op->SparsifyOp pattern
        const auto producer = sparsifyOp.input().getDefiningOp();
        if (!producer) {
            return;
        }
        if (!VPU::supportsSparseOutputs(producer)) {
            return;
        }
        if (!producer->hasOneUse()) {
            log.trace("Cant fuse '{0}', because of multiple consumers", sparsifyOp);
            return;
        }

        log.trace("Fusing '{0}' with '{1}'", sparsifyOp, producer->getName());
        producer->getResult(0).setType(sparsifyOp.output().getType());
        sparsifyOp.replaceAllUsesWith(producer->getResult(0));
        sparsifyOp.erase();
    });
}

//
// safeRunOnFunc
//

void FuseSparsityOpsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getFunction();
    if (_fuseSparsify.getValueOr(false)) {
        ::fuseSparsify(func, _log);
    } else {
        ::fuseDesparsify(func, _log);
    }
}

}  // namespace

//
// createFuseSparsityOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseSparsityOpsPass(Optional<bool> fuseSparsify, Logger log) {
    return std::make_unique<FuseSparsityOpsPass>(fuseSparsify, log);
}
