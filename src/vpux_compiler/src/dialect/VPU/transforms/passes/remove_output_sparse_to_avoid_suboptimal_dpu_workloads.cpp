//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_REMOVEOUTPUTSPARSETOAVOIDSUBOPTIMALDPUWORKLOADSPASS
#define GEN_PASS_DEF_REMOVEOUTPUTSPARSETOAVOIDSUBOPTIMALDPUWORKLOADSPASS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

void removeOutputSparse(VPU::NCEOpInterface nceOp, Logger log) {
    auto removeSparsityFlag = VPU::shouldRemoveOutputSparsity(nceOp);
    if (removeSparsityFlag != VPU::SparsityRemovalFlag::Success) {
        log.trace("{0} at {1}: keeping output sparsity, reason={2}", nceOp->getName(), nceOp->getLoc(),
                  removeSparsityFlag);
        return;
    }
    log.trace("{0} at {1}: removing output sparsity", nceOp->getName(), nceOp->getLoc());

    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    const auto dataType = mlir::cast<VPU::SparseTensorType>(clusteredOp->getResult(0).getType()).getData();

    auto recursivelyRemoveSparseOutput = [&](VPU::ClusteredOpInterface clusteredOp) -> void {
        clusteredOp->getResult(0).setType(dataType);
        log.nest().trace("Remove output sparsity for op {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());

        auto users = to_small_vector(clusteredOp->getUsers());
        while (!users.empty()) {
            auto currentOp = users.back();
            users.pop_back();
            if (auto unrolledTypeOp = mlir::dyn_cast_or_null<VPU::UnrolledTypeOp>(currentOp)) {
                auto outputSparseType = mlir::dyn_cast<VPU::SparseTensorType>(unrolledTypeOp.getOutput().getType());
                if (outputSparseType != nullptr) {
                    unrolledTypeOp->getResult(0).setType(outputSparseType.getData());
                }

                auto nextOps = to_small_vector(unrolledTypeOp->getUsers());
                users.insert(users.end(), nextOps.begin(), nextOps.end());
            } else if (mlir::isa_and_nonnull<VPU::ViewLikeOpInterface>(currentOp)) {
                inferReturnTypes(currentOp, InferShapedTypeMode::ALL);
                auto nextOps = to_small_vector(currentOp->getUsers());
                users.insert(users.end(), nextOps.begin(), nextOps.end());
            }
        }
    };

    recursivelyRemoveSparseOutput(clusteredOp);
}

//
// RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass
//
class RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass final :
        public VPU::impl::RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPassBase<
                RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass> {
public:
    explicit RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();

    // TODO: E#106239
    // This pass could remove activation sparsity after strategy manager. With these changes, multi-clustering and
    // tiling are done with the cost of activation sparsity being present while sparsity can be reverted. This can have
    // an impact over the performance. Hopefully in the future we can look into refactoring the strategy manager to also
    // take the decision on whether to enable activation sparsity or not.
    func->walk([&](VPU::NCEOpInterface op) {
        removeOutputSparse(op, _log);
    });
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log) {
    return std::make_unique<RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass>(log);
}
