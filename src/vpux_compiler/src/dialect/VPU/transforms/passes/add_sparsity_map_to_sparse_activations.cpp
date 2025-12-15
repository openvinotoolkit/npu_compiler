//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_ADDSPARSITYMAPTOSPARSEACTIVATIONS
#define GEN_PASS_DEF_ADDSPARSITYMAPTOSPARSEACTIVATIONS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

void updateUsers(mlir::OpResult& result) {
    for (auto user : result.getUsers()) {
        // If sparsity consumer then do not go further
        if (mlir::isa<VPU::NCEOpInterface, VPU::DesparsifyOp, mlir::func::ReturnOp>(user)) {
            continue;
        }
        // Can propagate type, do it recursively
        vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);
        for (auto result : user->getResults()) {
            updateUsers(result);
        }
    }
    return;
}

//
// AddSparsityMapToSparseActivations
//

class AddSparsityMapToSparseActivationsPass final :
        public VPU::impl::AddSparsityMapToSparseActivationsBase<AddSparsityMapToSparseActivationsPass> {
public:
    explicit AddSparsityMapToSparseActivationsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void AddSparsityMapToSparseActivationsPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    func->walk([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::SparseOpInterface, VPU::SparsifyOp, VPU::GroupSparseTensorOp>(op)) {
            return;
        }

        for (auto result : op->getOpResults()) {
            const auto sparseType = mlir::dyn_cast_or_null<VPU::SparseTensorType>(result.getType());

            if (sparseType == nullptr) {
                return;
            }

            if (sparseType.getSparsityMap() != nullptr) {
                return;
            }

            auto isSEOnlyOp = mlir::isa<VPU::GroupSparseTensorOp>(op) && sparseType.getSparsityMap() == nullptr &&
                              sparseType.getStorageElementTable() != nullptr;
            if (VPU::isSEOnlyWithoutSMSupported(config::getArch(op)) && isSEOnlyOp) {
                return;
            }

            _log.trace("Adding sparsity map to sparse type '{0}' produced by '{1}' at '{2}'", sparseType, op->getName(),
                       op->getLoc());

            const auto sparsityMapElementType = mlir::IntegerType::get(&ctx, 1, mlir::IntegerType::Signless);

            auto dataType = mlir::cast<vpux::NDTypeInterface>(sparseType.getData());
            auto sparsityMapType = dataType.changeElemType(sparsityMapElementType);

            const auto updatedSparseType = VPU::SparseTensorType::get(dataType, sparsityMapType);

            result.setType(updatedSparseType);

            // Propagare type through all users until the first consumer of sparse type
            updateUsers(result);
        }
    });
}

}  // namespace

//
// createAddSparsityMapToSparseActivationsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAddSparsityMapToSparseActivationsPass(Logger log) {
    return std::make_unique<AddSparsityMapToSparseActivationsPass>(log);
}
