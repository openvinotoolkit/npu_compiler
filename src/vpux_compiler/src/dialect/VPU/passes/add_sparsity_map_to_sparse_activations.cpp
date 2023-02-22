//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPU/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPU/sparsity_strategy.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

#include <typeinfo>

using namespace vpux;

namespace {

void updateUsers(mlir::OpResult& result) {
    for (auto user : result.getUsers()) {
        // If sparsity consumer then do not go further
        if (mlir::isa<VPU::NCEOpInterface, VPU::DesparsifyOp, mlir::ReturnOp>(user)) {
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
        public VPU::AddSparsityMapToSparseActivationsBase<AddSparsityMapToSparseActivationsPass> {
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
    auto func = getFunction();
    auto& ctx = getContext();

    func->walk([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::SparseOpInterface, VPU::SparsifyOp, VPU::GroupSparseTensorOp>(op)) {
            return;
        }

        for (auto result : op->getOpResults()) {
            const auto sparseType = result.getType().dyn_cast_or_null<VPU::SparseTensorType>();

            if (sparseType == nullptr) {
                return;
            }

            if (sparseType.getSparsityMap() != nullptr) {
                return;
            }

            _log.trace("Adding sparsity map to sparse type '{0}' produced by '{1}' at '{2}'", sparseType, op->getName(),
                       op->getLoc());

            const auto sparsityMapElementType = mlir::IntegerType::get(&ctx, 1, mlir::IntegerType::Signless);

            auto dataType = sparseType.getData().cast<vpux::NDTypeInterface>();
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
