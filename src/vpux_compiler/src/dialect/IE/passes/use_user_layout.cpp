//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// UseUserLayoutPass
//

class UseUserLayoutPass final : public IE::UseUserLayoutBase<UseUserLayoutPass> {
public:
    explicit UseUserLayoutPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnModule
//

void UseUserLayoutPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    const auto funcType = netFunc.getType();

    auto userInputs = netInfo.getInputsInfo();
    auto userOutputs = netInfo.getOutputsInfo();

    const auto getTypesWithUserLayout = [](SmallVector<IE::DataInfoOp, 1>& userDataInfo,
                                           ArrayRef<mlir::Type> originTypes, SmallVector<mlir::Type>& newTypes) {
        for (const auto& p : userDataInfo | indexed) {
            const auto ind = checked_cast<uint32_t>(p.index());

            const auto origType = originTypes[ind].cast<vpux::NDTypeInterface>();
            const auto userDimsOrder = p.value().getDimsOrder();

            newTypes[ind] = origType.changeDimsOrder(userDimsOrder);
        }
    };

    SmallVector<mlir::Type> newArgTypes(userInputs.size());
    getTypesWithUserLayout(userInputs, funcType.getInputs(), newArgTypes);

    SmallVector<mlir::Type> newResultTypes(userOutputs.size());
    getTypesWithUserLayout(userOutputs, funcType.getResults(), newResultTypes);

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val,
                                 vpux::NDTypeInterface newType) -> mlir::Operation* {
        const auto order = newType.getDimsOrder();
        return builder.create<IE::ReorderOp>(loc, newType, val, order.toAffineMap(builder.getContext()));
    };

    if (mlir::failed(convertFunc(netFunc, newArgTypes, newResultTypes, cvtOpBuilder, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUseUserLayoutPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseUserLayout(Logger log) {
    return std::make_unique<UseUserLayoutPass>(log);
}
