//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/IE/locations.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_USEUSERPRECISION
#define GEN_PASS_DEF_USEUSERPRECISION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// UseUserPrecisionPass
//

class UseUserPrecisionPass final : public IE::impl::UseUserPrecisionBase<UseUserPrecisionPass> {
public:
    explicit UseUserPrecisionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

//
// safeRunOnModule
//

void UseUserPrecisionPass::safeRunOnModule() {
    auto module = getOperation();

    IE::CNNNetworkOp netInfo;
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netInfo, netFunc);

    auto userInputs = netInfo.getInputsDataInfo();
    auto userOutputs = netInfo.getOutputsDataInfo();

    const auto funcType = netFunc.getFunctionType();

    SmallVector<mlir::Type> newArgTypes(netFunc.getNumArguments());

    for (const auto& p : funcType.getInputs() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = p.value().cast<vpux::NDTypeInterface>();
        const auto userType = userInputs[ind].getUserType().cast<vpux::NDTypeInterface>();

        const auto newType = origType.changeElemType(userType.getElementType());
        newArgTypes[ind] = newType;
    }

    SmallVector<mlir::Type> newResultTypes(netFunc.getNumResults());

    for (const auto& p : funcType.getResults() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = p.value().cast<vpux::NDTypeInterface>();
        const auto userType = userOutputs[ind].getUserType().cast<vpux::NDTypeInterface>();

        const auto newType = origType.changeElemType(userType.getElementType());
        newResultTypes[ind] = newType;
    }

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value val,
                                 vpux::NDTypeInterface newType) -> mlir::Operation* {
        const auto dstType = mlir::TypeAttr::get(newType.getElementType());
        const auto newLocation = appendLoc(baseLoc, "converted_to_{0}", dstType);
        return builder.create<IE::ConvertOp>(newLocation, newType, val, dstType);
    };

    if (mlir::failed(convertFunc(netFunc, newArgTypes, newResultTypes, cvtOpBuilder, _log))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUseUserPrecisionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createUseUserPrecisionPass(Logger log) {
    return std::make_unique<UseUserPrecisionPass>(log);
}
