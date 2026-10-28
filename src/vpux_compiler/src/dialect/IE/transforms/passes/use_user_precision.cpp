//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
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

    auto [netInfo, netFunc] = net::getFromModule(module);

    auto userInputs = netInfo.getInputsDataInfo();
    auto userOutputs = netInfo.getOutputsDataInfo();

    const auto funcType = netFunc.getFunctionType();

    SmallVector<mlir::Type> newArgTypes(netFunc.getNumArguments());

    for (const auto& p : funcType.getInputs() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = mlir::cast<vpux::NDTypeInterface>(p.value());
        const auto userType = mlir::cast<vpux::NDTypeInterface>(userInputs[ind].getUserType());

        const auto newType = origType.changeElemType(userType.getElementType());
        newArgTypes[ind] = newType;
    }

    SmallVector<mlir::Type> newResultTypes(netFunc.getNumResults());

    for (const auto& p : funcType.getResults() | indexed) {
        const auto ind = checked_cast<uint32_t>(p.index());

        const auto origType = mlir::cast<vpux::NDTypeInterface>(p.value());
        const auto userType = mlir::cast<vpux::NDTypeInterface>(userOutputs[ind].getUserType());

        const auto newType = origType.changeElemType(userType.getElementType());
        newResultTypes[ind] = newType;
    }

    const auto cvtOpBuilder = [](mlir::OpBuilder& builder, mlir::Location baseLoc, mlir::Value val,
                                 vpux::NDTypeInterface newType) -> mlir::Operation* {
        const auto dstType = mlir::TypeAttr::get(newType.getElementType());
        const auto newLocation = appendLoc(baseLoc, "converted_to_{0}", dstType);
        return IE::ConvertOp::create(builder, newLocation, newType, val, dstType);
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
