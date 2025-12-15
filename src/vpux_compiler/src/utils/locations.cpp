//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/locations.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

mlir::Location vpux::createLayerLocation(mlir::MLIRContext* ctx, const std::string& layerName,
                                         const std::string& layerType) {
    const auto layerNameAttr = mlir::StringAttr::get(ctx, layerName);
    const auto nameLoc = mlir::NameLoc::get(layerNameAttr);

    SmallVector<mlir::NamedAttribute> fields;
    fields.emplace_back(mlir::StringAttr::get(ctx, "type"), mlir::StringAttr::get(ctx, layerType));
    fields.emplace_back(mlir::StringAttr::get(ctx, "name"), layerNameAttr);
    auto metadata = mlir::DictionaryAttr::get(ctx, fields);

    return mlir::FusedLoc::get(ctx, {nameLoc}, metadata);
}

mlir::Location vpux::getValueLocation(mlir::Value val) {
    // value is produced by real operation, so use it
    if (auto producerOp = val.getDefiningOp()) {
        if (producerOp->getNumResults() < 2) {
            return producerOp->getLoc();
        }
        for (auto p : producerOp->getResults() | indexed) {
            if (p.value() == val) {
                return takeOpLoc(producerOp, StringLiteral("res_{0}"), p.index());
            }
        }
        VPUX_THROW("Unsupported number of results");
    }
    // value is a block argument, so a function argument
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
        const size_t inputNum = arg.getArgNumber();

        const auto ownerOp = mlir::dyn_cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
        VPUX_THROW_WHEN(ownerOp == nullptr,
                        "Invalid type of parent operation, expected to get mlir::func::FuncOp, but got {0}",
                        arg.getOwner()->getParentOp());
        auto moduleOp = getModuleOp(ownerOp);
        auto netInfoOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
        if (netInfoOps.size() != 1) {
            if constexpr (vpux::isDeveloperBuild()) {
                vpux::Logger::global().warning("Can't get location for input. If it isn't a test, please, debug this.");
                const std::string inputName = "generated_input_" + std::to_string(inputNum);
                return createLayerLocation(moduleOp->getContext(), inputName, "Parameter");
            } else {
                VPUX_THROW("Can't get location for input.");
            }
        }

        net::NetworkInfoOp netInfo;
        mlir::func::FuncOp netFunc;
        net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);

        if (ownerOp != netFunc) {
            // Note: one cannot provably deduce a single location as a
            // non-net-func could be called multiple times, thus, return the
            // location of this function.
            return appendLoc(ownerOp->getLoc(), "arg_{0}", arg.getArgNumber());
        }

        auto inputsInfo = to_small_vector(netInfo.getInputsInfo().getOps<net::DataInfoOp>());

        return inputsInfo[inputNum]->getLoc();
    }
    VPUX_THROW("Can't get location of '{0}'", val);
}
