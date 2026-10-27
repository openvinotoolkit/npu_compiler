//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/dialect.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <utility>

namespace vpux::HostExec {
#define GEN_PASS_DECL_EXTRACTRETURNSHAPES
#define GEN_PASS_DEF_EXTRACTRETURNSHAPES
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

class ExtractReturnShapesPass final : public HostExec::impl::ExtractReturnShapesBase<ExtractReturnShapesPass> {
public:
    explicit ExtractReturnShapesPass(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

private:
    Logger _log;
};

void ExtractReturnShapesPass::safeRunOnModule() {
    auto module = getOperation();

    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFunc;
    net::NetworkInfoOp::getFromModule(module, netInfo, mainFunc);

    SmallVector<mlir::Type> resultMainTypes(mainFunc.getResultTypes());
    SmallVector<mlir::Type> fromElementsOpTypes;
    SmallVector<mlir::Value> fromElementsOps;
    // No need to check for nullptr because getTerminator() in findReturnOp func asserts
    // in case a func block has no valid terminator
    auto returnOp = findReturnOp(mainFunc);

    mlir::OpBuilder builder(returnOp);
    for (auto& operand : returnOp->getOpOperands()) {
        auto operandType = operand.get().getType();

        SmallVector<mlir::Value> indexCastOpValues;
        auto operandTypeInterface = mlir::cast<NDTypeInterface>(operandType);
        for (int64_t idx = 0; idx < operandTypeInterface.getRank(); ++idx) {
            // Create tensor.dim op for each tensor's dimension of a returnOp's operand
            auto dimOp = mlir::tensor::DimOp::create(builder, appendLoc(operand.get().getLoc(), "dim_{0}", idx),
                                                     operand.get(), idx);
            // Cast tensor.dim's index type to int64
            auto toI64 = mlir::arith::IndexCastOp::create(builder, appendLoc(operand.get().getLoc(), "to_i64_{0}", idx),
                                                          vpux::getInt64Type(builder.getContext()), dimOp);
            indexCastOpValues.push_back(toI64->getResult(0));
        }

        mlir::ValueRange indexCastOpValueRange(indexCastOpValues);

        // Create tensor.from_elements ops from tensor.dim ops
        auto fromElementsOp = mlir::tensor::FromElementsOp::create(
                builder, appendLoc(operand.get().getLoc(), "from_elements"),
                mlir::RankedTensorType::get({operandTypeInterface.getRank()}, vpux::getInt64Type(builder.getContext())),
                indexCastOpValueRange);

        fromElementsOps.push_back(fromElementsOp);
        resultMainTypes.push_back(fromElementsOp.getType());
        fromElementsOpTypes.push_back(fromElementsOp.getType());
    }

    // We need to return new tensor.from_elements ops to avoid them being eliminated as unused
    mainFunc.setType(mlir::FunctionType::get(mainFunc->getContext(), mainFunc.getArgumentTypes(), resultMainTypes));
    returnOp.getOperandsMutable().append(fromElementsOps);

    // Update NetworkInfo op with new outputs
    auto& outputsRegion = netInfo.getOutputsInfo();
    mlir::OpBuilder netInfoBuilder(netInfo);
    netInfoBuilder.setInsertionPointToEnd(&outputsRegion.back());

    for (size_t i = 0; i < fromElementsOpTypes.size(); ++i) {
        auto name = formatv("out_{0}", i).str();
        net::DataInfoOp::create(netInfoBuilder, appendLoc(mainFunc.getLoc(), name), name, fromElementsOpTypes[i]);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::HostExec::createExtractReturnShapesPass(Logger log) {
    return std::make_unique<ExtractReturnShapesPass>(std::move(log));
}
