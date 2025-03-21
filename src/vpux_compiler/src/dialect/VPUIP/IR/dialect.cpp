//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include "vpux/compiler/dialect/IERT/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Transforms/BufferizationUtils.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace vpux;

namespace {

//
// VPUIPInlinerInterface
//

struct VPUIPInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};

struct VPUIPBufferizerInterface : mlir::DialectBufferizerInterface {
    using mlir::DialectBufferizerInterface::DialectBufferizerInterface;

    mlir::Type getTensorTypeFromMemRefType(mlir::Type type) const final {
        return reconstructTensorType(type);
    }
};

}  // namespace

//
// initialize
//

void vpux::VPUIP::VPUIPDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUIP/ops.cpp.inc>
            >();

    registerAttributes();
    registerTypes();

    addInterfaces<VPUIPInlinerInterface, VPUIPBufferizerInterface>();
}

//
// Operation executor attributes
//

namespace {

constexpr StringLiteral executorAttrName = "VPUIP.executor";
constexpr StringLiteral executorInstanceMaskAttrName = "VPUIP.executorIdx";

}  // namespace

void vpux::VPUIP::VPUIPDialect::setExecutor(mlir::async::ExecuteOp execOp, IndexedSymbolAttr executor) {
    VPUX_THROW_UNLESS(executor != nullptr, "Got an empty executor");
    execOp->setAttr(executorAttrName, executor);
}

void vpux::VPUIP::VPUIPDialect::setExecutorInstanceMask(mlir::async::ExecuteOp execOp,
                                                        mlir::ArrayAttr executorInstanceMask) {
    VPUX_THROW_UNLESS(executorInstanceMask != nullptr, "Got an empty executor instance");
    execOp->setAttr(executorInstanceMaskAttrName, executorInstanceMask);

    // In case of some DMA tasks port is also maintained as part of operation iself
    if (getExecutorKind(execOp) == VPU::ExecutorKind::DMA_NN) {
        if (executorInstanceMask.getValue().size() != 1) {
            // In case there are multiple executors subsequent passes should take care
            // of handling this
            return;
        }

        auto portIdxAttr = executorInstanceMask[0].cast<mlir::IntegerAttr>();

        auto* bodyBlock = execOp.getBody();
        for (auto& op : bodyBlock->getOperations()) {
            auto opToCheck = &op;
            if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(opToCheck)) {
                opToCheck = nceClustOp.getInnerTaskOp();
            }

            if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(opToCheck)) {
                dmaOp.setPortAttribute(portIdxAttr);
            }
        }
    }
}

llvm::StringLiteral vpux::VPUIP::VPUIPDialect::getExecutorAttrName() {
    return executorAttrName;
}

IndexedSymbolAttr vpux::VPUIP::VPUIPDialect::getExecutor(mlir::async::ExecuteOp execOp) {
    const auto executor = execOp->getAttr(executorAttrName);
    VPUX_THROW_UNLESS(executor != nullptr, "Can't find Executor attributes for Operation at '{0}'", execOp->getLoc());

    const auto executorSymbol = executor.dyn_cast<IndexedSymbolAttr>();
    VPUX_THROW_UNLESS(executorSymbol != nullptr, "Unsupported Executor attribute '{0}'", executorSymbol);

    return executorSymbol;
}

VPU::ExecutorKind vpux::VPUIP::VPUIPDialect::getExecutorKind(mlir::async::ExecuteOp execOp) {
    const auto executor = getExecutor(execOp);
    const auto maybeExecutorKind = vpux::VPU::symbolizeExecutorKind(executor.getLeafNameAttr());
    VPUX_THROW_WHEN(!maybeExecutorKind.has_value(), "Unsupported Executor Kind attribute '{0}'",
                    executor.getLeafNameAttr());

    return maybeExecutorKind.value();
}

bool vpux::VPUIP::VPUIPDialect::hasExecutorInstanceMask(mlir::async::ExecuteOp execOp) {
    return (execOp->getAttr(executorInstanceMaskAttrName) != nullptr);
}

bool vpux::VPUIP::VPUIPDialect::isComputeExecutorKind(VPU::ExecutorKind executorKind) {
    static const llvm::DenseSet<VPU::ExecutorKind> computeExecutors = {
            VPU::ExecutorKind::DPU, VPU::ExecutorKind::SHAVE_ACT, VPU::ExecutorKind::M2I};
    return computeExecutors.find(executorKind) != computeExecutors.end();
}

mlir::ArrayAttr vpux::VPUIP::VPUIPDialect::getExecutorInstanceMask(mlir::async::ExecuteOp execOp) {
    const auto executorInstanceMask = execOp->getAttr(executorInstanceMaskAttrName);
    VPUX_THROW_UNLESS(executorInstanceMask != nullptr, "Can't find Executor Instance attributes for Operation at '{0}'",
                      execOp->getLoc());

    const auto executorInstanceMaskSymbol = executorInstanceMask.dyn_cast<mlir::ArrayAttr>();
    VPUX_THROW_UNLESS(executorInstanceMaskSymbol != nullptr, "Unsupported Executor Instance attribute '{0}'",
                      executorInstanceMaskSymbol);

    return executorInstanceMaskSymbol;
}

//
// materializeConstant
//

mlir::Operation* vpux::VPUIP::VPUIPDialect::materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value,
                                                                mlir::Type type, mlir::Location loc) {
    if (!mlir::isa<Const::ContentAttr>(value)) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant from Attribute '{0}'", value);
        return nullptr;
    }

    if (!type.isa<mlir::MemRefType>()) {
        (void)errorAt(loc, "Can't materialize VPUIP Constant for Type '{0}'", type);
        return nullptr;
    }

    return builder.create<Const::DeclareOp>(loc, type, mlir::cast<Const::ContentAttr>(value));
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIP/dialect.cpp.inc>
