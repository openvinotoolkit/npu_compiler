//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// SetMemorySpacePass
//

class SetMemorySpacePass final : public VPUIP::SetMemorySpaceBase<SetMemorySpacePass> {
public:
    SetMemorySpacePass(VPUIP::MemKindCreateFunc memKindCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void updateFunction(mlir::FuncOp func, const AliasesInfo& aliasInfo) const;
    void updateAliases(AliasesInfo& aliasInfo, mlir::Value value) const;
    void safeRunOnFunc() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPU::MemoryKind _memKind{};
};

SetMemorySpacePass::SetMemorySpacePass(VPUIP::MemKindCreateFunc memKindCb, Logger log)
        : _memKindCb(std::move(memKindCb)) {
    VPUX_THROW_UNLESS(_memKindCb != nullptr, "Missing memKindCb");
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult SetMemorySpacePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    const auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.hasValue()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.getValue();
    return mlir::success();
}

void SetMemorySpacePass::updateFunction(mlir::FuncOp func, const AliasesInfo& aliasInfo) const {
    VPUX_THROW_UNLESS(func.getNumArguments() >= func.getNumResults(), "Function '{0}' is not bufferized", func);
    const auto numInputs = func.getNumArguments() - func.getNumResults();

    const auto updateArgTypes = [&](mlir::ValueRange args, SmallVector<mlir::Type>& newTypes) {
        for (auto arg : args) {
            const auto argType = arg.getType().cast<vpux::NDTypeInterface>();
            const auto newArgType = argType.changeMemSpace(_memKind);

            newTypes.push_back(newArgType);
            const auto& aliases = aliasInfo.getAllAliases(arg);
            for (auto var : aliases) {
                auto aliasType = var.getType();
                mlir::Type newAliasType;
                if (auto asyncType = aliasType.dyn_cast<mlir::async::ValueType>()) {
                    newAliasType = mlir::async::ValueType::get(
                            asyncType.getValueType().cast<vpux::NDTypeInterface>().changeMemSpace(_memKind));
                } else {
                    newAliasType = aliasType.cast<vpux::NDTypeInterface>().changeMemSpace(_memKind);
                }

                var.setType(newAliasType);
            }
        }
    };

    SmallVector<mlir::Type> newArgTypes;
    SmallVector<mlir::Type> newReturnTypes;

    updateArgTypes(func.getArguments(), newArgTypes);
    updateArgTypes(func.getArguments().drop_front(numInputs), newReturnTypes);

    VPUX_THROW_UNLESS(updateFunctionSignature(func, newArgTypes, newReturnTypes, _log).succeeded(),
                      "Fail to update function signature. new input types: '{0}'; new return types: '{1}'", newArgTypes,
                      newReturnTypes);
}

void SetMemorySpacePass::updateAliases(AliasesInfo& aliasInfo, mlir::Value value) const {
    const auto& aliases = aliasInfo.getAllAliases(value);

    for (auto var : aliases) {
        _log.nest().trace("Process alias buffer '{0}'", var);

        if (const auto futureType = var.getType().dyn_cast<mlir::async::ValueType>()) {
            const auto origType = futureType.getValueType().dyn_cast<vpux::NDTypeInterface>();
            VPUX_THROW_UNLESS(origType != nullptr, "Got non vpux::NDTypeInterface Type '{0}'", var.getType());

            const auto newType = origType.changeMemSpace(_memKind);
            const auto newFutureType = mlir::async::ValueType::get(newType);

            var.setType(newFutureType);
        } else {
            const auto origType = var.getType().dyn_cast<vpux::NDTypeInterface>();
            VPUX_THROW_UNLESS(origType != nullptr, "Got non vpux::NDTypeInterface Type '{0}'", var.getType());

            const auto newType = origType.changeMemSpace(_memKind);
            var.setType(newType);
        }
    }
}

void SetMemorySpacePass::safeRunOnFunc() {
    auto& aliasInfo = getAnalysis<AliasesInfo>();
    auto func = getFunction();

    updateFunction(func, aliasInfo);

    const auto allocOpCallback = [&](mlir::memref::AllocOp allocOp) {
        _log.trace("Got Alloc Operation '{0}'", allocOp->getLoc());

        if (allocOp.getType().getMemorySpace() != nullptr) {
            _log.nest().trace("It already has a memory space '{0}'", allocOp.getType().getMemorySpace());
            return;
        }

        updateAliases(aliasInfo, allocOp.memref());
    };

    const auto groupOpCallback = [&](vpux::GroupedViewOpInterface groupOp) {
        _log.trace("Got grouping operation '{0}'", groupOp->getLoc());

        // For grouping op memory space is set only if one of the buffers already has memory space set
        auto isMemSpaceSet = llvm::any_of(groupOp->getOperands(), [&](mlir::Value operand) {
            const auto operandMemSpaceAttr = operand.getType().cast<vpux::NDTypeInterface>().getMemSpace();
            if (operandMemSpaceAttr == nullptr) {
                return false;
            }
            const auto operandMemSpace =
                    VPU::symbolizeEnum<VPU::MemoryKind>(operandMemSpaceAttr.getLeafName()).getValue();
            return operandMemSpace == _memKind;
        });
        if (!isMemSpaceSet) {
            return;
        }

        for (auto operand : groupOp->getOperands() | indexed) {
            const auto operandMemSpace = operand.value().getType().cast<vpux::NDTypeInterface>().getMemSpace();
            if (operandMemSpace != nullptr) {
                _log.nest().trace("Operand '{0}' already has a memory space '{1}'", operand.index(), operandMemSpace);
                continue;
            }

            _log.nest().trace("Updating memory space for operand '{0}'", operand.index());
            updateAliases(aliasInfo, operand.value());
        }
    };

    func.walk(allocOpCallback);
    func.walk(groupOpCallback);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetMemorySpacePass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<SetMemorySpacePass>(std::move(memKindCb), log);
}
