//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_SETMEMORYSPACE
#define GEN_PASS_DEF_SETMEMORYSPACE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// SetMemorySpacePass
//

mlir::Type changeMemSpace(mlir::Type type, VPU::MemoryKind memKind) {
    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
        const auto newValueType = mlir::cast<vpux::NDTypeInterface>(asyncType.getValueType()).changeMemSpace(memKind);
        return mlir::async::ValueType::get(newValueType);
    }

    return mlir::cast<vpux::NDTypeInterface>(type).changeMemSpace(memKind);
}

class SetMemorySpacePass final : public VPUIP::impl::SetMemorySpaceBase<SetMemorySpacePass> {
public:
    SetMemorySpacePass(VPUIP::MemKindCreateFunc memKindCb, bool setMemorySpaceForFunctionBoundaries, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void updateFunction(mlir::func::FuncOp func, const AliasesInfo& aliasInfo) const;
    void updateAliases(
            AliasesInfo& aliasInfo, mlir::Value value, FuncRef<bool(mlir::Value)> pred = [](mlir::Value) {
                return true;
            }) const;
    void updateFunctionPreservingBoundaries(mlir::func::FuncOp func, AliasesInfo& aliasInfo) const;
    void safeRunOnModule() final;

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    bool _setMemorySpaceForFunctionBoundaries;
    VPU::MemoryKind _memKind{};
};

SetMemorySpacePass::SetMemorySpacePass(VPUIP::MemKindCreateFunc memKindCb, bool setMemorySpaceForFunctionBoundaries,
                                       Logger log)
        : _memKindCb(std::move(memKindCb)), _setMemorySpaceForFunctionBoundaries(setMemorySpaceForFunctionBoundaries) {
    VPUX_THROW_WHEN(_memKindCb == nullptr, "Missing memKindCb");
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult SetMemorySpacePass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    const auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.has_value()) {
        return mlir::failure();
    }

    if (setMemorySpaceForFunctionBoundaries.hasValue()) {
        _setMemorySpaceForFunctionBoundaries = setMemorySpaceForFunctionBoundaries.getValue();
    }

    _memKind = maybeMemKind.value();
    return mlir::success();
}

void SetMemorySpacePass::updateFunction(mlir::func::FuncOp func, const AliasesInfo& aliasInfo) const {
    VPUX_THROW_UNLESS(func.getNumArguments() >= func.getNumResults(), "Function '{0}' is not bufferized", func);
    const auto numInputs = func.getNumArguments() - func.getNumResults();

    const auto updateArgTypes = [&](mlir::ValueRange args, SmallVector<mlir::Type>& newTypes) {
        for (auto arg : args) {
            const auto newArgType = changeMemSpace(arg.getType(), _memKind);
            newTypes.push_back(newArgType);
            const auto& aliases = aliasInfo.getAllAliases(arg);
            for (auto var : aliases) {
                const auto newAliasType = changeMemSpace(var.getType(), _memKind);
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

void SetMemorySpacePass::updateAliases(AliasesInfo& aliasInfo, mlir::Value value,
                                       FuncRef<bool(mlir::Value)> pred) const {
    const auto& aliases = aliasInfo.getAllAliases(value);

    for (auto var : aliases) {
        if (!pred(var)) {
            continue;
        }

        _log.nest().trace("Process alias buffer '{0}'", var);

        const auto newType = changeMemSpace(var.getType(), _memKind);
        var.setType(newType);
    }
}

void SetMemorySpacePass::updateFunctionPreservingBoundaries(mlir::func::FuncOp func, AliasesInfo& aliasInfo) const {
    for (auto arg : func.getArguments()) {
        _log.nest().trace("Updating aliases of a function arg '{0}'", arg);
        updateAliases(aliasInfo, arg, [&](mlir::Value alias) {
            return alias != arg;
        });
    }

    _log.nest().trace("Casting arguments of non-host function '{0}' to memory space '{1}'", func.getSymName(),
                      _memKind);
    auto builder = mlir::OpBuilder::atBlockBegin(&func.getBody().front());
    for (auto arg : func.getArguments()) {
        const auto newType = changeMemSpace(arg.getType(), _memKind);
        // In HostCompile mode, block arguments must remain device-info-free
        // so that the function_type attribute stays consistent with the
        // call site (the caller passes buffers without an explicit memory
        // space). Instead of mutating the block argument type, insert a
        // Core.ReinterpretCast that adds the required memory space at this
        // specific use inside the function body.
        assert(!_setMemorySpaceForFunctionBoundaries && "This only applies to host-compile");
        // Note: ReinterpretCast is eventually transformed into a no-op, so
        // inserting it here unconditionally is not a significant problem.
        auto castOp = Core::ReinterpretCastOp::create(builder, appendLoc(arg.getLoc(), "memspace_cast"), newType, arg);
        _log.nest().trace("Inserted ReinterpretCast for device-info-free block arg: '{0}'", castOp.getResult());
        arg.replaceAllUsesExcept(castOp.getResult(), castOp);
        aliasInfo.addAlias(arg, castOp.getResult());
    }

    _log.nest().trace("Casting return values of non-host function '{0}' to memory space '{1}'", func.getSymName(),
                      _memKind);
    auto returnOp = findReturnOp(func);
    auto returnTypes = func.getFunctionType().getResults();
    builder.setInsertionPoint(returnOp);
    for (auto [operand, returnType] : llvm::zip(returnOp->getOperands(), returnTypes)) {
        auto castOp = Core::ReinterpretCastOp::create(builder, appendLoc(operand.getLoc(), "memspace_cast"), returnType,
                                                      operand);
        _log.nest().trace("Inserted ReinterpretCast for device-info-free return operand: '{0}'", castOp.getResult());
        operand.replaceAllUsesExcept(castOp.getResult(), castOp);
        aliasInfo.addAlias(operand, castOp.getResult());
    }
}

void SetMemorySpacePass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto entryPointFunc = net::getMainFunc(moduleOp);

    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        // Probably is ACT_SHAVE kernel
        if (funcOp.isExternal()) {
            return;
        }

        auto& aliasInfo = getChildAnalysis<AliasesInfo>(funcOp);
        // note: This is disabled in host-compile pipeline because the nestCallOp and host-side memrefs do not have
        // memory space locations.
        if (_setMemorySpaceForFunctionBoundaries || funcOp != entryPointFunc) {
            updateFunction(funcOp, aliasInfo);
        } else {
            updateFunctionPreservingBoundaries(funcOp, aliasInfo);
        }

        const auto allocOpCallback = [&](mlir::memref::AllocOp allocOp) {
            _log.trace("Got Alloc Operation '{0}'", allocOp->getLoc());

            if (allocOp.getType().getMemorySpace() != nullptr) {
                _log.nest().trace("It already has a memory space '{0}'", allocOp.getType().getMemorySpace());
                return;
            }

            updateAliases(aliasInfo, allocOp.getMemref());
        };

        const auto groupOpCallback = [&](vpux::GroupedViewOpInterface groupOp) {
            _log.trace("Got grouping operation '{0}'", groupOp->getLoc());

            // For grouping op memory space is set only if one of the buffers already has memory space set
            auto isMemSpaceSet = llvm::any_of(groupOp->getOperands(), [&](mlir::Value operand) {
                const auto operandMemSpaceAttr = mlir::cast<vpux::NDTypeInterface>(operand.getType()).getMemSpace();
                if (operandMemSpaceAttr == nullptr) {
                    return false;
                }
                const auto operandMemSpace =
                        VPU::symbolizeEnum<VPU::MemoryKind>(operandMemSpaceAttr.getLeafName()).value();
                return operandMemSpace == _memKind;
            });
            if (!isMemSpaceSet) {
                return;
            }

            for (auto operand : groupOp->getOperands() | indexed) {
                const auto operandMemSpace = mlir::cast<vpux::NDTypeInterface>(operand.value().getType()).getMemSpace();
                if (operandMemSpace != nullptr) {
                    _log.nest().trace("Operand '{0}' already has a memory space '{1}'", operand.index(),
                                      operandMemSpace);
                    continue;
                }

                _log.nest().trace("Updating memory space for operand '{0}'", operand.index());
                updateAliases(aliasInfo, operand.value());
            }
        };

        funcOp.walk(allocOpCallback);
        funcOp.walk(groupOpCallback);
    });
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createSetMemorySpacePass(MemKindCreateFunc memKindCb,
                                                                  bool setMemorySpaceForFunctionBoundaries,
                                                                  Logger log) {
    return std::make_unique<SetMemorySpacePass>(std::move(memKindCb), setMemorySpaceForFunctionBoundaries, log);
}
