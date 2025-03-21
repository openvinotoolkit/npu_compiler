//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/mem_live_range_info.hpp"

#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/utils/core/error.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <algorithm>

using namespace vpux;

vpux::MemLiveRangeInfo::MemLiveRangeInfo(mlir::func::FuncOp funcOp, mlir::AnalysisManager& am)
        : _log(Logger::global().nest("mem-live-range-info", 0)),
          _aliasInfo(am.getAnalysis<AliasesInfo, mlir::func::FuncOp>()) {
    buildRangeInfo(funcOp);
}

vpux::MemLiveRangeInfo::MemLiveRangeInfo(mlir::func::FuncOp funcOp, const AliasesInfo& aliasInfo)
        : _log(Logger::global().nest("mem-live-range-info", 0)), _aliasInfo(aliasInfo) {
    buildRangeInfo(funcOp);
}

vpux::MemLiveRangeInfo::MemLiveRangeInfo(mlir::func::FuncOp funcOp, const AliasesInfo& aliasInfo,
                                         std::optional<VPU::MemoryKind> memKind)
        : _log(Logger::global().nest("mem-live-range-info", 0)), _aliasInfo(aliasInfo), _memKind(memKind) {
    buildRangeInfo(funcOp);
}

void vpux::MemLiveRangeInfo::buildRangeInfo(mlir::func::FuncOp funcOp) {
    _log.trace("Collect all buffer allocations");
    _log = _log.nest();

    auto isTargetMemType = [&](mlir::Value buf) {
        auto bufType = buf.getType();
        if (const auto asyncType = bufType.dyn_cast<mlir::async::ValueType>()) {
            bufType = asyncType.getValueType();
        }

        auto bufNDType = bufType.dyn_cast<vpux::NDTypeInterface>();

        if (bufNDType == nullptr) {
            return false;
        }

        if (bufNDType.getMemoryKind() != _memKind) {
            return false;
        }

        return true;
    };

    auto updateConsProdMap = [&](mlir::SmallVector<mlir::Value> buffers, OpToUsedBuffersMap& map,
                                 mlir::async::ExecuteOp& execOp) {
        for (const auto& buffer : buffers) {
            if (_memKind.has_value() && !isTargetMemType(buffer)) {
                continue;
            }

            auto rootBuffer = _aliasInfo.getRoot(buffer);
            map[execOp].insert(rootBuffer);
        }
    };

    funcOp->walk([&](mlir::Operation* op) {
        if (!isBufAllocOp(op)) {
            if (auto curExecOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
                // Get live rages of input/output buffers per async::ExecuteOp
                auto* bodyBlock = curExecOp.getBody();
                for (auto& innerOp : bodyBlock->getOperations()) {
                    if (auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(innerOp)) {
                        auto inputs = vpux::to_small_vector(layerOp.getInputs());
                        if (auto nceTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(innerOp)) {
                            // in case of NCEClusterTaskOp we need to remove parent outputs from inputs
                            // in order to make depenendcy calculation work correctly
                            auto parentOutput = nceTaskOp.getParentOutput();
                            auto parentOutputSparsityMap = nceTaskOp.getParentOutputSparsityMap();
                            auto input = nceTaskOp.getInput();
                            auto inputSparsityMap = nceTaskOp.getInputSparsityMap();
                            auto weights = nceTaskOp.getWeights();
                            auto weightsSparsityMap = nceTaskOp.getWeightsSparsityMap();
                            llvm::SmallVector<mlir::Value> inputsToSanitize{};
                            inputsToSanitize.swap(inputs);
                            std::copy_if(inputsToSanitize.begin(), inputsToSanitize.end(), std::back_inserter(inputs),
                                         [&](mlir::Value value) {
                                             // For in-place eltwise op it might happen that parentOutput == input.
                                             // Check those first to make sure they don't get removed.
                                             if (value == input || value == inputSparsityMap || value == weights ||
                                                 value == weightsSparsityMap) {
                                                 return true;
                                             }
                                             return (value != parentOutput) && (value != parentOutputSparsityMap);
                                         });
                        }
                        auto outputs = layerOp.getOutputs();

                        updateConsProdMap(std::move(inputs), _opInputBuffersMap, curExecOp);
                        updateConsProdMap(outputs, _opOutputBuffersMap, curExecOp);
                    }
                }
            }
            return;
        }

        _log.trace("Got buffer value allocated by '{0}' at '{1}'", op->getName(), op->getLoc());
        _log = _log.nest();

        for (auto res : op->getResults()) {
            if (_memKind.has_value() && !isTargetMemType(res)) {
                continue;
            }
            addNewBuffer(res);
        }

        _log = _log.unnest();
    });

    _log = _log.unnest();
}

// call for each result of allocation operation in function
void vpux::MemLiveRangeInfo::addNewBuffer(mlir::Value val) {
    _log.trace("Collect all direct and indirect users");
    _log = _log.nest();

    auto* valRegion = val.getParentRegion();
    const auto& aliases = _aliasInfo.getAllAliases(val);
    auto& allUsers = _allUsersInBlock[val];

    for (auto alias : aliases) {
        _log.trace("Process alias '{0}'", alias);
        _log = _log.nest();

        if (alias.getParentBlock() == val.getParentBlock()) {
            _log.trace("The alias belongs to the same block, traverse its users");
            _log = _log.nest();

            for (auto* user : alias.getUsers()) {
                _log.trace("Got alias user '{0}' at '{1}'", user->getName(), user->getLoc());

                auto* userAncestor = valRegion->findAncestorOpInRegion(*user);
                VPUX_THROW_UNLESS(userAncestor != nullptr,
                                  "Alias user '{0}' doesn't belong to the same block or its sub-region as '{1}'",
                                  user->getLoc(), val);

                _log.trace("It has an ancestor '{0}' at '{1}' in root value parent region", userAncestor->getName(),
                           userAncestor->getLoc());

                allUsers.insert(userAncestor);
                _opBuffersMap[userAncestor].insert(val);
            }

            _log = _log.unnest();
        } else {
            _log.trace("The alias belongs to the sub-region of root block");

            auto* aliasParentOp = alias.getParentRegion()->getParentOp();
            VPUX_THROW_UNLESS(aliasParentOp != nullptr, "Alias '{0}' has no parent Operation", alias);

            _log.trace("It belongs to the operation '{0}' at '{1}'", aliasParentOp->getName(), aliasParentOp->getLoc());

            auto* parentAncestor = val.getParentRegion()->findAncestorOpInRegion(*aliasParentOp);
            VPUX_THROW_UNLESS(parentAncestor != nullptr,
                              "Alias '{0}' doesn't belong to the same block or its sub-region as '{1}'", alias, val);

            _log.trace("It has an ancestor '{0}' at '{1}' in root value parent region", parentAncestor->getName(),
                       parentAncestor->getLoc());

            allUsers.insert(parentAncestor);
            _opBuffersMap[parentAncestor].insert(val);
        }

        _log = _log.unnest();
    }

    _log = _log.unnest();
}

ValueOrderedSet vpux::MemLiveRangeInfo::getUsedBuffers(mlir::Operation* op) const {
    const auto it = _opBuffersMap.find(op);
    if (it != _opBuffersMap.end()) {
        return it->second;
    }

    return {};
}

ValueOrderedSet vpux::MemLiveRangeInfo::getInputBuffers(mlir::Operation* op) {
    const auto it = _opInputBuffersMap.find(op);
    if (it != _opInputBuffersMap.end()) {
        return it->second;
    }

    return {};
}

ValueOrderedSet vpux::MemLiveRangeInfo::getOutputBuffers(mlir::Operation* op) {
    const auto it = _opOutputBuffersMap.find(op);
    if (it != _opOutputBuffersMap.end()) {
        return it->second;
    }

    return {};
}

bool vpux::MemLiveRangeInfo::isBufferUsedByOp(mlir::Value val, mlir::Operation* op) const {
    const auto valIt = _allUsersInBlock.find(val);
    VPUX_THROW_UNLESS(valIt != _allUsersInBlock.end(), "Value '{0}' is not a buffer", val);
    auto& allUsers = valIt->second;

    return allUsers.find(op) != allUsers.end();
}

size_t vpux::MemLiveRangeInfo::eraseUser(mlir::Value val, mlir::Operation* op) {
    const auto valIt = _allUsersInBlock.find(val);
    VPUX_THROW_UNLESS(valIt != _allUsersInBlock.end(), "Value '{0}' is not a buffer", val);
    auto& allUsers = valIt->second;

    const auto opIt = _opBuffersMap.find(op);
    VPUX_THROW_UNLESS(opIt != _opBuffersMap.end(), "Operation '{0}' at '{1}' is not a buffer user", op->getName(),
                      op->getLoc());
    auto& allBufs = opIt->second;

    VPUX_THROW_UNLESS(allUsers.erase(op), "Operation '{0}' at '{1}' is not a buffer '{2}' user", op->getName(),
                      op->getLoc(), val);
    allBufs.erase(val);

    return allUsers.size();
}
