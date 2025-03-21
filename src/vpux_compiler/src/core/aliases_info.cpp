//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/types.hpp"
#include "vpux/compiler/dialect/core/interfaces/ops_interfaces.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <llvm/ADT/TypeSwitch.h>
#include <queue>
#include <stack>

using namespace vpux;

namespace {

std::string getValueForLog(mlir::Value val) {
    if (const auto arg = val.dyn_cast<mlir::BlockArgument>()) {
        return printToString("BlockArgument #{0} at '{1}'", arg.getArgNumber(), val.getLoc());
    }

    const auto res = val.cast<mlir::OpResult>();
    return printToString("Operation result #{0} for '{1}' at '{2}'", res.getResultNumber(), res.getOwner()->getName(),
                         val.getLoc());
}

void logAddAlias(const Logger& log, mlir::Value source, mlir::Value alias) {
    log.trace("Adding alias:");
    auto innerLog = log.nest();

    if (const auto arg = source.dyn_cast<mlir::BlockArgument>()) {
        innerLog.trace("- from: BlockArgument #{0} at '{1}'", arg.getArgNumber(), arg.getLoc());
    } else if (const auto res = source.cast<mlir::OpResult>()) {
        innerLog.trace("- from: Operation result #{0} for '{1}' at '{2}'", res.getResultNumber(),
                       res.getOwner()->getName(), res.getLoc());
    }

    if (const auto arg = alias.dyn_cast<mlir::BlockArgument>()) {
        innerLog.trace("- to: BlockArgument #{0} at '{1}'", arg.getArgNumber(), arg.getLoc());
    } else if (const auto res = alias.cast<mlir::OpResult>()) {
        innerLog.trace("- to: Operation result #{0} for '{1}' at '{2}'", res.getResultNumber(),
                       res.getOwner()->getName(), res.getLoc());
    }
}

}  // namespace

//
// AliasesInfoBase
//

const AliasesInfoBase::ValuesVector& AliasesInfoBase::getSources(mlir::Value val) const {
    const auto it = _sources.find(val);
    VPUX_THROW_UNLESS(it != _sources.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

mlir::Value AliasesInfoBase::getSource(mlir::Value val) const {
    const auto it = _sources.find(val);
    VPUX_THROW_UNLESS(it != _sources.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    VPUX_THROW_UNLESS(it->second.size() == 1, "Value '{0}' expected to have only one source. Got {1}",
                      getValueForLog(val), it->second.size());
    return *it->second.begin();
}

const AliasesInfoBase::ValuesVector& AliasesInfoBase::getRoots(mlir::Value val) const {
    const auto it = _roots.find(val);
    VPUX_THROW_UNLESS(it != _roots.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}

mlir::Value AliasesInfoBase::getRoot(mlir::Value val) const {
    const auto it = _roots.find(val);
    VPUX_THROW_UNLESS(it != _roots.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    VPUX_THROW_UNLESS(it->second.size() == 1, "Value '{0}' expected to have only one root. Got {1}",
                      getValueForLog(val), it->second.size());
    return *it->second.begin();
}

void AliasesInfoBase::uniqueAppend(ValuesVectorMap& map, mlir::Value key, mlir::Value value) {
    auto& values = map[key];
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.push_back(value);
}

void AliasesInfoBase::visitOp(mlir::Operation* op, bool ignoreInnerRegions = false) {
    VPUX_THROW_WHEN(op == nullptr, "NULL operation provided");
    std::function<bool(mlir::Type)> isBufferizedType = [&](mlir::Type type) -> bool {
        if (const auto asyncType = type.dyn_cast<mlir::async::ValueType>()) {
            return isBufferizedType(asyncType.getValueType());
        }

        return isBufferType(type);
    };

    llvm::TypeSwitch<mlir::Operation*, void>(op)
            .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewOp) {
                _log.trace("Got ViewLike Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                _log = _log.nest();

                const auto result = viewOp->getResult(0);
                const auto source = viewOp.getViewSource();

                VPUX_THROW_UNLESS(isBufferizedType(result.getType()),
                                  "AliasesInfo analysis works only with buffer types, got '{0}'", result.getType());
                VPUX_THROW_UNLESS(isBufferizedType(source.getType()),
                                  "AliasesInfo analysis works only with buffer types, got '{0}'", source.getType());

                addAlias(source, result);

                _log = _log.unnest();
            })
            .Case<MultiViewOpInterface>([&](MultiViewOpInterface viewOp) {
                _log.trace("Got MultiView Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                _log = _log.nest();

                for (const auto& result : viewOp->getResults()) {
                    _log.trace("Result #{0}", result.getResultNumber());

                    VPUX_THROW_UNLESS(isBufferizedType(result.getType()),
                                      "AliasesInfo analysis works only with buffer types, got '{0}'", result.getType());

                    const auto source = viewOp.getViewSource(result.getResultNumber());
                    if (source == nullptr) {
                        addAlias(result, result);
                        continue;
                    }

                    VPUX_THROW_UNLESS(isBufferizedType(source.getType()),
                                      "AliasesInfo analysis works only with buffer types, got '{0}'", source.getType());

                    addAlias(source, result);
                }

                _log = _log.unnest();
            })
            .Case<GroupedViewOpInterface>([&](GroupedViewOpInterface viewOp) {
                _log.trace("Got GroupedView Operation '{0}' at '{1}'", viewOp->getName(), viewOp->getLoc());
                _log = _log.nest();

                const auto result = viewOp->getResult(0);
                const auto sources = viewOp.getViewSources();

                VPUX_THROW_UNLESS(isBufferizedType(result.getType()),
                                  "AliasesInfo analysis works only with buffer types, got '{0}'", result.getType());
                for (const auto& source : sources) {
                    VPUX_THROW_UNLESS(isBufferizedType(source.getType()),
                                      "AliasesInfo analysis works only with buffer types, got '{0}'", source.getType());

                    addAlias(source, result);
                }

                _log = _log.unnest();
            })
            .Case<mlir::RegionBranchOpInterface>([&](mlir::RegionBranchOpInterface regionOp) {
                _log.trace("Got RegionBranch Operation '{0}' at '{1}'", regionOp->getName(), regionOp->getLoc());
                _log = _log.nest();

                SmallVector<mlir::RegionSuccessor> entries;
                regionOp.getSuccessorRegions(mlir::RegionBranchPoint::parent(), entries);

                for (const auto& entry : entries) {
                    auto* entryRegion = entry.getSuccessor();
                    VPUX_THROW_UNLESS(entryRegion != nullptr,
                                      "Entry region without an attached successor region at '{0}'", regionOp->getLoc());

                    const auto outerArgs = regionOp.getEntrySuccessorOperands(mlir::RegionBranchPoint(entryRegion));
                    const auto innerArgs = entry.getSuccessorInputs();

                    VPUX_THROW_UNLESS(outerArgs.size() == innerArgs.size(),
                                      "Mismatch between RegionBranch operands and its entry region arguments at '{0}'",
                                      regionOp->getLoc());

                    for (auto i : irange(outerArgs.size())) {
                        _log.trace("Check operand #{0} and corresponding region argument", i);

                        if (isBufferizedType(outerArgs[i].getType()) && isBufferizedType(innerArgs[i].getType())) {
                            addAlias(outerArgs[i], innerArgs[i]);
                        }
                    }
                }

                if (!ignoreInnerRegions) {
                    _log.trace("Traverse the RegionBranch inner regions");
                    _log = _log.nest();
                    for (auto& region : regionOp->getRegions()) {
                        for (auto& innerOp : region.getOps()) {
                            visitOp(&innerOp);
                        }
                    }
                    _log = _log.unnest();
                }

                for (auto& region : regionOp->getRegions()) {
                    SmallVector<mlir::RegionSuccessor> successors;
                    regionOp.getSuccessorRegions(mlir::RegionBranchPoint(&region), successors);

                    for (auto& successor : successors) {
                        for (auto& block : region) {
                            if (auto opInterface = llvm::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(
                                        block.getTerminator())) {
                                mlir::Region* pSuccessor = successor.getSuccessor();
                                auto branchPoint = (pSuccessor != nullptr) ? mlir::RegionBranchPoint(pSuccessor)
                                                                           : mlir::RegionBranchPoint::parent();
                                auto innerResults = opInterface.getSuccessorOperands(branchPoint);
                                const auto outerResults = successor.getSuccessorInputs();

                                VPUX_THROW_UNLESS(innerResults.size() == outerResults.size(),
                                                  "Mismatch between successor operands and its parent results at '{0}'",
                                                  regionOp->getLoc());

                                for (auto i : irange(innerResults.size())) {
                                    _log.trace("Check result #{0} and corresponding region result", i);

                                    if (isBufferizedType(innerResults[i].getType()) &&
                                        isBufferizedType(outerResults[i].getType())) {
                                        addAlias(innerResults[i], outerResults[i]);
                                    }
                                }

                                _log = _log.unnest();
                            }
                        }
                    }
                }
            })
            .Case<mlir::async::AwaitOp>([&](mlir::async::AwaitOp waitOp) {
                _log.trace("Got 'async.await' Operation at '{0}'", waitOp->getLoc());
                _log = _log.nest();

                if (const auto& result = waitOp.getResult()) {
                    const auto futureType = waitOp.getOperand().getType().dyn_cast<mlir::async::ValueType>();
                    VPUX_THROW_UNLESS(futureType != nullptr,
                                      "AliasesInfo analysis works only with !async.value<MemRef> types, got '{0}'",
                                      waitOp.getOperand().getType());

                    VPUX_THROW_UNLESS(isBufferizedType(futureType.getValueType()),
                                      "AliasesInfo analysis works only with buffer types, got '{0}'", futureType);
                    VPUX_THROW_UNLESS(isBufferizedType(result.getType()),
                                      "AliasesInfo analysis works only with buffer types, got '{0}'", result.getType());

                    addAlias(waitOp.getOperand(), result);
                }

                _log = _log.unnest();
            })
            .Default([&](mlir::Operation* op) {
                _log.trace("Got generic Operation '{0}' at '{1}'", op->getName(), op->getLoc());
                _log = _log.nest();

                for (const auto& result : op->getResults()) {
                    if (isBufferizedType(result.getType())) {
                        addAlias(result, result);
                    }
                }

                _log = _log.unnest();
            });
}

void AliasesInfoBase::addFuncArgAlias(mlir::Value funcArg) {
    _log.trace("Argument #{0}", getValueForLog(funcArg));
    bool isValidArgType = isBufferType(funcArg.getType());

    VPUX_THROW_UNLESS(
            isValidArgType,
            "AliasesInfo analysis works only with MemRef, SparseBuffer and DistributedBuffer types, got '{0}'",
            funcArg.getType());
    addAlias(funcArg, funcArg);
}

//
// ValueSourceInfo
//

ValueSourceInfo::ValueSourceInfo(mlir::Value val): AliasesInfoBase(Logger::global().nest("value-source-info", 0)) {
    std::queue<mlir::Value> q;
    q.push(val);

    while (!q.empty()) {
        auto currVal = q.front();
        q.pop();

        auto currOp = currVal.getDefiningOp();
        if (currOp == nullptr) {
            auto currSource = _sources.find(currVal);
            if (currSource != _sources.end()) {
                currOp = (*currSource->second.begin()).getDefiningOp();
            } else if (auto blockArg = currVal.dyn_cast<mlir::BlockArgument>()) {
                currOp = blockArg.getOwner()->getParentOp();
            }
        }

        if (mlir::isa_and_nonnull<mlir::func::FuncOp>(currOp)) {
            addFuncArgAlias(currVal);
            continue;
        }

        visitOp(currOp, true);

        auto values = _sources[currVal];
        for (auto& newVal : values) {
            if (newVal == nullptr) {
                continue;
            }
            q.push(newVal);
        }
    }
    updateRoots(val);
}

void ValueSourceInfo::addAlias(mlir::Value source, mlir::Value alias) {
    logAddAlias(_log, source, alias);
    if (source == alias) {
        uniqueAppend(_sources, alias, nullptr);
    } else {
        uniqueAppend(_sources, alias, source);
    }
}

void ValueSourceInfo::updateRoots(mlir::Value val) {
    std::stack<mlir::Value> s;
    s.push(val);
    while (!s.empty()) {
        auto key = s.top();
        s.pop();
        const auto sources = getSources(key);
        if (sources.size() == 1 && *sources.begin() == nullptr) {
            uniqueAppend(_roots, val, key);
        }
        for (const auto& source : sources) {
            if (source == nullptr) {
                continue;
            }
            s.push(source);
        }
    }
}

//
// AliasesInfo
//

AliasesInfo::AliasesInfo(mlir::func::FuncOp func): AliasesInfoBase(Logger::global().nest("alias-info", 0)) {
    init(func);
}

AliasesInfo::AliasesInfo(mlir::func::FuncOp func, VPU::MemoryKind memKind)
        : AliasesInfoBase(Logger::global().nest("alias-info", 0), memKind) {
    init(func);
}

void AliasesInfo::init(mlir::func::FuncOp func) {
    _log.trace("Analyze aliases for Function '@{0}'", func.getName());
    _log = _log.nest();

    _log.trace("Function arguments are roots for themselves");
    _log = _log.nest();
    for (const auto& funcArg : func.getArguments()) {
        addFuncArgAlias(funcArg);
    }
    _log = _log.unnest();

    _log.trace("Traverse the Function body");
    _log = _log.nest();

    auto ops = func.getOps();
    for (auto& op : ops) {
        visitOp(&op);
    }
}

void AliasesInfo::addAlias(mlir::Value source, mlir::Value alias) {
    if (_memKind.has_value()) {
        auto isTargetMemType = [&](mlir::Value buffer) {
            auto type = buffer.getType();

            if (const auto asyncType = type.dyn_cast<mlir::async::ValueType>()) {
                type = asyncType.getValueType();
            }

            auto ndType = type.dyn_cast<vpux::NDTypeInterface>();

            if (ndType == nullptr) {
                return false;
            }

            if (ndType.getMemoryKind() != _memKind) {
                return false;
            }
            return true;
        };

        if (!isTargetMemType(source) || !isTargetMemType(alias)) {
            return;
        }
    }

    logAddAlias(_log, source, alias);

    const auto roots = source == alias ? ValuesVector{alias} : getRoots(source);

    if (std::find(roots.begin(), roots.end(), alias) != roots.end()) {
        uniqueAppend(_sources, alias, nullptr);
    } else {
        uniqueAppend(_sources, alias, source);
    }
    for (const auto& root : roots) {
        _allAliases[root].insert(alias);
        uniqueAppend(_roots, alias, root);
    }
}

void AliasesInfo::removeAlias(mlir::Value alias) {
    const auto roots = getRoots(alias);

    auto rootsForAliasItr = _roots.find(alias);
    if (rootsForAliasItr != _roots.end()) {
        _roots.erase(rootsForAliasItr);
    }

    auto sourcesForAliasItr = _sources.find(alias);
    if (sourcesForAliasItr != _sources.end()) {
        _sources.erase(sourcesForAliasItr);
    }

    for (const auto& root : roots) {
        _allAliases[root].erase(alias);
    }
}

void AliasesInfo::remove(mlir::Value val) {
    _log.trace("Remove all info of a value '{0}''", getValueForLog(val));

    const auto roots = getRoots(val);

    for (const auto& root : roots) {
        _allAliases[root].erase(val);
        if (_allAliases[root].empty()) {
            _allAliases.erase(root);
        }
    }

    _allAliases.erase(val);
    _roots.erase(val);
    _sources.erase(val);
}

const AliasesInfo::ValuesSet& AliasesInfo::getAllAliases(mlir::Value val) const {
    const auto it = _allAliases.find(val);
    VPUX_THROW_UNLESS(it != _allAliases.end(), "Value '{0}' is not covered by aliases analysis", getValueForLog(val));
    return it->second;
}
