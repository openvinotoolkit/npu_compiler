//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/multi_consumer_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <functional>
#include <utility>

using namespace vpux;

namespace {

struct ViewLikeOpPathElem final {
    mlir::Operation* op = nullptr;
    unsigned resultIndex = 0;
    mlir::Value inputValue;
};

struct MPEMappableConsumerUsePath final {
    mlir::OpOperand* use = nullptr;
    mlir::Value sourceValue;
    SmallVector<ViewLikeOpPathElem> viewPath;
};

bool isFoldableQuantizedLayerOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    if (mlir::isa<IE::MaxPoolOp>(op)) {
        return false;
    }
    if (op->getNumResults() != 1 || !op->getResult(0).hasOneUse()) {
        return false;
    }
    return mlir::isa<IE::QuantizedLayerOpInterface>(op);
}

bool isMultiplyPartOfNormalizeL2Pattern(mlir::Operation* op) {
    if (!mlir::isa<IE::MultiplyOp>(op)) {
        return false;
    }

    const auto feedsNormalizeL2 = llvm::any_of(op->getResults(), [](mlir::Value result) {
        return llvm::any_of(result.getUsers(), [](mlir::Operation* user) {
            return mlir::isa<IE::NormalizeL2Op>(user);
        });
    });
    if (feedsNormalizeL2) {
        return true;
    }

    return llvm::any_of(op->getOperands(), [](mlir::Value operand) {
        return mlir::isa_and_present<IE::NormalizeL2Op>(operand.getDefiningOp());
    });
}

bool isSignedWeightsAsInputConsumer(mlir::Operation* terminalOp) {
    mlir::quant::QuantizedType firstQuantType;
    const auto weights = IE::findNCEOpWeightsAsInputOrConst(terminalOp, &firstQuantType);
    if (mlir::failed(weights) || !mlir::isa<mlir::BlockArgument>(weights.value())) {
        return false;
    }
    if (firstQuantType != nullptr) {
        return firstQuantType.isSigned();
    }
    const auto elemType = mlir::cast<NDTypeInterface>(weights.value().getType()).getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return intType.isSignedInteger();
    }
    return false;
}

mlir::Type getFakeQuantizeInputQuantizedType(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || inHighConst == nullptr) {
        return nullptr;
    }
    const auto realType = mlir::cast<vpux::NDTypeInterface>(fqOp.getInput().getType());
    if (realType == nullptr) {
        return nullptr;
    }
    if (auto realElemType = mlir::dyn_cast<mlir::FloatType>(realType.getElementType())) {
        return IE::getQuantizedType(inLowConst.getContentAttr(), inHighConst.getContentAttr(),
                                    fqOp.getLevelsOrLowFpTypeAttr(), realElemType, /*isSigned=*/false, fqOp.getLoc(),
                                    fqOp.getAutoBroadcast());
    }
    return nullptr;
}

SmallVector<mlir::Operation*> getDirectUsers(mlir::Operation* op) {
    SmallVector<mlir::Operation*> directUsers;
    if (op == nullptr) {
        return directUsers;
    }

    llvm::SmallPtrSet<mlir::Operation*, 8> directSeen;
    for (auto result : op->getResults()) {
        for (auto* userOp : result.getUsers()) {
            if (directSeen.insert(userOp).second) {
                directUsers.push_back(userOp);
            }
        }
    }

    return directUsers;
}

IE::MPEConsumerAnalysis getMPEEngineMappableConsumers(mlir::Operation* op) {
    if (op == nullptr) {
        return {};
    }
    const auto& strategyFactory = IE::getIEStrategyFactory(op->getContext());
    if (strategyFactory == nullptr) {
        return {};
    }
    const auto multiConsumerStrategy = strategyFactory->getMultiConsumerStrategy();
    if (multiConsumerStrategy == nullptr) {
        return {};
    }
    return multiConsumerStrategy->getMPEEngineMappableConsumers(op);
}

bool hasMPEEngineMappableDirectUser(mlir::Operation* producerOp, mlir::Operation* userOp) {
    if (producerOp == nullptr || userOp == nullptr) {
        return false;
    }

    const auto strategyResult = getMPEEngineMappableConsumers(producerOp);
    return llvm::is_contained(strategyResult.mappableDirectUsers, userOp);
}

SmallVector<MPEMappableConsumerUsePath> collectMPEEngineMappableConsumerUsesThroughViewLikes(mlir::Value rootValue) {
    SmallVector<MPEMappableConsumerUsePath> mappableUses;
    if (rootValue == nullptr) {
        return mappableUses;
    }

    SmallVector<ViewLikeOpPathElem> currentPath;
    llvm::SmallPtrSet<mlir::Operation*, 16> activeViewOps;

    const std::function<void(mlir::Value)> collect = [&](mlir::Value value) {
        for (auto& use : llvm::make_early_inc_range(value.getUses())) {
            auto* userOp = use.getOwner();
            if (IE::isElemTypeTransparentViewLikeOp(userOp)) {
                if (!activeViewOps.insert(userOp).second) {
                    continue;
                }
                for (auto result : userOp->getResults()) {
                    currentPath.push_back(ViewLikeOpPathElem{userOp, result.getResultNumber(), value});
                    collect(result);
                    currentPath.pop_back();
                }
                activeViewOps.erase(userOp);
                continue;
            }

            // Only split uses that are behind at least one view-like op. Direct users are handled as a cheaper single
            // clone in splitMPEEngineMappableConsumers().
            if (!currentPath.empty() && hasMPEEngineMappableDirectUser(value.getDefiningOp(), userOp)) {
                mappableUses.push_back(MPEMappableConsumerUsePath{&use, value, currentPath});
            }
        }
    };

    collect(rootValue);
    return mappableUses;
}

mlir::Value cloneViewLikePathFromValue(mlir::PatternRewriter& rewriter, mlir::Value initialValue,
                                       ArrayRef<ViewLikeOpPathElem> viewPath) {
    auto currentValue = initialValue;
    mlir::IRMapping mapper;
    for (const auto& pathElem : viewPath) {
        VPUX_THROW_WHEN(pathElem.op == nullptr, "Cannot clone a null view-like operation");
        VPUX_THROW_UNLESS(pathElem.resultIndex < pathElem.op->getNumResults(),
                          "Invalid result index {0} for view-like operation '{1}'", pathElem.resultIndex,
                          pathElem.op->getLoc());
        // Only the single input value is remapped below, so a clone of a multi-operand op would keep the
        // remaining operands attached to the original graph.
        VPUX_THROW_UNLESS(pathElem.op->getNumOperands() == 1 && pathElem.op->getOperand(0) == pathElem.inputValue,
                          "View-like operation '{0}' must have exactly one operand matching the remapped value",
                          pathElem.op->getLoc());

        mapper.map(pathElem.inputValue, currentValue);
        rewriter.setInsertionPoint(pathElem.op);
        auto* clonedViewOp = rewriter.clone(*pathElem.op, mapper);
        extendOpLoc(clonedViewOp, "mappable_consumers");
        currentValue = clonedViewOp->getResult(pathElem.resultIndex);
    }

    return currentValue;
}

}  // namespace

bool vpux::IE::canFakeQuantizeFoldUpstream(IE::FakeQuantizeOp fqOp, bool seOpsEnabled, Logger log) {
    if (fqOp == nullptr) {
        return false;
    }
    auto quantizedElemType = getFakeQuantizeInputQuantizedType(fqOp);
    if (quantizedElemType == nullptr) {
        return false;
    }

    auto* op = fqOp.getInput().getDefiningOp();
    while (op != nullptr) {
        if (isMultiplyPartOfNormalizeL2Pattern(op)) {
            return false;
        }
        if (isFoldableQuantizedLayerOp(op)) {
            return true;
        }
        if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
            !IE::isValidToPropagateQuantize(op, seOpsEnabled, quantizedElemType, log)) {
            return false;
        }
        op = op->getOperand(0).getDefiningOp();
    }
    return false;
}

IE::MPEConsumerAnalysis vpux::IE::collectMPEEngineMappableConsumersThroughViewLikes(
        mlir::Operation* op, llvm::function_ref<bool(mlir::Operation*)> isTerminalMappable) {
    MPEConsumerAnalysis result;
    if (op == nullptr) {
        return result;
    }

    const auto directUsers = getDirectUsers(op);
    auto terminalMappableUsers = 0;

    SmallVector<mlir::Operation*> mappableUsers;
    for (auto* directUser : directUsers) {
        // Collect terminal consumers reachable from directUser by walking through element-type transparent view-like
        // ops until a non-view op is reached. If any branch terminates on an op that is neither transparent nor
        // accepted by isTerminalMappable, directUser is not a pure candidate.
        SmallVector<mlir::Operation*> worklist{directUser};
        llvm::SmallPtrSet<mlir::Operation*, 8> visited;
        bool sawComputeConsumer = false;
        while (!worklist.empty()) {
            auto* userOp = worklist.pop_back_val();
            // Skip already-traversed ops to guard against cycles and duplicate enqueues.
            if (!visited.insert(userOp).second) {
                continue;
            }
            // Terminal consumer found - record it, do not expand further.
            if (isTerminalMappable(userOp)) {
                sawComputeConsumer = true;
                terminalMappableUsers++;
                result.mappableTerminalOps.push_back(userOp);
                continue;
            }
            // Op forwards element type info from a single input to a single result (Reshape, AffineReshape,
            // Transpose, Reorder, etc.) - keep walking the use-def chain.
            if (IE::isElemTypeTransparentViewLikeOp(userOp)) {
                for (auto res : userOp->getResults()) {
                    for (auto* nextUser : res.getUsers()) {
                        worklist.push_back(nextUser);
                    }
                }
                continue;
            }
            // Any other consumer breaks the "produces only into mappable terminals" invariant.
            sawComputeConsumer = false;
            break;
        }

        // directUser is mappable only when every traversed branch ended at a mappable terminal and at least one such
        // terminal was found.
        if (sawComputeConsumer) {
            mappableUsers.push_back(directUser);
        }
    }
    result.terminalMappableUserCount = terminalMappableUsers;
    result.mappableDirectUsers = std::move(mappableUsers);
    return result;
}

bool vpux::IE::splitMPEEngineMappableConsumers(mlir::PatternRewriter& rewriter, mlir::Operation* op, bool seOpsEnabled,
                                               Logger log) {
    if (op == nullptr || op->getNumResults() == 0) {
        return false;
    }

    const auto mappableInfo = classifyMPEEngineMappableConsumers(op, seOpsEnabled, log);
    if (mappableInfo.terminalMappableUserCount > 1 && mappableInfo.eligibility == MPEConsumerEligibility::MIXED) {
        return false;
    }

    mlir::Operation* clonedOp = nullptr;
    const auto getOrCreateClonedOp = [&]() -> mlir::Operation* {
        if (clonedOp == nullptr) {
            rewriter.setInsertionPoint(op);
            clonedOp = rewriter.clone(*op);
            extendOpLoc(clonedOp, "mappable_consumers");
        }
        return clonedOp;
    };

    bool consumersSplit = false;
    // Mixed direct consumers: create one dedicated clone for all directly mappable users. The original op keeps feeding
    // non-mappable users and can later be lowered/removed by the caller.
    if (mappableInfo.eligibility == MPEConsumerEligibility::MIXED) {
        const auto isMappableDirectUser = [&](mlir::OpOperand& opOperand) {
            return llvm::is_contained(mappableInfo.mappableDirectUsers, opOperand.getOwner());
        };
        rewriter.replaceOpUsesWithIf(op, getOrCreateClonedOp()->getResults(), isMappableDirectUser);
        consumersSplit = true;
    }

    // Mixed consumers can also appear behind a shared transparent chain:
    //   op -> View -> mappable op
    //              -> non-mappable op
    // In that case clone the root op once and clone the required view-like path for each mappable terminal use. Reuse
    // an already cloned path when multiple mappable users consume the same terminal view result.
    if (!mappableInfo.suppressViewLikePathSplitting) {
        for (auto rootResult : op->getResults()) {
            const auto mappableUsePaths = collectMPEEngineMappableConsumerUsesThroughViewLikes(rootResult);
            SmallVector<std::pair<mlir::Value, mlir::Value>> clonedPathValues;
            for (const auto& mappableUsePath : mappableUsePaths) {
                auto replacementValueIt = llvm::find_if(clonedPathValues, [&](const auto& clonedPathValue) {
                    return clonedPathValue.first == mappableUsePath.sourceValue;
                });

                mlir::Value replacementValue;
                if (replacementValueIt == clonedPathValues.end()) {
                    const auto clonedRootValue = getOrCreateClonedOp()->getResult(rootResult.getResultNumber());
                    replacementValue = cloneViewLikePathFromValue(rewriter, clonedRootValue, mappableUsePath.viewPath);
                    clonedPathValues.push_back({mappableUsePath.sourceValue, replacementValue});
                } else {
                    replacementValue = replacementValueIt->second;
                }

                rewriter.modifyOpInPlace(mappableUsePath.use->getOwner(), [&]() {
                    mappableUsePath.use->set(replacementValue);
                });
                consumersSplit = true;
            }
        }
    }

    if (clonedOp == nullptr) {
        return false;
    }

    const auto clonedOpHasUses = llvm::any_of(clonedOp->getResults(), [](mlir::Value result) {
        return !result.use_empty();
    });
    if (!clonedOpHasUses) {
        rewriter.eraseOp(clonedOp);
        return false;
    }

    return consumersSplit;
}

IE::MPEConsumerAnalysis vpux::IE::classifyMPEEngineMappableConsumers(mlir::Operation* op, bool seOpsEnabled,
                                                                     Logger log) {
    MPEConsumerAnalysis info;
    if (op == nullptr) {
        return info;
    }
    const auto directUsers = getDirectUsers(op);
    if (directUsers.empty()) {
        return info;
    }

    const auto strategyResult = getMPEEngineMappableConsumers(op);
    info.terminalMappableUserCount = strategyResult.terminalMappableUserCount;

    llvm::SmallPtrSet<mlir::Operation*, 8> mappableSet(strategyResult.mappableDirectUsers.begin(),
                                                       strategyResult.mappableDirectUsers.end());
    for (auto* directUser : directUsers) {
        if (mappableSet.contains(directUser)) {
            info.mappableDirectUsers.push_back(directUser);
        }
    }

    if (info.mappableDirectUsers.empty()) {
        info.eligibility = MPEConsumerEligibility::NONE;
    } else if (info.mappableDirectUsers.size() == directUsers.size()) {
        info.eligibility = MPEConsumerEligibility::ALL;
    } else {
        info.eligibility = MPEConsumerEligibility::MIXED;
    }

    if (info.eligibility == MPEConsumerEligibility::MIXED &&
        llvm::any_of(strategyResult.mappableTerminalOps, isSignedWeightsAsInputConsumer)) {
        info.eligibility = MPEConsumerEligibility::NONE;
        info.mappableDirectUsers.clear();
        info.suppressViewLikePathSplitting = true;
    }

    if (strategyResult.terminalMappableUserCount > 1) {
        const auto fqOp = mlir::dyn_cast<IE::FakeQuantizeOp>(op);
        if (!canFakeQuantizeFoldUpstream(fqOp, seOpsEnabled, log)) {
            info.eligibility = MPEConsumerEligibility::NONE;
            info.mappableDirectUsers.clear();
        }
    }

    return info;
}
