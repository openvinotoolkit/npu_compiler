//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

using namespace vpux;

namespace {
size_t combineHash(size_t h1, size_t h2) {
    return h1 ^ (h2 << 1);
}

template <typename... Args>
size_t combineHash(size_t h1, Args... args) {
    return combineHash(h1, combineHash(args...));
}

size_t hashType(NDTypeInterface type) {
    const auto asStr = llvm::formatv("{0}", type).str();
    return std::hash<std::string>()(asStr);
}

struct FuncSignatureHashes {
    SmallVector<size_t> inputHashes;
    size_t outputHash;
};

// Generates hashes for model inputs/outputs
// Hash is computes as combination of stringified type, io name and position
FuncSignatureHashes buildFuncSignatureHashes(mlir::func::FuncOp funcOp) {
    auto functionType = funcOp.getFunctionType();

    const size_t numInputs = functionType.getNumInputs();
    SmallVector<size_t> inputHashes(numInputs, 0);
    for (size_t i = 0; i < numInputs; ++i) {
        inputHashes[i] = hashType(functionType.getInput(i)) ^ i;  // ^ i adds positional change
    }

    size_t outHash = 0;
    for (size_t i = 0; i < functionType.getNumResults(); ++i) {
        const size_t resHash = hashType(functionType.getResult(i));
        outHash = combineHash(outHash, resHash);
    }

    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    auto netOps = to_small_vector(moduleOp.getOps<IE::CNNNetworkOp>());

    if (netOps.size() == 1) {
        auto cnnOp = netOps.front();
        auto inputsInfo = to_small_vector(cnnOp.getInputsInfo().getOps<IE::DataInfoOp>());
        for (size_t i = 0; i < numInputs; ++i) {
            const std::string name = inputsInfo[i].getNameAttrName().str();
            const size_t inputNameHash = std::hash<std::string>()(name);
            inputHashes[i] = combineHash(inputHashes[i], inputNameHash);
        }

        auto outputsInfo = to_small_vector(cnnOp.getOutputsInfo().getOps<IE::DataInfoOp>());
        for (size_t i = 0; i < functionType.getNumResults(); ++i) {
            const std::string name = outputsInfo[i].getNameAttrName().str();
            const size_t outputNameHash = std::hash<std::string>()(name);
            outHash = combineHash(outHash, outputNameHash);
        }
    }
    return {std::move(inputHashes), outHash};
}

// Within common context StringAttr describing same strings will have same pImpl and same hash, so no need to convert to
// string and hash again
struct HashLUT {
    size_t getHash(mlir::StringAttr attr) {
        if (lut.find(attr) != lut.end()) {
            return lut[attr];
        }
        const size_t hash = std::hash<std::string>()(attr.str());
        lut[attr] = hash;
        return hash;
    }
    mlir::DenseMap<mlir::StringAttr, size_t> lut;
};

// Exclude attributes from serialization. We need to layer hash to serialize MC strategy, so hash must be independ from
// MCS
mlir::DictionaryAttr getRequiredAttrDict(mlir::DictionaryAttr dictAttr) {
    SmallVector<mlir::NamedAttribute> attrs;
    const SmallVector<StringLiteral> ignoredAttr = {vpux::multiClusterStrategy, vpux::tilingStrategy};
    for (auto attr : dictAttr) {
        if (llvm::find(ignoredAttr, attr.getName().getValue()) == ignoredAttr.end()) {
            attrs.push_back(attr);
        }
    }
    return mlir::DictionaryAttr::get(dictAttr.getContext(), attrs);
}

// Implements first stage of hashing. At this stage layer hash depends on combination of
// - origin layer
// - origin layer type
// - hash of op output types
// - attributes of op
// This hash is depend only on operation itself and independ from topology
struct FirstStageHasher {
    HashLUT namesLut;
    HashLUT typesLut;
    mlir::DenseMap<mlir::Operation*, size_t> strategyOpsHashes;
    mlir::DenseMap<mlir::Operation*, size_t> otherOpsHashes;

    std::unordered_set<size_t> collisionDetector;

    void handleReturnOp(mlir::Operation* op, size_t opHash) {
        otherOpsHashes[op] = opHash;
    }

    size_t getOpHash(mlir::Operation* op, bool strategyOp = false) {
        if (strategyOpsHashes.count(op) > 0) {
            return strategyOpsHashes[op];
        }

        if (otherOpsHashes.count(op) > 0) {
            return otherOpsHashes[op];
        }

        size_t locationBasedHash = 0;
        if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc())) {
            auto metadata = fusedLoc.getMetadata();
            if (metadata != nullptr) {
                auto metaDict = mlir::dyn_cast<mlir::DictionaryAttr>(fusedLoc.getMetadata());
                VPUX_THROW_WHEN(metaDict == nullptr || metaDict.empty(), "Empty metadata");

                auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(metaDict.get("name"));
                auto typeAttr = mlir::dyn_cast<mlir::StringAttr>(metaDict.get("type"));
                locationBasedHash = combineHash(namesLut.getHash(nameAttr), typesLut.getHash(typeAttr));
            }
        } else {
            VPUX_THROW_WHEN(strategyOp, "Can't get layer metadata for '{0}'", op->getLoc());
        }

        size_t operationTypeHash = std::hash<std::string>()(op->getName().getStringRef().str());
        size_t opHash = combineHash(locationBasedHash, operationTypeHash);
        if (!mlir::isa<Const::DeclareOp>(op)) {
            auto filteredAttrs = getRequiredAttrDict(op->getAttrDictionary());
            const auto asStr = llvm::formatv("{0}", filteredAttrs).str();
            size_t attrsHash = std::hash<std::string>()(asStr);
            opHash = combineHash(opHash, attrsHash);
        }
        for (auto resultType : op->getResultTypes()) {
            opHash = combineHash(opHash, hashType(resultType));
        }

        if (strategyOp) {
            strategyOpsHashes[op] = opHash;
        } else {
            otherOpsHashes[op] = opHash;
        }

        return opHash;
    };
};

// Second stage of hashing. During this stage extend hash with hashes of neighbors. It helps to get unique hashes for
// each operation based on topology If collision was detected, we perform new round again using hash from previous
// round.Each round we increase perceptivity for each operation until we don't detect any duplicate
HashStageResult hashRound(const FuncSignatureHashes& funcSignature, FirstStageHasher& fsHasher,
                          mlir::DenseMap<mlir::Operation*, size_t> previousIteration) {
    size_t hits = 0;
    mlir::DenseMap<mlir::Operation*, size_t> newOpsToHash;
    const auto hashWithFallBack = [&](mlir::Operation* op) {
        if (previousIteration.count(op) > 0) {
            ++hits;
            return previousIteration[op];
        }
        size_t initialHash = fsHasher.getOpHash(op, false);
        newOpsToHash[op] = initialHash;
        return initialHash;
    };

    mlir::DenseMap<mlir::Operation*, size_t> localizedHashes;
    mlir::DenseMap<size_t, mlir::Operation*> collisionDetector;
    bool hasCollision = false;

    const auto rehashGroup = [&](const mlir::DenseMap<mlir::Operation*, size_t>& hashGroup) {
        for (const auto& item : hashGroup) {
            mlir::Operation* op = item.first;
            size_t opHash = item.second;

            for (auto operand : op->getOperands()) {
                size_t operandHash = 0;
                if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                    operandHash = funcSignature.inputHashes[blockArg.getArgNumber()];
                } else {
                    auto defOp = operand.getDefiningOp();
                    VPUX_THROW_WHEN(defOp == nullptr, "Invalid operand in second stage");
                    operandHash = hashWithFallBack(defOp);
                }
                opHash = combineHash(opHash, operandHash);
            }

            for (auto result : op->getResults()) {
                for (auto user : result.getUsers()) {
                    size_t resultHash = hashWithFallBack(user);
                    opHash = combineHash(opHash, resultHash);
                }
            }
            if (fsHasher.strategyOpsHashes.count(op) > 0 && collisionDetector.count(opHash) > 0) {
                hasCollision = true;
            }

            collisionDetector[opHash] = op;
            localizedHashes[op] = opHash;
        }
    };
    auto opsToHash = previousIteration.empty() ? fsHasher.strategyOpsHashes : previousIteration;
    rehashGroup(opsToHash);
    // create a copy of newOpsToHash to avoid in-flight modifications
    const auto newDiscoveredOperations = newOpsToHash;
    rehashGroup(newDiscoveredOperations);

    return {!hasCollision, std::move(localizedHashes), std::move(collisionDetector)};
}

};  // namespace

HashStageResult vpux::hashFunctionLayers(mlir::func::FuncOp funcOp) {
    const auto funcSignature = buildFuncSignatureHashes(funcOp);

    FirstStageHasher fsHasher;
    funcOp->walk([&](VPU::LayerOpInterface op) {
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            fsHasher.handleReturnOp(op, funcSignature.outputHash);
        }

        auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op.getOperation());
        auto isSWOp = mlir::isa<VPU::SWOpInterface>(op.getOperation());
        // Avoid cluttering dump with irrelevant layers
        if (!isNCEOp && !isSWOp) {
            return;
        }
        // hash operations which could have strategy assigned
        fsHasher.getOpHash(op, true);
    });

    mlir::DenseMap<mlir::Operation*, size_t> previousIteration;
    HashStageResult hashingResult;
    const size_t NUM_ROUNDS = 10;
    for (size_t i = 0; i < NUM_ROUNDS; ++i) {
        hashingResult = hashRound(funcSignature, fsHasher, previousIteration);
        if (hashingResult.succeed) {
            break;
        }
        previousIteration = hashingResult.localizedHashes;
    }

    return hashingResult;
}

bool vpux::loadPreConfiguredStrategy(vpux::Logger log, mlir::func::FuncOp func, StringRef modelHash) {
    auto maybeStrategy = vpux::maybeGetStrategyFor(modelHash);
    if (!maybeStrategy.has_value()) {
        log.trace("Cannot find pre-configured model settings");
        return false;
    }
    HashStageResult layerHashes;
    // Fail on error for dev builds, otherwise fallback to strategy assignment
#ifdef VPUX_DEVELOPER_BUILD
    layerHashes = hashFunctionLayers(func);
#else
    try {
        layerHashes = hashFunctionLayers(func);
    } catch (...) {
        log.trace("Got invalid IR during hashing");
        return false;
    }
#endif
    if (!layerHashes.succeed) {
        log.trace("Cannot get unique hash values for ops");
        return false;
    }

    const auto& opToHash = layerHashes.localizedHashes;
    const auto strategies = maybeStrategy.value();

    bool applied = true;
    func->walk([&](VPU::LayerOpInterface op) {
        auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op.getOperation());
        auto isSWOp = mlir::isa<VPU::SWOpInterface>(op.getOperation());
        // Avoid cluttering dump with irrelevant layers
        if (!isNCEOp && !isSWOp) {
            return mlir::WalkResult::advance();
        }
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation());
        if (clusteredOp == nullptr) {
            return mlir::WalkResult::advance();
        }
        if (opToHash.find(op.getOperation()) == opToHash.end()) {
            log.trace("Cannot find hash for layer '{0}'", op->getLoc());
            applied = false;
            return mlir::WalkResult::interrupt();
        }

        const auto opHash = opToHash.at(op.getOperation());
        if (strategies.find(opHash) == strategies.end()) {
            applied = false;
            log.trace("Cannot find strategy for layer '{0}'. Looking for hash: {1}", op->getLoc(), opHash);
            return mlir::WalkResult::interrupt();
        }
        const auto strat = strategies.at(opHash);
        if (strat.has_value()) {
            clusteredOp.setMultiClusterStrategy(strat.value());
        }
        return mlir::WalkResult::advance();
    });
    return applied;
}
