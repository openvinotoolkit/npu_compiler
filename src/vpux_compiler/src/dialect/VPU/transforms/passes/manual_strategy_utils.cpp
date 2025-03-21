//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/string_ref.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include "vpux/compiler/core/developer_build_utils.hpp"

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

namespace vpux::VPU {
#define GEN_PASS_DECL_MANUALSTRATEGYUTILS
#define GEN_PASS_DEF_MANUALSTRATEGYUTILS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// ManualStrategyUtilsPass
//

class ManualStrategyUtilsPass final : public VPU::impl::ManualStrategyUtilsBase<ManualStrategyUtilsPass> {
public:
    ManualStrategyUtilsPass()
            : _writeStrategyToJSON(false),
              _writeStrategyFileLocation(),
              _readStrategyFromJSON(false),
              _readStrategyFileLocation(),
              _updateStrategyForOutputPipelining(false),
              _enableSideLoadDump(false),
              _modelHash(){};
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, bool enableSideLoadDump, StringRef modelHash,
                            Logger log);
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, bool updateStrategyForOutputPipelining,
                            bool enableSideLoadDump, StringRef modelHash, Logger log);

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnFunc() final;

private:
    bool _writeStrategyToJSON;
    std::string _writeStrategyFileLocation;
    bool _readStrategyFromJSON;
    std::string _readStrategyFileLocation;
    bool _updateStrategyForOutputPipelining;
    bool _enableSideLoadDump;
    std::string _modelHash;
};

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 bool enableSideLoadDump, StringRef modelHash, Logger log)
        // NOTE: currently called after two/three strategy passes, flags in all must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation.str()),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation.str()),
          _updateStrategyForOutputPipelining(false),
          _enableSideLoadDump(enableSideLoadDump),
          _modelHash(modelHash) {
    Base::initLogger(log, Base::getArgumentName());
}

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 bool updateStrategyForOutputPipelining, bool enableSideLoadDump,
                                                 StringRef modelHash, Logger log)
        // NOTE: currently called after two/three strategy passes, flags in all must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation.str()),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation.str()),
          _updateStrategyForOutputPipelining(updateStrategyForOutputPipelining),
          _enableSideLoadDump(enableSideLoadDump),
          _modelHash(modelHash) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult ManualStrategyUtilsPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }

    if (writeStrategyToJSON.hasValue()) {
        _writeStrategyToJSON = writeStrategyToJSON.getValue();
    }

    if (writeStrategyFileLocation.hasValue()) {
        _writeStrategyFileLocation = writeStrategyFileLocation.getValue();
    }

    if (readStrategyFromJSON.hasValue()) {
        _readStrategyFromJSON = readStrategyFromJSON.getValue();
    }

    if (readStrategyFileLocation.hasValue()) {
        _readStrategyFileLocation = readStrategyFileLocation.getValue();
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void ManualStrategyUtilsPass::safeRunOnFunc() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_NPU_WRITE_STRATEGY_JSON", _writeStrategyToJSON);
    parseEnv("IE_NPU_WRITE_STRATEGY_JSON_LOC", _writeStrategyFileLocation);
    parseEnv("IE_NPU_READ_STRATEGY_JSON", _readStrategyFromJSON);
    parseEnv("IE_NPU_READ_STRATEGY_JSON_LOC", _readStrategyFileLocation);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    auto func = getOperation();
    if (_enableSideLoadDump) {
        const auto layerHashes = hashFunctionLayers(func);
        VPUX_THROW_UNLESS(layerHashes.succeed, "Can't generate unique hashes for side-load dump");
        saveMCSideLoadStrategyToFile(func, "mc_side_load_dump.json", layerHashes.localizedHashes, _modelHash);
    }

    if (!_writeStrategyToJSON && !_readStrategyFromJSON) {
        _log.trace("Flags to write and read disabled, skipping pass");
        return;
    }

    if (_readStrategyFromJSON && _readStrategyFileLocation.empty()) {
        _log.error("Invalid read location for manual strategy");
        signalPassFailure();
        return;
    }

    if (_writeStrategyToJSON && _writeStrategyFileLocation.empty()) {
        _log.error("Invalid write location for manual strategy");
        signalPassFailure();
        return;
    }

    _log.trace("Starting Manual Strategy Pass");
    _log.nest(1).trace("Option to write strategy: '{0}'", _writeStrategyToJSON);
    _log.nest(1).trace("Strategy write file location: '{0}'", _writeStrategyFileLocation);
    _log.nest(1).trace("Option to read strategy: '{0}'", _readStrategyFromJSON);
    _log.nest(1).trace("Strategy read file location: '{0}'", _readStrategyFileLocation);

    // store operations with Location as key to enable Location based mapping
    llvm::MapVector<mlir::Location, mlir::Operation*> operations;
    llvm::MapVector<mlir::Location, mlir::Operation*> outputPipeliningOps;

    bool operationsWrappedInClusterTiling = false;
    bool operationsHaveTilingAttr = false;

    func->walk([&](VPU::LayerOpInterface op) {
        auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op.getOperation());
        auto isSWOp = mlir::isa<VPU::SWOpInterface>(op.getOperation());
        // Avoid cluttering dump with irrelevant layers
        if (!isNCEOp && !isSWOp) {
            return;
        }
        // store unique operations (tiled operations are merged)
        mlir::Location opLoc = op.getLoc();
        if (op->hasAttr(tilingStrategy)) {
            const auto fused = opLoc.dyn_cast<mlir::FusedLoc>();
            VPUX_THROW_UNLESS(fused, "Tiled operation has location not fused");
            // store only 1 tile
            opLoc = fused.getLocations().front();
        } else {
            if (operations.find(opLoc) != operations.end()) {
                // if duplicate locations, create unique
                opLoc = appendLoc(opLoc, "unique_{0}", operations.count(opLoc));
                op->setLoc(opLoc);
            }
        }
        operations.insert({opLoc, op.getOperation()});
        if (!operationsWrappedInClusterTiling && op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
            _log.nest(2).trace("Operations wrapped in cluster tiling exist");
            operationsWrappedInClusterTiling = true;
        }
        if (!operationsHaveTilingAttr && op->hasAttr(tilingStrategy)) {
            _log.nest(2).trace("Tiled operations exist");
            operationsHaveTilingAttr = true;
        }

        // collect all operations that have tilingStrategy attribute after the number of tiles increases for output
        // pipelining
        // at this stage, VF tiling has been applied and VF ops do not have tilingStrategy, so VF ops are not collected
        // Layers' strategies will be saved into JSON file when _writeStrategyToJSON is 'true'
        // Layers' strategies will be overwritten with manual attributes from JSON file when _readStrategyFromJSON is
        // 'true'
        if (_updateStrategyForOutputPipelining && op->hasAttr(tilingStrategy)) {
            outputPipeliningOps.insert({opLoc, op.getOperation()});
        }
    });

    if (_writeStrategyToJSON) {
        llvm::json::Value json(nullptr);
        if (_updateStrategyForOutputPipelining) {
            _log.nest(1).trace("Update strategy for output pipelining in JSON");
            auto expectedJson = readManualStrategyJSON(_writeStrategyFileLocation);
            if (expectedJson) {
                json = expectedJson.get();
                updateTilingStrategyInJSONForOperations(json, outputPipeliningOps);
                writeManualStrategyJSON(_writeStrategyFileLocation, json);
            }
        } else {
            _log.nest(1).trace("Writing strategy to JSON");
            // pass attributes name and default value for creating JSON - filter
            // currently supported attributes
            //  - multiClusterStrategy
            //  - tilingStrategy
            //  - verticalFusion
            //  - verticalFusionHash
            //  - layerType
            DenseMap<StringRef, StringRef> attributes = {{multiClusterStrategy, defaultNoValue},
                                                         {tilingStrategy, defaultNoValue},
                                                         {verticalFusion, "False"},
                                                         {verticalFusionHash, defaultNoValue},
                                                         {layerTypeName, defaultNoValue}};

            if (operationsWrappedInClusterTiling && operationsHaveTilingAttr) {
                // read strategies from first strategy pass and append new strategies
                _log.nest(2).trace("Appending to strategies from first strategy pass");
                auto expectedJson = readManualStrategyJSON(_writeStrategyFileLocation);
                if (expectedJson) {
                    json = expectedJson.get();
                }
            }

            // writing current strategy to json from first pass
            createStrategyJSONFromOperations(json, operations, attributes);
            writeManualStrategyJSON(_writeStrategyFileLocation, json);
        }
    }

    if (_readStrategyFromJSON) {
        auto expectedManualStrategy = readManualStrategyJSON(_readStrategyFileLocation);
        // overwriting operation attributes
        if (expectedManualStrategy) {
            llvm::json::Value manualStrategy = expectedManualStrategy.get();
            Logger::global().warning("WARNING: Experimental mode - assigning manual strategies");
            overwriteManualStrategy(manualStrategy,
                                    _updateStrategyForOutputPipelining ? outputPipeliningOps : operations);
        }
    }
}

}  // namespace

//
// createManualStrategyUtilsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createManualStrategyUtilsPass() {
    return std::make_unique<ManualStrategyUtilsPass>();
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(
        bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
        StringRef readStrategyFileLocation, bool enableSideLoadDump, StringRef modelHash, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, writeStrategyFileLocation,
                                                     readStrategyFromJSON, readStrategyFileLocation, enableSideLoadDump,
                                                     modelHash, log);
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(
        bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
        StringRef readStrategyFileLocation, bool updateStrategyForOutputPipelining, bool enableSideLoadDump,
        StringRef modelHash, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(
            writeStrategyToJSON, writeStrategyFileLocation, readStrategyFromJSON, readStrategyFileLocation,
            updateStrategyForOutputPipelining, enableSideLoadDump, modelHash, log);
}
