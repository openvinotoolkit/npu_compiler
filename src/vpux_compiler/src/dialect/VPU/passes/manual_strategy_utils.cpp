//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPU/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Block.h>

#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// ManualStrategyUtilsPass
//

class ManualStrategyUtilsPass final : public ManualStrategyUtilsBase<ManualStrategyUtilsPass> {
public:
    ManualStrategyUtilsPass() = default;
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, Logger log);

private:
    void safeRunOnFunc() final;

private:
    bool _writeStrategyToJSON;
    StringRef _writeStrategyFileLocation;
    bool _readStrategyFromJSON;
    StringRef _readStrategyFileLocation;
};

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 Logger log)
        // NOTE: currently called after two strategy passes, flags in both must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation) {
    Base::initLogger(log, Base::getArgumentName());
}

//
// safeRunOnFunc
//

void ManualStrategyUtilsPass::safeRunOnFunc() {
    auto func = getFunction();

    if (!_writeStrategyToJSON && !_readStrategyFromJSON) {
        _log.trace("Flags to write and read disabled, skipping pass");
        return;
    }

    if (_readStrategyFromJSON && _readStrategyFileLocation.empty()) {
        _log.trace("Invalid read location for manual strategy, skipping pass");
        return;
    }

    if (_writeStrategyToJSON && _writeStrategyFileLocation.empty()) {
        _log.trace("Invalid write location for manual strategy, skipping pass");
        return;
    }

    _log.trace("Starting Manual Strategy Pass");
    _log.nest(1).trace("Option to write strategy: '{0}'", _writeStrategyToJSON);
    _log.nest(1).trace("Strategy write file location: '{0}'", _writeStrategyFileLocation);
    _log.nest(1).trace("Option to read strategy: '{0}'", _readStrategyFromJSON);
    _log.nest(1).trace("Strategy read file location: '{0}'", _readStrategyFileLocation);

    // store operations with Location as key to enable Location based mapping
    llvm::DenseMap<mlir::Location, mlir::Operation*> operations;

    bool operationsWrappedInClusterTiling = false;
    bool operationsHaveTilingAttr = false;

    func->walk([&](VPU::NCEOpInterface op) {
        // store unique operations (tiled operations are merged)
        mlir::Location opLoc = nullptr;
        if (const auto fused = op.getLoc().dyn_cast<mlir::FusedLoc>()) {
            // fused are unique
            opLoc = fused.getLocations().front();
        } else {
            opLoc = op.getLoc();
            if (operations.find(opLoc) != operations.end()) {
                // if duplicate locations, create unique
                opLoc = appendLoc(opLoc, llvm::formatv("unique_{0}", operations.count(opLoc)).str());
                // op.getOperation()->setLoc(opLoc);
            }
        }
        operations.insert({opLoc, op.getOperation()});
        if (op->hasAttr("manualTilingStrategy")) {
            op->setAttr("tilingStrategy", op->getAttr("manualTilingStrategy"));
            op->removeAttr("manualTilingStrategy");
        }
        if (!operationsWrappedInClusterTiling && op->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
            _log.nest(2).trace("Operations wrapped in cluster tiling exist");
            operationsWrappedInClusterTiling = true;
        }
        if (!operationsHaveTilingAttr && op->hasAttr("tilingStrategy")) {
            _log.nest(2).trace("Tiled operations exist");
            operationsHaveTilingAttr = true;
        }
    });

    if (_writeStrategyToJSON) {
        _log.nest(1).trace("Writing strategy to JSON");
        // pass attributes name for creating JSON - filter
        // currently supported attributes
        //  - multiClusterStrategy
        //  - tilingStrategy
        SmallVector<StringRef> strategyAttributes = {"multiClusterStrategy", "tilingStrategy"};

        Json json;
        if (operationsWrappedInClusterTiling) {
            // read stategies from first strategy pass and append new strategies
            _log.nest(2).trace("Appending to strategies from first strategy pass");
            json = readManualStrategyJSON(_writeStrategyFileLocation);
        }
        // writing current strategy to json
        json = createStrategyJSONFromOperations(json, operations, strategyAttributes);
        writeManualStrategyJSON(_writeStrategyFileLocation, json);
    }

    if (_readStrategyFromJSON) {
        _log.nest(1).trace("Reading strategy from JSON");
        if (!operationsWrappedInClusterTiling && !operationsHaveTilingAttr) {
            // reading strategy from json only during first pass call
            auto manualStrategy = readManualStrategyJSON(_readStrategyFileLocation);

            // overwriting operation attributes
            if (!manualStrategy.is_null()) {
                Logger::global().warning("WARNING: Experimental mode - assigning manual strategies");
                overwriteManualStrategy(manualStrategy, operations);
            }
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

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                               StringRef writeStrategyFileLocation,
                                                               bool readStrategyFromJSON,
                                                               StringRef readStrategyFileLocation, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, writeStrategyFileLocation,
                                                     readStrategyFromJSON, readStrategyFileLocation, log);
}
