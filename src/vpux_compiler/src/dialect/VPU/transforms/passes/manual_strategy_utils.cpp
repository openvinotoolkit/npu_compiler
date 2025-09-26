//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mutex>
#include <shared_mutex>
#include "llvm/Support/MD5.h"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
#include "vpux/compiler/core/developer_build_utils.hpp"
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include <mutex>

namespace vpux::VPU {
#define GEN_PASS_DECL_MANUALSTRATEGYUTILS
#define GEN_PASS_DEF_MANUALSTRATEGYUTILS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

// mutex to protect file r/w for multi-threading
static std::shared_mutex writeStrategyFileMutex;
static std::shared_mutex dumpStrategyFileMutex;

// Function to get the model ID based on the NetworkInfoOp
std::string getModelID(mlir::ModuleOp moduleOp) {
    static std::mutex modelIdMutex;
    std::lock_guard<std::mutex> guard(modelIdMutex);

    if (!moduleOp) {
        return "unknown_model";
    }

    net::NetworkInfoOp networkInfo = nullptr;
    moduleOp->walk([&](net::NetworkInfoOp op) {
        networkInfo = op;
        return mlir::WalkResult::interrupt();
    });

    if (!networkInfo) {
        return "unknown_model";
    }

    std::string serializedOp;
    llvm::raw_string_ostream serializedOpStream(serializedOp);
    networkInfo.print(serializedOpStream);

    llvm::MD5 md5;
    md5.update(serializedOp);
    llvm::MD5::MD5Result result;
    md5.final(result);

    llvm::SmallString<32> hashStr;
    llvm::MD5::stringifyResult(result, hashStr);
    return hashStr.str().str();
}

// Count how many func.func ops are in the module
int countFunctions(mlir::ModuleOp moduleOp) {
    VPUX_THROW_UNLESS(moduleOp != nullptr, "ModuleOp is null");
    auto funcOps = moduleOp.getOps<mlir::func::FuncOp>();
    return std::distance(funcOps.begin(), funcOps.end());
}

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
              _dumpStrategyToLog(false) {
    }
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, Logger log);
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, bool updateStrategyForOutputPipelining, Logger log);
    ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
                            StringRef readStrategyFileLocation, bool updateStrategyForOutputPipelining,
                            bool dumpStrategyToLog, std::string contextId, Logger log);

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnFunc() final;

private:
    bool _writeStrategyToJSON;
    std::string _writeStrategyFileLocation;
    bool _readStrategyFromJSON;
    std::string _readStrategyFileLocation;
    bool _updateStrategyForOutputPipelining;
    bool _dumpStrategyToLog;
    // count how many func.func ops processed now under current context (special pass position)
    // format: {"_contextId": <counter>}
    static std::unordered_map<std::string, std::atomic<int>> _sharedFuncCounter;
    // context id to identify the pass position
    std::string _contextId = "default";
};

std::unordered_map<std::string, std::atomic<int>> ManualStrategyUtilsPass::_sharedFuncCounter;

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 Logger log)
        // NOTE: currently called after two/three strategy passes, flags in all must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation.str()),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation.str()),
          _updateStrategyForOutputPipelining(false),
          _dumpStrategyToLog(false) {
    Base::initLogger(log, Base::getArgumentName());
}

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 bool updateStrategyForOutputPipelining, Logger log)
        // NOTE: currently called after two/three strategy passes, flags in all must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation.str()),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation.str()),
          _updateStrategyForOutputPipelining(updateStrategyForOutputPipelining),
          _dumpStrategyToLog(false) {
    Base::initLogger(log, Base::getArgumentName());
}

ManualStrategyUtilsPass::ManualStrategyUtilsPass(bool writeStrategyToJSON, StringRef writeStrategyFileLocation,
                                                 bool readStrategyFromJSON, StringRef readStrategyFileLocation,
                                                 bool updateStrategyForOutputPipelining, bool dumpStrategyToLog,
                                                 std::string contextId, Logger log)
        // NOTE: currently called after two/three strategy passes, flags in all must match.
        : _writeStrategyToJSON(writeStrategyToJSON),
          _writeStrategyFileLocation(writeStrategyFileLocation.str()),
          _readStrategyFromJSON(readStrategyFromJSON),
          _readStrategyFileLocation(readStrategyFileLocation.str()),
          _updateStrategyForOutputPipelining(updateStrategyForOutputPipelining),
          _dumpStrategyToLog(dumpStrategyToLog),
          _contextId(std::move(contextId)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult ManualStrategyUtilsPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
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
    auto module = vpux::getTopModuleOp(func);
    const int funcCount = countFunctions(module);

    if (!_writeStrategyToJSON && !_readStrategyFromJSON && !_dumpStrategyToLog) {
        _log.trace("Flags to write and dump and read disabled, skipping pass");
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

    // 1) We need it to store VF tiling strategies to avoid losing info after unrolling
    // TODO: it can be removed after output pipeline tiling pass are moved before VF, see E#163863
    // 2) Append model ID (a hash value) to avoid race writing when multiple models are compiled in parallel such
    // as in CI
    // 3) static declare for only once initialization when multithreading running
    static const std::string dumpStrategyFileLocation =
            _dumpStrategyToLog ? ("strategy_dump_" + getModelID(module) + ".json") : "";

    _log.trace("Starting Manual Strategy Pass");
    _log.nest(1).trace("Option to write strategy: '{0}'", _writeStrategyToJSON);
    _log.nest(1).trace("Strategy write file location: '{0}'", _writeStrategyFileLocation);
    _log.nest(1).trace("Option to read strategy: '{0}'", _readStrategyFromJSON);
    _log.nest(1).trace("Strategy read file location: '{0}'", _readStrategyFileLocation);
    _log.nest(1).trace("Option to dump strategy: '{0}'", _dumpStrategyToLog);
    _log.nest(1).trace("Strategy dump file loction: '{0}'", dumpStrategyFileLocation);

    // store operations with Location as key to enable Location based mapping
    llvm::MapVector<mlir::Location, mlir::Operation*> operations;
    llvm::MapVector<mlir::Location, mlir::Operation*> outputPipeliningOps;

    func->walk([&](VPU::LayerOpInterface op) {
        auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op.getOperation());
        auto isSWOp = mlir::isa<VPU::SWOpInterface>(op.getOperation());
        // Avoid cluttering dump with irrelevant layers
        if (!isNCEOp && !isSWOp) {
            return;
        }
        // store unique operations (tiled operations are merged)
        mlir::Location opLoc = op.getLoc();
        if (operations.find(opLoc) != operations.end()) {
            // if duplicate locations, create unique
            opLoc = appendLoc(opLoc, "unique_{0}", operations.count(opLoc));
            op->setLoc(opLoc);
        }
        operations.insert({opLoc, op.getOperation()});

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

    llvm::json::Value json(nullptr);
    if (_writeStrategyToJSON || _dumpStrategyToLog) {
        if (_updateStrategyForOutputPipelining) {
            _log.nest(1).trace("Update strategy for output pipelining in JSON");
            if (_dumpStrategyToLog) {
                std::shared_lock<std::shared_mutex> readLock(dumpStrategyFileMutex);
                auto expectedJson = readManualStrategyJSON(dumpStrategyFileLocation);
                if (expectedJson) {
                    json = expectedJson.get();
                }
            } else {
                std::shared_lock<std::shared_mutex> readLock(writeStrategyFileMutex);
                auto expectedJson = readManualStrategyJSON(_writeStrategyFileLocation);
                if (expectedJson) {
                    json = expectedJson.get();
                }
            }

            if (json != nullptr) {
                updateTilingStrategyInJSONForOperations(json, outputPipeliningOps);
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

            // writing current strategy to json
            createStrategyJSONFromOperations(json, operations, attributes);
        }
    }

    if (_dumpStrategyToLog) {
        // unique lock to assure only one thread to write and no thread to read
        std::unique_lock<std::shared_mutex> writeLock(dumpStrategyFileMutex);
        // We need it to store VF tiling strategies to avoid losing info after unrolling
        // TODO: it can be removed after output pipeline tiling pass are moved before VF, see E#163863
        writeManualStrategyJSON(dumpStrategyFileLocation, json);
        if (_sharedFuncCounter.find(_contextId) == _sharedFuncCounter.end()) {
            _sharedFuncCounter[_contextId] = 0;
        }
        ++_sharedFuncCounter[_contextId];
        // Print final strategies to log
        if (_updateStrategyForOutputPipelining && (_sharedFuncCounter[_contextId] == funcCount)) {
            _log.info("Final strategies for output pipelining:");
            _log.info("{0}", llvm::formatv("\n{0:2}", json).str());
        }
    }

    if (_writeStrategyToJSON) {
        // unique lock to assure only one thread to write and no thread to read
        std::unique_lock<std::shared_mutex> writeLock(writeStrategyFileMutex);
        writeManualStrategyJSON(_writeStrategyFileLocation, json);
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

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                               StringRef writeStrategyFileLocation,
                                                               bool readStrategyFromJSON,
                                                               StringRef readStrategyFileLocation, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, writeStrategyFileLocation,
                                                     readStrategyFromJSON, readStrategyFileLocation, log);
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                               StringRef writeStrategyFileLocation,
                                                               bool readStrategyFromJSON,
                                                               StringRef readStrategyFileLocation,
                                                               bool updateStrategyForOutputPipelining, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(writeStrategyToJSON, writeStrategyFileLocation,
                                                     readStrategyFromJSON, readStrategyFileLocation,
                                                     updateStrategyForOutputPipelining, log);
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(bool writeStrategyToJSON,
                                                               StringRef writeStrategyFileLocation,
                                                               bool readStrategyFromJSON,
                                                               StringRef readStrategyFileLocation,
                                                               bool dumpStrategyToLog,
                                                               bool updateStrategyForOutputPipelining, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(
            writeStrategyToJSON, writeStrategyFileLocation, readStrategyFromJSON, readStrategyFileLocation,
            updateStrategyForOutputPipelining, dumpStrategyToLog, "default", log);
}

std::unique_ptr<mlir::Pass> VPU::createManualStrategyUtilsPass(
        bool writeStrategyToJSON, StringRef writeStrategyFileLocation, bool readStrategyFromJSON,
        StringRef readStrategyFileLocation, bool dumpStrategyToLog, bool updateStrategyForOutputPipelining,
        std::string contextId, Logger log) {
    return std::make_unique<ManualStrategyUtilsPass>(
            writeStrategyToJSON, writeStrategyFileLocation, readStrategyFromJSON, readStrategyFileLocation,
            updateStrategyForOutputPipelining, dumpStrategyToLog, contextId, log);
}
