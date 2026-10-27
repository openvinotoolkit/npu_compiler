//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/core/utils/dump_intermediate_values.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/core/IR/indexed_symbol_attr.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/logger/common_logger.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/YAMLTraits.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassInstrumentation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace vpux;

namespace {

class DumpIntermediateValuesImpl {
    struct FuncOpInfo {
        mlir::func::ReturnOp returnOp;
        SmallVector<mlir::func::CallOp> callOps;
        size_t origNumResults{};
    };

    using FuncInfoMap = std::map<mlir::func::FuncOp, FuncOpInfo>;
    using FuncDumpedResults = std::map<mlir::func::FuncOp, SmallVector<mlir::Value>>;

    std::vector<OpFilter> _filters;
    Logger _log;

public:
    DumpIntermediateValuesImpl(std::vector<OpFilter> filters, Logger log)
            : _filters(std::move(filters)), _log(std::move(log)) {
        _log.setName("dump-intermediate-values");
    }

    mlir::LogicalResult process(mlir::ModuleOp moduleOp) {
        _log.info("Operation filters:");
        if (_filters.empty()) {
            _log.nest().info("No filters specified (dump all operations)");
        } else {
            for (auto filter : _filters) {
                if (filter.locations.empty()) {
                    _log.nest().info("Type '{0}' with no location filters (dump all ops of this type)", filter.name);
                } else {
                    _log.nest().info("Type '{0}' with location filters: '{1}'", filter.name, filter.locations);
                }
            }
        }

        FuncInfoMap funcInfo;
        FuncDumpedResults dumpedResults;
        if (mlir::failed(identifyTargetResults(moduleOp, funcInfo, dumpedResults))) {
            return mlir::failure();
        }
        if (dumpedResults.empty()) {
            _log.warning("No intermediate values matched the specified filters. Nothing to dump.");
            return mlir::success();
        }
        if (mlir::failed(addNewResultValues(funcInfo, dumpedResults))) {
            return mlir::failure();
        }
        if (mlir::failed(updateNetworkInfoOp(moduleOp, funcInfo))) {
            return mlir::failure();
        }
        return mlir::success();
    }

private:
    // Identifies the result values that should be dumped
    // Also collects information about the function operations in the IR (associated call and return operations)
    mlir::LogicalResult identifyTargetResults(mlir::ModuleOp moduleOp, FuncInfoMap& funcInfo,
                                              FuncDumpedResults& dumpedResults) {
        _log.info("Searching for intermediate values to dump based on the filters");
        size_t totalDumpedResults = 0;
        auto status = moduleOp.walk([&](mlir::Operation* op) {
            if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
                auto funcOp = returnOp->getParentOfType<mlir::func::FuncOp>();
                assert(funcOp != nullptr && "Return operation should be part of a function");
                assert(funcInfo[funcOp].returnOp == nullptr &&
                       "There should be only one return operation per function");
                funcInfo[funcOp].returnOp = returnOp;
                funcInfo[funcOp].origNumResults = returnOp.getNumOperands();
                return mlir::WalkResult::advance();
            }

            if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
                auto calledFuncOp = vpux::getCalledFunction(callOp);
                funcInfo[calledFuncOp].callOps.push_back(callOp);
                return mlir::WalkResult::advance();
            }

            // No filters means that all results should be dumped
            if (!_filters.empty()) {
                const auto opName = op->getName();
                const auto opLoc = vpux::stringifyPrimaryLocation(op->getLoc());
                auto found = llvm::find_if(_filters, [opName, opLoc = StringRef(opLoc)](const OpFilter& filter) {
                                 llvm::Regex opNameRegex(filter.name);
                                 if (!opNameRegex.match(opName.getStringRef())) {
                                     return false;
                                 }
                                 if (filter.locations.empty()) {
                                     return true;
                                 }
                                 for (const auto& location : filter.locations) {
                                     llvm::Regex locationRegex(location);
                                     if (locationRegex.match(opLoc)) {
                                         return true;
                                     }
                                 }
                                 return false;
                             }) != _filters.end();
                if (!found) {
                    return mlir::WalkResult::skip();
                }
            }

            _log.nest().info("Found operation '{0}' at '{1}'", op->getName(), op->getLoc());

            auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
            if (funcOp == nullptr) {
                _log.nest(2).info("The operation is not part of a function. Skipping");
                return mlir::WalkResult::skip();
            }

            size_t addedResults = 0;
            for (auto result : op->getResults()) {
                if (llvm::any_of(result.getUsers(), [](mlir::Operation* userOp) {
                        return mlir::isa<mlir::func::ReturnOp>(userOp);
                    })) {
                    _log.nest(2).info("The operation's result number {0} is already returned by the function. Skipping",
                                      result.getResultNumber());
                    continue;
                }
                dumpedResults[funcOp].emplace_back(result);
                ++addedResults;
            }
            totalDumpedResults += addedResults;
            _log.nest(2).info("Added {0} result(s) from the operation to be dumped", addedResults);

            return mlir::WalkResult::advance();
        });
        if (status.wasInterrupted()) {
            return mlir::failure();
        }
        _log.nest().info("Total intermediate values to dump: {0}", totalDumpedResults);
        return mlir::success();
    }

    mlir::Operation* getProducerOp(mlir::Value value, const Logger& log) {
        auto producerOp = value.getDefiningOp();
        if (producerOp == nullptr) {
            log.warning("Missing producer operation for dumped result '{0}'", value);
            return nullptr;
        }
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(producerOp)) {
            auto calledFuncOp = vpux::getCalledFunction(callOp);
            if (calledFuncOp == nullptr) {
                log.warning("Missing called function for call operation '{0}'", callOp->getLoc());
                return nullptr;
            }
            const auto resultIdx = mlir::cast<mlir::OpResult>(value).getResultNumber();
            auto returnedValue = calledFuncOp.getBody().front().getTerminator()->getOperand(resultIdx);
            return getProducerOp(returnedValue, log);
        }
        return producerOp;
    }

    mlir::LogicalResult addNewResultValues(FuncInfoMap& funcInfo, const FuncDumpedResults& dumpedResults) {
        const auto introduceCopyIfNeeded = [&](mlir::Value value) -> mlir::Value {
            auto type = mlir::cast<NDTypeInterface>(value.getType());
            if (type.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
                return value;
            }
            auto producerOp = getProducerOp(value, _log);
            auto builder = mlir::OpBuilder(producerOp->getContext());
            builder.setInsertionPointAfter(producerOp);
            const auto ddrMemSpace =
                    vpux::IndexedSymbolAttr::get(producerOp->getContext(), stringifyEnum(VPU::MemoryKind::DDR));
            auto ddrType = type;
            if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(ddrType)) {
                ddrType = mlir::cast<NDTypeInterface>(distType.getCompactType());
            }
            ddrType = ddrType.changeMemSpace(ddrMemSpace);
            return VPU::CopyOp::create(builder, value.getLoc(), ddrType, value, ddrMemSpace)->getResult(0);
        };

        for (auto& dumpedResult : dumpedResults) {
            auto funcOp = dumpedResult.first;
            auto& results = dumpedResult.second;
            _log.debug("Adding {0} new result values for function '{1}'", results.size(), funcOp.getSymName());

            SmallVector<mlir::Type> newReturnedTypes;
            if (funcInfo.find(funcOp) == funcInfo.end()) {
                _log.error("Missing info for function '{0}'", funcOp.getSymName());
                return mlir::failure();
            }
            auto returnOp = funcInfo.at(funcOp).returnOp;
            for (auto result : results) {
                VPUX_THROW_UNLESS((mlir::isa<mlir::TensorType, VPU::DistributedTensorType>(result.getType())),
                                  "Only Tensor and DistributedTensor types are supported, got '{0}'", result.getType());
                auto newResult = introduceCopyIfNeeded(result);
                returnOp.getOperandsMutable().append(newResult);
                newReturnedTypes.push_back(newResult.getType());
                _log.nest().debug("Added result of type '{0}' at position {1}", newResult.getType(),
                                  static_cast<int64_t>(returnOp->getNumOperands()) - 1);
            }

            const auto funcType = funcOp.getFunctionType();
            const auto newResultTypes =
                    to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), newReturnedTypes));
            const auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), funcType.getInputs(), newResultTypes);
            funcOp.setType(newFuncType);

            for (auto callOp : funcInfo.at(funcOp).callOps) {
                SmallVector<mlir::Value> origResults;
                SmallVector<mlir::Type> resultTypes;
                for (auto origResult : callOp.getResults()) {
                    origResults.push_back(origResult);
                    resultTypes.push_back(origResult.getType());
                }
                resultTypes.append(newReturnedTypes);

                mlir::OpBuilder builder(callOp);
                const auto newCallOp = mlir::func::CallOp::create(builder, callOp.getLoc(), callOp.getCalleeAttr(),
                                                                  resultTypes, callOp->getOperands());
                for (const auto& [origResult, newResult] : zip(origResults, newCallOp->getResults())) {
                    origResult.replaceAllUsesWith(newResult);
                }

                newCallOp->setAttrs(callOp->getAttrs());

                // Update the original call op with the new one
                for (auto& info : funcInfo) {
                    auto& origCallOps = info.second.callOps;
                    for (auto idx : irange(origCallOps.size())) {
                        if (origCallOps[idx] == callOp) {
                            origCallOps[idx] = newCallOp;
                        }
                    }
                }

                callOp.erase();

                auto parentFuncOp = newCallOp->getParentOfType<mlir::func::FuncOp>();
                if (funcInfo.find(parentFuncOp) == funcInfo.end()) {
                    _log.error("Missing info for parent function '{0}'", parentFuncOp.getSymName());
                    return mlir::failure();
                }
                auto parentReturnOp = funcInfo.at(parentFuncOp).returnOp;
                auto returnOperands = parentReturnOp.getOperandsMutable();
                for (auto res : newCallOp->getResults()) {
                    if (res.getResultNumber() < origResults.size()) {
                        continue;
                    }
                    returnOperands.append(res);
                }

                const auto parentFuncType = parentFuncOp.getFunctionType();
                const auto newResultTypes =
                        to_small_vector(llvm::concat<const mlir::Type>(parentFuncType.getResults(), newReturnedTypes));
                const auto newParentFuncType =
                        mlir::FunctionType::get(parentFuncOp.getContext(), parentFuncType.getInputs(), newResultTypes);
                parentFuncOp.setType(newParentFuncType);
            }
        }
        return mlir::success();
    }

    mlir::LogicalResult updateNetworkInfoOp(mlir::ModuleOp moduleOp, const FuncInfoMap& funcInfo) {
        net::NetworkInfoOp netOp;
        mlir::func::FuncOp mainFuncOp;
        net::NetworkInfoOp::getFromModule(moduleOp, netOp, mainFuncOp);

        auto builder = mlir::OpBuilder::atBlockBegin(&moduleOp->getRegion(0).front());
        auto outputsInfoBuilder = mlir::OpBuilder::atBlockEnd(&netOp.getOutputsInfo().front(), builder.getListener());

        if (funcInfo.find(mainFuncOp) == funcInfo.end()) {
            _log.error("Missing info for function '{0}'", mainFuncOp.getSymName());
            return mlir::failure();
        }
        auto returnOp = funcInfo.at(mainFuncOp).returnOp;
        auto origNumResults = funcInfo.at(mainFuncOp).origNumResults;
        auto newResults = returnOp->getOperands().slice(origNumResults, returnOp.getNumOperands() - origNumResults);
        _log.debug("Adding {0} new output info entries to NetworkInfo", newResults.size());

        for (const auto& p : newResults | indexed) {
            auto resultIdx = p.index();
            auto result = p.value();

            auto producerOp = getProducerOp(result, _log.nest());
            VPUX_THROW_WHEN(producerOp == nullptr, "Missing producer operation for dumped result '{0}'", result);
            const auto producerLoc = producerOp->getLoc();
            const auto producerName = getLayerNameFromLocation(producerLoc);
            const auto producerType = getLayerTypeFromLocation(producerLoc);
            const auto loc = appendLoc(producerLoc, llvm::formatv("dump_{0}", resultIdx));
            const auto name = producerName.empty()
                                      ? formatv("dump_{0}_{1}", resultIdx, producerType).str()
                                      : formatv("dump_{0}_{1}_{2}", resultIdx, producerType, producerName).str();
            auto type = mlir::cast<NDTypeInterface>(result.getType());
            TypeComponents typeComponents;
            if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType())) {
                typeComponents.setElementType(normalizeQuantStorageType(qType));
            }
            typeComponents.setShape(ShapeRef(type.getMemShape().raw()));
            typeComponents.setStrides(StridesRef(type.getMemStrides().raw()));
            typeComponents.setDimsOrder(DimsOrder::fromNumDims(type.getRank()));
            type = type.changeTypeComponents(typeComponents);
            net::DataInfoOp::create(outputsInfoBuilder, loc, name, type);
            _log.nest().debug("Added output info: name='{0}', type='{1}'", name, type);
        }
        return mlir::success();
    }
};

class DumpIntermediateValuesInstrumentation final : public mlir::PassInstrumentation {
public:
    void addPass(StringRef passName, ArrayRef<OpFilter> filters, LogLevel logLevel) {
        _passName = passName;
        _filters = filters;
        _logLevel = logLevel;
    }

public:
    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
        if (pass->getName() != _passName) {
            return;
        }

        Logger log = Logger("DumpIntermediateValuesInstrumentation", _logLevel);
        log.info("Dumping intermediate values after pass {0}", pass->getName());

        mlir::ModuleOp moduleOp = nullptr;
        if (auto modOp = mlir::dyn_cast<mlir::ModuleOp>(op)) {
            moduleOp = modOp;
        } else if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
            log.nest().warning(
                    "The instrumentation was added after a function pass. This may behave incorrectly if "
                    "there are multiple functions in the IR, as it accesses the parent module operation directly.");
            moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
        } else {
            log.nest().error("Unexpected operation type: {0}", op->getName());
            return;
        }

        if (mlir::failed(dumpIntermediateValues(moduleOp, _filters, log))) {
            log.nest().error("Failed to dump intermediate values");
        } else {
            log.nest().info("Successfully dumped intermediate values");
        }
    }

private:
    std::string _passName;
    std::vector<OpFilter> _filters;
    LogLevel _logLevel = LogLevel::Info;
};

}  // namespace

mlir::FailureOr<DumpIntermediateValuesConfig> vpux::parseYaml(StringRef fileName) {
    std::ifstream f(fileName.str());
    if (!f.is_open()) {
        return mlir::failure();
    }
    std::string fileContent((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    llvm::yaml::Input input(fileContent);
    DumpIntermediateValuesConfig config;
    input >> config;
    if (input.error()) {
        return mlir::failure();
    }
    return config;
}

void vpux::addIntermediateValueDumper(mlir::PassManager& pm, StringRef configFilePath, const Logger& log) {
    const auto config = parseYaml(configFilePath);
    if (mlir::failed(config)) {
        log.error("Failed to parse YAML file: '{0}'", configFilePath);
        return;
    }
    log.trace("Successfully parsed YAML file: '{0}'", configFilePath);

    auto instrumentation = std::make_unique<DumpIntermediateValuesInstrumentation>();
    instrumentation->addPass(config->passName, config->filters, log.level());
    pm.addInstrumentation(std::move(instrumentation));
}

mlir::LogicalResult vpux::dumpIntermediateValues(mlir::ModuleOp moduleOp, ArrayRef<OpFilter> filters,
                                                 const Logger& log) {
    DumpIntermediateValuesImpl impl(filters, log);
    return impl.process(moduleOp);
}
