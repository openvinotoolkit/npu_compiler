//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/core/IR/ops.hpp>
#include <vpux/compiler/dialect/core/transforms/passes.hpp>
#include <vpux/compiler/dialect/core/types.hpp>
#include <vpux/compiler/utils/func_dialect.hpp>
#include <vpux/compiler/utils/types.hpp>
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Core {
#define GEN_PASS_DECL_PACKNESTEDMODULES
#define GEN_PASS_DEF_PACKNESTEDMODULES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

// Make this pass deterministic by
// 1. Sorting the operations in every cluster by the order in which the operations appear in the block
// 2. Sorting the clusters by the order in which their first operations appear in the blocks. Since
//    clusters don't share any operations, this is well defined.
//
// To achieve this we use std::map and std::set instead of the unordered versions and order the ops
// using isBeforeInBlock.

using FuncOrModule = mlir::Operation*;

// Note: Do not use FuncOrModuleComparator in another context as it will crash or cause UB if 2 ops appear in different
// blocks.
struct FuncOrModuleComparator {
    bool operator()(FuncOrModule a, FuncOrModule b) const {
        assert(a->getBlock() == b->getBlock());
        return a->isBeforeInBlock(b);
    }
};

using CallGraph = std::map<FuncOrModule, SmallVector<FuncOrModule>, FuncOrModuleComparator>;
using FuncOrModuleOpSet = std::set<FuncOrModule, FuncOrModuleComparator>;

/// Returns true if there exists a FuncOp in `from` that has an edge to any FuncOp in `to` according to callGraph.
/// Note that ModuleOps are ignore here.
bool hasSpanningEdges(const FuncOrModuleOpSet& from, const FuncOrModuleOpSet& to, const CallGraph& callGraph) {
    for (const auto& u : from) {
        if (const auto it = callGraph.find(u); it != callGraph.end()) {
            for (const auto& v : it->second) {
                if (to.find(v) != to.end()) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Merges the set src into dst.
void mergeInto(FuncOrModuleOpSet& dst, const FuncOrModuleOpSet& src) {
    dst.insert(src.begin(), src.end());
}

bool hasDynamicTypeMetadata(vpux::NDTypeInterface type) {
    return mlir::isa<Core::BoundedTensorType>(type);
}

// Check a single Core.ReinterpretCast user and return its result type when it carries
// dynamic-shape metadata (bounds).
vpux::NDTypeInterface getDynamicTypeFromReinterpretUsers(mlir::Value root) {
    for (auto* user : root.getUsers()) {
        auto reinterpretCast = mlir::dyn_cast<Core::ReinterpretCastOp>(user);
        if (reinterpretCast == nullptr) {
            continue;
        }

        auto castType = mlir::dyn_cast<vpux::NDTypeInterface>(reinterpretCast.getResult().getType());
        if (castType != nullptr && hasDynamicTypeMetadata(castType)) {
            return castType;
        }
    }

    return nullptr;
}

// Check a single Core.ReinterpretCast producer and return its input type when it carries
// dynamic-shape metadata.
vpux::NDTypeInterface getDynamicTypeFromReinterpretInputs(mlir::Value root) {
    auto reinterpretCast = root.getDefiningOp<Core::ReinterpretCastOp>();
    if (reinterpretCast == nullptr) {
        return nullptr;
    }

    auto castInputType = mlir::dyn_cast<vpux::NDTypeInterface>(reinterpretCast.getInput().getType());
    if (castInputType != nullptr && hasDynamicTypeMetadata(castInputType)) {
        return castInputType;
    }

    return nullptr;
}

vpux::NDTypeInterface getInputNetworkInfoType(mlir::func::FuncOp funcOp, unsigned argIndex) {
    VPUX_THROW_WHEN(argIndex >= funcOp.getNumArguments(), "Input index {0} is out of range for function '{1}'",
                    argIndex, funcOp.getSymName());

    // This pass only propagates dynamic bounds into NetworkInfo when they can be recovered from the
    // reinterpret-cast.
    if (auto dynamicType = getDynamicTypeFromReinterpretUsers(funcOp.getArgument(argIndex)); dynamicType != nullptr) {
        return dynamicType;
    }

    return nullptr;
}

vpux::NDTypeInterface getOutputNetworkInfoType(mlir::func::FuncOp funcOp, unsigned resultIndex) {
    VPUX_THROW_WHEN(resultIndex >= funcOp.getNumResults(), "Output index {0} is out of range for function '{1}'",
                    resultIndex, funcOp.getSymName());

    // This pass only propagates dynamic bounds into NetworkInfo when they can be recovered from the
    // reinterpret-cast.
    for (auto returnOp : funcOp.getOps<mlir::func::ReturnOp>()) {
        if (resultIndex >= returnOp.getNumOperands()) {
            continue;
        }

        // Return operands may be wrapped by reinterpret casts; recover dynamic metadata.
        auto current = returnOp.getOperand(resultIndex);
        if (auto dynamicType = getDynamicTypeFromReinterpretInputs(current); dynamicType != nullptr) {
            return dynamicType;
        }
    }

    return nullptr;
}

mlir::Type buildNetworkInfoType(vpux::NDTypeInterface baseShapeType, vpux::NDTypeInterface recoveredType) {
    mlir::Type newType = mlir::RankedTensorType::get(baseShapeType.getShape(), baseShapeType.getElementType(), nullptr);

    if (!baseShapeType.getShape().isDynamic() || recoveredType == nullptr) {
        return newType;
    }

    const auto bounds = mlir::dyn_cast<Core::BoundedTensorType>(recoveredType);
    if (bounds == nullptr) {
        return newType;
    }

    const auto baseType = mlir::cast<vpux::NDTypeInterface>(newType);
    const auto typeComponents = TypeComponents().setBounds(Bounds(bounds.getBounds().raw()));
    return baseType.changeTypeComponents(typeComponents);
}

//
// PackNestedModules
//

struct NestedClusterInfo {
    FuncOrModuleOpSet functions;
    std::string moduleName;
    mlir::func::FuncOp entryPointFunc = nullptr;
    Core::NestingMode mode = Core::NestingMode::Default;
    bool isFunctionToPack = false;  // True if created from FunctionToPack attribute
};

struct Clusters {
    FuncOrModuleOpSet topCluster;
    SmallVector<NestedClusterInfo> nestedClusters;
};

class PackNestedModules : public vpux::Core::impl::PackNestedModulesBase<PackNestedModules> {
public:
    explicit PackNestedModules(Logger log, Core::NestingMode nestingMode, bool enableProfiling)
            : _nestingMode(nestingMode), _enableProfiling(enableProfiling), _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit PackNestedModules(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    // utility functions
    mlir::Operation* getFirstClusterInsertionPoint(FuncOrModuleOpSet& topCluster);
    std::string getSubModuleName(size_t index);
    bool belongsIntoTopCluster(mlir::func::FuncOp caller, mlir::func::FuncOp mainFuncOp);
    // clustering
    CallGraph collectCallGraph(mlir::func::FuncOp root);
    SmallVector<FuncOrModuleOpSet> createClusters(const CallGraph& callGraph, mlir::func::FuncOp mainFuncOp);
    SmallVector<FuncOrModuleOpSet> createMainCluster(const CallGraph& callGraph);
    Clusters collectClusters(mlir::func::FuncOp mainFuncOp);
    Clusters collectClustersFromAttributes();
    void patchCallSites(const Clusters& clusters);
    mlir::ModuleOp nestCluster(mlir::OpBuilder& builder, NestedClusterInfo&& clusterInfo);
    // updating PipelineOptions, Resources and NetworkInfo
    void clonePipelineOptionsOp(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp);
    void cloneResourcesOps(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp);
    void createNetInfoForFuncOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp, bool enableProfiling = false);
    void nestNetworkInfo(mlir::ModuleOp subModuleOp, net::NetworkInfoOp& netInfo);
    void removeNetInfoOp(mlir::ModuleOp moduleOp);

    void safeRunOnModule() final;

private:
    Core::NestingMode _nestingMode = Core::NestingMode::Default;
    bool _enableProfiling = false;
    Logger _log;
};

mlir::LogicalResult PackNestedModules::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (mode.hasValue()) {
        _nestingMode = Core::parseNestingMode(mode.getValue());
    }

    if (enableProfiling.hasValue()) {
        _enableProfiling = enableProfiling.getValue();
    }

    return mlir::success();
}

bool PackNestedModules::belongsIntoTopCluster(mlir::func::FuncOp caller, mlir::func::FuncOp mainFuncOp) {
    return (_nestingMode == Core::NestingMode::EntryPoint ? false : caller == mainFuncOp) ||
           caller->hasAttrOfType<mlir::UnitAttr>("do_not_nest");
}

std::string PackNestedModules::getSubModuleName(size_t index) {
    return _nestingMode == Core::NestingMode::EntryPoint ? Core::NPU_MODULE_NAME.str() : formatv("Module{0}", index);
}

mlir::Operation* PackNestedModules::getFirstClusterInsertionPoint(FuncOrModuleOpSet& topCluster) {
    if (_nestingMode == Core::NestingMode::EntryPoint) {
        // We assume that the top module is not empty.
        return &getOperation().getBodyRegion().front().back();
    } else {
        assert(!topCluster.empty());
        return *topCluster.begin();
    }
}

/// Returns a map of FuncOp -> Vector<FuncOp|ModuleOp> that models the call graph using cycle-aware BFS.
CallGraph PackNestedModules::collectCallGraph(mlir::func::FuncOp root) {
    auto topModuleOp = getOperation();

    CallGraph callGraph;

    mlir::DenseSet<mlir::func::FuncOp> visited;

    std::queue<mlir::func::FuncOp> workList;
    workList.push(root);

    while (!workList.empty()) {
        auto funcOp = workList.front();
        workList.pop();

        if (bool firstOccurrence = visited.insert(funcOp).second; !firstOccurrence) {
            continue;
        }

        auto& calleeVec = callGraph[funcOp];

        funcOp.walk([&](mlir::CallOpInterface callOp) {
            const auto calleeSymRef = mlir::cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
            const auto firstSymName =
                    calleeSymRef.getRootReference();  // first symbol in a chain @module_0::...::@module_N::@calleeFunc
            const auto calleeOp = topModuleOp.lookupSymbol(firstSymName);
            assert(calleeOp != nullptr);
            calleeVec.push_back(calleeOp);
            // We can ignore ModuleOps because we cannot call functions of a top module from nested modules
            if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(calleeOp); func != nullptr) {
                workList.push(func);
            }
        });
    }

    return callGraph;
}

/// Returns a vector of sets of FuncOps.
///
///     Assume we have the following call graph:
///
///         main* -> [f, h]
///
///         f -> g*
///
///         h -> i
///
///         i -> j
///         j -> i
///
///     Functions annotated with (*) will be put into the "top" cluster (i.e. they will not be nested into submodules),
///     either because that function is the entry point or it contains the attribute "do_not_nest".
/// 1)
///     The result will be initialized as follows. All FuncOps that belong into the top cluster are put into the first
///     set of the vector first. All other functions are then put into their own respective singleton set. In this
///     example:
///         [{main*, g*}, {f}, {h}, {i}, {j}]
/// 2)
///     Then all singleton sets are merged into the top cluster set when they have an edge into the top cluster set.
///     In our example, because of (f -> g*), we would get {main*, g*, f}. To account for transitive edges, this step
///     is done until a full iteration is performed without any merges.
/// 3)
///     Then the remaining clusters (except for the top cluster) are merged pairwise if they have edges from on set to
///     the other or vice versa. This is done until no more clusters can be merged In this example:
///         [{main*, g*, f}, {h}, {i}, {j}]     {i} and {j}    are merged because of i -> j and j -> i
///         [{main*, g*, f}, {h}, {i, j}]       {h} and {i, j} are merged because of h -> i
///         [{main*, g*, f}, {h, i, j}]         Nothing to merge
/// 4)
///     Done
///
SmallVector<FuncOrModuleOpSet> PackNestedModules::createClusters(const CallGraph& callGraph,
                                                                 mlir::func::FuncOp mainFuncOp) {
    SmallVector<FuncOrModuleOpSet> clusters{{}};

    for (auto [caller, callees] : callGraph) {
        const auto callerFuncOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(caller);
        if (callerFuncOp != nullptr && belongsIntoTopCluster(callerFuncOp, mainFuncOp)) {
            clusters[0].insert(caller);
        } else {
            clusters.push_back({});
            clusters.back().insert(caller);
        }
    }

    // merge singletons into top cluster
    auto mergeSingletonsIntoTopCluster = [&]() -> bool {
        for (auto src = clusters.begin() + 1; src < clusters.end(); src++) {
            auto dst = clusters.begin();
            const auto doMerge = hasSpanningEdges(*src, *dst, callGraph);
            if (doMerge) {
                mergeInto(*dst, *src);
                clusters.erase(src);
                return true;
            }
        }
        return false;
    };
    while (mergeSingletonsIntoTopCluster()) {
    }

    // correctly merge pairwise until fixed point is reached (ignore top cluster)
    auto mergePairwise = [&]() -> bool {
        for (auto dst = std::next(clusters.begin()); dst < clusters.end(); dst++) {
            for (auto src = std::next(dst); src < clusters.end(); src++) {
                const auto doMerge = hasSpanningEdges(*dst, *src, callGraph) || hasSpanningEdges(*src, *dst, callGraph);
                if (doMerge) {
                    mergeInto(*dst, *src);
                    clusters.erase(src);
                    return true;
                }
            }
        }
        return false;
    };
    while (mergePairwise()) {
    }

    return clusters;
}

/// Creates one main cluster for the entryPoint function
SmallVector<FuncOrModuleOpSet> PackNestedModules::createMainCluster(const CallGraph& callGraph) {
    FuncOrModuleOpSet mainCluster;

    for (auto& [caller, callees] : callGraph) {
        mainCluster.insert(caller);
        mainCluster.insert(callees.begin(), callees.end());
    }

    auto topModuleOp = getOperation();
    FuncOrModuleOpSet topCluster;
    // topCluster contains all func ops of topModule that doesn't belong to callGraph.
    // We can ignore ModuleOps because we don't need to update their callOps
    for (const auto& funcOp : topModuleOp.getOps<mlir::func::FuncOp>()) {
        auto funcOrModule = funcOp;
        if (callGraph.count(funcOrModule) == 0) {
            topCluster.insert(funcOrModule);
        }
    }

    return {std::move(topCluster), std::move(mainCluster)};
}

// Returns Clusters with FuncOps grouped for nesting. Every nested cluster represents one set of operations
// that will be nested inside a submodule. The resulting clusters will be sorted to ensure deterministic IR.
Clusters PackNestedModules::collectClusters(mlir::func::FuncOp mainFuncOp) {
    const auto callGraph = collectCallGraph(mainFuncOp);
    const auto clustersSet = _nestingMode == Core::NestingMode::EntryPoint ? createMainCluster(callGraph)
                                                                           : createClusters(callGraph, mainFuncOp);

    VPUX_THROW_WHEN(clustersSet.empty(), "Expected at least one cluster (top cluster), but got empty set");

    Clusters result;
    result.topCluster = std::move(clustersSet.front());

    // Convert FuncOrModuleOpSet to NestedClusterInfo with auto-generated module names
    for (size_t index = 1; index < clustersSet.size(); ++index) {
        NestedClusterInfo clusterInfo;
        clusterInfo.functions = std::move(clustersSet[index]);
        clusterInfo.moduleName = getSubModuleName(index - 1);
        clusterInfo.entryPointFunc = nullptr;
        clusterInfo.mode = _nestingMode;
        result.nestedClusters.push_back(std::move(clusterInfo));
    }

    return result;
}

/// Collects clusters from functions marked with the FunctionToPack attribute.
/// Functions are grouped by their target module name. Functions without the attribute go into the top cluster.
Clusters PackNestedModules::collectClustersFromAttributes() {
    auto topModuleOp = getOperation();

    llvm::StringMap<NestedClusterInfo> moduleGroups;
    Clusters result;

    for (auto funcOp : topModuleOp.getOps<mlir::func::FuncOp>()) {
        auto targetModule = config::getFunctionToPackTargetModule(funcOp);
        if (targetModule.empty()) {
            result.topCluster.insert(funcOp.getOperation());
            continue;
        }

        _log.trace("Function '{0}' will be packed into module '{1}'", funcOp.getSymName(), targetModule);
        auto& clusterInfo = moduleGroups[targetModule];
        if (clusterInfo.moduleName.empty()) {
            clusterInfo.moduleName = targetModule;
        }
        clusterInfo.functions.insert(funcOp.getOperation());

        // Check if this function is designated as entry point
        if (config::hasFunctionToPackEntryPointAttribute(funcOp)) {
            VPUX_THROW_WHEN(clusterInfo.entryPointFunc != nullptr,
                            "Multiple functions marked as FunctionToPackEntryPoint in module '{0}'. "
                            "Found '{1}' and '{2}'. Only one entry point is allowed.",
                            targetModule, clusterInfo.entryPointFunc.getSymName(), funcOp.getSymName());
            clusterInfo.entryPointFunc = funcOp;
            _log.trace("Function '{0}' designated as entry point for module '{1}'", funcOp.getSymName(), targetModule);
        }

        config::removeFunctionToPackAttribute(funcOp);
    }

    _log.debug("Collected module groups: {0}", moduleGroups.size());
    for (auto& entry : moduleGroups) {
        VPUX_THROW_WHEN(topModuleOp.lookupSymbol<mlir::ModuleOp>(entry.second.moduleName) != nullptr,
                        "Module with name '{0}' already exists. Cannot create a new module for FunctionToPack "
                        "functions.",
                        entry.second.moduleName);
        entry.second.mode = Core::NestingMode::Default;
        entry.second.isFunctionToPack = true;
        result.nestedClusters.push_back(std::move(entry.second));
    }

    return result;
}

/// Replaces all mlir::CallOps in the top cluster with Core::NestedCallOps that point to the newly nested functions.
void PackNestedModules::patchCallSites(const Clusters& clusters) {
    // Build mapping from function/module name to target module name
    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    llvm::DenseMap<mlir::StringAttr, mlir::SymbolRefAttr> nestedClusterMap;
    for (const auto& clusterInfo : clusters.nestedClusters) {
        for (auto funcOrModuleOp : clusterInfo.functions) {
            auto symName = mlir::cast<mlir::SymbolOpInterface>(funcOrModuleOp).getNameAttr();
            nestedClusterMap[symName] = mlir::SymbolRefAttr::get(builder.getStringAttr(clusterInfo.moduleName));
        }
    }

    for (auto funcOrModuleOp : clusters.topCluster) {
        funcOrModuleOp->walk([&](mlir::CallOpInterface callOp) {
            const auto calleeOpSymRef = mlir::cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());

            const auto firstSymName = calleeOpSymRef.getRootReference();
            auto it = nestedClusterMap.find(firstSymName);
            if (it == nestedClusterMap.end()) {
                return;
            }

            const auto& moduleSymRef = it->second;

            // Construct a vector of FlatSymbolRefAttr from callee op reference symbols and then
            // build a new SymbolRefAttr as @Module + {symVec}
            SmallVector<mlir::FlatSymbolRefAttr> symVec{mlir::FlatSymbolRefAttr::get(firstSymName)};
            symVec.append(calleeOpSymRef.getNestedReferences().begin(), calleeOpSymRef.getNestedReferences().end());
            const auto nestedSymbolAttr = mlir::SymbolRefAttr::get(moduleSymRef.getRootReference(), symVec);

            builder.setInsertionPoint(callOp);
            const auto nestedCallOp = builder.create<Core::NestedCallOp>(
                    callOp.getLoc(), nestedSymbolAttr, callOp->getResultTypes(), callOp->getOperands());

            callOp->replaceAllUsesWith(nestedCallOp);
            callOp->erase();
        });
    }
}

/// Creates a new ModuleOp at the builder's current location and moves all FuncOps in the cluster into it.
mlir::ModuleOp PackNestedModules::nestCluster(mlir::OpBuilder& builder, NestedClusterInfo&& clusterInfo) {
    const auto& moduleName = clusterInfo.moduleName;

    auto topModuleOp = getOperation();
    auto subModuleOp = builder.create<mlir::ModuleOp>(appendLoc(topModuleOp.getLoc(), "{0}", moduleName), moduleName);
    auto& targetOps = subModuleOp.getBody()->getOperations();

    for (auto funcOrModuleOp : clusterInfo.functions) {
        targetOps.splice(targetOps.end(), funcOrModuleOp->getBlock()->getOperations(), funcOrModuleOp->getIterator());
    }

    // Mark module as created via FunctionToPack for later identification by unpack pass
    if (clusterInfo.isFunctionToPack) {
        config::setPackedModuleAttribute(subModuleOp);
    }

    return subModuleOp;
}

void PackNestedModules::createNetInfoForFuncOp(mlir::OpBuilder& builder, mlir::func::FuncOp funcOp,
                                               bool enableProfiling) {
    _log.trace("Creating new networkInfo for the function: {0}", funcOp.getName());
    auto netInfo = builder.create<net::NetworkInfoOp>(
            appendLoc(funcOp.getLoc(), "nested_network_info"),
            mlir::FlatSymbolRefAttr::get(funcOp->getContext(), funcOp.getName()), enableProfiling);
    net::setupSections(netInfo, enableProfiling);

    auto funcType = funcOp.getFunctionType();

    // Handle inputs
    auto& inputRegion = netInfo.getInputsInfo();
    builder.setInsertionPointToStart(&inputRegion.front());

    // These will be replaced with core dialect definitions when dynamic strides are removed
    llvm::StringRef funcArgDynamicStridesAttrName = vpux::HostExec::HOST_EXEC_FUNC_ARG_DYNAMIC_STRIDES_ATTR_NAME;
    mlir::StringAttr funcArgDynamicStridesAttrNameAttr = builder.getStringAttr(funcArgDynamicStridesAttrName);
    llvm::StringRef dynamicStridesAttrName = vpux::HostExec::HOST_EXEC_DYNAMIC_STRIDES_ATTR_NAME;

    auto createDataInfoOp = [&](const std::string& name, vpux::NDTypeInterface argType,
                                vpux::NDTypeInterface netInfoArgType) {
        mlir::Type newType = buildNetworkInfoType(argType, netInfoArgType);
        auto dataInfoOp = builder.create<net::DataInfoOp>(appendLoc(funcOp.getLoc(), name), name, newType);

        SmallVector<mlir::Attribute> tensorNamesStringAttrs;
        tensorNamesStringAttrs.push_back(builder.getStringAttr(name));
        auto tensorNamesArrayAttr = builder.getArrayAttr(tensorNamesStringAttrs);
        dataInfoOp.setTensorNamesAttr(tensorNamesArrayAttr);
        return dataInfoOp;
    };

    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
        auto argType = mlir::cast<vpux::NDTypeInterface>(funcType.getInput(i));
        auto netInfoArgType = getInputNetworkInfoType(funcOp, i);
        auto name = formatv("in_{0}", i).str();
        auto dataInfoOp = createDataInfoOp(name, argType, netInfoArgType);

        // If func op has dynamicStrides attribute for arguments then set the same attribute to inputsInfo
        auto dynamicStridesAttr =
                mlir::dyn_cast_or_null<mlir::BoolAttr>(funcOp.getArgAttr(i, funcArgDynamicStridesAttrName));
        if (dynamicStridesAttr && dynamicStridesAttr.getValue()) {
            dataInfoOp->setAttr(dynamicStridesAttrName, mlir::UnitAttr::get(dataInfoOp.getContext()));
            funcOp.removeArgAttr(i, funcArgDynamicStridesAttrNameAttr);
        }
    }

    // Handle outputs
    auto& outputsRegion = netInfo.getOutputsInfo();
    builder.setInsertionPointToStart(&outputsRegion.front());

    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
        auto resType = mlir::cast<vpux::NDTypeInterface>(funcType.getResult(i));
        auto netInfoResType = getOutputNetworkInfoType(funcOp, i);
        auto name = formatv("out_{0}", i).str();
        auto dataInfoOp = createDataInfoOp(name, resType, netInfoResType);

        // If func op has dynamicStrides attribute for results then set the same attribute to outputsInfo
        auto dynamicStridesAttr =
                mlir::dyn_cast_or_null<mlir::BoolAttr>(funcOp.getResultAttr(i, funcArgDynamicStridesAttrName));
        if (dynamicStridesAttr && dynamicStridesAttr.getValue()) {
            dataInfoOp->setAttr(dynamicStridesAttrName, mlir::UnitAttr::get(dataInfoOp.getContext()));
            funcOp.removeResultAttr(i, funcArgDynamicStridesAttrNameAttr);
        }
    }
}

void PackNestedModules::clonePipelineOptionsOp(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp) {
    auto topPipelineOptions = topModuleOp.getOps<config::PipelineOptionsOp>();
    auto optionsCount = std::distance(topPipelineOptions.begin(), topPipelineOptions.end());
    VPUX_THROW_WHEN(optionsCount != 1, "No valid count of config.PipelineOptionsOp found {0}", optionsCount);

    auto topPipelineOptionsOp = *topPipelineOptions.begin();
    builder.clone(*topPipelineOptionsOp);
}

void PackNestedModules::cloneResourcesOps(mlir::OpBuilder& builder, mlir::ModuleOp topModuleOp) {
    for (auto reservedResource : topModuleOp.getOps<config::ResourcesOp>()) {
        builder.clone(*reservedResource);
    }
}

void PackNestedModules::nestNetworkInfo(mlir::ModuleOp subModuleOp, net::NetworkInfoOp& netInfo) {
    auto& targetOps = subModuleOp.getBody()->getOperations();
    targetOps.splice(targetOps.begin(), netInfo->getBlock()->getOperations(), netInfo->getIterator());
}

void PackNestedModules::removeNetInfoOp(mlir::ModuleOp moduleOp) {
    SmallVector<net::NetworkInfoOp> netInfoOps(moduleOp.getOps<net::NetworkInfoOp>());
    for (auto netInfo : netInfoOps) {
        netInfo->erase();
    }
}

void PackNestedModules::safeRunOnModule() {
    auto topModuleOp = getOperation();
    auto [netInfo, mainFuncOp] = net::getFromModule(topModuleOp);

    // Dispatch cluster collection based on whether FunctionToPack attributes are present
    auto clusters = collectClustersFromAttributes();
    if (clusters.nestedClusters.empty()) {
        clusters = collectClusters(mainFuncOp);
        _log.debug("Collected top clusters: {0}, nested clusters: {1}", clusters.topCluster.size(),
                   clusters.nestedClusters.size());
    }

    VPUX_THROW_WHEN(_nestingMode == Core::NestingMode::EntryPoint && clusters.nestedClusters.size() != 1,
                    "Cannot nest entryPoint function: no valid count of nested clusters found {0}",
                    clusters.nestedClusters.size());

    patchCallSites(clusters);

    mlir::OpBuilder::Listener listener;
    mlir::OpBuilder builder(&getContext(), &listener);
    builder.setInsertionPoint(getFirstClusterInsertionPoint(clusters.topCluster));

    for (auto&& clusterInfo : clusters.nestedClusters) {
        _log.trace("Process nested module: {0}", clusterInfo.moduleName);
        mlir::func::FuncOp entryPointFunc = clusterInfo.entryPointFunc;
        Core::NestingMode clusterMode = clusterInfo.mode;
        bool isFunctionToPack = clusterInfo.isFunctionToPack;

        // move because the FuncOps in cluster will be invalidated
        auto subModuleOp = nestCluster(builder, std::move(clusterInfo));

        // subModule ops should have the same attributes as the top module
        // that have not already been set
        for (auto attr : topModuleOp->getAttrs()) {
            if (!subModuleOp->hasAttr(attr.getName())) {
                subModuleOp->setAttr(attr.getName(), attr.getValue());
            }
        }

        builder.setInsertionPointToStart(subModuleOp.getBody());

        // If no explicit entry point was set, default to the first function in the module
        if (entryPointFunc == nullptr) {
            auto funcOps = subModuleOp.getOps<mlir::func::FuncOp>();
            assert(std::distance(funcOps.begin(), funcOps.end()) > 0 &&
                   "A nested module must have at least 1 function");
            entryPointFunc = *funcOps.begin();
        }

        if (clusterMode == Core::NestingMode::EntryPoint) {
            // If we nest entryPoint func then networkInfo should be moved to the subModule as well
            nestNetworkInfo(subModuleOp, netInfo);
        } else if (isFunctionToPack) {
            // FunctionToPack mode: use the designated entry point function
            createNetInfoForFuncOp(builder, entryPointFunc, _enableProfiling);
        } else {
            // In case of the default nesting mode we use the func of each subModule op as its new entryPoint
            // and create new NetworkInfo operations
            auto funcOps = subModuleOp.getOps<mlir::func::FuncOp>();
            auto funcOpsCount = std::distance(funcOps.begin(), funcOps.end());
            if (funcOpsCount != 1) {
                continue;
            }
            createNetInfoForFuncOp(builder, entryPointFunc, _enableProfiling);
        }

        builder.setInsertionPointToStart(subModuleOp.getBody());

        clonePipelineOptionsOp(builder, topModuleOp);
        cloneResourcesOps(builder, topModuleOp);

        // nestCluster invalidates IR so we have to update the insertion point
        builder.setInsertionPointAfter(subModuleOp);
    }
    // Get rid of any inconsistencies:
    // if there are no entryPoint but a module still has NetInfo in topModule, we need to remove it
    // and vice versa
    auto topModuleNetInfos = to_small_vector(topModuleOp.getOps<net::NetworkInfoOp>());
    _log.debug("Check whether the top module has got entry point function after all, netInfos: {0}",
               topModuleNetInfos.size());
    if (topModuleNetInfos.size() > 0) {
        auto netFunc = topModuleOp.lookupSymbol<mlir::func::FuncOp>(topModuleNetInfos.front().getEntryPointAttr());
        if (netFunc == nullptr) {
            _log.debug("Remove NetworkInfo section completely as there is no any entry point function");
            removeNetInfoOp(topModuleOp);
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createPackNestedModulesPass(Logger log, Core::NestingMode nestingMode,
                                                                    bool enableProfiling) {
    return std::make_unique<PackNestedModules>(log, nestingMode, enableProfiling);
}
