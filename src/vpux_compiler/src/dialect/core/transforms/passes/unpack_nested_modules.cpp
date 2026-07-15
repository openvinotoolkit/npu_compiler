//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/array_ref.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Visitors.h>
#include <variant>

namespace vpux::Core {
#define GEN_PASS_DECL_UNPACKNESTEDMODULES
#define GEN_PASS_DEF_UNPACKNESTEDMODULES
#include "vpux/compiler/dialect/core/passes.hpp.inc"
}  // namespace vpux::Core

using namespace vpux;

namespace {

mlir::func::FuncOp getFirstFunctionDirectlyInModule(mlir::ModuleOp moduleOp) {
    auto allFuncs = moduleOp.getOps<mlir::func::FuncOp>();
    assert(!allFuncs.empty() && "Module must have at least one operation");
    auto firstFunc = *allFuncs.begin();
    assert(getModuleOp(firstFunc) == moduleOp && "Module operations are assumed to be non-nested");
    return firstFunc;
}

// Helper structure for convenient manipulation of mlir::SymbolRefAttr. It stores symbols as std::string's in
// a container and provides utilities for querying, removing, adding symbol name prefixes.
struct NestedSymbolHelper {
    SmallVector<std::string> prefixParts;

    NestedSymbolHelper() = default;

    NestedSymbolHelper(mlir::SymbolRefAttr symbol) {
        prefixParts.push_back(symbol.getRootReference().str());
        for (auto nestedRef : symbol.getNestedReferences()) {
            prefixParts.push_back(nestedRef.getAttr().str());
        }
    }

    NestedSymbolHelper(llvm::StringRef symbolRef) {
        prefixParts.push_back(symbolRef.str());
    }

    NestedSymbolHelper(const std::string& symbolName) {
        prefixParts.push_back(symbolName);
    }

    // Appends a symbol of FlatSymbolRefAttr type to prefixParts
    void append(mlir::FlatSymbolRefAttr flatSymbol) {
        prefixParts.push_back(flatSymbol.getAttr().str());
    }

    // Returns true if `other` is a prefix of this nested symbol
    bool hasPrefix(const NestedSymbolHelper& other) const {
        if (prefixParts.size() < other.prefixParts.size()) {
            return false;
        }

        for (size_t i = 0; i < other.prefixParts.size(); ++i) {
            if (prefixParts[i] != other.prefixParts[i]) {
                return false;
            }
        }

        return true;
    }

    // Removes `other` prefix from this in-place
    void removePrefix(const NestedSymbolHelper& other) {
        if (hasPrefix(other)) {
            prefixParts.erase(prefixParts.begin(), prefixParts.begin() + other.prefixParts.size());
        }
    }

    // Returns an attribute of type `SymbolRefAttr` or `FlatSymbolRefAttr` constructed from prefixParts
    mlir::Attribute makeSymbolAttr(mlir::MLIRContext* ctx) const {
        if (prefixParts.empty()) {
            return nullptr;
        }
        if (prefixParts.size() == 1) {
            return mlir::FlatSymbolRefAttr::get(ctx, prefixParts[0]);
        }

        // prefixParts.size() > 1
        auto newRootRef = mlir::StringAttr::get(ctx, prefixParts[0]);
        SmallVector<mlir::FlatSymbolRefAttr> newNestedRefs;
        for (size_t i = 1; i < prefixParts.size(); ++i) {
            newNestedRefs.push_back(mlir::FlatSymbolRefAttr::get(ctx, prefixParts[i]));
        }
        return mlir::SymbolRefAttr::get(newRootRef, newNestedRefs);
    }

    // Creates a new NestedSymbolHelper object by concatenating this and `other`s prefix parts.
    NestedSymbolHelper concat(const NestedSymbolHelper& other) const {
        NestedSymbolHelper result(*this);
        for (const auto& otherPrefix : other.prefixParts) {
            result.prefixParts.push_back(otherPrefix);
        }
        return result;
    }

    // Custom comparison operator for sorting items in std::set. It prioritizes deeper (longer)
    // and lexicographically greater prefixes, allowing more nested symbols to come first.
    bool operator<(const NestedSymbolHelper& other) const {
        for (size_t i = 0; i < std::min(prefixParts.size(), other.prefixParts.size()); ++i) {
            if (prefixParts[i] < other.prefixParts[i]) {
                return false;
            } else if (other.prefixParts[i] < prefixParts[i]) {
                return true;
            }
        }

        return prefixParts.size() > other.prefixParts.size();
    }
};

class UnpackNestedModulesPass final : public Core::impl::UnpackNestedModulesBase<UnpackNestedModulesPass> {
    /// Returns all "top-level" nested modules within a specified module.
    /// Top-level nested modules are direct children of the module.
    SmallVector<mlir::ModuleOp> collectTopLevelNestedModules(mlir::ModuleOp mainModule);

    /// Moves any discovered nested functions into the specified module.
    void moveNestedFunctionsIntoMainModule(SmallVector<mlir::ModuleOp> nestedModules,
                                           std::set<NestedSymbolHelper>& nestedModuleNames, mlir::ModuleOp mainModule);

    /// Converts any calls to nested functions into calls without
    /// nested modules names in their symbols
    void flattenCallSites(mlir::ModuleOp mainModule, const std::set<NestedSymbolHelper>& nestedModuleNames);

    /// Erases any nested modules.
    void eraseNestedModules(ArrayRef<mlir::ModuleOp> nestedModules);

    /// Move network info before the first Module or Func op if possible
    void moveNetworkInfoToTheTopOfModule(mlir::ModuleOp moduleOp);

    /// Removes PipelineOptionsOp and ResourcesOps from the nested module
    void removePipelineOptionsOp(mlir::ModuleOp moduleOp);
    void removeResourcesOp(mlir::ModuleOp moduleOp);
    void removeNetInfoOp(mlir::ModuleOp moduleOp);

public:
    explicit UnpackNestedModulesPass(const Logger& log, Core::NestingMode nestingMode): _nestingMode(nestingMode) {
        Base::initLogger(log, Base::getArgumentName());
    }
    explicit UnpackNestedModulesPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnModule() final;

private:
    Core::NestingMode _nestingMode = Core::NestingMode::Default;
};

mlir::LogicalResult UnpackNestedModulesPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (mode.hasValue()) {
        _nestingMode = Core::parseNestingMode(mode.getValue());
    }

    return mlir::success();
}

SmallVector<mlir::ModuleOp> UnpackNestedModulesPass::collectTopLevelNestedModules(mlir::ModuleOp mainModule) {
    if (_nestingMode == Core::NestingMode::EntryPoint) {
        auto nestedModule = mainModule.lookupSymbol<mlir::ModuleOp>(Core::NPU_MODULE_NAME);
        return (nestedModule == nullptr) ? SmallVector<mlir::ModuleOp>{} : SmallVector<mlir::ModuleOp>({nestedModule});
    }

    SmallVector<mlir::ModuleOp> nestedWithPackedAttribute;
    SmallVector<mlir::ModuleOp> topLevelNested;

    // Note: assumed to be "fast" since we only walk modules.
    mainModule.walk([&](mlir::ModuleOp nestedModule) {
        if (nestedModule == mainModule) {  // Note: walk visits "self" as well
            return mlir::WalkResult::advance();
        }

        auto funcOps = nestedModule.getOps<mlir::func::FuncOp>();
        auto it = funcOps.begin();

        // Module without funcOp indicates reserved memory module. Skip this pass
        // for such modules.
        if (std::distance(it, funcOps.end()) == 0) {
            return mlir::WalkResult::advance();
        }

        const bool directChildOfMainModule = (nestedModule->getParentOfType<mlir::ModuleOp>() == mainModule);
        if (directChildOfMainModule) {
            _log.trace("Found top-level nested module '{0}' inside '{1}'", nestedModule.getSymName(),
                       mainModule.getSymName());
            topLevelNested.push_back(nestedModule);

            const bool hasPackedModuleAttribute = config::hasPackedModuleAttribute(nestedModule);
            if (hasPackedModuleAttribute) {
                _log.trace("Module '{0}' has PackedModule attribute, will be unpacked", nestedModule.getSymName());
                nestedWithPackedAttribute.push_back(nestedModule);
            }
        }
        return mlir::WalkResult::skip();
    });

    if (nestedWithPackedAttribute.empty()) {
        _log.trace("No modules with PackedModule attribute found, falling back to unpacking all nested modules for "
                   "backward compatibility");
        return topLevelNested;
    }

    return nestedWithPackedAttribute;
}

// Recursive walk function that moves func ops to the top module and collects nested modules names to std::set
// so that they can be "erased" from callOp's callee symbols.
// In each recursion step we collect a current module name and "prefix" (chain of previous module names) + current
// module name. For example, for ths structure
// ```mlir
// module @Nested {
//  module @Nested2 {
//      func.func private @bar(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> { ... }
//  }
//
//  func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf16> { ... }
// }
// ```
// we collect ["@Nested::@Nested2", "@Nested2", "@Nested"], because we can call
// @Nested::@Nested2::@bar, @Nested2::@bar and @Nested::@foo
void walkNestedModulesAndMoveFuncs(mlir::ModuleOp moduleOp, mlir::ModuleOp mainModule,
                                   std::set<NestedSymbolHelper>& nestedModuleNames, mlir::func::FuncOp referenceFuncOp,
                                   const NestedSymbolHelper& prefix = {}) {
    auto currentModuleSymName = NestedSymbolHelper(mlir::SymbolRefAttr::get(moduleOp.getSymNameAttr()));
    auto nestedSymName = prefix.concat(currentModuleSymName);

    nestedModuleNames.insert(nestedSymName);
    nestedModuleNames.insert(std::move(currentModuleSymName));

    for (auto subModuleOp : moduleOp.getOps<mlir::ModuleOp>()) {
        walkNestedModulesAndMoveFuncs(subModuleOp, mainModule, nestedModuleNames, referenceFuncOp, nestedSymName);
    }

    SmallVector<mlir::func::FuncOp> funcOps(moduleOp.getOps<mlir::func::FuncOp>());
    for (auto funcOp : funcOps) {
        [[maybe_unused]] auto nestedModule = getModuleOp(funcOp);
        assert(nestedModule != mainModule);
        // Note: move nested function to a place relative to the "reference"
        // function (some function that is guaranteed to be in the main module).
        funcOp->moveBefore(referenceFuncOp);
        funcOp.setNested();
    }
}

void UnpackNestedModulesPass::moveNestedFunctionsIntoMainModule(SmallVector<mlir::ModuleOp> nestedModules,
                                                                std::set<NestedSymbolHelper>& nestedModuleNames,
                                                                mlir::ModuleOp mainModule) {
    if (_nestingMode == Core::NestingMode::EntryPoint) {
        auto existingTargetNetOps = mainModule.getOps<net::NetworkInfoOp>();
        auto& targetOps = mainModule.getBodyRegion().front().getOperations();
        // It is checked in `safeRunOnModule()` that nestedModules is guaranteed to have only one element.
        auto nestedModuleOp = nestedModules.begin();

        // No need to move PipelineOptionsOp and ResourcesOps to the top module
        removePipelineOptionsOp(*nestedModuleOp);
        removeResourcesOp(*nestedModuleOp);
        if (!existingTargetNetOps.empty()) {
            _log.debug("Despite having the nesting mode chosen as EntryPoint, the top module already has entryPoint, "
                       "skip nested entryPoint");
            removeNetInfoOp(*nestedModuleOp);
        }

        targetOps.splice(targetOps.end(), nestedModuleOp->getBodyRegion().front().getOperations());
        nestedModuleNames.insert(nestedModuleOp->getSymNameAttr().str());
        return;
    }

    // Note: "reference" here means some function that is guaranteed to be in the main module
    auto referenceFuncOp = getFirstFunctionDirectlyInModule(mainModule);
    for (auto nested : nestedModules) {
        // Note: walk here is "recursive". That is, given the below structure:
        // ```mlir
        // module @Nested {
        //  module @Nested2 {
        //      func.func private @bar(%arg: tensor<2x2xf16>) -> tensor<2x2xf16> { ... }
        //  }
        //
        //  func.func private @foo(%arg: tensor<2x2xf32>) -> tensor<2x2xf16> { ... }
        // }
        // ```
        // *both* @foo and @bar would be visited.
        walkNestedModulesAndMoveFuncs(nested, mainModule, nestedModuleNames, referenceFuncOp);
    }
}

void UnpackNestedModulesPass::flattenCallSites(mlir::ModuleOp mainModule,
                                               const std::set<NestedSymbolHelper>& nestedModuleNames) {
    mainModule.walk([&](Core::NestedCallOp callOp) {
        _log.trace("Unnestifying the call-site '{0}'", callOp);
        auto calleeSymbol = callOp.getCallee();  // @Module0::..::@ModuleN::@func
        auto calleeNestedSym = NestedSymbolHelper(calleeSymbol);
        // Elements in `nestedModuleNames` set are sorted from longest to smallest prefix
        auto foundPrefix =
                std::find_if(nestedModuleNames.begin(), nestedModuleNames.end(), [&](const NestedSymbolHelper& symbol) {
                    return calleeNestedSym.hasPrefix(symbol);
                });

        if (foundPrefix == nestedModuleNames.end()) {
            return mlir::WalkResult::advance();
        }

        calleeNestedSym.removePrefix(*foundPrefix);
        const auto symbolAttrVal = calleeNestedSym.makeSymbolAttr(mainModule->getContext());
        if (symbolAttrVal == nullptr) {
            _log.trace("Cannot make symbol attribute from `calleeNestedSym`");
            return mlir::WalkResult::advance();
        }

        mlir::OpBuilder builder(callOp);
        mlir::ValueRange newCallResults;

        if (auto flatSymbolAttr = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(symbolAttrVal); flatSymbolAttr != nullptr) {
            const auto newCallOp = builder.create<mlir::func::CallOp>(callOp.getLoc(), flatSymbolAttr,
                                                                      callOp->getResultTypes(), callOp->getOperands());
            newCallResults = newCallOp->getResults();
        } else if (auto symbolAttr = mlir::dyn_cast<mlir::SymbolRefAttr>(symbolAttrVal); symbolAttr != nullptr) {
            const auto newCoreCallOp = builder.create<Core::NestedCallOp>(
                    callOp.getLoc(), symbolAttr, callOp->getResultTypes(), callOp->getOperands());
            newCallResults = newCoreCallOp->getResults();
        } else {
            _log.trace("Failed to construct an attr of type `FlatSymbolRefAttr` or `SymbolRefAttr`");
            return mlir::WalkResult::advance();
        }
        callOp->replaceAllUsesWith(newCallResults);
        callOp->erase();
        return mlir::WalkResult::skip();
    });
}

void UnpackNestedModulesPass::removePipelineOptionsOp(mlir::ModuleOp moduleOp) {
    auto pipelineOptions = moduleOp.getOps<config::PipelineOptionsOp>();
    auto optionsCount = std::distance(pipelineOptions.begin(), pipelineOptions.end());
    VPUX_THROW_WHEN(optionsCount != 1, "No valid count of config.PipelineOptionsOp found {0}", optionsCount);

    auto pipelineOptionsOp = *pipelineOptions.begin();
    pipelineOptionsOp->erase();
}

void UnpackNestedModulesPass::removeResourcesOp(mlir::ModuleOp moduleOp) {
    SmallVector<config::ResourcesOp> resourcesOps(moduleOp.getOps<config::ResourcesOp>());
    for (auto reservedResource : resourcesOps) {
        reservedResource->erase();
    }
}

void UnpackNestedModulesPass::removeNetInfoOp(mlir::ModuleOp moduleOp) {
    SmallVector<net::NetworkInfoOp> netInfoOps(moduleOp.getOps<net::NetworkInfoOp>());
    for (auto netInfo : netInfoOps) {
        netInfo->erase();
    }
}

void UnpackNestedModulesPass::eraseNestedModules(ArrayRef<mlir::ModuleOp> nestedModules) {
    for (auto nested : nestedModules) {
        _log.trace("Erasing nested module '{0}'", nested.getSymName());
        nested.erase();
    }
}

void UnpackNestedModulesPass::moveNetworkInfoToTheTopOfModule(mlir::ModuleOp moduleOp) {
    auto it = llvm::find_if(moduleOp.getOps(), [](auto& op) {
        return mlir::isa<mlir::func::FuncOp, mlir::ModuleOp>(op);
    });
    auto [netInfo, mainFuncOp] = net::getFromModule(moduleOp);
    if (it != moduleOp.getOps().end()) {
        netInfo->moveBefore(&(*it));
    }
}

// Note: `safeRunOnModule()` is assumed to run *only* for the "main" (the most
// top-level) module. Thus, this pass should run exactly once and unnestify
// "everything", leaving just the "main" module.
void UnpackNestedModulesPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    const auto nestedModules = collectTopLevelNestedModules(moduleOp);
    VPUX_THROW_WHEN(
            _nestingMode == Core::NestingMode::EntryPoint && (nestedModules.empty() || nestedModules.size() != 1),
            "No valid count of nested modules named '{0}' with 'EntryPoint' nesting mode found {1}",
            Core::NPU_MODULE_NAME, nestedModules.size());

    std::set<NestedSymbolHelper> nestedModuleNames;
    moveNestedFunctionsIntoMainModule(nestedModules, nestedModuleNames, moduleOp);
    flattenCallSites(moduleOp, nestedModuleNames);
    if (_nestingMode == Core::NestingMode::EntryPoint) {
        moveNetworkInfoToTheTopOfModule(moduleOp);
    }
    eraseNestedModules(nestedModules);
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Core::createUnpackNestedModulesPass(const Logger& log,
                                                                      Core::NestingMode nestingMode) {
    return std::make_unique<UnpackNestedModulesPass>(log, nestingMode);
}
