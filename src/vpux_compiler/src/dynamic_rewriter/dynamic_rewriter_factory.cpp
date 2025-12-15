//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace vpux;
namespace vpux {

//
// RewriterRegistry implementation
//

void RewriterRegistry::storeRewriter(StringRef name, RewriterFactory factory) {
    VPUX_THROW_WHEN(_rewriterSets.find(name) != _rewriterSets.end(),
                    "Rewriter name '{0}' conflicts with existing set name", name.str());
    VPUX_THROW_WHEN(factory == nullptr, "RewriterFactory is null for rewriter '{0}'", name.str());
    _rewriters[name] = std::move(factory);

    // If currently registering a set, add this rewriter to that set
    if (!_currentRewriterSet.empty()) {
        _rewriterSets[_currentRewriterSet].push_back(name.str());
    }
}

bool RewriterRegistry::addRewriter(mlir::MLIRContext* ctx, Logger&, StringRef name,
                                   mlir::RewritePatternSet& patterns) const {
    auto it = _rewriters.find(name);
    if (it != _rewriters.end()) {
        it->second(patterns, ctx);
        return true;
    }
    return false;
}

bool RewriterRegistry::addRewriters(mlir::MLIRContext* ctx, Logger& log, const std::vector<std::string>& names,
                                    mlir::RewritePatternSet& patterns) const {
    bool allSucceeded = true;
    for (const auto& name : names) {
        if (!addRewriter(ctx, log, name, patterns)) {
            log.error("Rewriter '{0}' not found in registry", name);
            allSucceeded = false;
        }
    }
    return allSucceeded;
}

void RewriterRegistry::addAllRewriters(mlir::MLIRContext* ctx, Logger& log, mlir::RewritePatternSet& patterns) const {
    for (const auto& [name, factory] : _rewriters) {
        VPUX_THROW_WHEN(factory == nullptr, "RewriterFactory is null for rewriter '{0}'", name.str());
        factory(patterns, ctx);
        log.trace("Added rewriter: {0}", name.str());
    }
}

bool RewriterRegistry::addRewritersFromString(mlir::MLIRContext* ctx, Logger& log, StringRef str,
                                              mlir::RewritePatternSet& patterns) const {
    if (str.empty()) {
        return true;
    }

    const auto names = vpux::splitAndTrimStringByDelimiter(str.str());
    bool allSucceeded = true;
    for (const auto& name : names) {
        // Try set first, then rewriter
        if (!addRewriterSet(ctx, log, name, patterns) && !addRewriter(ctx, log, name, patterns)) {
            log.error("Neither rewriter nor set '{0}' found in registry", name);
            allSucceeded = false;
        }
    }

    return allSucceeded;
}

std::vector<std::string> RewriterRegistry::getRegisteredRewriters() const {
    std::vector<std::string> rewriters;
    rewriters.reserve(_rewriters.size());
    for (const auto& [name, _] : _rewriters) {
        rewriters.push_back(name.str());
    }
    return rewriters;
}

bool RewriterRegistry::hasRewriter(StringRef name) const {
    return _rewriters.find(name) != _rewriters.end();
}

void RewriterRegistry::clear() {
    _rewriters.clear();
    _rewriterSets.clear();
}

bool RewriterRegistry::addRewriterSet(mlir::MLIRContext* ctx, Logger& log, StringRef setName,
                                      mlir::RewritePatternSet& patterns) const {
    auto it = _rewriterSets.find(setName);
    if (it == _rewriterSets.end()) {
        return false;
    }

    bool allSucceeded = true;
    for (const auto& rewriterName : it->second) {
        if (!addRewriter(ctx, log, rewriterName, patterns)) {
            log.error("Rewriter '{0}' in set '{1}' not found in registry", rewriterName, setName.str());
            allSucceeded = false;
        }
    }
    return allSucceeded;
}

bool RewriterRegistry::hasRewriterSet(StringRef setName) const {
    return _rewriterSets.find(setName) != _rewriterSets.end();
}

const llvm::StringMap<std::vector<std::string>>& RewriterRegistry::getRegisteredRewriterSets() const {
    return _rewriterSets;
}

//
// RegistryManager implementation
//

RewriterRegistry& RegistryManager::getGlobalRegistry() {
    static RewriterRegistry globalRegistry;
    return globalRegistry;
}

std::unique_ptr<RewriterRegistry> RegistryManager::createCustomRegistry() {
    return std::make_unique<RewriterRegistry>();
}

RewriterRegistry& RegistryManager::getEffectiveRegistry(RewriterRegistry* customRegistry) {
    return customRegistry ? *customRegistry : getGlobalRegistry();
}

//
// RewriterExecutorInterfaceBase implementation
//

void RewriterExecutorInterfaceBase::setRewriterName(StringRef name) {
    _rewriterName = name.str();
}

StringRef RewriterExecutorInterfaceBase::getRewriterName() const {
    return _rewriterName;
}

void displayRegisteredRewriters(const RewriterRegistry& registry, Logger& log) {
    auto registeredRewriterSets = registry.getRegisteredRewriterSets();
    if (!registeredRewriterSets.empty()) {
        log.debug("Registered rewriter sets:");
        for (const auto& [rewriterSet, rewriters] : registeredRewriterSets) {
            log.debug("Rewriter set: {0}", rewriterSet.str());
            log.nest(1).debug("Rewriters:");
            for (const auto& rewriter : rewriters) {
                log.nest(2).debug("- {0}", rewriter);
            }
        }
    } else {
        auto registeredRewriters = registry.getRegisteredRewriters();
        log.debug("Registered rewriters:");
        for (const auto& rewriter : registeredRewriters) {
            log.nest(1).debug("- {0}", rewriter);
        }
    }
}

mlir::LogicalResult RewriterExecutorInterfaceBase::addRewritersToPatterns(mlir::MLIRContext* ctx, Logger& log,
                                                                          RewriterRegistry& registry,
                                                                          mlir::RewritePatternSet& patterns) {
    if (_rewriterName.empty()) {
        log.error("Rewriter name must be specified");
        displayRegisteredRewriters(registry, log);
        return mlir::failure();
    }

    if (_rewriterName == "all") {
        log.debug("Adding all registered rewriters");
        registry.addAllRewriters(ctx, log, patterns);
    } else {
        log.debug("Adding rewriters from string: {0}", _rewriterName);
        if (!registry.addRewritersFromString(ctx, log, _rewriterName, patterns)) {
            log.error("Failed to add rewriter(s) from string: {0}", _rewriterName);
            displayRegisteredRewriters(registry, log);
            return mlir::failure();
        }
    }

    return mlir::success();
}

mlir::LogicalResult RewriterExecutorInterfaceBase::executeRewriters(mlir::MLIRContext* ctx, Logger& log,
                                                                    mlir::func::FuncOp funcOp,
                                                                    RewriterRegistry* customRegistry) {
    mlir::RewritePatternSet patterns(ctx);

    auto& registry = RegistryManager::getEffectiveRegistry(customRegistry);
    auto registeredRewriters = registry.getRegisteredRewriters();

    if (registeredRewriters.empty()) {
        log.error("No rewriters registered in the registry");
        return mlir::failure();
    }

    if (mlir::failed(addRewritersToPatterns(ctx, log, registry, patterns))) {
        return mlir::failure();
    }

    // Initialize patterns that need function context
    for (auto& pattern : patterns.getNativePatterns()) {
        if (auto* rewriter = dynamic_cast<IInitializableRewriter*>(pattern.get())) {
            log.trace("Initializing rewriter: {0}", pattern->getDebugName());
            rewriter->initialize(funcOp);
        }
    }

    auto greedyRewriteConfig = getDefaultGreedyRewriteConfig();
    if (mlir::failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), greedyRewriteConfig))) {
        log.error("Failed to apply rewriters");
        return mlir::failure();
    }
    return mlir::success();
}

//
// RewriterExecutorInterface implementation
//

mlir::LogicalResult RewriterExecutorInterface::addRewritersToPatterns(mlir::MLIRContext* ctx, Logger& log,
                                                                      RewriterRegistry& registry,
                                                                      mlir::RewritePatternSet& patterns) {
    if (getRewriterName().empty()) {
        setRewriterName("all");
    }
    return RewriterExecutorInterfaceBase::addRewritersToPatterns(ctx, log, registry, patterns);
}

}  // namespace vpux
