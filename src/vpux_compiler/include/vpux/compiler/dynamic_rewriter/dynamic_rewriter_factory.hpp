//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/scope_exit.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/StringMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <functional>
#include <memory>
#include <string>

namespace vpux {

using RewriterFactory = std::function<void(mlir::RewritePatternSet&, mlir::MLIRContext*)>;

//
// RewriterRegistry holds a collection of rewriter factories
//

class RewriterRegistry {
public:
    RewriterRegistry() = default;
    virtual ~RewriterRegistry() = default;
    RewriterRegistry(const RewriterRegistry&) = delete;
    RewriterRegistry& operator=(const RewriterRegistry&) = delete;
    RewriterRegistry(RewriterRegistry&&) = default;
    RewriterRegistry& operator=(RewriterRegistry&&) = default;

    template <typename RewriterClass, typename... Args>
    void registerRewriter(StringRef name, Args&&... args);
    template <typename Func, typename... Args>
    void registerRewriterSet(StringRef setName, Func&& registrationFunc, Args&&... args);

    bool addRewriter(mlir::MLIRContext* ctx, Logger& log, StringRef name, mlir::RewritePatternSet& patterns) const;
    bool addRewriters(mlir::MLIRContext* ctx, Logger& log, const std::vector<std::string>& names,
                      mlir::RewritePatternSet& patterns) const;
    void addAllRewriters(mlir::MLIRContext* ctx, Logger& log, mlir::RewritePatternSet& patterns) const;

    // Adds rewriters from a comma-separated string of rewriter names and/or set names
    // e.g. valid string:
    // single rewriter: "rewriterA"
    // multiple rewriters: "rewriterA,rewriterB"
    // single set: "set1"
    // multiple sets: "set1,set2"
    // mix of rewriters and sets: "rewriterA,set1,rewriterB,set2"
    bool addRewritersFromString(mlir::MLIRContext* ctx, Logger& log, StringRef str,
                                mlir::RewritePatternSet& patterns) const;
    bool addRewriterSet(mlir::MLIRContext* ctx, Logger& log, StringRef setName,
                        mlir::RewritePatternSet& patterns) const;

    std::vector<std::string> getRegisteredRewriters() const;
    const llvm::StringMap<std::vector<std::string>>& getRegisteredRewriterSets() const;

    bool hasRewriter(StringRef name) const;
    bool hasRewriterSet(StringRef setName) const;
    void clear();

private:
    void storeRewriter(StringRef name, RewriterFactory factory);

private:
    llvm::StringMap<RewriterFactory> _rewriters;
    llvm::StringMap<std::vector<std::string>> _rewriterSets;
    // Helper to track the current rewriter set being registered
    // Used to add rewriters to the _rewriterSets automatically
    std::string _currentRewriterSet;
};

// Helper to create a RewriterFactory for a given rewriter class and constructor arguments
// This function does not really add rewriters into pattern set, but creates a lambda function
// that can be called later by such as `addRewriter` to do the actual addition
template <typename RewriterClass, typename... Args>
RewriterFactory createRewriterFactory(Args&&... args) {
    // [args...] creates copies of args every time the lambda is invoked
    // Currently, args are expected to be boolean or Logger which is cheap to copy
    return [args...](mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
        patterns.add<RewriterClass>(ctx, args...);
    };
}

// Register a rewriter with the given name and constructor arguments
// It firstly creates a RewriterFactory and then stores the factory
// and the rewriter name in the hash map
template <typename RewriterClass, typename... Args>
void RewriterRegistry::registerRewriter(StringRef name, Args&&... args) {
    auto factory = createRewriterFactory<RewriterClass>(std::forward<Args>(args)...);
    storeRewriter(name, std::move(factory));
}

// Register a set of rewriters under a set name
// The registrationFunc is expected to call registerRewriter() multiple times
// to register rewriters that belong to this set
template <typename Func, typename... Args>
void RewriterRegistry::registerRewriterSet(StringRef setName, Func&& registrationFunc, Args&&... args) {
    std::string previousSet = std::move(_currentRewriterSet);
    _currentRewriterSet = setName.str();

    VPUX_SCOPE_EXIT {
        _currentRewriterSet = std::move(previousSet);
    };

    VPUX_THROW_WHEN(_rewriters.find(_currentRewriterSet) != _rewriters.end(),
                    "Set name '{0}' conflicts with existing rewriter name", _currentRewriterSet);
    VPUX_THROW_WHEN(_rewriterSets.find(_currentRewriterSet) != _rewriterSets.end(), "Set '{0}' is already registered",
                    _currentRewriterSet);

    // Set up _currentRewriterSet for tracking rewriters added during this set registration
    // This makes sure all rewriters registered within the registrationFunc are added to this set
    try {
        registrationFunc(std::forward<Args>(args)...);
    } catch (const std::exception& error) {
        _rewriterSets.erase(_currentRewriterSet);
        VPUX_THROW("Cannot register set '{0}': {1}", _currentRewriterSet, error.what());
    }
}

//
// RegistryManager manages the global and custom rewriter registries
//

class RegistryManager {
public:
    static RewriterRegistry& getGlobalRegistry();
    static std::unique_ptr<RewriterRegistry> createCustomRegistry();

    // If a custom registry is provided, use it; otherwise, use the global registry
    static RewriterRegistry& getEffectiveRegistry(RewriterRegistry* customRegistry = nullptr);
};

//
// RewriterExecutorInterfaceBase
//

class RewriterExecutorInterfaceBase {
public:
    virtual ~RewriterExecutorInterfaceBase() = default;

    void setRewriterName(StringRef name);

    StringRef getRewriterName() const;

protected:
    // Common rewriter execution logic called from safeRunOnFunc
    mlir::LogicalResult executeRewriters(mlir::MLIRContext* ctx, Logger& log, mlir::func::FuncOp funcOp,
                                         RewriterRegistry* customRegistry = nullptr);

    // Default implementation with dynamic rewriter selection
    virtual mlir::LogicalResult addRewritersToPatterns(mlir::MLIRContext* ctx, Logger& log, RewriterRegistry& registry,
                                                       mlir::RewritePatternSet& patterns);

private:
    std::string _rewriterName;
};

class RewriterExecutorInterface : public RewriterExecutorInterfaceBase {
public:
    virtual ~RewriterExecutorInterface() = default;

protected:
    mlir::LogicalResult addRewritersToPatterns(mlir::MLIRContext* ctx, Logger& log, RewriterRegistry& registry,
                                               mlir::RewritePatternSet& patterns) final;
};

//
// IInitializableRewriter
// Interface for patterns that need initialization with function-level context
//

class IInitializableRewriter {
public:
    virtual ~IInitializableRewriter() = default;

    virtual void initialize(mlir::func::FuncOp funcOp) = 0;
};

}  // namespace vpux
