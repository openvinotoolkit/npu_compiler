//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/constant_tracing.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Debug/ExecutionContext.h>
#include <mlir/IR/MLIRContext.h>
#include <memory>

namespace vpux::Const {

std::string gatherTrace();

// Cache for storing call stacks for constant transformations. Only used in developer builds,
// as call stack gathering can be expensive. API function getSpecificCallStack() allows to query
// the call stack for a specific constant transformation.
// The mapping can be iterated as well to retrieve all transformations.
// Note: this cache is shared across the entire MLIRContext, so it can be used to correlate call
// stacks between different constants and transformations.
class CallStackCache final : public mlir::DialectInterface::Base<CallStackCache> {
public:
    using TransformationTy = std::tuple</* Transformation */ Const::TransformAttrInterface, /* Trace */ std::string>;
    using CallStackTy =
            std::unordered_map</* Constant */ TraceId, /* Transformation list */ std::vector<TransformationTy>>;

private:
    CallStackTy _callStack;
    mutable std::mutex _callStackMutex;

public:
    // required by MLIR's internal type-id infrastructure:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CallStackCache)

    CallStackCache(mlir::Dialect* dialect): Base(dialect) {
    }

    CallStackTy& getCallStack();
    std::mutex& callStackCacheMutex();
    std::string getSpecificCallStack(mlir::ElementsAttr baseContent, Const::TransformAttrInterface transformation);
};

struct CallStackObserver : public mlir::tracing::ExecutionContext::Observer {
    void beforeExecute(const mlir::tracing::ActionActiveStack*, mlir::tracing::Breakpoint*, bool) final;
    void afterExecute(const mlir::tracing::ActionActiveStack*) final;
};

}  // namespace vpux::Const
