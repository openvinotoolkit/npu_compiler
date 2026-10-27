//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"

#include <mlir/IR/MLIRContext.h>

namespace vpux {

template <typename Cache, typename Dialect>
Cache& getCache(mlir::MLIRContext* ctx) {
    auto* dialect = ctx->getOrLoadDialect<Dialect>();

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    auto dialectTypeName = llvm::getTypeName<Dialect>().str();
    VPUX_THROW_WHEN(dialect == nullptr, "{0} must be present in the context", dialectTypeName);
#endif

    auto* iface = dialect->template getRegisteredInterface<Cache>();
    assert(iface != nullptr && "The requested cache must be registered in the context");
    return *iface;
}

}  // namespace vpux
