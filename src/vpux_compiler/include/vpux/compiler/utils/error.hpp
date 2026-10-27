//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace vpux {

template <typename... Args>
mlir::LogicalResult errorAt(mlir::Location loc, StringLiteral format, Args&&... args) {
    return printTo(mlir::emitError(loc), format, std::forward<Args>(args)...);
}

template <typename... Args>
mlir::LogicalResult errorAt(mlir::Operation* op, StringLiteral format, Args&&... args) {
    return printTo(op->emitError(), format, std::forward<Args>(args)...);
}

template <typename... Args>
mlir::LogicalResult matchFailed(Logger log, mlir::RewriterBase& rewriter, mlir::Operation* op, StringLiteral format,
                                Args&&... args) {
    log.trace(format, args...);
    return rewriter.notifyMatchFailure(op, formatv(format.data(), std::forward<Args>(args)...));
}

}  // namespace vpux
