//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"

#include <mlir/Pass/Pass.h>

namespace vpux {

std::unique_ptr<mlir::Pass> createDynamicRewriterExecutorPass(Logger log = Logger::global());
// For unit test purposes only
std::unique_ptr<mlir::Pass> createDynamicRewriterExecutorPass(StringRef rewriterName, Logger log = Logger::global());

void registerDynamicRewriterExecutorPass();

}  // namespace vpux
