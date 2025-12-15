//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/IR/dialect.hpp"

#include <mlir/IR/MLIRContext.h>

#include <cassert>

using namespace vpux::config;

namespace {

ConfigCache* getRegisteredInterface(mlir::MLIRContext* context) {
    auto dialect = context->getOrLoadDialect<ConfigDialect>();
    assert(dialect != nullptr && "ConfigDialect must be present in the context");

    auto registeredInterface = dialect->getRegisteredInterface<ConfigCache>();
    assert(registeredInterface != nullptr && "The requested ConfigCache must be registered in the context");
    return registeredInterface;
}

}  // namespace

void vpux::config::setNPUConstraints(mlir::MLIRContext* context, const NPUConstraints& constraint) {
    auto registeredInterface = getRegisteredInterface(context);
    registeredInterface->setConstraints(constraint);
}

const NPUConstraints& vpux::config::getNPUConstraints(mlir::MLIRContext* context) {
    auto registeredInterface = getRegisteredInterface(context);
    return registeredInterface->getConstraints();
}
