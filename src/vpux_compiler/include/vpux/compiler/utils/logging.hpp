//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Builders.h>

namespace vpux {

void addLogging(mlir::MLIRContext& ctx, Logger& log);

class OpBuilderLogger final : public mlir::OpBuilder::Listener {
public:
    explicit OpBuilderLogger(Logger log): _log(log) {
    }

public:
    void notifyOperationInserted(mlir::Operation* op, mlir::OpBuilder::InsertPoint previous) final;
    void notifyBlockInserted(mlir::Block* block, mlir::Region* previous, mlir::Region::iterator previousIt) final;

private:
    Logger _log;
};

}  // namespace vpux
