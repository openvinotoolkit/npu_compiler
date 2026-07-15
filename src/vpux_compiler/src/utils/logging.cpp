//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

using namespace vpux;

//
// Context logging
//

void vpux::addLogging(mlir::MLIRContext& ctx, Logger& log) {
    auto& diagEngine = ctx.getDiagEngine();

    diagEngine.registerHandler([log](mlir::Diagnostic& diag) -> mlir::LogicalResult {
        const auto severity = diag.getSeverity();
        const auto msgLevel = [severity]() -> LogLevel {
            switch (severity) {
            case mlir::DiagnosticSeverity::Note:
            case mlir::DiagnosticSeverity::Remark:
                return LogLevel::Info;

            case mlir::DiagnosticSeverity::Warning:
                return LogLevel::Warning;

            case mlir::DiagnosticSeverity::Error:
                return LogLevel::Error;
            default:
                return LogLevel::None;
            }
        }();

#ifdef VPUX_DEVELOPER_BUILD
        const auto loc = diag.getLocation();
        log.addEntry(msgLevel, "Got Diagnostic at {0} : {1}", loc, diag);
#else
        log.addEntry(msgLevel, "{0}", diag);
#endif

        // Suppress the default handler
        return mlir::success();
    });
}

//
// OpBuilderLogger
//

void vpux::OpBuilderLogger::notifyOperationInserted(mlir::Operation* op, mlir::OpBuilder::InsertPoint previous) {
    (void)previous;

    _log.trace("Add new Operation {0}", op->getLoc());
}

void vpux::OpBuilderLogger::notifyBlockInserted(mlir::Block* block, mlir::Region* previous,
                                                mlir::Region::iterator previousIt) {
    (void)previous;
    (void)previousIt;

    if (auto* parent = block->getParentOp()) {
        _log.trace("Add new Block for Operation {0}", parent->getLoc());
    } else {
        _log.trace("Add new Block without parent Operation");
    }
}
