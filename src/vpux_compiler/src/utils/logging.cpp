//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/options.hpp"

using namespace vpux;

//
// Context logging
//

void vpux::addLogging(mlir::MLIRContext& ctx, Logger log) {
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

        const auto loc = diag.getLocation();
        log.addEntry(msgLevel, "Got Diagnostic at {0} : {1}", loc, diag);

        // Propagate diagnostic to following handlers
        return mlir::failure();
    });
}

//
// PassLogging
//

namespace {

class PassLogging final : public mlir::PassInstrumentation {
public:
    explicit PassLogging(Logger log): _log(log) {
    }

    void runBeforePipeline(std::optional<mlir::OperationName> name, const PipelineParentInfo&) final {
        if (name.has_value()) {
            _log.debug("Start Pass Pipeline {0}", *name);
        }
    }

    void runAfterPipeline(std::optional<mlir::OperationName> name, const PipelineParentInfo&) final {
        if (name.has_value()) {
            _log.debug("End Pass Pipeline {0}", *name);
        }
    }

    void runBeforePass(mlir::Pass* pass, mlir::Operation* op) final {
        if (pass->getName() != "mlir::detail::OpToOpPassAdaptor") {
            _log.debug("Start Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
        if (pass->getName() != "mlir::detail::OpToOpPassAdaptor") {
            _log.debug("End Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runAfterPassFailed(mlir::Pass* pass, mlir::Operation* op) override {
        if (pass->getName() == "mlir::detail::OpToOpPassAdaptor") {
            return;
        }
        auto module = op->getParentOfType<mlir::ModuleOp>();
        if (config::getWorkloadManagementStatus(module) == WorkloadManagementStatus::FAILED) {
            _log.warning("WLM Failed Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        } else {
            _log.error("Failed Pass {0} on Operation {1}", pass->getName(), op->getLoc());
        }
    }

    void runBeforeAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) override {
        _log.trace("Start Analysis {0} on Operation {1}", name, op->getLoc());
    }

    void runAfterAnalysis(StringRef name, mlir::TypeID, mlir::Operation* op) override {
        _log.trace("End Analysis {0} on Operation {1}", name, op->getLoc());
    }

private:
    Logger _log;
};

}  // namespace

void vpux::addLogging(mlir::PassManager& pm, Logger log) {
    pm.addInstrumentation(std::make_unique<PassLogging>(log));
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
