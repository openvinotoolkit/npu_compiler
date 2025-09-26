//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/statistics_collection.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/Operation.h>

#include <memory>

namespace vpux::IE {
#define GEN_PASS_DECL_DUMPSTATISTICSOFIEOPS
#define GEN_PASS_DEF_DUMPSTATISTICSOFIEOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
namespace {
// custom pretty printer
std::string stringifyElemType(mlir::Value value) {
    const auto type = mlir::cast<NDTypeInterface>(value.getType()).getElementType();

    // Note: when quantized type, squash it to a bare minimum to not overly
    // populate the logs. this is especially important for per-axis types that
    // could have an enormous length due to scales / zero-points being printed.
    if (auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(type)) {
        // Note: this is a workaround because quantized types do not properly
        // store storage types
        const auto stringifyStorageType = [&](mlir::Type type) {
            if (qType.isSigned()) {
                return formatv("{0}", type).str();
            }
            return formatv("u{0}", type).str();
        };

        const auto storageType = qType.getStorageType();
        const auto expressedType = qType.getExpressedType();

        // Note: do not print scale / zp to avoid lengthy logs
        if (auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(qType)) {
            const auto axis = perAxisType.getQuantizedDimension();
            return formatv("qtype<{0}:{1}:{2}, per-axis>", stringifyStorageType(storageType), expressedType, axis)
                    .str();
        }

        const auto [scales, zeroPoints] = extractScalesAndZeroPoints(type);
        return formatv("qtype<{0}:{1}, {2:f3}:{3}>", stringifyStorageType(storageType), expressedType, scales.front(),
                       zeroPoints.front())
                .str();
    }
    return formatv("{0}", type).str();
}

struct DumpStatisticsOfIeOpsPass final : IE::impl::DumpStatisticsOfIeOpsBase<DumpStatisticsOfIeOpsPass> {
    DumpStatisticsOfIeOpsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnModule() final;
};

struct ExternalCounterState {
    size_t globalCount{0};
};

struct PercentageCounter final : utils::OpCounter {
    PercentageCounter(const ExternalCounterState& state, const std::string& category,
                      utils::OpCounter::IsOperationSuitable pred,
                      utils::OpCounter::HandleUnrecognizedCounter handleUnrecognized = {})
            : utils::OpCounter(category, std::move(pred), std::move(handleUnrecognized)), _state(state) {
    }

    void printStatistics(const vpux::Logger& log) const override {
        if (_count == 0) {
            return;
        }
        log.info("{0} - {1} ({2:P})", _category, _count, asFraction(_count));
    }

    void printUnrecognizedStatistics(const vpux::Logger& log) const override {
        for (const auto& [name, count] : _unrecognizedCounters) {
            log.info("{0} - {1} ({2:P})", name, count, asFraction(count));
        }
    }

    size_t getCount() const {
        return _count;
    }

private:
    const ExternalCounterState& _state;

    double asFraction(size_t count) const {
        return (static_cast<double>(count) / static_cast<double>(_state.globalCount));
    }
};

template <typename... Args>
std::unique_ptr<utils::OpCounter> makePercentageCounter(Args&&... args) {
    return std::make_unique<PercentageCounter>(std::forward<Args>(args)...);
}

// more specialized counter that also tracks the split by input -> output type
// for a given operation
template <typename Op>
utils::OpCounterTree::Node makeComputationalCounter(const ExternalCounterState& state) {
    const auto isThisOp = [](mlir::Operation* op) {
        return mlir::isa<Op>(op);
    };
    const auto extractTypeConversion = [](mlir::Operation* op) {
        return formatv("{0} -> {1}", stringifyElemType(op->getOperand(0)), stringifyElemType(op->getResult(0))).str();
    };
    return {makePercentageCounter(state, Op::getOperationName().str(), isThisOp, extractTypeConversion), {}};
}

std::string getOpName(mlir::Operation* op) {
    return op->getName().getStringRef().str();
}

utils::OpCounterTree collectCounters(const ExternalCounterState& state) {
    // the hierarchy:
    // IE ops
    //  |- Non-computational ops ("isPureViewOp")
    //  |- Computational ops
    //      |- IE.Convert - special*
    //      |- IE.AvgPool - special
    //
    // *- ops marked special also print types of input and output
    std::vector<utils::OpCounterTree::Node> ieOps;
    ieOps.emplace_back(makePercentageCounter(
            state, "Non-computational",
            [&](mlir::Operation* op) {
                return IE::isPureViewOp(op);
            },
            /* unrecognized op handler = */ getOpName));

    std::vector<utils::OpCounterTree::Node> specialComputationalOps;
    specialComputationalOps.emplace_back(makeComputationalCounter<IE::ConvertOp>(state));
    specialComputationalOps.emplace_back(makeComputationalCounter<IE::AvgPoolOp>(state));
    ieOps.emplace_back(makePercentageCounter(
                               state, "Computational",
                               [&](mlir::Operation* op) {
                                   return !IE::isPureViewOp(op);
                               },
                               /* unrecognized op handler = */ getOpName),
                       std::move(specialComputationalOps));

    std::vector<utils::OpCounterTree::Node> roots;
    roots.emplace_back(makePercentageCounter(
                               state, IE::IEDialect::getDialectNamespace().str(),
                               [&](mlir::Operation* op) {
                                   // Note: ignore meta-info ops (e.g. config.MemoryResource)
                                   return mlir::isa<IE::IEDialect>(op->getDialect()) &&
                                          !op->hasTrait<config::ResourceOpInterface::Trait>();
                               },
                               /* unrecognized op handler = */ getOpName),
                       std::move(ieOps));
    return utils::OpCounterTree(std::move(roots));
}

void DumpStatisticsOfIeOpsPass::safeRunOnModule() {
    if (!_log.isActive(LogLevel::Info)) {
        // Note: pass logs to INFO - when INFO logging is disabled, there is
        // nothing to be logged -> nothing to do in the pass
        return;
    }

    ExternalCounterState state;  // Note: required to report percentages
    auto counters = collectCounters(state);

    auto moduleOp = getOperation();
    moduleOp->walk([&](mlir::Operation* op) {
        utils::AddOpRecordVisitor collector(op);
        counters.apply(collector);
    });

    // Note: counters here are *always* percentage counters
    state.globalCount = static_cast<PercentageCounter*>(counters.roots().back().data().get())->getCount();

    _log.info("IE dialect statistics:");
    utils::PrintOpRecordVisitor printer(_log);
    counters.apply(printer);
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createDumpStatisticsOfIeOpsPass(Logger log) {
    return std::make_unique<DumpStatisticsOfIeOpsPass>(log);
}
