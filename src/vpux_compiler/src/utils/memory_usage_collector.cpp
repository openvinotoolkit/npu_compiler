//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/memory_usage_collector.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/utils/core/memory_usage.hpp"

#include <mlir/Pass/PassInstrumentation.h>

using namespace vpux;

namespace {

//
// CollectMemoryUsageInstrumentation
//

class CollectMemoryUsageInstrumentation final : public mlir::PassInstrumentation {
public:
    explicit CollectMemoryUsageInstrumentation(Logger log): _log(log) {
        _log.setName("memory-usage-collector");
        _peakMemUsageKB = getPeakMemoryUsage().count();
    }

    void runAfterPass(mlir::Pass* pass, mlir::Operation*) override {
        auto peakMemUsageKB = getPeakMemoryUsage().count();
        if (auto delta = peakMemUsageKB - _peakMemUsageKB) {
            _log.info("Peak memory usage after '{0}' pass: {1} kB, increase {2} kB", pass->getName(), peakMemUsageKB,
                      delta);
            _peakMemUsageKB = peakMemUsageKB;
        }
    }

private:
    Logger _log;
    int64_t _peakMemUsageKB;
};

}  // namespace

void vpux::addMemoryUsageCollector(mlir::PassManager& pm, Logger log) {
    auto instr = std::make_unique<CollectMemoryUsageInstrumentation>(log);
    pm.addInstrumentation(std::move(instr));
}
