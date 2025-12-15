//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/barrier_info.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/ring_buffer.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/SmallSet.h>

#include <cassert>
#include <cstdint>
#include <queue>
#include <unordered_set>

namespace vpux {
namespace VPURT {

static constexpr int64_t MAX_DMA_ENGINES = 2;

struct BarrierConfig final {
    mlir::Location loc;

    int64_t realId = -1;
    int64_t nextSameId = -1;
    int64_t producerCount = 0;
    int64_t consumerCount = 0;

    explicit BarrierConfig(mlir::Location loc, int64_t realId = -1): loc(loc), realId(realId) {
    }
};

struct BarrierUserConfig final {
    std::unordered_set<int64_t> waits;
    std::unordered_set<int64_t> posts;
    int64_t startAfter = -1;
    int64_t cleanAfter = -1;
    int64_t virtualDep = -1;
    int64_t count = 0;

    BarrierUserConfig() = default;

    explicit BarrierUserConfig(int64_t virtualDep, int64_t count = 1): virtualDep(virtualDep), count(count) {
    }
};

//
// VirtualDependencyTracker
//

class VirtualDependencyTracker final {
public:
    using Range = std::pair<int64_t, int64_t>;

    struct Dependency final {
        Range consumer;
        Range producer;
    };

public:
    VirtualDependencyTracker(): _deps(1) {
    }

public:
    int64_t add(VPURT::TaskOp taskOp);
    int64_t add(ArrayRef<int64_t> waits, ArrayRef<int64_t> posts);

    int64_t clone(int64_t i);

public:
    bool validId(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _ids.size();
    }
    bool validDep(int64_t i) const {
        return i >= 0 && checked_cast<size_t>(i) < _deps.size();
    }

    int64_t id(int64_t i) const {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }
    int64_t& id(int64_t i) {
        assert(validId(i));
        return _ids[checked_cast<size_t>(i)];
    }

    const Dependency& dep(int64_t i) const {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }
    Dependency& dep(int64_t i) {
        assert(validDep(i));
        return _deps[checked_cast<size_t>(i)];
    }

public:
    void print(Logger log) const;

private:
    SmallVector<int64_t> _ids;
    SmallVector<Dependency> _deps;
};

//
// BarrierWlmHandler
// Class that check if WLM restrictions are satisfied for PID assignment
//
class BarrierWlmHandler {
public:
    BarrierWlmHandler() = delete;
    BarrierWlmHandler(BarrierInfo& barrierInfo);
    BarrierWlmHandler(BarrierInfoTest& barrierInfoTest);

    bool canBarrierReusePidFromBarrier(size_t vid, size_t prevVid);

private:
    BarrierInfo _barrierInfo;

    // Store information about task control map that will be initialized through
    // barrierInfo for given control graph split block index
    size_t _blockIdxOfTaskControlMap;
    std::pair<SmallVector<llvm::BitVector>, size_t> _taskControlMapAndOffset;
};

//
// BarrierSimulator
//

class BarrierSimulator final {
public:
    explicit BarrierSimulator(mlir::func::FuncOp funcOp);
    // wlmFlag is used to enabled PID assignment restrictions for WLM
    explicit BarrierSimulator(mlir::func::FuncOp funcOp, bool wlmFlag, BarrierInfo& barrierInfo);

public:
    bool isDynamicBarriers() const {
        return _isDynamicBarriers;
    }

public:
    // Used to enable PID assignment restrictions for WLM
    void configureForWlm(BarrierInfo& barrierInfo);
    const BarrierConfig& getConfig(mlir::Value bar) const;

public:
    mlir::LogicalResult checkProducerCount(Logger log) const;
    mlir::LogicalResult checkProducerAndConsumerCount(Logger log) const;
    mlir::LogicalResult simulateBarriers(Logger log, std::optional<int64_t> numBarriers = std::nullopt,
                                         std::optional<bool> barrierLegalization = std::nullopt);
    mlir::LogicalResult simulateBarriersForWlmPageApproach(Logger log, int64_t numBarriers);
    mlir::LogicalResult generateBarrierOrderWithSimulation(Logger log, int64_t numBarriers,
                                                           SmallVector<size_t>& virtualToPhysicalBarrierMapping);

    void updateBarrierOrderInIr();
    SmallVector<mlir::DenseSet<VPURT::DeclareVirtualBarrierOp>> getBarrierBatchesToLegalize();
    void linkNextIds(Logger log);
    void calculateBarrierBatchesForParallelTasks();

private:
    void parseBarriers(mlir::Operation* parentOp);
    void parseTasks(mlir::Operation* parentOp);

private:
    enum class Status { Success, Skip, Fail };

    Status processSim(const VirtualDependencyTracker::Dependency& dep, BarrierUserConfig& user, int64_t count,
                      StringRef taskType, int64_t index, SmallVectorImpl<int64_t>& toVirtual,
                      RingBuffer<int64_t>& nextReal, Logger log);

    void calculateBarrierBatchesForParallelTasks(const std::set<int64_t>& activeBarriers);

private:
    // DMA task index {queue index, task index in queue}
    using DmaTaskIdx = std::pair<int64_t, size_t>;

    int64_t _barrierProducerSlotCount = 0;
    int64_t _barrierTotalSlotCount = 0;
    int64_t _availableBarriers = 0;
    int64_t _usedBarriers = 0;

    DenseMap<mlir::Operation*, size_t> _barriersMap;
    SmallVector<BarrierConfig> _barriers;
    bool _isDynamicBarriers = false;
    bool _wlmPidProgramming = false;
    bool _wlmPageApproach = false;

    SmallVector<SmallVector<BarrierUserConfig>> _dmaTasks;
    SmallVector<BarrierUserConfig> _nceTasks;
    SmallVector<BarrierUserConfig> _actTasks;
    SmallVector<BarrierUserConfig> _m2iTasks;

    VirtualDependencyTracker _vdt;

    // barrier legalisation
    SmallVector<std::set<int64_t>> _barrierBatchesToLegalize;
    DenseMap<size_t, mlir::Operation*> _barrierVID;

    // Mapping of barriers (VIDs) to page index (WLM subgraph approach)
    DenseMap<size_t, SmallVector<size_t>> _pageToBarVidVecMap;

    // For each barrier PID (map key) store queue of VID instances
    // This is created only when generateBarrierOrderWithSimulation() is called
    DenseMap<size_t, std::queue<size_t>> _barrierPidToVidInstancesQueueMap;

    // When for barrier simulation VID to PID mapping is provided return order
    // of barrier reprogramming. This order is expected to later be represented in IR
    // This is created only when generateBarrierOrderWithSimulation() is called
    SmallVector<size_t> _barrierProgrammingOrder;

    // Module for managing restrictions on barrier PID reuse in case of WLM
    std::optional<BarrierWlmHandler> _barrierWlmHandlerOpt = std::nullopt;
    BarrierInfo _barrierInfo;
};

}  // namespace VPURT
}  // namespace vpux
