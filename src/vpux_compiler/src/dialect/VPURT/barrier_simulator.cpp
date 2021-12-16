//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPURT/barrier_simulator.hpp"

#include "vpux/compiler/dialect/VPU/attributes.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/string_ref.hpp"

using namespace vpux;

//
// Virtual ID manipulation
//

namespace {

constexpr StringLiteral virtualIdAttrName = "VPURT.virtualId";

int64_t getVirtualId(mlir::Operation* op) {
    const auto attr = op->getAttr(virtualIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
    VPUX_THROW_WHEN(attr == nullptr, "The barrier operation at '{0}' doesn't have attrribute '{1}'", op->getLoc(),
                    virtualIdAttrName);

    return attr.getValue().getSExtValue();
}

}  // namespace

//
// VirtualDependencyTracker
//

int64_t vpux::VPURT::VirtualDependencyTracker::add(VPURT::TaskOp taskOp) {
    auto extract = [&](Range& range, mlir::ValueRange barriers) {
        range.first = _ids.size();

        for (const auto bar : barriers) {
            const auto vid = getVirtualId(bar.getDefiningOp());
            _ids.push_back(vid);
            ++range.second;
        }
    };

    Dependency d;
    extract(d.consumer, taskOp.waitBarriers());
    extract(d.producer, taskOp.updateBarriers());

    if (d.consumer.second != 0 || d.producer.second != 0) {
        _deps.push_back(d);
        return checked_cast<int64_t>(_deps.size() - 1);
    }

    return 0;
}

int64_t vpux::VPURT::VirtualDependencyTracker::add(ArrayRef<int64_t> waits, ArrayRef<int64_t> posts) {
    auto extract = [&](Range& range, ArrayRef<int64_t> barriers) {
        range.first = _ids.size();
        range.second = barriers.size();
        _ids.insert(_ids.end(), barriers.begin(), barriers.end());
    };

    if (!waits.empty() || !posts.empty()) {
        Dependency d;
        extract(d.consumer, waits);
        extract(d.producer, posts);

        _deps.push_back(d);
        return checked_cast<int64_t>(_deps.size() - 1);
    }

    return 0;
}

int64_t vpux::VPURT::VirtualDependencyTracker::clone(int64_t i) {
    _deps.push_back(dep(i));
    return checked_cast<int64_t>(_deps.size() - 1);
}

void vpux::VPURT::VirtualDependencyTracker::print(Logger log) const {
    log.trace("{0} dependencies:", _deps.size());

    for (const auto& p : _deps | indexed) {
        const auto& d = p.value();
        const auto ind = p.index();

        log.nest().trace("Dependency[{0}]: consumer: {1} - {2}, producer: {3} - {4}", ind, d.consumer.first,
                         d.consumer.second, d.producer.first, d.producer.second);
    }
}

//
// BarrierSimulator
//

namespace {

const EnumMap<VPU::ArchKind, int64_t> MAX_BARRIERS_PER_INFERENCE = {
        {VPU::ArchKind::KMB, 64 / 2},  // half barries are used (runtime limitation)
        {VPU::ArchKind::TBH, 64 / 2},  // half barries are used (runtime limitation)
        {VPU::ArchKind::MTL, 64},      //
        {VPU::ArchKind::LNL, 64},      //
};

int64_t getNumAvailableBarriers(mlir::Operation* parentOp) {
    const auto arch = VPU::getArch(parentOp);

    auto module = parentOp->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto nceAttr = VPU::ExecutorKindAttr::get(parentOp->getContext(), VPU::ExecutorKind::NCE);
    auto nceResOp = resOp.getExecutor(nceAttr);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE Executor information");

    const auto numClusters = nceResOp.count();

    const auto maxNumClustersForArch = VPU::getMaxDPUClusterNum(module);
    VPUX_THROW_UNLESS(maxNumClustersForArch != 0, "Failed to get maxNumClustersForArch");

    const auto barIt = MAX_BARRIERS_PER_INFERENCE.find(arch);
    VPUX_THROW_WHEN(barIt == MAX_BARRIERS_PER_INFERENCE.end(), "Unsupported VPU architecture '{0}'", arch);

    const auto maxBarriersPerInference = barIt->second;

    const auto barriersPerCluster = maxBarriersPerInference / maxNumClustersForArch;
    const auto maxNumBarriers = std::min(maxBarriersPerInference, barriersPerCluster * numClusters);

    return maxNumBarriers;
}

}  // namespace

vpux::VPURT::BarrierSimulator::BarrierSimulator(mlir::Operation* parentOp)
        : _log(Logger::global().nest("BarrierSimulator", 0)) {
    _availableBarriers = getNumAvailableBarriers(parentOp);

    assignVirtualIds(parentOp);
    parseBarriers(parentOp);
    parseTasks(parentOp);
    cleanUpVirtualIds(parentOp);
}

const VPURT::BarrierConfig& vpux::VPURT::BarrierSimulator::getConfig(mlir::Value bar) const {
    const auto it = _barriersMap.find(bar.getDefiningOp());
    VPUX_THROW_WHEN(it == _barriersMap.end(), "Barrier at '{0}' was not covered by BarrierSimulator", bar.getLoc());
    return _barriers[it->second];
}

void vpux::VPURT::BarrierSimulator::assignVirtualIds(mlir::Operation* parentOp) {
    _log.trace("Assign virtual IDs to the barriers");

    int64_t vid = 0;

    parentOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            VPUX_THROW_WHEN(vid > std::numeric_limits<short>::max(), "Barrier virtual id '{0}' is too large", vid);

            op->setAttr(virtualIdAttrName, getIntAttr(op->getContext(), vid++));
        }
    });
}

void vpux::VPURT::BarrierSimulator::cleanUpVirtualIds(mlir::Operation* parentOp) {
    parentOp->walk([&](mlir::Operation* op) {
        if (mlir::isa<VPURT::DeclareVirtualBarrierOp, VPURT::ConfigureBarrierOp>(op)) {
            op->removeAttr(virtualIdAttrName);
        }
    });
}

void vpux::VPURT::BarrierSimulator::parseBarriers(mlir::Operation* parentOp) {
    _log.trace("Parse barriers");

    _isDynamicBarriers = false;

    parentOp->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        _isDynamicBarriers = true;

        const auto vid = getVirtualId(barrierOp);
        VPUX_THROW_UNLESS(vid == checked_cast<int64_t>(_barriers.size()),
                          "Got wrong virtual ID for the barrier at '{0}'", barrierOp->getLoc());

        _barriersMap.insert({barrierOp, _barriers.size()});
        _barriers.emplace_back(BarrierConfig());
    });

    parentOp->walk([&](VPURT::ConfigureBarrierOp barrierOp) {
        VPUX_THROW_WHEN(_isDynamicBarriers, "Can't have both dynamic and static barriers at the same time");

        const auto vid = getVirtualId(barrierOp);
        VPUX_THROW_UNLESS(vid == checked_cast<int64_t>(_barriers.size()),
                          "Got wrong virtual ID for the barrier at '{0}'", barrierOp->getLoc());

        _barriersMap.insert({barrierOp, _barriers.size()});
        _barriers.emplace_back(barrierOp.id());

        _usedBarriers = std::max(_usedBarriers, barrierOp.id() + 1);
    });
}

void vpux::VPURT::BarrierSimulator::parseTasks(mlir::Operation* parentOp) {
    _log.trace("Parse tasks");

    const auto updateBarrierConfigs = [&](VPURT::TaskOp taskOp, const int64_t count = 1) {
        for (const auto bar : taskOp.waitBarriers()) {
            const auto v = getVirtualId(bar.getDefiningOp());
            _barriers[v].consumerCount += count;
        }

        for (const auto bar : taskOp.updateBarriers()) {
            const auto v = getVirtualId(bar.getDefiningOp());
            _barriers[v].producerCount += count;
        }
    };

    parentOp->walk([&](VPURT::TaskOp taskOp) {
        auto* wrappedTaskOp = taskOp.getInnerTaskOp().getOperation();

        const auto virtualDep = _vdt.add(taskOp);

        switch (taskOp.getExecutorKind()) {
        case VPU::ExecutorKind::DMA_NN: {
            int64_t port = 0;
            if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(wrappedTaskOp)) {
                port = dmaOp.port();
            } else if (auto compressedDmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(wrappedTaskOp)) {
                port = compressedDmaOp.port();
            } else {
                VPUX_THROW("Could not cast to DMA task");
            }

            VPUX_THROW_UNLESS(port < MAX_DMA_ENGINES,
                              "NNDMAOp port value '{0}' larger than maximum number of engines '{1}'", port,
                              MAX_DMA_ENGINES);

            _dmaTasks[port].emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        case VPU::ExecutorKind::NCE: {
            auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(wrappedTaskOp);
            VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");

            _nceTasks.emplace_back(virtualDep, nceOp.getNumVariants());
            updateBarrierConfigs(taskOp, nceOp.getNumVariants());
            break;
        }

        case VPU::ExecutorKind::SHAVE_ACT: {
            _actTasks.emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        case VPU::ExecutorKind::SHAVE_UPA: {
            _upaTasks.emplace_back(virtualDep);
            updateBarrierConfigs(taskOp);
            break;
        }

        default:
            VPUX_THROW("Unsupported executor '{0}'", taskOp.getExecutorKind());
        }
    });
}

//
// The limitation is not related to HW capabilities or FIFO depth, but to the fact that the runtime needs to know when a
// workload is completed, in order to replace it with another one in NN CMX.
//
// Since there's no other efficient feedback mechanism from DPU/SNN to LNN, LNN monitors the barrier production of DPU
// tasks and recycles the storage when the corresponding barrier gets produced. The limitation comes from how many
// invariants/variants can be stored in NN CMX at the same time. For single cluster inferences these counts are 64/512,
// while for 4-cluster inferences 128/512. If too many invariants/variants contribute to the same barrier, the runtime
// will not receive the confirmation that it may recycle the storage to bring in the next workloads, hence the deadlock.
//
// Since the storage area is double buffered, and the workloads in question may start at any index in the buffer, it's
// only safe for at most <storage_size / 2 + 1> consecutive invariants/variants to produce the same barrier. So finally,
// the limits are:
//
// On single cluster:
//   32 + 1 invariants
//   256 + 1 variants
// On 4 clusters:
//   64 + 1 invariants
//   256 + 1 variants
//
void vpux::VPURT::BarrierSimulator::checkProducerCount() const {
    static constexpr int64_t MAX_PRODUCER_COUNT = 256;

    for (auto vid : irange(_barriers.size())) {
        const auto producerCount = _barriers[vid].producerCount;
        VPUX_THROW_WHEN(producerCount > MAX_PRODUCER_COUNT, "Barrier {0} has {1} barriers (max {2})", vid,
                        producerCount, MAX_PRODUCER_COUNT);
    }
}

void vpux::VPURT::BarrierSimulator::simulateBarriers() {
    _log.trace("Simulating barrier flow");

    SmallVector<int64_t> toVirtual(_availableBarriers, -1);

    RingBuffer<int64_t> nextReal(toVirtual.size());
    while (!nextReal.full()) {
        nextReal.push(checked_cast<int64_t>(nextReal.size()));
    }

    size_t bar = 0;
    size_t dma[MAX_DMA_ENGINES] = {0};
    size_t dpu = 0;
    size_t act = 0;
    size_t upa = 0;
    bool progressed = false;

    for (; bar < _barriers.size() || dma[0] < _dmaTasks[0].size() || dma[1] < _dmaTasks[1].size() ||
           dpu < _nceTasks.size() || act < _actTasks.size() || upa < _upaTasks.size();
         progressed = false) {
        _log.nest(1).trace(
                "DMA: {0} / {1}, {2} / {3}; DPU: {4} / {5}; ACT: {6} / {7}; UPA: {8} / {9}; BAR: {10} / {11}", dma[0],
                _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), dpu, _nceTasks.size(), act, _actTasks.size(), upa,
                _upaTasks.size(), bar, _barriers.size());

        // Static vs dynamic barriers need a different loop exit condition
        const auto barrierCheck = [&]() {
            if (_isDynamicBarriers) {
                return !nextReal.empty();
            } else {
                return toVirtual[_barriers[bar].realId] == -1;
            }
        };

        // Map new barriers
        _log.nest(2).trace("Map new barriers");
        for (; bar < _barriers.size() && barrierCheck(); ++bar, progressed = true) {
            int64_t real = 0;

            if (_isDynamicBarriers) {
                real = nextReal.front();
                nextReal.pop();

                _barriers[bar].realId = real;
            } else {
                real = _barriers[bar].realId;
            }

            toVirtual[checked_cast<size_t>(real)] = checked_cast<int64_t>(bar);

            _log.nest(3).trace("VID[{0}]: PID {1}, producers {2}, consumers {3}", bar, real,
                               _barriers[bar].producerCount, _barriers[bar].consumerCount);
        }

        // Process DMAs
        _log.nest(2).trace("Process DMAs");
        for (int64_t e = 0; e < MAX_DMA_ENGINES; ++e) {
            for (; dma[e] < _dmaTasks[e].size(); ++dma[e], progressed = true) {
                auto& dt = _dmaTasks[e][dma[e]];

                const auto status =
                        processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "DMA", dma[e], toVirtual, nextReal);

                VPUX_THROW_WHEN(status == Status::Fail, "Barrier simulation failed");

                if (status == Status::Skip) {
                    break;
                }

                _log.nest(3).trace("DMA[{0}][{1}]: wait: {2}, post: {3}", e, dma[e], dt.waitMask, dt.postMask);
            }
        }

        // Process DPUs
        _log.nest(2).trace("Process DPUs");
        for (; dpu < _nceTasks.size(); ++dpu, progressed = true) {
            auto& dt = _nceTasks[dpu];

            const auto status = processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "DPU", dpu, toVirtual, nextReal);

            VPUX_THROW_WHEN(status == Status::Fail, "Barrier simulation failed, see log for details");

            if (status == Status::Skip) {
                break;
            }

            _log.nest(3).trace("DPU[{0}]: wait: {1}, post: {2}, variants: {3}", dpu, dt.waitMask, dt.postMask,
                               dt.count);
        }

        // Process Act Kernels
        _log.nest(2).trace("Process DPUs");
        for (; act < _actTasks.size(); ++act, progressed = true) {
            auto& dt = _actTasks[act];

            const auto status = processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "ACT", act, toVirtual, nextReal);

            VPUX_THROW_WHEN(status == Status::Fail, "Barrier simulation failed");

            if (status == Status::Skip) {
                break;
            }

            _log.nest(3).trace("ACT[{0}]: wait: {1}, post: {2}", act, dt.waitMask, dt.postMask);
        }

        // Process UPA tasks
        _log.nest(2).trace("Process UPA tasks");
        for (; upa < _upaTasks.size(); ++upa, progressed = true) {
            auto& dt = _upaTasks[upa];

            const auto status = processSim(_vdt.dep(dt.virtualDep), dt, dt.count, "UPA", upa, toVirtual, nextReal);

            VPUX_THROW_WHEN(status == Status::Fail, "Barrier simulation failed");

            if (status == Status::Skip) {
                break;
            }

            _log.nest(3).trace("UPA[{0}]: wait: {1}, post: {2}", upa, dt.waitMask, dt.postMask);
        }

        if (!progressed) {
            _log.error("Barrier simulation blocked at DMA: {0} / {1}, {2} / {3}; DPU: {4} / {5}; ACT: {6} / {7}; UPA: "
                       "{8} / {9}; BAR: {10} / {11}",
                       dma[0], _dmaTasks[0].size(), dma[1], _dmaTasks[1].size(), dpu, _nceTasks.size(), act,
                       _actTasks.size(), upa, _upaTasks.size(), bar, _barriers.size());

            for (size_t b = 0; b < bar; ++b) {
                if (_barriers[b].producerCount != 0 || _barriers[b].consumerCount != 0)
                    _log.error("Barrier {0} mapped to real {1} with remaining producers: {2}, consumers: {3}", b,
                               _barriers[b].realId, _barriers[b].producerCount, _barriers[b].consumerCount);
            }

            VPUX_THROW("Barrier simulation failed, see log for details");
        }
    }
}

VPURT::BarrierSimulator::Status vpux::VPURT::BarrierSimulator::processSim(
        const VirtualDependencyTracker::Dependency& dep, BarrierUserConfig& user, int64_t count, StringRef taskType,
        int64_t index, SmallVectorImpl<int64_t>& toVirtual, RingBuffer<int64_t>& nextReal) {
    auto barrierCheck = [&](bool dynamicCond, bool staticCond) {
        return dynamicCond && (staticCond || !_isDynamicBarriers);
    };

    for (auto i : irange(dep.consumer.second)) {
        const auto v = _vdt.id(dep.consumer.first + i);
        const auto r = _barriers[v].realId;

        if (barrierCheck(checked_cast<int64_t>(toVirtual.size()) <= r, toVirtual[r] != v)) {
            _log.nest(3).trace("{0}[{1}] is waiting for consumer barrier {2} to be mapped", taskType, index, v);
            return Status::Skip;
        }

        if (_barriers[v].producerCount > 0) {
            _log.nest(3).trace("{0}[{1}] waiting for barrier {2} to be produced, {3} remaining", taskType, index, v,
                               _barriers[v].producerCount);
            return Status::Skip;
        }
    }

    for (auto i : irange(dep.producer.second)) {
        const auto v = _vdt.id(dep.producer.first + i);
        const auto r = _barriers[v].realId;

        if (barrierCheck(checked_cast<int64_t>(toVirtual.size()) <= r, toVirtual[r] != v)) {
            _log.nest(3).trace("{0}[{1}] is waiting for consumer barrier {2} to be mapped", taskType, index, v);
            return Status::Skip;
        }
    }

    user.startAfter = 0;

    for (auto i : irange(dep.consumer.second)) {
        const auto v = _vdt.id(dep.consumer.first + i);
        const auto r = _barriers[v].realId;

        if (barrierCheck(r < checked_cast<int64_t>(toVirtual.size()), toVirtual[r] == v)) {
            // barrier not ready to be consumed
            if ((_barriers[v].producerCount != 0) || (_barriers[v].consumerCount < count)) {
                _log.error(
                        "Simulate barriers failed - barrier {0} not ready to be consumed (producers {1} consumers {2})",
                        v, _barriers[v].producerCount, _barriers[v].consumerCount);
                return Status::Fail;
            }

            _barriers[v].consumerCount -= count;
            user.waitMask |= 1 << r;
            user.startAfter = std::max(user.startAfter, v + 1);

            if (_barriers[v].consumerCount == 0) {
                toVirtual[r] = -1;

                if (_isDynamicBarriers) {
                    nextReal.push(r);
                }
            }
        } else {
            _log.error("Virtual barrier {0} still not mapped", v);
            return Status::Fail;
        }
    }

    user.cleanAfter = _barriers.size();

    for (auto i : irange(dep.producer.second)) {
        const auto v = _vdt.id(dep.producer.first + i);
        const auto r = _barriers[v].realId;

        if (barrierCheck(r < checked_cast<int64_t>(toVirtual.size()), toVirtual[r] == v)) {
            if (_barriers[v].producerCount < count) {
                _log.error("Simulate barriers failed - barrier {0} producer count ({1} is lower then number of tasks "
                           "which are producing it ({2})",
                           v, _barriers[v].producerCount, count);
                return Status::Fail;
            }

            _barriers[v].producerCount -= count;
            user.postMask |= 1 << r;
            user.startAfter = std::max(user.startAfter, v + 1);
            user.cleanAfter = std::min(user.cleanAfter, v);
        } else {
            _log.error("Virtual barrier {0} still not mapped", v);
            return Status::Fail;
        }
    }

    return Status::Success;
}

void vpux::VPURT::BarrierSimulator::linkNextIds() {
    _log.trace("Create links to the next config of the same barrier");

    for (auto i : irange(_barriers.size())) {
        auto& current = _barriers[i];
        current.nextSameId = -1;

        for (auto j = i + 1; j < _barriers.size(); ++j)
            if (_barriers[j].realId == current.realId) {
                current.nextSameId = j;
                break;
            }

        _log.nest().trace("VID: {0}, PHYS ID: {1}, NEXT SAME PHYS ID: {2}", i, current.realId, current.nextSameId);
    }
}
