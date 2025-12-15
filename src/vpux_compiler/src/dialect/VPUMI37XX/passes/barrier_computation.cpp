//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <npu_37xx_nnrt.hpp>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>

#include <vector>

namespace vpux::VPUMI37XX {
#define GEN_PASS_DECL_BARRIERCOMPUTATION
#define GEN_PASS_DEF_BARRIERCOMPUTATION
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp.inc"
}  // namespace vpux::VPUMI37XX

using namespace vpux;
using namespace npu37xx;

namespace {

struct VirtualDependencyTracker {
    using Range = std::pair<unsigned int, unsigned int>;

    struct Dependency {
        Range consumer_;
        Range producer_;
    };

    VirtualDependencyTracker(): ids_(), deps_(1) {
    }

    template <typename TaskOpType>
    unsigned int add(TaskOpType taskOp) {
        auto extract = [&](Range& range, ::mlir::ValueRange barriers) -> bool {
            range.first = ids_.size();

            if (barriers.empty()) {
                return true;
            }

            for (auto bv : barriers) {
                auto vv = mlir::dyn_cast<VPUMI37XX::ConfigureBarrierOp>(bv.getDefiningOp());
                VPUX_THROW_UNLESS(vv, "Encountered unexpected non barrier");

                auto v = vv.getType().getValue();

                VPUX_THROW_UNLESS(v < std::numeric_limits<uint32_t>::max(), "Barrier virtual id '{0}' is too large", v);
                ids_.push_back(v);
                ++range.second;
            }

            return true;
        };

        Dependency d{};

        if (!extract(d.consumer_, taskOp.getWaitBarriers())) {
            return UINT_MAX;
        }

        if (!extract(d.producer_, taskOp.getUpdateBarriers())) {
            return UINT_MAX;
        }

        if (d.consumer_.second || d.producer_.second) {
            deps_.push_back(d);
            return deps_.size() - 1;
        } else {
            return 0;
        }

        return 0;
    }

    uint32_t& id(unsigned int i) {
        return ids_[i];
    }
    Dependency& dep(unsigned int i) {
        return deps_[i];
    }

private:
    std::vector<uint32_t> ids_;
    std::vector<Dependency> deps_;
};

using TaskVector = std::vector<std::tuple<mlir::Operation*, nn_public::VpuTaskSchedulingBarrierConfig, unsigned int>>;

// Below structure is used to gather information when
// given barrier (VID) gets fully consumed. This will later be used
// when setting clean_after field
struct BarrierConsumptionEventData {
    BarrierConsumptionEventData(int64_t numOfBarriers) {
        barVidIndexToConsumptionEventIndexVec.resize(numOfBarriers, numOfBarriers - 1);
        nextConsumptionEventIndex = 0;
    };
    void barrierVidConsumed(int64_t barVid) {
        VPUX_THROW_WHEN(barVid >= static_cast<int64_t>(barVidIndexToConsumptionEventIndexVec.size()),
                        "Wrong VID - '{0}'", barVid);
        barVidIndexToConsumptionEventIndexVec[barVid] = nextConsumptionEventIndex++;
    };

    int64_t getConsumptionEventIndex(int64_t barVid) {
        VPUX_THROW_WHEN(barVid >= static_cast<int64_t>(barVidIndexToConsumptionEventIndexVec.size()),
                        "Wrong VID - '{0}'", barVid);
        return barVidIndexToConsumptionEventIndexVec[barVid];
    };
    int64_t nextConsumptionEventIndex;
    SmallVector<int64_t> barVidIndexToConsumptionEventIndexVec;
};

bool processSim(VirtualDependencyTracker& vdt_, const std::vector<nn_public::VpuBarrierCountConfig>& barriersConfig,
                std::vector<nn_public::VpuBarrierCountConfig>& counts, const VirtualDependencyTracker::Dependency& dep,
                nn_public::VpuTaskSchedulingBarrierConfig& bar_sched, unsigned short count,
                std::vector<int64_t>& to_virtual, BarrierConsumptionEventData& barrierConsumptionEventData) {
    auto barrierCheck = [&](bool dynamicCond) {
        return dynamicCond;
    };

    for (unsigned int i = 0; i < dep.consumer_.second; ++i) {
        unsigned v = vdt_.id(dep.consumer_.first + i);
        auto r = barriersConfig[v].real_id_;

        if (barrierCheck(to_virtual.size() <= r)) {
            return false;
        }

        if (counts[v].producer_count_ > 0) {
            return false;
        }
    }

    for (unsigned int i = 0; i < dep.producer_.second; ++i) {
        unsigned v = vdt_.id(dep.producer_.first + i);
        auto r = barriersConfig[v].real_id_;

        if (to_virtual.size() <= r || (to_virtual[r] != static_cast<int64_t>(v))) {
            if (r < to_virtual.size()) {
            } else {
            }
            return false;
        }
    }

    bar_sched.start_after_ = 0;

    for (unsigned int i = 0; i < dep.consumer_.second; ++i) {
        unsigned v = vdt_.id(dep.consumer_.first + i);
        auto r = barriersConfig[v].real_id_;

        if (barrierCheck(r < to_virtual.size())) {
            // barrier not ready to be consumed
            if ((counts[v].producer_count_ != 0) || (counts[v].consumer_count_ < count)) {
                VPUX_THROW("v = {0} counts[v].producer_count_ = {1} counts[v].consumer_count_ = {2}", v,
                           counts[v].producer_count_, counts[v].consumer_count_);
            }

            counts[v].consumer_count_ -= count;
            bar_sched.start_after_ = std::max<uint32_t>(bar_sched.start_after_, v + 1);

            if (counts[v].consumer_count_ == 0) {
                barrierConsumptionEventData.barrierVidConsumed(v);
                to_virtual[r] = -1;
            }
        } else {
            VPUX_THROW("r = {0} to_virtual.size() = {1}", static_cast<int>(r), to_virtual.size());
        }
    }

    // Initialize clean_after with largest VID. Actual value will be configured
    // at the end of simulation after all tasks and barriers are processed
    bar_sched.clean_after_ = counts.empty() ? 0 : static_cast<uint32_t>(counts.size() - 1);

    for (unsigned int i = 0; i < dep.producer_.second; ++i) {
        unsigned v = vdt_.id(dep.producer_.first + i);
        auto r = barriersConfig[v].real_id_;

        if (barrierCheck(r < to_virtual.size())) {
            if (counts[v].producer_count_ < count) {
                VPUX_THROW("v = {0} counts[v].producer_count_ = {1}", v, counts[v].producer_count_);
            }
            counts[v].producer_count_ -= count;
            bar_sched.start_after_ = std::max<uint32_t>(bar_sched.start_after_, v + 1);
        } else {
            VPUX_THROW("r = {0} to_virtual.size() = {1}", static_cast<int>(r), to_virtual.size());
        }
    }

    return true;
}

void simulateBarriers(const std::vector<nn_public::VpuBarrierCountConfig>& barriersConfigs, unsigned char nn_barriers_,
                      TaskVector& dmas0, TaskVector& dmas1, TaskVector& dpus, TaskVector& acts,
                      VirtualDependencyTracker& vdt_) {
    auto counts = barriersConfigs;
    std::vector<int64_t> to_virtual(nn_barriers_, -1);

    BarrierConsumptionEventData barrierConsumptionEventData(counts.size());

    auto dmaCurr0 = dmas0.begin();
    auto dmaCurr1 = dmas1.begin();
    auto dpuCurr = dpus.begin();
    auto actCurr = acts.begin();

    bool progressed = false;

    auto processNextTask = [&](auto& currentIterator, auto endIterator) {
        if (currentIterator == endIterator) {
            return;
        }

        auto& current = *currentIterator;
        const auto& op = std::get<0>(current);
        auto& barrierConfig = std::get<1>(current);
        const auto dependencyIndex = std::get<2>(current);

        const auto barrierHitsCount =
                mlir::dyn_cast<vpux::VPUMI37XX::ExecutableTaskOpInterface>(op).getBarrierHitsCount();
        if (processSim(vdt_, barriersConfigs, counts, vdt_.dep(dependencyIndex), barrierConfig, barrierHitsCount,
                       to_virtual, barrierConsumptionEventData)) {
            ++currentIterator;
            progressed = true;
        }
    };

    for (unsigned int bar = 0; bar < counts.size() || dmaCurr0 != dmas0.end() || dmaCurr1 != dmas1.end() ||
                               dpuCurr != dpus.end() || actCurr != acts.end();
         progressed = false) {
        // Static vs dynamic barriers need a different loop exit condition
        auto cond = [&]() {
            return to_virtual[barriersConfigs[bar].real_id_] == -1;
        };

        // map new barriers
        for (; bar < counts.size() && cond(); ++bar, progressed = true) {
            to_virtual[barriersConfigs[bar].real_id_] = static_cast<int64_t>(bar);
        }

        // Process each HW FIFO one by one to reflect parallelism between those engines
        // This will allow to identify order of full barrier consumption events needed
        // for proper setting of clean_after field
        processNextTask(dmaCurr0, dmas0.end());
        processNextTask(dmaCurr1, dmas1.end());
        processNextTask(dpuCurr, dpus.end());
        processNextTask(actCurr, acts.end());

        if (!progressed) {
            VPUX_THROW("Did not progress");
        }
    }

    // Traverse tasks again to configure clean_after field based
    // on stored order of each barrier (VID) consumption event (consumer count gets decremented to 0)

    auto updateCleanAfterField = [&](TaskVector& tasks) {
        for (auto& task : tasks) {
            auto& barrierConfig = std::get<1>(task);
            const auto dependencyIndex = std::get<2>(task);
            for (unsigned int i = 0; i < vdt_.dep(dependencyIndex).producer_.second; ++i) {
                unsigned v = vdt_.id(vdt_.dep(dependencyIndex).producer_.first + i);

                // In case task updates multiple barriers pick the one with earliest
                // consumption event
                if (barrierConsumptionEventData.getConsumptionEventIndex(v) <
                    barrierConsumptionEventData.getConsumptionEventIndex(barrierConfig.clean_after_)) {
                    barrierConfig.clean_after_ = v;
                }
            }
        }
    };

    updateCleanAfterField(dmas0);
    updateCleanAfterField(dmas1);
    updateCleanAfterField(dpus);
    updateCleanAfterField(acts);
}

class BarrierComputationPass final : public VPUMI37XX::impl::BarrierComputationBase<BarrierComputationPass> {
public:
    explicit BarrierComputationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    // Set next_same_id attribute and previousSameId operand for each ConfigureBarrier operation,
    // and here we don't need to verify barrier if it has same previousSameId with other same physical id barrier,
    // because the previousSameId operand is continuously increasing
    void setBarrierIDs(mlir::MLIRContext* ctx, mlir::func::FuncOp funcOp) {
        auto MAX_PID = VPUIP::getNumAvailableBarriers(funcOp);

        std::vector<VPUMI37XX::ConfigureBarrierOp> lastAssignedBarrier(MAX_PID);

        for (auto op : funcOp.getOps<VPUMI37XX::ConfigureBarrierOp>()) {
            auto vid = mlir::cast<vpux::VPURegMapped::IndexType>(op.getOperation()->getResult(0).getType()).getValue();
            auto pid = op.getId();

            auto& lastBarrier = lastAssignedBarrier[pid];
            if (lastBarrier != nullptr) {
                op.getPreviousSameIdMutable().assign(lastBarrier.getOperation()->getResult(0));
                lastBarrier.setNextSameIdAttr(
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), vid));
            }

            lastBarrier = op;
        }
    }

    template <typename TaskOpType, typename Condition = FuncRef<bool(TaskOpType)>>
    TaskVector buildTaskVector(
            mlir::func::FuncOp funcOp, VirtualDependencyTracker& tracker, Condition&& condition = [](TaskOpType) {
                return true;
            }) {
        TaskVector vector;
        for (auto op : funcOp.getOps<TaskOpType>()) {
            if (condition(op)) {
                vector.emplace_back(op, nn_public::VpuTaskSchedulingBarrierConfig{0, 0}, tracker.add(op));
            }
        }
        return vector;
    }

    void setBarrierAttributes(const TaskVector& tasks, mlir::MLIRContext* ctx) {
        for (auto task : tasks) {
            auto& op = std::get<0>(task);
            const auto& barrierConfig = std::get<1>(task);

            auto newStartAfterAttr = mlir::IntegerAttr::get(
                    mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), barrierConfig.start_after_);

            op->setAttr("start_after", newStartAfterAttr);

            auto newCleanAfterAttr = mlir::IntegerAttr::get(
                    mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned), barrierConfig.clean_after_);

            op->setAttr("clean_after", newCleanAfterAttr);
        }
    }

    void safeRunOnFunc() final {
        auto funcOp = getOperation();
        mlir::MLIRContext* ctx = &(getContext());

        setBarrierIDs(ctx, funcOp);

        VirtualDependencyTracker vdt_;

        auto mappedInferenceOps = funcOp.getOps<VPUMI37XX::MappedInferenceOp>();
        VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");
        auto mappedInferenceOp = *(mappedInferenceOps.begin());
        auto dmaTasks = mappedInferenceOp.getDmaTasks();
        auto dmaCount = parseIntArrayAttr<int64_t>(mappedInferenceOp.getDmaCount());

        TaskVector dmas0(dmaCount[0]), dmas1(dmaCount.size() < 2 ? 0 : dmaCount[1]);
        for (auto head : dmaTasks) {
            auto headDMA = mlir::cast<VPUMI37XX::NNDMAOp>(head.getDefiningOp());
            auto& list = headDMA.getPort() == 0 ? dmas0 : dmas1;
            while (head) {
                auto dma = mlir::cast<VPUMI37XX::NNDMAOp>(head.getDefiningOp());
                auto value = std::make_tuple(dma, nn_public::VpuTaskSchedulingBarrierConfig{0, 0}, vdt_.add(dma));
                auto index = mlir::cast<VPURegMapped::IndexType>(dma.getIndex().getType()).getValue();
                list[index] = value;
                head = dma.getNextDMAIdx();
            }
        }

        auto dpus = buildTaskVector<VPUMI37XX::DPUInvariantOp>(funcOp, vdt_);
        auto acts = buildTaskVector<VPUMI37XX::ActKernelInvocationOp>(funcOp, vdt_);

        std::vector<nn_public::VpuBarrierCountConfig> barriersConfigs;
        unsigned char nn_barriers_ = 0;
        for (auto op : funcOp.getOps<VPUMI37XX::ConfigureBarrierOp>()) {
            barriersConfigs.push_back(nn_public::VpuBarrierCountConfig{std::numeric_limits<uint32_t>::max(),
                                                                       op.getProducerCount().value(),
                                                                       op.getConsumerCount().value(),
                                                                       op.getId(),
                                                                       {0}});
            nn_barriers_ = std::max<unsigned char>(nn_barriers_, op.getId() + 1);
        }

        simulateBarriers(barriersConfigs, nn_barriers_, dmas0, dmas1, dpus, acts, vdt_);

        setBarrierAttributes(dmas0, ctx);
        setBarrierAttributes(dmas1, ctx);
        setBarrierAttributes(dpus, ctx);
        setBarrierAttributes(acts, ctx);
    }
};

}  // namespace

//
// createBarrierComputationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI37XX::createBarrierComputationPass(Logger log) {
    return std::make_unique<BarrierComputationPass>(log);
}
