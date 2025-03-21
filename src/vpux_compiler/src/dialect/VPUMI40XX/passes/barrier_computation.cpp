//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/wlm_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <vector>

namespace vpux::VPUMI40XX {
#define GEN_PASS_DECL_BARRIERCOMPUTATION
#define GEN_PASS_DEF_BARRIERCOMPUTATION
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc"
}  // namespace vpux::VPUMI40XX

using namespace vpux;

namespace {

struct VirtualDependencyTracker {
    using Range = std::pair<size_t, size_t>;

    struct Dependency {
        Range consumer_;
        Range producer_;
    };

    VirtualDependencyTracker(): ids_(), deps_(1) {
    }

    template <typename TaskOpType>
    size_t add(TaskOpType taskOp) {
        auto extract = [&](Range& range, ::mlir::ValueRange barriers) -> bool {
            range.first = ids_.size();

            if (barriers.empty()) {
                return true;
            }

            for (auto bv : barriers) {
                auto vv = mlir::dyn_cast<VPUMI40XX::ConfigureBarrierOp>(bv.getDefiningOp());
                VPUX_THROW_UNLESS(vv, "Encountered unexpected non barrier");

                auto v = vv.getType().getValue();

                VPUX_THROW_UNLESS(v < std::numeric_limits<uint32_t>::max(), "Barrier virtual id '{0}' is too large", v);
                ids_.push_back(v);
                ++range.second;
            }

            return true;
        };

        Dependency d{};

        if (!extract(d.consumer_, taskOp.getWaitBarriers()))
            return UINT_MAX;

        if (!extract(d.producer_, taskOp.getUpdateBarriers()))
            return UINT_MAX;

        if (d.consumer_.second || d.producer_.second) {
            deps_.push_back(d);
            return deps_.size() - 1;
        } else
            return 0;

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

struct TaskSchedulingBarrierConfig {
    uint32_t start_after_;
    uint32_t clean_after_;
};

using TaskVector = std::vector<std::tuple<mlir::Operation*, TaskSchedulingBarrierConfig, unsigned int>>;

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
                        "1 Wrong VID - '{0}'", barVid);
        barVidIndexToConsumptionEventIndexVec[barVid] = nextConsumptionEventIndex++;
    };

    int64_t getConsumptionEventIndex(int64_t barVid) {
        VPUX_THROW_WHEN(barVid >= static_cast<int64_t>(barVidIndexToConsumptionEventIndexVec.size()),
                        "2 Wrong VID - '{0}'", barVid);
        return barVidIndexToConsumptionEventIndexVec[barVid];
    };
    int64_t nextConsumptionEventIndex;
    SmallVector<int64_t> barVidIndexToConsumptionEventIndexVec;
};

struct BarrierCountConfig {
    uint16_t producer_count_;
    uint16_t consumer_count_;
    uint8_t real_id_;
};

bool processSim(VirtualDependencyTracker& vdt_, const std::vector<BarrierCountConfig>& barriersConfig,
                std::vector<BarrierCountConfig>& counts, const VirtualDependencyTracker::Dependency& dep,
                TaskSchedulingBarrierConfig& bar_sched, size_t count, std::vector<int64_t>& to_virtual,
                BarrierConsumptionEventData& barrierConsumptionEventData) {
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

            counts[v].consumer_count_ -= checked_cast<unsigned short>(count);
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
    bar_sched.clean_after_ = static_cast<uint32_t>(counts.size() - 1);

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

void simulateBarriers(const std::vector<BarrierCountConfig>& barriersConfigs, unsigned char nn_barriers_,
                      std::vector<TaskVector>& dmas, TaskVector& dpus, TaskVector& acts, TaskVector& m2is,
                      VirtualDependencyTracker& vdt_) {
    auto counts = barriersConfigs;
    std::vector<int64_t> to_virtual(nn_barriers_, -1);

    BarrierConsumptionEventData barrierConsumptionEventData(counts.size());

    SmallVector<TaskVector::iterator> dmaCurrs;
    for (auto& dma : dmas) {
        dmaCurrs.push_back(dma.begin());
    }
    auto dpuCurr = dpus.begin();
    auto actCurr = acts.begin();
    auto m2iCurr = m2is.begin();

    bool progressed = false;

    auto processTasks = [&](auto& currentIterator, auto endIterator) {
        for (; currentIterator != endIterator; ++currentIterator, progressed = true) {
            auto& current = *currentIterator;
            const auto& op = std::get<0>(current);
            auto& barrierConfig = std::get<1>(current);
            const auto dependencyIndex = std::get<2>(current);

            const auto barrierHitsCount =
                    mlir::dyn_cast<vpux::VPUMI40XX::ExecutableTaskOpInterface>(op).getBarrierHitsCount();
            if (!processSim(vdt_, barriersConfigs, counts, vdt_.dep(dependencyIndex), barrierConfig, barrierHitsCount,
                            to_virtual, barrierConsumptionEventData)) {
                break;
            }
        }
    };

    auto checkDMAStatus = [&]() {
        for (auto item : dmaCurrs | indexed) {
            auto index = item.index();
            auto& dmaCur = item.value();
            if (dmaCur != dmas[index].end()) {
                return true;
            }
        }
        return false;
    };

    for (unsigned int bar = 0; bar < counts.size() || checkDMAStatus() || dpuCurr != dpus.end() ||
                               actCurr != acts.end() || m2iCurr != m2is.end();
         progressed = false) {
        // Static vs dynamic barriers need a different loop exit condition
        auto cond = [&]() {
            return to_virtual[barriersConfigs[bar].real_id_] == -1;
        };

        // map new barriers
        for (; bar < counts.size() && cond(); ++bar, progressed = true) {
            to_virtual[barriersConfigs[bar].real_id_] = static_cast<int64_t>(bar);
        }

        for (auto item : dmaCurrs | indexed) {
            auto index = item.index();
            auto& dmaCurr = item.value();
            processTasks(dmaCurr, dmas[index].end());
        }
        processTasks(dpuCurr, dpus.end());
        processTasks(actCurr, acts.end());
        processTasks(m2iCurr, m2is.end());

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

    for (auto& dma : dmas) {
        updateCleanAfterField(dma);
    }
    updateCleanAfterField(dpus);
    updateCleanAfterField(acts);
    updateCleanAfterField(m2is);
}

class BarrierComputationPass final : public VPUMI40XX::impl::BarrierComputationBase<BarrierComputationPass> {
public:
    explicit BarrierComputationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename TaskOpType, typename Condition = FuncRef<bool(TaskOpType)>>
    TaskVector buildTaskVector(
            mlir::func::FuncOp funcOp, VirtualDependencyTracker& tracker, Condition&& condition = [](TaskOpType) {
                return true;
            }) {
        TaskVector vector;
        for (auto op : funcOp.getOps<TaskOpType>()) {
            if (condition(op)) {
                vector.emplace_back(op, TaskSchedulingBarrierConfig{0, 0}, tracker.add(op));
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

        vpux::VPUMI40XX::setBarrierIDs(ctx, funcOp);

        VirtualDependencyTracker vdt_;

        // Get channel id from index, in which the list value 0: DDR list, 1: CMX list
        auto getChannelId = [](VPUMI40XX::NNDMAOp dmaOp) {
            auto index = dmaOp.getIndexType();
            return index.getListIdx();
        };

        std::set<int64_t> dmaTasksQueueIndex;
        for (auto dmaOp : funcOp.getOps<VPUMI40XX::NNDMAOp>()) {
            auto port = dmaOp.getPort();
            auto dmaQueueId = getDMAQueueIdEncoding(port, checked_cast<int64_t>(getChannelId(dmaOp)));
            if (dmaTasksQueueIndex.find(dmaQueueId) == dmaTasksQueueIndex.end()) {
                dmaTasksQueueIndex.insert(dmaQueueId);
            }
        }

        std::vector<TaskVector> dmas;
        for (auto dmaQueueId : dmaTasksQueueIndex) {
            auto dmaList = buildTaskVector<VPUMI40XX::NNDMAOp>(funcOp, vdt_, [&](VPUMI40XX::NNDMAOp dma) {
                return dmaQueueId == getDMAQueueIdEncoding(dma.getPort(), checked_cast<int64_t>(getChannelId(dma)));
            });
            dmas.push_back(std::move(dmaList));
        }

        auto dpus = buildTaskVector<VPUMI40XX::DPUInvariantOp>(funcOp, vdt_);
        auto acts = buildTaskVector<VPUMI40XX::ActKernelInvocationOp>(funcOp, vdt_);
        auto m2is = buildTaskVector<VPUMI40XX::M2IOp>(funcOp, vdt_);

        std::vector<BarrierCountConfig> barriersConfigs;
        unsigned char nn_barriers_ = 0;
        for (auto op : funcOp.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
            barriersConfigs.push_back(
                    BarrierCountConfig{op.getProducerCount().value(), op.getConsumerCount().value(), op.getId()});
            nn_barriers_ = std::max<unsigned char>(nn_barriers_, op.getId() + 1);
        }

        simulateBarriers(barriersConfigs, nn_barriers_, dmas, dpus, acts, m2is, vdt_);

        for (auto& dma : dmas) {
            setBarrierAttributes(dma, ctx);
        }
        setBarrierAttributes(dpus, ctx);
        setBarrierAttributes(acts, ctx);
        setBarrierAttributes(m2is, ctx);
    }
};

}  // namespace

//
// createBarrierComputationPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createBarrierComputationPass(Logger log) {
    return std::make_unique<BarrierComputationPass>(log);
}
