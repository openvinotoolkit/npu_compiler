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
#pragma once

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>

#include "vpux/compiler/core/barrier_resource_state.hpp"
#include "vpux/compiler/core/op_resource_state.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

namespace vpux {

struct opResourceState;
static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
class FeasibleBarrierScheduler {
public:
    struct operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

            return uniqueId1 < uniqueId2;
        }
    };
    struct HeapElement {
        HeapElement(mlir::Operation* op = NULL, schedule_time_t t = 0UL): op_(op), time_(t) {
        }

        mlir::Operation* op_;
        schedule_time_t time_;
    };

    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a.time_ > b.time_;
        }
    };

    struct ScheduledOpInfo {
        schedule_time_t schedule_time_;
        mlir::Operation* op_;
        size_t barrier_index_;
        size_t slot_count_;
    };

    struct barrierInfo {
        barrierInfo(size_t bindex = 0UL, size_t slot_count = 0UL): bindex_(bindex), slot_count_(slot_count) {
        }
        size_t bindex_;
        size_t slot_count_;
    }; /*struct barrierInfo*/

    class barrierTransitionStructure {
    public:
        barrierTransitionStructure(mlir::FuncOp func, FeasibleBarrierScheduler& feasibleBarrierScheduler,
                                   schedule_time_t time = std::numeric_limits<schedule_time_t>::max());

        void init();
        // bool process_next_scheduled_op(const BarrierScheduleGenerator::schedule_info_t& sinfo,
        //                                mlir::OpBuilder& builder);
        void close_barrier_producer_list();
        struct operation_comparator_t {
            bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
                int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op1)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());
                int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op2)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());

                return uniqueId1 < uniqueId2;
            }
        };

        using producers_t = std::set<mlir::Operation*, operation_comparator_t>;
        using producer_iterator_t = typename producers_t::const_iterator;

    private:
        // void maintain_invariant_temporal_change(const BarrierScheduleGenerator::schedule_info_t& sinfo,
        //                                         mlir::OpBuilder& builder);
        // inline void process_current_barrier_producer_list_close_event(mlir::Operation* bop_curr,
        //                                                               mlir::Operation* bop_prev);
        // void add_scheduled_op_to_producer_list(const BarrierScheduleGenerator::schedule_info_t& sinfo);
        // mlir::Operation* create_new_barrier_task(const BarrierScheduleGenerator::schedule_info_t& sinfo,
        //                                         mlir::OpBuilder& builder);

        mlir::FuncOp _func;
        // Outer class
        FeasibleBarrierScheduler& feasibleBarrierScheduler_;
        schedule_time_t time_;
        mlir::Operation* curr_barrier_task_;
        mlir::Operation* prev_barrier_task_;
        producers_t producers_;
    };

    typedef mlir::Operation const* operation_t;
    typedef std::unordered_map<operation_t, barrierInfo> active_barrier_map_t;
    typedef size_t resource_t;
    typedef std::unordered_map<size_t, barrierTransitionStructure> barrierAssociationTable;
    using delay_t = size_t;
    using schedulable_ops_t = std::list<mlir::Operation*>;
    using schedulable_ops_iterator_t = typename schedulable_ops_t::iterator;
    using processed_ops_t = std::set<mlir::Operation*>;
    using schedule_heap_t = std::vector<HeapElement>;
    using operation_in_degree_t = std::map<mlir::Operation*, size_t, operation_comparator_t>;
    using priority_map_t = std::map<mlir::Operation*, size_t, operation_comparator_t>;
    using resource_utility_map_t = std::unordered_map<mlir::Operation*, unsigned>;
    using schedule_time_t = size_t;

    struct opResourceState {
        opResourceState(size_t n = 0UL, size_t m = 0UL)
                : barrier_map_(), state_(), barrier_count_(n), slots_per_barrier_(m) {
            Logger::global().error(
                    "Initializing op_resource_state in Barrier_Schedule_Generator with barrier count {0} "
                    "slots_per_barrie {1}",
                    barrier_count_, slots_per_barrier_);
        }

        void init(const opResourceState& other) {
            barrier_map_.clear();
            barrier_count_ = other.barrier_count_;
            slots_per_barrier_ = other.slots_per_barrier_;
            state_.init(barrier_count_, slots_per_barrier_);
        }

        bool is_resource_available(const resource_t& demand) const {
            Logger::global().error("Looking for a barrier with free slots");
            return state_.has_barrier_with_slots(demand);
        }

        bool schedule_operation(const operation_t& op, resource_t& demand) {
            Logger::global().error("Scheduling an operation");
            assert(is_resource_available(demand));
            if (barrier_map_.find(op) != barrier_map_.end()) {
                return false;
            }
            size_t bid = state_.assign_slots(demand);
            barrier_map_.insert(std::make_pair(op, barrierInfo(bid, demand)));
            return true;
        }

        bool unschedule_operation(const operation_t& op) {
            auto itr = barrier_map_.find(op);
            if (itr == barrier_map_.end()) {
                return false;
            }
            const barrierInfo& binfo = itr->second;
            bool ret = state_.unassign_slots(binfo.bindex_, binfo.slot_count_);
            assert(ret);
            (void)ret;
            barrier_map_.erase(itr);
            return true;
        }

        const barrierInfo& get_barrier_info(const operation_t& op) const {
            auto itr = barrier_map_.find(op);

            assert(itr != barrier_map_.end());
            return itr->second;
        }

        active_barrier_map_t barrier_map_;
        BarrierResourceState state_;
        size_t barrier_count_;
        size_t slots_per_barrier_;
    }; /*struct opResourceState*/

    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, const resource_state_t& rstate, Logger log);
    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log, size_t numberOfBarriers,
                             size_t maxProducersPerBarrier);
    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log);

    void operator++();
    void getAllBarriersProducersAndConsumers();
    void initResourceState(const resource_state_t& start_state);
    void computeOpIndegree(operation_in_degree_t& in_degree);
    void addToCandidateSet(mlir::Operation* op);
    void computeOperationPriorities();
    void createResourceUtilityTable();
    void addOutGoingOperationsToCandidateList(mlir::Operation* op);
    void assignUniqueIds();
    void pushToHeap(const HeapElement& elem);
    HeapElement popFromHeap();

    bool operator==(const FeasibleBarrierScheduler& o) const;
    bool reached_end() const;
    bool nextSchedulableOperation();
    //bool init(const resource_state_t& upper_bound);
    bool init();
    bool doesOpRunOnNCE(mlir::Operation* op);

    mlir::Operation*& operator*();
    size_t currentTime() const;
    const resource_state_t& resourceState() const;
    bool isValidOp(schedulable_ops_iterator_t itr) const;
    schedulable_ops_iterator_t find_schedulable_op();
    unsigned countProducerConsumerTasks(mlir::Operation* op);
    static SmallVector<mlir::Operation*> getConsumerOps(mlir::Operation* op);
    static std::string printOpType(VPURT::TaskOp taskOp);
    static mlir::IntegerAttr getUniqueID(mlir::Operation* op);
    size_t schedule();
    void populateScheduledOps(mlir::Operation* scheduledOp);
    void removeRedundantWaitBarriers();
    void removeRedundantDependencies();
    void initializeBarrierAssociationTable();

protected:
    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    operation_in_degree_t _in_degree;
    schedule_heap_t _heap;
    schedule_time_t _current_time;
    schedulable_ops_t _candidates;
    resource_state_t _resource_state;
    MinHeapOrdering _heap_ordering;
    mlir::Operation* _schedulable_op;
    processed_ops_t _processed_ops;
    priority_map_t _priority;
    resource_utility_map_t _resource_utility_map;
    std::map<mlir::Operation*, size_t> _outDegreeTable;
    SmallVector<IERT::LayerOpInterface> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;
    size_t _barrierCount;
    size_t _slotsPerBarrier;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierProducersMap;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierConsumersMap;

    const resource_state_t _startState;
    // container for the schedule output
    SmallVector<ScheduledOpInfo> _scheduledOps;
    barrierAssociationTable _barrierAssociationTable;

    /*Stores every barrier's associated update and wait operations*/
    //std::map<mlir::Operation*,std::pair<std::set<mlir::Operation*, task_operation_comparator_t>, std::set<mlir::Operation*, task_operation_comparator_t>>,operation_comparator_t> configureBarrierOpUpdateWaitMap;  // update,wait
};

}  // namespace vpux