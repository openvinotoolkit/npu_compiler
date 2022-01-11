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

#include "vpux/compiler/core/barrier_scheduler.hpp"

using namespace vpux::VPURT;

BarrierScheduler::barrierTransitionStructure::barrierTransitionStructure(BarrierScheduler& feasibleBarrierScheduler,
                                                                         size_t time)
        : _feasibleBarrierScheduler(feasibleBarrierScheduler), _time(time), _producers() {
}

void BarrierScheduler::barrierTransitionStructure::init() {
    _time = std::numeric_limits<size_t>::max();
    _previousBarrierTask = NULL;
    _currentBarrierTask = NULL;
    _producers.clear();
}

bool BarrierScheduler::barrierTransitionStructure::processNextScheduledTask(const ScheduledOpInfo& sinfo,
                                                                            mlir::OpBuilder& builder) {
    size_t currentTime = sinfo._scheduleTime;
    bool createdNewBarrierTask = false;

    _feasibleBarrierScheduler._log.trace("The scheduled time is {0}, the global time is {1}, the task is {2} the "
                                         "barrier index is {3} and the slot cout is {4}",
                                         sinfo._scheduleTime, _time, BarrierScheduler::getUniqueID(sinfo._op),
                                         sinfo._barrierIndex, sinfo._producerSlotCount);

    if (_time != currentTime) {
        _feasibleBarrierScheduler._log.trace("Case 1: temporal transition happened, create a new barrier task");

        // Case-1: a temporal transition happened
        createdNewBarrierTask = true;
        maintainInvariantTemporalChange(sinfo, builder);
        _time = currentTime;
    } else {
        // Case-2: a trival case
        _feasibleBarrierScheduler._log.trace("Case-2: trival case - adding the scheduled task to the producer list");
        addScheduledTaskToProducerList(sinfo);
    }
    return createdNewBarrierTask;
}

void BarrierScheduler::barrierTransitionStructure::closeBarrierProducerList() {
    if (_currentBarrierTask == NULL) {
        return;
    }
    processCurrentBarrierProducerListCloseEvent(_currentBarrierTask, _previousBarrierTask);
}

inline void BarrierScheduler::barrierTransitionStructure::processCurrentBarrierProducerListCloseEvent(
        mlir::Operation* currentBarrier, mlir::Operation* previousBarrier) {
    _feasibleBarrierScheduler._log.trace("Process current barrier producer list close event");

    mlir::Operation* barrierEnd = NULL;
    VPUX_THROW_UNLESS(currentBarrier != barrierEnd, "Eror the current barrier is Null");

    // Get the barrier object for the three barrier tasks 
    mlir::Operation* barrierPrevious = NULL;

    if (previousBarrier != barrierEnd) {
        barrierPrevious = previousBarrier;
    }

    _feasibleBarrierScheduler._log.trace("The ID of barrier b_curr is {0}", currentBarrier->getAttr("id"));

    for (producerIteratorType producer = _producers.begin(); producer != _producers.end(); ++producer) {
        mlir::Operation* source = *producer;

        // Step-1.2 (a): producers
        auto barrierProducersItr = _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.find(currentBarrier);

        if (barrierProducersItr != _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.end()) {
            _feasibleBarrierScheduler._log.trace("Adding producer task with ID {0} to barrier ID {1}",
                                                 BarrierScheduler::getUniqueID(source), currentBarrier->getAttr("id"));

            barrierProducersItr->second.first.insert(source);
        } else {
            VPUX_THROW("Error unable to find the update tasks for barrier ID {0}", currentBarrier->getAttr("id"));
        }

        // Step-1.2 (b): consumers
        auto barrierConsumersItr = _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.find(currentBarrier);

        if (barrierConsumersItr != _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.end()) {
            auto opConsumers = _feasibleBarrierScheduler.getConsumerOps(source);

            for (auto consumer = opConsumers.begin(); consumer != opConsumers.end(); ++consumer) {
                _feasibleBarrierScheduler._log.trace("Step-1.2 Adding consumer task ID {0} to barrier ID {1}",
                                                     BarrierScheduler::getUniqueID(*consumer),
                                                     currentBarrier->getAttr("id"));

                barrierConsumersItr->second.second.insert(*consumer);
            }
        } else {
            VPUX_THROW("Error unable to find the wait tasks for barrier ID {0}", currentBarrier->getAttr("id"));
        }

        // Step-1.3
        if (barrierPrevious) {
            auto barrierConsumersItr = _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.find(barrierPrevious);

            if (barrierConsumersItr != _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.end()) {
                _feasibleBarrierScheduler._log.trace("Step-1.3 Adding consumer task ID {0} to barrier ID {1}",
                                                     BarrierScheduler::getUniqueID(source), barrierPrevious->getAttr("id"));

                barrierConsumersItr->second.second.insert(source);
            } else {
                VPUX_THROW("Not found");
            }
        }
    }  // foreach producer
}

void BarrierScheduler::barrierTransitionStructure::maintainInvariantTemporalChange(const ScheduledOpInfo& sinfo,
                                                                                   mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log.trace("Maintain an invariant temporal change, the scheduled time is {0}, task ID is "
                                         "{1} the barrier index is {2} and the slot cout is {3}",
                                         sinfo._scheduleTime, BarrierScheduler::getUniqueID(sinfo._op),
                                         sinfo._barrierIndex, sinfo._producerSlotCount);

    //              B_prev
    // curr_state : Prod_list={p_0, p_1, ... p_n}-->B_curr
    // event: Prod_list={q_0}->B_curr_new
    //
    // scheduler says it want to associate both B_old and B_new to the
    // same physical barrier.
    //
    // Restore Invariant:
    // Step-1.1: create a new barrier task (B_new).
    // Step-1.2: update B_curr
    //        a. producers: B_curr is now closed so update its producers
    //        b. consumers: for each (p_i, u) \in P_old x V
    //                      add u to the consumer list of B_old
    // Step-1.3: update B_prev
    //           consumers: add p_i \in P_old to the consumer list of
    //                      B_prev. This is because B_prev and B_curr
    //                      are associated with same physical barrier.
    // Step-2: B_prev = B_curr , B_curr = B_curr_new , Prod_list ={q0}
    mlir::Operation* previousBarrier = _previousBarrierTask;
    mlir::Operation* currentBarrier = _currentBarrierTask;
    mlir::Operation* barrierEnd = NULL;
    mlir::Operation* newCurrentBarrier = barrierEnd;

    newCurrentBarrier = createNewBarrierTask(sinfo, builder);
    VPUX_THROW_UNLESS(newCurrentBarrier != barrierEnd, "Error newly created barrier is Null");

    // STEP-1 
    if (currentBarrier != barrierEnd) {
        _feasibleBarrierScheduler._log.trace("The ID of barrier currentBarrier is {0}", currentBarrier->getAttr("id"));
        processCurrentBarrierProducerListCloseEvent(currentBarrier, previousBarrier);
    }

    // STEP-2 
    _previousBarrierTask = _currentBarrierTask;
    _currentBarrierTask = newCurrentBarrier;
    _producers.clear();
    addScheduledTaskToProducerList(sinfo);
}

void BarrierScheduler::barrierTransitionStructure::addScheduledTaskToProducerList(const ScheduledOpInfo& sinfo) {
    auto scheduled_op = sinfo._op;

    _feasibleBarrierScheduler._log.trace("Adding task {0} to the producer list of the barrier transition structure",
                                         _currentBarrierTask->getAttr("id"));
    _producers.insert(scheduled_op);
}

mlir::Operation* BarrierScheduler::barrierTransitionStructure::createNewBarrierTask(const ScheduledOpInfo& sinfo,
                                                                                    mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log.trace("Creating a new virtual barrier task");

    static size_t barrierTaskId = 1UL;

    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(sinfo._op->getLoc());
    newBarrier->setAttr(virtualIdAttrName, getIntAttr(newBarrier->getContext(), barrierTaskId));

    std::set<mlir::Operation*> newBarrierProducers{};
    std::set<mlir::Operation*> newBarrierConsumers{};

    _feasibleBarrierScheduler._configureBarrierOpUpdateWaitMap.insert(
            std::make_pair(newBarrier, std::make_pair(newBarrierProducers, newBarrierConsumers)));

    _feasibleBarrierScheduler._log.trace("Created a new barrier task with barrier ID {0} after task id {1}",
                                         barrierTaskId, BarrierScheduler::getUniqueID(sinfo._op));

    barrierTaskId++;
    return newBarrier;
}