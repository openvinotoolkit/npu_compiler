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

#include "vpux/compiler/core/prefetch_edge_generator.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

vpux::PrefetchEdgeGenerator::PrefetchEdgeGenerator(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo)
        : _log(Logger::global().nest("chain-pipelining", 0)), _scheduledOps(initialSchedule), _depsInfo(depsInfo) {
    // TODO: consider storing ops in a struct with
    // opIdx, time, out-degree, size, hasActiveResource, isData
    // sort by time -> out-degree -> size -> opIdx
}

bool vpux::PrefetchEdgeGenerator::prefetchConstraintsSatisifed(ScheduledOpInfo* dataOp, ScheduledOpInfo* computeOp,
                                                               size_t currentComputeOpLevel) {
    // constraints for prefetching limiting the prefetch so that operations are not prefetched
    // too early, this includes levels of compute operations where level 0 is the previous compute
    // operation scheduled, level 1 is the compute before level 0, etc. as well as arbitrarty time
    // constraint where operations can only be prefetched a certain time before

    if (dataOp->isDataOp()) {
        // if a data op, check level and time constraints
        if (_depsInfo.getOpDeps(dataOp->op_).empty()) {
            // const prefetching, operation has no dependencies
            if (currentComputeOpLevel > PREFETCH_LEVEL_LIMIT_CONST) {
                // const level difference constraint
                return false;
            }
        } else {
            // activation prefetching, operation contains dependencies
            if (currentComputeOpLevel > PREFETCH_LEVEL_LIMIT_ACT) {
                // activation level difference constraint
                return false;
            }
        }

        // time difference constraint
        if (dataOp->time_ - computeOp->time_ > PREFETCH_TIME_LIMIT) {
            return false;
        }
    }

    // NOTE: in future constraints taking into account number of cycles

    return true;
}

bool vpux::PrefetchEdgeGenerator::allDataOpDependenciesExecuted(operationIdxType dataIdx) {
    // condition will be false in cases of consts such as weight, weight table
    // however for tiled activations the DMAs will have a dependency, prevent
    // the activation from being prefetched and reducing the availible free NNCMX
    // size as it will not be scheduled at that time but some other data op might

    // check if all dependencies of the operations were executed
    for (auto opDepIdx : _depsInfo.getOpDeps(dataIdx)) {
        // if a dependency was not executed this op can not be prefetched at this time
        if (_executedOps.find(opDepIdx) == _executedOps.end()) {
            return false;
        }
    }

    return true;
}

bool vpux::PrefetchEdgeGenerator::canDataOpBePrefetched(ScheduledOpInfo* dataOp) {
    // check if the data op can be prefetched - satisfies all the below conditions

    // if not data op
    if (!dataOp->isDataOp()) {
        return false;
    }

    // if op has no active resources
    if (!dataOp->hasActiveResource()) {
        return false;
    }

    // if op already prefetched
    if (_prefetchedDataOps.find(dataOp->op_) != _prefetchedDataOps.end()) {
        return false;
    }

    // if operation has some unscheduled dependencies
    if (!allDataOpDependenciesExecuted(dataOp->op_)) {
        return false;
    }

    // if all conditions satisfied, the op can be prefetched
    return true;
}

bool vpux::PrefetchEdgeGenerator::isPrefetchSinkOperation(ScheduledOpInfo* op) {
    auto execOp = _depsInfo.getExecuteOpAtIndex(op->op_);
    uint32_t numUnits = 0;
    const auto executor = IERT::IERTDialect::getExecutor(execOp, numUnits);
    if (executor.getLeafReference() != VPU::ExecutorKindAttr::get(execOp->getContext(), VPU::ExecutorKind::NCE)) {
        return false;
    }
    return op->isTrueComputeOp() && op->hasActiveResource();
}

vpux::PrefetchEdgeGenerator::prefetchMap vpux::PrefetchEdgeGenerator::generatePrefetchEdges() {
    _log.trace("Creating pipelining chains");

    // track the level of the current compute op
    size_t currentComputeOpLevel;
    auto computeOp = _scheduledOps.begin();
    // skip input op, mark input as executed
    // _executedOps.insert(computeOp->op_);
    // ++computeOp;
    auto dataOp = computeOp;

    while (computeOp != _scheduledOps.end()) {
        // find compute op
        if (isPrefetchSinkOperation(computeOp)) {
            // find first possible data op to overlap with the compute
            currentComputeOpLevel = 1;
            // NOTE: data op must be after compute
            dataOp = computeOp;
            // advance to next op
            ++dataOp;
            // store max free size
            vpux::AddressType maxFreeSize = computeOp->freeCmx_;

            // iterate over the following ops
            while (dataOp != _scheduledOps.end()) {
                // 1. verify prefetching constraints met, if not move to next compute
                if (isPrefetchSinkOperation(dataOp) && dataOp->resourceSize() > 100) {
                    auto execOp = _depsInfo.getExecuteOpAtIndex(dataOp->op_);
                    auto nce = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(execOp.body().front().front());
                    if (nce && nce.task_type() != vpux::VPUIP::NCETaskType::ELTWISE) {
                        ++currentComputeOpLevel;
                    }
                }
                if (!prefetchConstraintsSatisifed(dataOp, computeOp, currentComputeOpLevel)) {
                    std::cout << "condition not satisifed" << std::endl;
                    break;
                }

                // 2. all constraints met, try to find a prefetch-able data op
                if (dataOp->isOriginalOp() && computeOp->isOriginalOp() && canDataOpBePrefetched(dataOp)) {
                    auto dataOpSize = dataOp->resourceSize();

                    // retrieve max free size to data op
                    auto temp = computeOp;
                    while (temp != dataOp && temp->time_ < dataOp->time_) {
                        maxFreeSize = std::min(maxFreeSize, temp->freeCmx_);
                        ++temp;
                    }

                    if (dataOpSize < maxFreeSize && computeOp->time_ < dataOp->time_) {
                        // ensure the data operation will fit through all ops scheduled intermediatly
                        _log.trace("data op = '{0}' will fit during compute = '{1}' with time dif = '{2}' and level "
                                   "dif '{3}'",
                                   dataOp->op_, computeOp->op_, (dataOp->time_ - computeOp->time_),
                                   currentComputeOpLevel);
                        std::cout << dataOp->op_ << " during " << computeOp->op_ << std::endl;
                        // store the prefetch edge
                        _prefetchEdges[computeOp->op_].insert(std::make_pair(dataOp->op_, dataOp->time_));
                        _prefetchedDataOps.insert(dataOp->op_);
                        // reduce max free size with this data op size
                        maxFreeSize = maxFreeSize - dataOpSize;

                        // update free size for all compute ops to the prefetch op
                        auto temp = computeOp;
                        while (temp != dataOp && temp->time_ < dataOp->time_) {
                            if (isPrefetchSinkOperation(temp)) {
                                temp->freeCmx_ -= dataOpSize;
                            }
                            ++temp;
                        }
                    } else {
                        std::cout << dataOp->op_ << " NOT during " << computeOp->op_ << " " << maxFreeSize << " "
                                  << dataOpSize << std::endl;
                    }
                }

                // advance to data next op
                ++dataOp;
            }
        }
        // mark the operation as executed to store dependencies
        _executedOps.insert(computeOp->op_);
        // advance to next compute op
        ++computeOp;
    }

    _log.trace("prefetch edge count: '{0}'", _prefetchEdges.size());

    return _prefetchEdges;
}
