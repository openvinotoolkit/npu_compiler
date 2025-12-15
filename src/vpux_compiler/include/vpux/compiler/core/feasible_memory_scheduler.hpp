//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"
#include "vpux/compiler/utils/partitioner.hpp"

namespace VPUNN {
class VPUCostModel;
}  // namespace VPUNN

namespace vpux {

class FeasibleMemoryScheduler final {
public:
    using operationIdxType = size_t;
    // Operation type
    enum class EOpType {
        ORIGINAL_OP = 0,
        ORIGINAL_PREFETCHED_OP = 1,
        ORIGINAL_SPILL_READ_OP = 2,
        ORIGINAL_SPILL_WRITE_OP = 3,
        IMPLICIT_SPILL_READ_OP = 4,
        IMPLICIT_SPILL_WRITE_OP = 5,
        IMPLICIT_SPILL_WRITE_FRAG_OP = 6
    };
    // QueueType represents independent execution queue identified by
    // executor kind and abstract ID that allows to differentiate
    // different FIFOs on same executor (e.g. 2 separate DMA channels on single
    // DMA engine)
    struct QueueType {
        VPU::ExecutorKind execKind;
        uint8_t id = 0;

        bool operator<(const QueueType& other) const {
            if (execKind == other.execKind) {
                return id < other.id;
            }
            return execKind < other.execKind;
        }
        bool operator==(const QueueType& other) const {
            return execKind == other.execKind && id == other.id;
        }
        bool operator!=(const QueueType& other) const {
            return !(*this == other);
        }
    };
    struct QueueAndCycleType {
        QueueType queueType;
        llvm::BitVector execMask;
        size_t cycle;
    };
    // Core struct in feasible memory scheduler
    struct HeapElement {
        explicit HeapElement(operationIdxType opIdx, const QueueAndCycleType& queueAndCycle, size_t cycleCost = 1UL,
                             EOpType op_type = EOpType::ORIGINAL_OP, mlir::Value spillBuffer = nullptr)
                : op_(opIdx),
                  executorInstanceMask_(queueAndCycle.execMask),
                  queueType_(queueAndCycle.queueType),
                  cycleBegin_(queueAndCycle.cycle),
                  cycleEnd_(queueAndCycle.cycle + cycleCost),
                  opType_(op_type),
                  spillBuffer_(spillBuffer) {
        }
        bool operator==(const HeapElement& other) const {
            return (op_ == other.op_) && (cycleBegin_ == other.cycleBegin_) && (spillBuffer_ == other.spillBuffer_);
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isSpillWriteOp() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) || (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillReadOp() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_READ_OP);
        }
        bool isSpillOp() const {
            return isSpillWriteOp() || isSpillReadOp();
        }
        operationIdxType op_;
        // Mask identifying indexes of given executor type to be used for running this operation
        llvm::BitVector executorInstanceMask_;
        QueueType queueType_;
        size_t cycleBegin_;
        size_t cycleEnd_;
        EOpType opType_;
        mlir::Value spillBuffer_;
    };

    // Sort pair by the second arg - operation index (=async-deps-index) which is aligned with order in IR
    struct operationIdxSort {
        // TODO will be replaced by DPU order heuristic
        bool operator()(const operationIdxType& op1, const operationIdxType& op2) const {
            return op1 < op2;
        }
    };
    // Sort heap by earliest begin cycle
    struct CycleBeginMinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) const;
    };
    // Sort heap by earliest end cycle
    struct CycleEndMinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) const;
    };
    // Struct storing CMX address space
    struct IntervalInfo {
        void invalidate() {
            begin_ = std::numeric_limits<size_t>::max();
            end_ = std::numeric_limits<size_t>::min();
        }
        size_t length() const {
            VPUX_THROW_UNLESS(begin_ <= end_, "Invalid resource interval");
            return end_ - begin_ + 1;
        }
        IntervalInfo(): begin_(), end_() {
            invalidate();
        }
        IntervalInfo(size_t ibeg, size_t iend): begin_(ibeg), end_(iend) {
        }
        IntervalInfo(size_t ibeg, size_t iend, mlir::Value buffer): begin_(ibeg), end_(iend), buffer_(buffer) {
        }
        bool operator==(const IntervalInfo& other) const {
            return (begin_ == other.begin_) && (end_ == other.end_) && (buffer_ == other.buffer_);
        }
        size_t begin_;
        size_t end_;
        mlir::Value buffer_;
    };
    // Struct used to output the scheduled op info
    struct ScheduledOpInfo {
        ScheduledOpInfo(operationIdxType op, EOpType type, size_t cycleBegin, size_t cycleEnd,
                        SmallVector<IntervalInfo> inputResourceInfo, SmallVector<IntervalInfo> outputResourceInfo)
                : op_(op),
                  opType_(type),
                  cycleBegin_(cycleBegin),
                  cycleEnd_(cycleEnd),
                  inputResourceInfo_(inputResourceInfo),
                  outputResourceInfo_(outputResourceInfo) {
        }

        ScheduledOpInfo(operationIdxType op, EOpType type, size_t cycleBegin, vpux::AddressType freeCmx, bool isDataOp)
                : op_(op), opType_(type), cycleBegin_(cycleBegin), freeCmx_(freeCmx), isDataOp_(isDataOp) {
        }

        ScheduledOpInfo() = default;

        bool operator==(const ScheduledOpInfo& other) const {
            return (other.op_ == op_) && (other.opType_ == opType_);
        }
        ScheduledOpInfo& operator=(const HeapElement& hElement) {
            op_ = hElement.op_;
            opType_ = hElement.opType_;
            return *this;
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP) ||
                   (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP) || (opType_ == EOpType::ORIGINAL_SPILL_READ_OP);
        }
        bool isOriginalSpillWriteOp() const {
            return (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP);
        }
        bool isOriginalSpillReadOp() const {
            return (opType_ == EOpType::ORIGINAL_SPILL_READ_OP);
        }
        bool isSpillWrite() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) || (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillWriteFrag() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillRead() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_READ_OP);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        StringLiteral opTypeName() const {
            if (opType_ == EOpType::ORIGINAL_OP) {
                return StringLiteral("ORIGINAL");
            } else if (opType_ == EOpType::ORIGINAL_PREFETCHED_OP) {
                return StringLiteral("ORIGINAL_PREFETCHED");
            } else if (opType_ == EOpType::ORIGINAL_SPILL_READ_OP) {
                return StringLiteral("ORIGINAL_SPILL_READ");
            } else if (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP) {
                return StringLiteral("ORIGINAL_SPILL_WRITE");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_READ_OP) {
                return StringLiteral("IMPLICIT_SPILL_READ");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) {
                return StringLiteral("IMPLICIT_SPILL_WRITE");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP) {
                return StringLiteral("IMPLICIT_SPILL_WRITE_FRAG");
            }
            return StringLiteral("UNDEFINED");
        }
        size_t numOfOutputResources() const {
            return outputResourceInfo_.size();
        }
        bool hasActiveOutputResource() const {
            for (size_t resourceIdx = 0; resourceIdx < numOfOutputResources(); resourceIdx++) {
                if (isActiveOutputResource(resourceIdx)) {
                    return true;
                }
            }
            return false;
        }
        bool isActiveOutputResource(size_t idx) const {
            return (outputResourceInfo_[idx].begin_ <= outputResourceInfo_[idx].end_);
        }
        bool isDataOp() const {
            return isDataOp_;
        }
        size_t resourceSize() const {
            size_t size = 0;
            for (size_t resourceIdx = 0; resourceIdx < numOfOutputResources(); resourceIdx++) {
                if (isActiveOutputResource(resourceIdx)) {
                    size += endOutputResource(resourceIdx) - beginOutputResource(resourceIdx);
                }
            }
            return size;
        }
        size_t beginOutputResource(size_t idx) const {
            return outputResourceInfo_[idx].begin_;
        }
        size_t endOutputResource(size_t idx) const {
            return outputResourceInfo_[idx].end_;
        }
        mlir::Value getOutputBuffer(size_t idx) const {
            return outputResourceInfo_[idx].buffer_;
        }

        size_t numOfInputResources() const {
            return inputResourceInfo_.size();
        }
        bool hasActiveInputResource() const {
            for (size_t resourceIdx = 0; resourceIdx < numOfInputResources(); resourceIdx++) {
                if (isActiveInputResource(resourceIdx)) {
                    return true;
                }
            }
            return false;
        }
        bool isActiveInputResource(size_t idx) const {
            return (inputResourceInfo_[idx].begin_ <= inputResourceInfo_[idx].end_);
        }
        size_t beginInputResource(size_t idx) const {
            return inputResourceInfo_[idx].begin_;
        }
        size_t endInputResource(size_t idx) const {
            return inputResourceInfo_[idx].end_;
        }
        mlir::Value getInputBuffer(size_t idx) const {
            return inputResourceInfo_[idx].buffer_;
        }

        operationIdxType op_{};
        EOpType opType_{EOpType::ORIGINAL_OP};
        int64_t cycleBegin_{};
        int64_t cycleEnd_{};
        vpux::AddressType freeCmx_{};
        bool isDataOp_{false};
        SmallVector<IntervalInfo> inputResourceInfo_;
        SmallVector<IntervalInfo> outputResourceInfo_;
        QueueType queueType{VPU::ExecutorKind::DMA_NN, 0};
        llvm::BitVector executorInstanceMask;
    };
    using ScheduledOpInfoVec = SmallVector<ScheduledOpInfo, 1>;

    // Struct storing buffer info for allocation order
    struct BufferOrder {
        BufferOrder(mlir::Value buffer, vpux::AddressType size, size_t outDegree)
                : buffer(buffer), size(size), outDegree(outDegree) {
        }
        mlir::Value buffer;
        vpux::AddressType size;
        size_t outDegree;
    };
    // Struct storing eviction policy info for buffers
    struct EvictionCandidate {
        EvictionCandidate(size_t priority, size_t earliestConsumerIdx, size_t size, operationIdxType bufferWriterIdx,
                          size_t outputIdx, mlir::Value buffer)
                : priority_(priority),
                  earliestConsumerIdx_(earliestConsumerIdx),
                  size_(size),
                  bufferWriterIdx_(bufferWriterIdx),
                  outputIdx_(outputIdx),
                  buffer_(buffer) {
        }
        size_t priority_;
        operationIdxType earliestConsumerIdx_;
        size_t size_;
        operationIdxType bufferWriterIdx_;
        size_t outputIdx_;
        mlir::Value buffer_;
    };
    // Eviction priority policy for buffers
    struct EvictionPriority {
        bool operator()(const EvictionCandidate& ec1, const EvictionCandidate& ec2) const {
            // higher eviction priority
            if (ec1.priority_ != ec2.priority_) {
                return ec1.priority_ > ec2.priority_;
            }

            // last first consumer
            if (ec1.earliestConsumerIdx_ != ec2.earliestConsumerIdx_) {
                return ec1.earliestConsumerIdx_ > ec2.earliestConsumerIdx_;
            }

            // smaller size
            if (ec1.size_ != ec2.size_) {
                return ec1.size_ < ec2.size_;
            }

            // place in IR
            if (ec1.bufferWriterIdx_ != ec2.bufferWriterIdx_) {
                return ec1.bufferWriterIdx_ > ec2.bufferWriterIdx_;
            }

            // same index, multiple outputs, check idx
            return ec1.outputIdx_ > ec2.outputIdx_;
        }
    };

    // Represents a snapshot of the scheduler's state, allowing rollback to a previous state
    struct ScheduleStateSnapshot;

public:
    FeasibleMemoryScheduler(VPU::MemoryKind memKind, VPU::MemoryKind secondLvlMemKind, MemLiveRangeInfo& liveRangeInfo,
                            AsyncDepsInfo& depsInfo, Logger log, LinearScan<mlir::Value, LinearScanHandler>& scan,
                            config::ArchKind arch, std::shared_ptr<VPUNN::VPUCostModel> costModel,
                            int64_t nceClusterCount, int64_t dmaCount, bool enableScheduleStatistics,
                            bool optimizeFragmentation, bool activelySpillForPrefetching);

public:
    ScheduledOpInfoVec generateSchedule();
    void cleanUpAndLogSchedule(ScheduledOpInfoVec& scheduledOps);
    void printFragmentFixCountLog() const;

private:
    bool init();
    void clearLists();
    void schedulingLoop();
    void initializeReadyLists();
    void identifyDataOps();

    // DAG maintenance
    SmallVector<operationIdxType> reduceInDegreeOfAdjacentOperations(operationIdxType opIdx);
    SmallVector<operationIdxType> unlockNewReadyOps(const HeapElement& hElement);
    void distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps);

    // heap maintenance
    void pushToCycleBeginHeap(const HeapElement& elem);
    bool unscheduledOpsOnQueue(const QueueType& queueType);
    void moveFromCycleBeginToCycleEndHeap();
    size_t findMinScheduledQueueCycle();

    // scheduling type utils
    bool isDataOp(operationIdxType opIdx);
    bool isNoInputDepComputeOp(operationIdxType opIdx);

    // cycle and executor utils
    VPU::ExecutorKind getExecutorType(operationIdxType opIdx);
    QueueType getQueueType(operationIdxType opIdx);
    size_t getCurrentCycle(operationIdxType opIdx, bool spilled = false);
    void insertInOpIdxCycleEndMap(const operationIdxType& opIdx, const size_t& endCycle);
    size_t getEarliestComputeBeginCycle(operationIdxType opIdx);
    // Based on operation/buffer properties determine how many executor instances are needed
    // to run given operation
    size_t getOpDemandForExecutorsInstances(operationIdxType opIdx, QueueType queueType);
    size_t getBufferDemandForExecutorsInstances(mlir::Value buffer, QueueType queueType);
    // Return mask of executor instances specifying on which executor indexes given operation
    // is to be executed on
    llvm::BitVector getExecutorInstanceMask(size_t numOfNeededInstances, QueueType queueType);
    llvm::BitVector getExecutorInstanceMaskForOp(operationIdxType opIdx, QueueType queueType);
    llvm::BitVector getExecutorInstanceMaskForBuffer(mlir::Value buffer, QueueType queueType);
    QueueAndCycleType getCurrentCycleAndExecutorInstanceMask(operationIdxType opIdx, size_t depEndCycle = 0);
    QueueAndCycleType getCurrentCycleAndExecutorInstanceMaskForSpill(mlir::Value buffer, EOpType spillType,
                                                                     size_t depEndCycle = 0);
    size_t spilledOperationCycleCost(mlir::Value spilledBuffer);
    size_t operationCycleCost(operationIdxType opIdx);
    void alignExecutors(size_t nextAvailableCycle);
    void updateCurrentCycleForExecutor(QueueType queueType, llvm::BitVector executorInstanceMask, size_t newCycle);

    // scheduling various operation types
    size_t scheduleSpilledOpBuffer(operationIdxType opIdx, mlir::Value* buffer);
    size_t scheduleDependencies(operationIdxType opIdx);
    size_t scheduleOp(operationIdxType opIdx, EOpType opType = EOpType::ORIGINAL_OP);
    bool freeMemoryResources(const HeapElement& hElement);

    // handling buffers
    mlir::DenseSet<mlir::Value> getBuffersToAllocateForOp(operationIdxType opIdx);
    bool canAllocBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate);
    void sortAndAllocateBuffers(mlir::DenseSet<mlir::Value>& buffersToAllocate);
    SmallVector<mlir::Value> getNonAliveBuffersUsedByOperation(operationIdxType opIdx);
    SmallVector<mlir::Value> sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers);
    void updateBufferCycleUseAndProducer(size_t opIdx, size_t opCycleEnd, const mlir::Value buffer,
                                         bool isNewProducer = false);
    void createBufferAsyncIdxMap();

    // scheduling loops
    void unscheduleAllCompletingOps();
    size_t getOperationLevel(operationIdxType opIdx, bool isSpilled = false);
    void prefetchOps(ArrayRef<std::pair<operationIdxType, size_t>> scheduledOps,
                     mlir::DenseSet<mlir::Value>& buffersToAllocate, bool checkMemoryFragmentation);
    void scheduleNonComputeOps();
    void scheduleComputeOps();

    // eviction utility
    void evictActiveOp(EvictionCandidate evictionCandidate);
    size_t evictionPriority(operationIdxType writerOpIdx, mlir::Value buffer);
    operationIdxType getEarliestConsumerIdx(operationIdxType opIdx);
    EvictionCandidate chooseCandidateForEviction(const mlir::DenseSet<mlir::Value>& aliveBuffers);
    void forceScheduleActiveOpEviction();
    size_t getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer);

    // reporting and schedule generation
    void populateScheduledOps(const HeapElement& scheduledOp);

    // Actively spill for aggressive prefetch to resolve memory fragmentation
    // When detected prefetching failure due to memory fragmentation, this scheduler actively spill out eviction
    // candidates until enough memory is available for prefetching
    // If no eviction candidate could resolve the fragmentation, the scheduler will roll back to the state before
    // prefetching and memory fragmentation happens
    std::optional<EvictionCandidate> chooseCandidateForEvictionToSupportPrefetch(
            mlir::DenseSet<mlir::Value>& aliveBuffers, mlir::DenseMap<mlir::Value, operationIdxType>& bufferProducer,
            LinearScan<mlir::Value, LinearScanHandler>& scan);
    void forceSpillingForPrefetch();
    // Prepare the variables to track the big-container changes/deltas during scheduling compute ops
    // This is needed to be able to rollback the changes when prefetching fails due to fragmentation
    void resetScheduleComputeOpsDeltas();
    // Spill the eviction candidate that is selected by `chooseCandidateForEvictionToSupportPrefetch`
    void performEvictionAndScheduling(const EvictionCandidate& evictionCandidate);
    size_t getOpCmxDemand(operationIdxType opIdx);

private:
    Logger _log;
    // memory space, which will be allocated
    VPU::MemoryKind _memKind;
    // second level mem space, used as source of data or destination for spilling
    VPU::MemoryKind _secondLvlMemKind;
    // information about op buffers
    MemLiveRangeInfo& _liveRangeInfo;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // architecture kind
    config::ArchKind _archKind;
    // VPUNN cost model
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    // NCE cluster count
    int64_t _nceClusterCount;
    // DMA ports count
    int64_t _numDMAPorts;
    // Flag for enabling additional statistic related logic
    bool _enableScheduleStatistics;
    // Flag for enabling fragmentation optimization
    bool _optimizeFragmentation;
    // there are 8 barriers per cluster
    // TODO: E93149 update barrier usage
    const int64_t _barrierPerCluster = 8;
    // TODO: E106645 issue with heap order, fix ordering issue and convert back to a heap
    // heap with earliest operation begin cycle
    std::set<HeapElement, CycleBeginMinHeapOrdering> _cycleBeginHeap;
    // heap with earliest operation end cycle
    std::set<HeapElement, CycleEndMinHeapOrdering> _cycleEndHeap;
    // compute operations with 0 in-degree, optimal to schedule, that strictly preserve IR order
    std::set<operationIdxType, operationIdxSort> _readyComputeOps;
    // compute DMA operations with 0 in-degree, that do not necessarily preserve IR order
    std::set<operationIdxType, operationIdxSort> _readyDMAOps;
    // data operations with 0 in-degree
    std::set<operationIdxType, operationIdxSort> _readyDataOps;
    // spilled operation which are ready to be rescheduled
    mlir::DenseMap<mlir::Value, operationIdxType> _readySpilledOps;
    // store operation spilled buffers
    llvm::DenseMap<operationIdxType, llvm::DenseSet<mlir::Value>> _spillBufferMap;
    // operations which do not belong to main compute chain for activations from network
    // input to output. Such operations need to be distinguished from other ops as scheduler
    // is focused on scheduling ops along compute chain. Such operation will only be considered
    // for scheduling once all input dependency data and/or compute ops have been executed
    std::set<operationIdxType, operationIdxSort> _nonComputeChainOps;
    // operation in-degree, number of incoming edges
    std::unordered_map<operationIdxType, size_t> _inDegreeTable;
    // operation out-degree, number of outgoing edges
    std::unordered_map<operationIdxType, size_t> _outDegreeTable;
    // level limit for op prefetching, do not check later ready operations
    const size_t _prefetchingLevelLimit = 35;
    // level of DMAs corresponding to how many DPUs will execute before this DMA is needed
    mlir::DenseMap<operationIdxType, size_t> _dataOpLevels;
    // contains the operation writing to the buffer
    mlir::DenseMap<mlir::Value, operationIdxType> _bufferProducer;
    // contains last used cycle of buffer
    mlir::DenseMap<mlir::Value, size_t> _bufferLastCycleUse;
    // container for the schedule output
    ScheduledOpInfoVec _scheduledOps;
    // outputs of the graph
    llvm::DenseSet<operationIdxType> _outputOps;
    // operation level vector
    mlir::SmallVector<size_t> _opLevelVec;
    // order for compute ops from IR
    std::map<QueueType, SmallVector<operationIdxType>> _computeOpOrder;
    // cycle pipeline for every executor. Vector element type is to support multiple instances of
    // same executor type in case where scheduler needs to be aware of it
    // TODO: Currently scheduler supports only multiple DMA executors (ports)
    std::map<QueueType, SmallVector<size_t>> _executorPipelines = {
            {{VPU::ExecutorKind::DMA_NN}, {1}},    {{VPU::ExecutorKind::DPU}, {1}},
            {{VPU::ExecutorKind::NCE}, {1}},       {{VPU::ExecutorKind::SHAVE_NN}, {1}},
            {{VPU::ExecutorKind::SHAVE_ACT}, {1}}, {{VPU::ExecutorKind::M2I}, {1}}};

    // spilled operation cycle cost
    mlir::DenseMap<mlir::Value, size_t> _spillBufferCycleCost;

    // Map of root buffers and vector of operations (operationIdx) writing to this buffer
    // This is used to sort the shared buffers based on their out-degree using the _outDegreeTable
    // so that we can prefetch them at the start in the CMX not somewhere in the middle to get a contigious CMX
    // space
    mlir::DenseMap<mlir::Value, SmallVector<operationIdxType>> _bufferOpIdxMap;

    mlir::DenseMap<operationIdxType, size_t> _opIdxEndCycleMap;

    std::set<EvictionCandidate, EvictionPriority> _evictionCandidatesCache;

    llvm::BitVector _isDataOp;

    //
    // Actively spilling
    // To resolve memory fragmentation, the scheduler can actively spill out eviction candidates
    // to make more consecutive memory for prefetching
    //
    // option to enable actively spilling
    bool _activelySpillForPrefetching;
    // flag to indicate if prefetching failed due to memory fragmentation
    // this flag is set true during prefetching if memory fragment is detected and used to trigger active spilling
    bool _prefetchFailedDueToFragmentation = false;
    size_t _prefetchFragmentationFailureCount = 0;
    size_t _prefetchFragmentationFailureFixedCount = 0;

    // track buffers fragmented during prefetching
    mlir::DenseSet<mlir::Value> _fragmentedBuffers;
    // actively spilled eviction candidate
    std::optional<FeasibleMemoryScheduler::EvictionCandidate> _evictionCandidateToSupportPrefetch;

    /**
     * @brief Snapshot related members for actively spilling
     *
     * Store the stages before and after the current compute ops scheduling to be able to roll back if actively spilling
     * is needed. Everytime `scheduleComputeOps` is called, these variables are reset. And during `scheduleComputeOps`
     * execution, the changes made to these variables are tracked. The reason of not saving _bufferProducer,
     * _bufferLastCycleUse and _opIdxEndCycleMap to snapshot is to save the compilation time. Saving the whole maps is
     * more time-consuming than using the deltas to revert the changes.
     * `_new*` variables store newly added buffers/ops during `scheduleComputeOps`
     * `_original*` variables store previous scheduled but changed buffers/ops during `scheduleComputeOps`
     */

    // _bufferProducer deltas
    mlir::DenseSet<mlir::Value> _newBufferProducersFromScheduleComputeOps;
    mlir::DenseMap<mlir::Value, operationIdxType> _originalBufferProducersFromScheduleComputeOps;
    // _bufferLastCycleUse deltas
    mlir::DenseSet<mlir::Value> _newBufferLastCycleUsesFromScheduleComputeOps;
    mlir::DenseMap<mlir::Value, size_t> _originalBufferLastCycleUseFromScheduleComputeOps;
    // _opIdxEndCycleMap deltas
    mlir::DenseMap<operationIdxType, size_t> _originalOpIdxEndCycleMapFromScheduleComputeOps;
    mlir::DenseSet<operationIdxType> _newOpsFromScheduleComputeOps;
};

}  // namespace vpux
