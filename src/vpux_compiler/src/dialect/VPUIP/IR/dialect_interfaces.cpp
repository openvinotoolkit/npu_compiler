//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace detail {

/*
 * Dispatch pre-inlining callOp processing to
 * a specific processor if conditions are met
 */
struct CallOPPreInliner;
struct CallOPPreInlinerVisitor {
    CallOPPreInlinerVisitor(Logger log = Logger::global());

    template <class PreInliner, class... Args>
    void addPreInliner(Args&&... args) {
        preInliners.push_back(std::make_unique<PreInliner>(_log, std::forward<Args>(args)...));
    }

    void visit(mlir::Operation* op, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const;

private:
    Logger _log;
    std::vector<std::unique_ptr<CallOPPreInliner>> preInliners;
};

/*
 * Once callOp is categorized as a part of batched processing,
 * which means is has `debatched` tag at the moment of the implementation,
 * this preprocessor takes care about suitable resource mapping.
 * By default during each callOp compilation it's supposed to occupy
 * whole NPU resources like tiles/CMX, DDR and so on.
 * Having multiple callOp responsible for processing different lines of a batched tensor,
 * this means that we face to resource concurrency among different callOp because
 * eventually all of them are mapped on the same CMXs and DDR addresses using same offsets.
 * To overcome this resouce allocation limitation this preInliner was conceived.
 *
 * The preinliner responsibilities are the following:
 *  a) detemine which a callOp index I is from a batched dimension range [0...N]
 *  b) remap CMX, cluster_id, section_id etc. from the range [0...T], where T is a compilation tile count, to the range
 * [0 + I... (T + I) / N] c) Apply similar logic to DDR allocation (TODO - E###-131884)
 */
struct CallOPPreInliner {
    CallOPPreInliner(Logger&& log): _log(log), dispatcher(_log) {
    }
    virtual ~CallOPPreInliner() = default;

    struct CMXTypeModifier {
        mutable Logger _log;
        CMXTypeModifier(Logger& log): _log(log) {
        }
        CMXTypeModifier(Logger&& log): _log(std::move(log)) {
        }

        static size_t recalculateIndex(size_t index, const DebatchedCallOpData& callOpData,
                                       size_t totalAvailableTilesCount);
        mlir::Type transform(mlir::Type type, const DebatchedCallOpData& callOpData,
                             size_t totalAvailableTilesCount) const;

        template <class ClusterIdFunctor>
        static SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> modifyOutwardHaloAttrs(
                ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos, ClusterIdFunctor modifier, Logger log);

        template <class ClusterIdFunctor>
        static SmallVector<vpux::VPUIP::HaloRegionAttr> modifyHaloAttrsClusterId(
                ArrayRef<vpux::VPUIP::HaloRegionAttr> haloAttrs, ClusterIdFunctor modifier, Logger log);
    };

    struct FunctionAnalyticBase {
        size_t totalAvailableTilesCount;
        size_t maxDDRBytesAvailable;
        std::map<size_t, std::optional<size_t>> argsOffset;
        std::string to_string() const;
        static FunctionAnalyticBase create(::mlir::func::CallOp callOp);
    };

    struct ResourceDescriptor : public FunctionAnalyticBase {
        DebatchedCallOpData callOpData;
        std::optional<size_t> singleFunctionDDRConsumptionBytes;
        std::string to_string() const;
        static ResourceDescriptor create(::mlir::func::CallOp callOp,
                                         ::mlir::iterator_range<::mlir::Region::iterator> inlinedBlocks, Logger log);

    private:
        ResourceDescriptor() = delete;
    };
    /*
     * Delegates processing to appropriate handler if conditions apt the hadler invocation,
     * depends on operation type and attributes it carrying
     */
    class Dispatcher {
    public:
        struct OpExtractor {
            OpExtractor(mlir::Operation& op): _op(op), _opCounter(0) {
            }
            mlir::Operation* next() {
                // once being called, it returns an operation itself
                if (_opCounter == 0) {
                    _opCounter++;
                    return &_op;
                }

                // Subsequent calls return further stacked operations from enclosed region:
                // only single stacked op is supported as part of an encompassed operation, see TaskOp.
                // In general, we will need to extract next stacked/enclosed operations until exist but at the moment
                // any task op comprises only one operation inside
                if (_opCounter == 1) {
                    _opCounter++;
                    if (auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(_op); taskOp != nullptr) {
                        if (auto innerTaskOp = taskOp.getInnerTaskOp(); innerTaskOp != nullptr) {
                            return innerTaskOp;
                        }
                    }
                }
                return nullptr;
            }

        private:
            mlir::Operation& _op;
            size_t _opCounter = 0;
        };

        struct SpecificOpPreInliner {
            virtual ~SpecificOpPreInliner() = default;
            virtual bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const = 0;
            virtual void reset() {
            }
        };

        struct GreedyModifier final : public SpecificOpPreInliner, private CMXTypeModifier {
            GreedyModifier(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            mutable Logger _log;
        };

        struct CMXModifierForDeclareOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForDeclareOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            bool applyCMX(VPURT::DeclareBufferOp& op, const DebatchedCallOpData& callOpData,
                          size_t totalAvailableTilesCount) const;
            bool applyDDR(VPURT::DeclareBufferOp& op, const DebatchedCallOpData& callOpData,
                          size_t offsetDDRAllocationBytes, size_t maxDDRBytesAvailable) const;
            mutable Logger _log;
        };

        struct CMXModifierForNCEClusterTaskOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForNCEClusterTaskOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            mutable Logger _log;
        };

        struct CMXModifierForSWKernelOp final : public SpecificOpPreInliner, private CMXTypeModifier {
            CMXModifierForSWKernelOp(Logger&& log): CMXTypeModifier(log), _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;

        private:
            mutable Logger _log;
        };

        struct FuncArgumentDeclareModifier final : public SpecificOpPreInliner {
            FuncArgumentDeclareModifier(Logger&& log): _log(std::move(log)) {
            }
            bool apply(mlir::Operation& op, const ResourceDescriptor& resource) const override;
            void reset() override;

        private:
            mutable std::map<size_t, std::optional<size_t>> argsLastRelativeOffsetInCluster;
            mutable Logger _log;
        };

        Dispatcher(Logger& log);
        ~Dispatcher() = default;

        template <class PreInlinerProcessor, class... Args>
        void addPreInlinerProcessor(Args&&... args) {
            specificPreInliners.push_back(std::make_unique<PreInlinerProcessor>(std::forward<Args>(args)...));
        }
        void dispatch(mlir::Operation& op, const ResourceDescriptor& resourse) const;
        void reset();

    private:
        mutable Logger _log;
        std::vector<std::unique_ptr<SpecificOpPreInliner>> specificPreInliners;
    };

    virtual bool isApplicable(mlir::Operation*) const = 0;
    virtual void apply(mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const;

protected:
    Logger _log;
    Dispatcher dispatcher;
};

struct FuncArgOffsetPreInliner : public CallOPPreInliner {
    FuncArgOffsetPreInliner(Logger& log);
    bool isApplicable(mlir::Operation* call) const override;
};

struct BatchedCallOpReorderingPreInliner : public CallOPPreInliner {
    BatchedCallOpReorderingPreInliner(Logger& log);
    bool isApplicable(mlir::Operation* call) const override;
};

/*
 * FuncArgOffsetPreInliner
 */
FuncArgOffsetPreInliner::FuncArgOffsetPreInliner(Logger& log)
        : CallOPPreInliner(log.nest("func-arg-offset-preinliner")) {
    dispatcher.addPreInlinerProcessor<Dispatcher::FuncArgumentDeclareModifier>(log.nest());
}

bool FuncArgOffsetPreInliner::isApplicable(mlir::Operation* call) const {
    if (mlir::isa_and_nonnull<mlir::func::CallOp>(call)) {
        return !vpux::DebatchedCallOpAttributeView::hasReorderingAttr(mlir::dyn_cast<::mlir::func::CallOp>(call));
        ;
    }
    return false;
}

/*
 * BatchedCallOpReorderingPreInliner
 */
BatchedCallOpReorderingPreInliner::BatchedCallOpReorderingPreInliner(Logger& log)
        : CallOPPreInliner(log.nest("batch-reordering-preinliner")) {
    dispatcher.addPreInlinerProcessor<Dispatcher::CMXModifierForDeclareOp>(log.nest());
    dispatcher.addPreInlinerProcessor<Dispatcher::CMXModifierForNCEClusterTaskOp>(log.nest());
    dispatcher.addPreInlinerProcessor<Dispatcher::CMXModifierForSWKernelOp>(log.nest());
    dispatcher.addPreInlinerProcessor<Dispatcher::FuncArgumentDeclareModifier>(log.nest());
    dispatcher.addPreInlinerProcessor<Dispatcher::GreedyModifier>(log.nest());
}

bool BatchedCallOpReorderingPreInliner::isApplicable(mlir::Operation* call) const {
    if (mlir::isa_and_nonnull<mlir::func::CallOp>(call) && call->hasAttr(vpux::DebatchedCallOpAttributeView::name())) {
        return vpux::DebatchedCallOpAttributeView::hasReorderingAttr(mlir::dyn_cast<::mlir::func::CallOp>(call));
    }
    return false;
}

/*
 * CallOPPreInliner
 */

mlir::func::FuncOp getCalledFunction(mlir::func::CallOp callOp) {
    mlir::SymbolRefAttr sym = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
    if (!sym) {
        return nullptr;
    }
    return mlir::dyn_cast_or_null<mlir::func::FuncOp>(mlir::SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

void CallOPPreInliner::apply(mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    ResourceDescriptor resourse =
            ResourceDescriptor::create(mlir::dyn_cast<::mlir::func::CallOp>(call), inlinedBlocks, _log);
    _log.debug("apply CallOPPreInliner on {0}: analytic: {1}", *call, resourse.to_string());

    size_t inlinedBlocksCountIndex = 0;
    std::unordered_map<size_t /*block index*/, size_t /*op index*/> opPerBlockCount;
    using OperationPtrIndexPair = std::pair<mlir::Operation*, size_t>;
    std::optional<OperationPtrIndexPair> openTagOperationCandidate{};
    std::optional<OperationPtrIndexPair> closeTagOperationCandidate{};
    for (mlir::Block& block : inlinedBlocks) {
        opPerBlockCount[inlinedBlocksCountIndex] = 0;
        for (auto& op : block.getOperations()) {
            dispatcher.dispatch(op, resourse);

            if (!openTagOperationCandidate.has_value()) {
                // find first operation for inserting TAG, skip Constants
                // as they will be optimized by canonizerPass
                // and won't appear in a final IR providing that we loose the TAG,
                // as well as beginning of batched-processing operation block
                if (!mlir::isa<vpux::Const::DeclareOp>(op)) {
                    openTagOperationCandidate = std::make_pair(&op, opPerBlockCount[inlinedBlocksCountIndex]);
                    closeTagOperationCandidate = std::make_pair(&op, opPerBlockCount[inlinedBlocksCountIndex]);
                }
            } else if (closeTagOperationCandidate.has_value() && !mlir::isa<mlir::func::ReturnOp>(op)) {
                // find a last operation in the block standing before ReturnOp
                // as it won't be inlined so that we will loose the TAG, as well
                // as an end of batched-processing operation block
                closeTagOperationCandidate.value().first = &op;
                closeTagOperationCandidate.value().second++;
            }
            opPerBlockCount[inlinedBlocksCountIndex]++;
        }
        _log.info("CallOPPreInliner has inlined operations: {0} from block: {1}",
                  opPerBlockCount[inlinedBlocksCountIndex], inlinedBlocksCountIndex);
        VPUX_THROW_UNLESS(closeTagOperationCandidate.has_value() && openTagOperationCandidate.has_value(),
                          "InlinedTagAttribute candidate operation must have been determined");
        _log.debug("InlinedTagAttributes candidate positions: {0} and {1}", openTagOperationCandidate.value().second,
                   closeTagOperationCandidate.value().second);
        inlinedBlocksCountIndex++;
    }
    if (openTagOperationCandidate.has_value()) {
        static constexpr const char* inlinedLocDebugFormat{
                "_inlined_{0}_{1}_{2}"};  // opIndexInOrigFunc / callIndex / totalCalls
        openTagOperationCandidate.value().first->setLoc(
                vpux::appendLoc(openTagOperationCandidate.value().first->getLoc(),
                                formatv(inlinedLocDebugFormat, 0, resourse.callOpData.getCallIndex(),
                                        resourse.callOpData.getBatchSize())
                                        .str()));
        closeTagOperationCandidate.value().first->setLoc(vpux::appendLoc(
                closeTagOperationCandidate.value().first->getLoc(),
                formatv(inlinedLocDebugFormat,
                        closeTagOperationCandidate.value().second -
                                openTagOperationCandidate.value()
                                        .second /*as a relative position in the block starting from the beginning*/,
                        resourse.callOpData.getCallIndex(), resourse.callOpData.getBatchSize())
                        .str()));
    }
}

/*
 * CallOPPreInliner::ResourceDescriptor
 */
std::string CallOPPreInliner::FunctionAnalyticBase::to_string() const {
    std::stringstream ss;
    ss << "DDR available bytes: ";
    if (maxDDRBytesAvailable == 0) {
        ss << "UNDETERMINED";
    } else {
        ss << maxDDRBytesAvailable;
    }
    ss << ", Available Tiles count: ";
    if (totalAvailableTilesCount == 0) {
        ss << "UNDETERMINED";
    } else {
        ss << totalAvailableTilesCount;
    }
    if (!argsOffset.empty()) {
        ss << "\nFunction arguments: " << argsOffset.size() << std::endl;
        for (auto [argIdx, argOffset] : argsOffset) {
            ss << "argIdx: " << argIdx;
            if (argOffset.has_value()) {
                ss << " has an actual offset: " << argOffset.value();
            } else {
                ss << "has an UNDETERMINED offset";
            }
            ss << std::endl;
        }
    }
    return ss.str();
}
CallOPPreInliner::FunctionAnalyticBase CallOPPreInliner::FunctionAnalyticBase::create(::mlir::func::CallOp call) {
    size_t tileExecutorCount = 0;
    uint64_t maxDDRBytesAvailable = 0;
    auto module = vpux::getModuleOp(call);
    auto tileOp = config::getTileExecutor(module);
    if (tileOp) {
        tileExecutorCount = static_cast<size_t>(tileOp.getCount());
    }
    auto memOp = config::getAvailableMemory(module, vpux::VPU::MemoryKind::DDR);
    if (memOp) {
        maxDDRBytesAvailable = checked_cast<uint64_t>(memOp.getByteSize());
    }
    size_t argIdx = 0;
    std::map<size_t, std::optional<size_t>> argsOffset;
    for (auto op : call->getOperands()) {
        auto argDefiningOp = op.getDefiningOp();
        std::optional<size_t> argBytesOffset;
        if (mlir::isa<VPURT::DeclareBufferOp>(argDefiningOp)) {
            auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(argDefiningOp);
            argBytesOffset = std::make_optional<size_t>(declareOp.getByteOffset());
        }
        argsOffset[argIdx] = std::move(argBytesOffset);
        argIdx++;
    }
    return FunctionAnalyticBase{tileExecutorCount, maxDDRBytesAvailable, std::move(argsOffset)};
}

std::string CallOPPreInliner::ResourceDescriptor::to_string() const {
    std::stringstream ss;
    ss << callOpData.to_string();
    if (singleFunctionDDRConsumptionBytes.has_value()) {
        ss << ", DDR offset: " << singleFunctionDDRConsumptionBytes.value();
    } else {
        ss << ", DDR offset: UNDETERMINED";
    }
    ss << ", " << FunctionAnalyticBase::to_string();
    return ss.str();
}

CallOPPreInliner::ResourceDescriptor CallOPPreInliner::ResourceDescriptor::create(
        ::mlir::func::CallOp call, ::mlir::iterator_range<::mlir::Region::iterator> inlinedBlocks, Logger log) {
    FunctionAnalyticBase commonFuncData = FunctionAnalyticBase::create(call);
    auto debatchedAttr = DebatchedCallOpAttributeView::extract(call);
    if (!debatchedAttr.has_value()) {
        return ResourceDescriptor{std::move(commonFuncData), DebatchedCallOpData{0, 1}, {}};
    }
    VPUX_THROW_UNLESS(debatchedAttr.has_value(), "CallOPPreInliner::apply expected an attribute: {0}",
                      DebatchedCallOpAttributeView::name());
    const DebatchedCallOpData& callOpData = debatchedAttr.value().getCallData();
    log.debug("Proceed with gathering all DDR buffer allocations to determine a device memory occupation range");
    std::map<size_t, size_t> allocationsOffsetSize;
    try {
        for (mlir::Block& block : inlinedBlocks) {
            for (auto& op : block.getOperations()) {
                if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
                    continue;
                }

                auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);

                auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(declareOp.getType());
                if (ndType.getMemoryKind() != VPU::MemoryKind::DDR) {
                    continue;
                }
                size_t offset = declareOp.getByteOffset();
                int64_t shapeSize = calcTotalShapeSize(ndType.getShape());
                int64_t memorySize = shapeSize * getElemTypeSize(ndType.getElementType()).to<Byte>().count();
                allocationsOffsetSize.emplace(offset, memorySize);
            }
        }
    } catch (const std::exception& ex) {
        log.debug("Cannot calculate DDR allocation due to exception: {1}", ex.what());
        allocationsOffsetSize.clear();
    }
    if (allocationsOffsetSize.empty()) {
        log.debug("No DDR allocations, no any offset recalculation required");
        return ResourceDescriptor{std::move(commonFuncData), callOpData, {}};
    }

    log.debug("collected DDR allocations: {0}", allocationsOffsetSize.size());
    size_t index = 0;
    std::pair<size_t, size_t> occupiedDDRAdressesRange{std::numeric_limits<size_t>::max(), 0};
    for (auto [offset, size] : allocationsOffsetSize) {
        log.trace("{0}: {1}, {2}", index, offset, size);
        index++;
        occupiedDDRAdressesRange.first = std::min(offset, occupiedDDRAdressesRange.first);           // left border
        occupiedDDRAdressesRange.second = std::max(offset + size, occupiedDDRAdressesRange.second);  // right border
        log.trace("occupied DDR adresses range: [{0},{1}]", occupiedDDRAdressesRange.first,
                  occupiedDDRAdressesRange.second);
    }
    log.debug("calculated occupied DDR adresses range: [{0},{1}]", occupiedDDRAdressesRange.first,
              occupiedDDRAdressesRange.second);
    VPUX_THROW_WHEN(occupiedDDRAdressesRange.first > occupiedDDRAdressesRange.second,
                    "DDR adress range determined incorrectly, left border: {0} cann't be greater than right: {1}",
                    occupiedDDRAdressesRange.first, occupiedDDRAdressesRange.second);
    return ResourceDescriptor{std::move(commonFuncData), callOpData,
                              occupiedDDRAdressesRange.first + occupiedDDRAdressesRange.second};
}

/*
 * CallOPPreInliner::Dispatcher
 */

CallOPPreInliner::Dispatcher::Dispatcher(Logger& log): _log(log.nest()) {
}

void CallOPPreInliner::Dispatcher::dispatch(mlir::Operation& op, const ResourceDescriptor& res) const {
    OpExtractor extractor(op);
    for (auto innerOp = extractor.next(); innerOp != nullptr; innerOp = extractor.next()) {
        for (const auto& p : specificPreInliners) {
            if (p->apply(*innerOp, res)) {
                break;
            }
        }
    }
}

void CallOPPreInliner::Dispatcher::reset() {
    for (auto& p : specificPreInliners) {
        p->reset();
    }
}
/*
 * CallOPPreInliner::CMXTypeModifier
 */

size_t CallOPPreInliner::CMXTypeModifier::recalculateIndex(size_t index, const DebatchedCallOpData& callOpData,
                                                           size_t totalAvailableTilesCount) {
    return index + (callOpData.getCallIndex() * totalAvailableTilesCount) / callOpData.getBatchSize();
}

mlir::Type CallOPPreInliner::CMXTypeModifier::transform(mlir::Type type, const DebatchedCallOpData& callOpData,
                                                        size_t totalAvailableTilesCount) const {
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(type);
    if (ndType == nullptr || ndType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        return type;
    }

    auto memSpace = ndType.getMemSpace();
    VPUX_THROW_UNLESS(memSpace.getRootName() == "CMX_NN" && memSpace.getLeafName() == "CMX_NN",
                      "Expected memspace CMX_NN, got: {0}/{1}", memSpace.getRootName(), memSpace.getLeafName());
    if (!memSpace.getIndex().has_value()) {
        return type;
    }
    auto calculateClusterId = [&callOpData, totalAvailableTilesCount](const mlir::IntegerAttr& attr) -> size_t {
        return CMXTypeModifier::recalculateIndex(parseIntAttr<size_t>(attr), callOpData, totalAvailableTilesCount);
    };

    auto calculateCmxIndex = [calculateClusterId](const IndexedSymbolAttr& memSpace) -> size_t {
        return calculateClusterId(memSpace.getIndexAttr().value());
    };
    auto newCMXIndex = calculateCmxIndex(memSpace);
    _log.trace("mem kind: {0} has tile index: {1}, new index: {2}", stringifyEnum(ndType.getMemoryKind()),
               memSpace.getIndex().value(), newCMXIndex);
    auto newMemSpace = vpux::IndexedSymbolAttr::get(ndType.getContext(), memSpace.getRootName(), newCMXIndex);
    auto newType = ndType.changeMemSpace(newMemSpace);

    auto itiBufferTypeFromOp = mlir::dyn_cast<vpux::VPUIP::ITIBufferType>(type);
    if (itiBufferTypeFromOp) {
        // HALO regions require for additional processing
        ArrayRef<vpux::VPUIP::HaloRegionAttr> inwardHalos = itiBufferTypeFromOp.getInwardHaloRegions();
        SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardsHaloAttr =
                modifyHaloAttrsClusterId(inwardHalos, calculateClusterId, _log);

        ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos = itiBufferTypeFromOp.getOutwardHaloRegions();
        SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> newOutwardsHaloAttr =
                modifyOutwardHaloAttrs(outwardHalos, calculateClusterId, _log);

        auto newItiBufferTypeFromOp = VPUIP::ITIBufferType::get(
                itiBufferTypeFromOp.getContext(),
                itiBufferTypeFromOp.getShape()
                        .raw(),  // Despite ctor requires memShape(), we pass getShape() to avoid dimensions reordering
                itiBufferTypeFromOp.getElementType(), itiBufferTypeFromOp.getLayout(), newMemSpace,
                itiBufferTypeFromOp.getIduSegmentation(), newInwardsHaloAttr, newOutwardsHaloAttr);
        newType = newItiBufferTypeFromOp;
    }

    return newType;
}

template <class ClusterIdFunctor>
SmallVector<vpux::VPUIP::HaloRegionAttr> CallOPPreInliner::CMXTypeModifier::modifyHaloAttrsClusterId(
        ArrayRef<vpux::VPUIP::HaloRegionAttr> haloAttrs, ClusterIdFunctor modifier, Logger log) {
    SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardsHaloAttr;
    newInwardsHaloAttr.reserve(haloAttrs.size());
    auto nestedLog = log.nest();
    nestedLog.trace("HaloRegionAttrs: {0}", haloAttrs.size());
    for (auto&& haloAttr : haloAttrs) {
        auto newCMXIndex = modifier(haloAttr.getClusterId());
        nestedLog.trace("HaloRegionAttr has `cluster_id`: {0}, new value: {1}",
                        parseIntAttr<size_t>(haloAttr.getClusterId()), newCMXIndex);
        auto newClusterIdAttr = getIntAttr(haloAttr.getContext(), newCMXIndex);
        auto newHaloAttr = vpux::VPUIP::HaloRegionAttr::get(haloAttr.getContext(), haloAttr.getShape(),
                                                            haloAttr.getOffset(), newClusterIdAttr);
        newInwardsHaloAttr.push_back(newHaloAttr);
    }
    return newInwardsHaloAttr;
}

template <class ClusterIdFunctor>
SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> CallOPPreInliner::CMXTypeModifier::modifyOutwardHaloAttrs(
        ArrayRef<vpux::VPUIP::OutwardHaloRegionAttr> outwardHalos, ClusterIdFunctor modifier, Logger log) {
    SmallVector<vpux::VPUIP::OutwardHaloRegionAttr> newOutwardsHaloAttr;
    newOutwardsHaloAttr.reserve(outwardHalos.size());
    Logger _log = log.nest();
    _log.trace("ITIBufferType has outwardHaloRegions: {0}", outwardHalos.size());
    for (auto& haloAttr : outwardHalos) {
        auto newCMXIndex = modifier(haloAttr.getClusterId());
        _log.trace("Outward halo region has `cluster_id`: {0}, new value: {1}",
                   parseIntAttr<size_t>(haloAttr.getClusterId()), newCMXIndex);
        auto newClusterIdAttr = getIntAttr(haloAttr.getContext(), newCMXIndex);
        auto inwardFromOutwardsAttrs = haloAttr.getInwardHaloRegions();

        SmallVector<vpux::VPUIP::HaloRegionAttr> newInwardHalos = CMXTypeModifier::modifyHaloAttrsClusterId(
                parseCustomAttrArray<vpux::VPUIP::HaloRegionAttr>(inwardFromOutwardsAttrs), modifier, _log);
        SmallVector<mlir::Attribute> newInwardHalosAttrs;
        newInwardHalosAttrs.reserve(newInwardHalos.size());
        std::transform(newInwardHalos.begin(), newInwardHalos.end(), std::back_inserter(newInwardHalosAttrs),
                       [](vpux::VPUIP::HaloRegionAttr attr) {
                           return attr;
                       });
        auto newHaloAttr = vpux::VPUIP::OutwardHaloRegionAttr::get(
                haloAttr.getContext(), haloAttr.getShape(), haloAttr.getOffset(), newClusterIdAttr,
                mlir::ArrayAttr::get(haloAttr.getContext(), newInwardHalosAttrs));
        newOutwardsHaloAttr.push_back(newHaloAttr);
    }
    return newOutwardsHaloAttr;
}

/*
 * CallOPPreInliner::Dispatcher::GreedyModifier
 */

bool CallOPPreInliner::Dispatcher::GreedyModifier::apply(mlir::Operation& op,
                                                         const ResourceDescriptor& resource) const {
    _log.trace("GreedyModifier of: {0} started", op.getName());
    for (auto&& result : op.getResults()) {
        auto newType = transform(result.getType(), resource.callOpData, resource.totalAvailableTilesCount);
        result.setType(newType);
    }
    _log.trace("GreedyModifier processing of: {0} finished", op.getName());
    return true;
}

/*
 * CallOPPreInliner::Dispatcher::CMXModifierForDeclareOp
 */

bool CallOPPreInliner::Dispatcher::CMXModifierForDeclareOp::apply(mlir::Operation& op,
                                                                  const ResourceDescriptor& resource) const {
    if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
        return false;
    }

    auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(declareOp.getType());
    if (ndType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        return applyCMX(declareOp, resource.callOpData, resource.totalAvailableTilesCount);
    } else if (ndType.getMemoryKind() == VPU::MemoryKind::DDR &&
               (declareOp.getSection() != VPURT::BufferSection::FunctionInput &&
                declareOp.getSection() != VPURT::BufferSection::FunctionOutput)) {
        return applyDDR(declareOp, resource.callOpData, resource.singleFunctionDDRConsumptionBytes.value(),
                        resource.maxDDRBytesAvailable);
    }

    return false;
}

bool CallOPPreInliner::Dispatcher::CMXModifierForDeclareOp::applyCMX(VPURT::DeclareBufferOp& op,
                                                                     const DebatchedCallOpData& callOpData,
                                                                     size_t totalAvailableTilesCount) const {
    mlir::ModuleOp module = vpux::getModuleOp(op);
    auto ctx = module.getContext();

    auto sectionArrayAttr = op.getSectionIndexAttr();
    auto sectionArray = parseIntArrayAttr<size_t>(sectionArrayAttr);
    _log.trace("{0} current section value: {1}", op->getName(), sectionArray);
    std::transform(sectionArray.begin(), sectionArray.end(), sectionArray.begin(),
                   [&callOpData, totalAvailableTilesCount](size_t index) {
                       return CMXTypeModifier::recalculateIndex(index, callOpData, totalAvailableTilesCount);
                   });

    auto declareOpResult = op->getResult(0);
    auto newType = transform(declareOpResult.getType(), callOpData, totalAvailableTilesCount);
    declareOpResult.setType(newType);

    _log.trace("{0} new section value: {1}", op->getName(), sectionArray),
            op.setSectionIndexAttr(getIntArrayAttr(ctx, sectionArray));
    return true;
}

bool CallOPPreInliner::Dispatcher::CMXModifierForDeclareOp::applyDDR(VPURT::DeclareBufferOp& op,
                                                                     const DebatchedCallOpData& callOpData,
                                                                     size_t offsetDDRAllocationBytes,
                                                                     size_t maxDDRBytesAvailable) const {
    size_t currentBatchedCallDDROffset = callOpData.getCallIndex() * offsetDDRAllocationBytes;
    auto opBytesOffset = op.getByteOffset();
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getType());
    int64_t shapeSize = calcTotalShapeSize(ndType.getShape());
    int64_t memorySize = shapeSize * getElemTypeSize(ndType.getElementType()).to<Byte>().count();
    VPUX_THROW_WHEN(
            currentBatchedCallDDROffset + opBytesOffset + memorySize > maxDDRBytesAvailable,
            "Cannot substitute DDR offset: {0} by a new one: {1} as available memory range is limited by: {2} bytes."
            "Please try on \"debatching-inlining-method=naive\" instead",
            opBytesOffset, currentBatchedCallDDROffset + opBytesOffset, maxDDRBytesAvailable);
    _log.debug("{0} DDR offset old: {1}, new: {2}", op->getName(), opBytesOffset,
               currentBatchedCallDDROffset + opBytesOffset);
    op.setByteOffset(currentBatchedCallDDROffset + opBytesOffset);
    return false;
}

/*
 * CallOPPreInliner::Dispatcher::CMXModifierForNCEClusterTaskOp
 */

bool CallOPPreInliner::Dispatcher::CMXModifierForNCEClusterTaskOp::apply(mlir::Operation& op,
                                                                         const ResourceDescriptor& resource) const {
    if (!mlir::isa<vpux::VPUIP::NCEClusterTaskOp>(op)) {
        return false;
    }

    auto nceTaskOp = mlir::dyn_cast<vpux::VPUIP::NCEClusterTaskOp>(op);
    for (auto result : nceTaskOp.getResults()) {
        auto newType = transform(result.getType(), resource.callOpData, resource.totalAvailableTilesCount);
        result.setType(newType);
    }

    auto dpuTasks = nceTaskOp.getVariants().getOps<VPUIP::DPUTaskOp>();
    _log.trace("{0} has DPUTaskOps: {1}", op.getName(), !dpuTasks.empty());
    for (auto&& dpuTaskOp : dpuTasks) {
        auto clusterId = dpuTaskOp.getClusterId();
        if (clusterId.has_value()) {
            auto newClusterId = CMXTypeModifier::recalculateIndex(clusterId.value(), resource.callOpData,
                                                                  resource.totalAvailableTilesCount);
            _log.trace("{0} has `cluster_id`: {1}, new value: {2}", dpuTaskOp->getName(), clusterId.value(),
                       newClusterId);
            dpuTaskOp.setClusterId(newClusterId);
        }
    }
    return true;
}

/*
 * CallOPPreInliner::Dispatcher::CMXModifierForSWKernelOp
 */

bool CallOPPreInliner::Dispatcher::CMXModifierForSWKernelOp::apply(mlir::Operation& op,
                                                                   const ResourceDescriptor& resource) const {
    if (!mlir::isa<vpux::VPUIP::SwKernelOp>(op)) {
        return false;
    }

    auto swKernelTaskOp = mlir::dyn_cast<vpux::VPUIP::SwKernelOp>(op);
    for (auto result : swKernelTaskOp.getResults()) {
        mlir::Type newType = transform(result.getType(), resource.callOpData, resource.totalAvailableTilesCount);
        result.setType(newType);
    }

    auto tileIndex = swKernelTaskOp.getTileIndex();
    if (tileIndex.has_value()) {
        size_t newTileIndex = CMXTypeModifier::recalculateIndex(tileIndex.value(), resource.callOpData,
                                                                resource.totalAvailableTilesCount);
        swKernelTaskOp.setTileIndex(newTileIndex);
        _log.trace("{0}, has `tile_index`: {1}, new index: {2}", swKernelTaskOp->getName(), tileIndex.value(),
                   newTileIndex);
    }

    auto swKernelRuns = swKernelTaskOp.getBody().getOps<VPUIP::SwKernelRun>();
    _log.trace("{0} has SwKernelRun: {0}", swKernelTaskOp->getName(), !swKernelRuns.empty());
    for (auto&& kernelRun : swKernelRuns) {
        auto operands = kernelRun->getOperands();
        _log.trace("operands: {0}", operands.size());
        for (auto operand : operands) {
            auto kernelRunNewType =
                    transform(operand.getType(), resource.callOpData, resource.totalAvailableTilesCount);
            operand.setType(kernelRunNewType);
        }
    }
    return true;
}

/*
 * CallOPPreInliner::Dispatcher::FuncArgumentDeclareModifier
 */

bool CallOPPreInliner::Dispatcher::FuncArgumentDeclareModifier::apply(mlir::Operation& op,
                                                                      const ResourceDescriptor& resource) const {
    if (!mlir::isa<VPURT::DeclareBufferOp>(op)) {
        return false;
    }
    auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);
    if (declareOp.getSection() != VPURT::BufferSection::FunctionInput &&
        declareOp.getSection() != VPURT::BufferSection::FunctionOutput) {
        return false;
    }

    const auto& funcArgAttr = declareOp.getSectionIndexAttr();
    VPUX_THROW_UNLESS(funcArgAttr,
                      "BufferSection::FunctionInput from operation {0} must contains a section index attribute");

    auto sectionArrayAttr = declareOp.getSectionIndexAttr();
    auto sectionArray = parseIntArrayAttr<size_t>(sectionArrayAttr);
    _log.trace("DeclareOp current section attribute: {0}", sectionArray);
    VPUX_THROW_UNLESS(
            sectionArray.size() >= 1,
            "DeclareOp current section attribute of BufferSection::FunctionInput must contains at least 1 element");

    size_t argIndex = sectionArray[0];
    if (auto funcArgOffsetIt = resource.argsOffset.find(argIndex); funcArgOffsetIt != resource.argsOffset.end()) {
        auto declareOp = mlir::dyn_cast<VPURT::DeclareBufferOp>(op);

        // clear FunctionInput/Output section
        declareOp.setSection(VPURT::BufferSection::DDR);
        declareOp.setSectionIndexAttr(::mlir::ArrayAttr());

        VPUX_THROW_UNLESS(funcArgOffsetIt->second.has_value(), "Func arguments: {0} has no determined offset",
                          argIndex);
        auto opBytesOffset = declareOp.getByteOffset();

        // If unroll-distributed-ops has been executed, an original buffer at a fixed offset position
        // will be superseded by its shards, which occupy the fixed offset position plus a shard specific offset
        // so that we must preserve this offset during our func args inlining
        size_t newOffset = 0;
        if (argsLastRelativeOffsetInCluster[argIndex].has_value()) {
            newOffset =
                    opBytesOffset - argsLastRelativeOffsetInCluster[argIndex].value() + funcArgOffsetIt->second.value();
        } else {
            newOffset = funcArgOffsetIt->second.value();
            argsLastRelativeOffsetInCluster[argIndex] = std::make_optional<size_t>(opBytesOffset);
        }
        _log.debug("fix the func argument by index: {1}. DDR offset old: {2}, new: {3}", op.getName(), argIndex,
                   opBytesOffset, newOffset);
        declareOp.setByteOffset(newOffset);
    }
    return true;
}
void CallOPPreInliner::Dispatcher::FuncArgumentDeclareModifier::reset() {
    argsLastRelativeOffsetInCluster.clear();
}

CallOPPreInlinerVisitor::CallOPPreInlinerVisitor(Logger log): _log(log.nest()) {
    addPreInliner<FuncArgOffsetPreInliner>();
    addPreInliner<BatchedCallOpReorderingPreInliner>();
}

void CallOPPreInlinerVisitor::visit(mlir::Operation* op,
                                    mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    VPUX_THROW_UNLESS(op != nullptr, "Empty operation");
    _log.trace("CallOPPreInlinerVisitor started");
    for (const auto& p : preInliners) {
        if (p->isApplicable(op)) {
            p->apply(op, inlinedBlocks);
        }
    }
    _log.trace("CallOPPreInlinerVisitor finished");
}
}  // namespace detail

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

bool VPUIP::FuncInlinerInterface::isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const {
    return true;
}

void VPUIP::FuncInlinerInterface::handleTerminator(mlir::Operation*, mlir::ValueRange) const {
}

void VPUIP::FuncInlinerInterface::processInlinedCallBlocks(
        mlir::Operation* call, mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const {
    auto parentOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(parentOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    ::detail::CallOPPreInlinerVisitor preProc;
    preProc.visit(call, inlinedBlocks);

    DenseMap<VPURT::TaskQueueType, std::pair<VPURT::TaskOp, VPURT::TaskOp>> taskQueuesFirstAndLastOpMap;
    for (mlir::Block& block : inlinedBlocks) {
        for (auto& op : block.getOperations()) {
            if (mlir::isa<mlir::func::ReturnOp, VPURT::DeclareBufferOp, VPURT::DeclareVirtualBarrierOp,
                          Const::DeclareOp>(op)) {
                continue;
            }

            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(op);
            VPUX_THROW_WHEN(taskOp == nullptr, "Unexpected operation type: {0}", op.getName());

            const auto taskQueueType = VPURT::getTaskQueueType(taskOp, false);

            // DPU and Shave tasks are all expected to be guarded by barriers so in case such tasks in given inlined
            // block don't have either wait or update barrier connect them to parent task barriers. Logic for finding
            // first and last op in a queue does not handle Shave tasks to full extent due to lack of explicit tasks
            // list for multiple shave engines on single NCE cluster. In such case there might be multiple first or last
            // Shave task that should be connected to a parent barrier.
            if (taskQueueType.type != VPU::ExecutorKind::DMA_NN) {
                if (taskOp.getWaitBarriers().empty()) {
                    taskOp.getWaitBarriersMutable().append(parentOp.getWaitBarriers());
                }
                if (taskOp.getUpdateBarriers().empty()) {
                    taskOp.getUpdateBarriersMutable().append(parentOp.getUpdateBarriers());
                }
                continue;
            }

            if (taskQueuesFirstAndLastOpMap.find(taskQueueType) == taskQueuesFirstAndLastOpMap.end()) {
                // First occurrence of task on this queue
                taskQueuesFirstAndLastOpMap[taskQueueType] = std::make_pair(taskOp, taskOp);
            } else {
                // In case new task spotted, update last task info
                taskQueuesFirstAndLastOpMap[taskQueueType].second = taskOp;
            }
        }
    }
    // Identify first and last task on each execution queue.
    // For first tasks if they do no wait on any barrier connect them with start barrier
    // For end tasks if they do not update any barrier connect then to end barrier
    for (auto& taskQueuesFirstAndLastOp : taskQueuesFirstAndLastOpMap) {
        auto queueFirstOp = taskQueuesFirstAndLastOp.second.first;
        auto queueLastOp = taskQueuesFirstAndLastOp.second.second;
        if (queueFirstOp.getWaitBarriers().empty() && !parentOp.getWaitBarriers().empty()) {
            // Empty "waits" barriers means
            // this operation is one of the first operations from the callable region
            // Add "waits" barriers(if exist) from the parent VPURT::TaskOp
            // to wait operators from the previous callable region
            queueFirstOp.getWaitBarriersMutable().append(parentOp.getWaitBarriers());
        }

        if (queueLastOp.getUpdateBarriers().empty() && !parentOp.getUpdateBarriers().empty()) {
            // Empty "update" barriers means
            // this operation is one of the last operations from the callable region
            // Add "update" barriers(if exist) from the parent VPURT::TaskOp
            // to notify operators from the next callable region
            queueLastOp.getUpdateBarriersMutable().append(parentOp.getUpdateBarriers());
        }
    }
}

std::tuple<mlir::Block*, mlir::Block::iterator> VPUIP::FuncInlinerInterface::getInlineBlockAndPoint(
        mlir::Operation* call) const {
    VPUX_THROW_WHEN(call == nullptr, "fun.call must not be empty");
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    return std::make_tuple(taskOp->getBlock(), std::next(taskOp->getIterator()));
}

void VPUIP::FuncInlinerInterface::eraseCall(mlir::Operation* call) const {
    VPUX_THROW_WHEN(call == nullptr, "fun.call must not be empty");
    auto taskOp = call->getParentOfType<VPURT::TaskOp>();
    VPUX_THROW_WHEN(taskOp == nullptr, "fun.call must have parent VPURT::TaskOp");

    taskOp->erase();
}
