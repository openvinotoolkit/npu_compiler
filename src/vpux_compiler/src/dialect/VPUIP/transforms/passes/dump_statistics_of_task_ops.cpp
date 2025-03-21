//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/profiling/common.hpp"

#include <functional>
#include <map>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DUMPSTATISTICSOFTASKOPSPASS
#define GEN_PASS_DEF_DUMPSTATISTICSOFTASKOPSPASS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

using IsSpecificOpFunc = std::function<bool(mlir::Operation*)>;
using RemainderHandlerFunc = std::function<std::string(mlir::Operation*)>;

//
// Declare utility functions
//

std::string convertBytesToReadableSize(uint64_t);
bool printDMASizes(const std::string&, const uint64_t&);

class SpecificCategoryCounter {
public:
    using CounterPtr = std::shared_ptr<SpecificCategoryCounter>;
    using CountersVec = std::vector<CounterPtr>;

    SpecificCategoryCounter(const std::string& category, IsSpecificOpFunc predicate,
                            const CountersVec& nestedCounters = {},
                            std::optional<RemainderHandlerFunc> maybeRemainderHandler = std::nullopt)
            : _category(category),
              _predicate(std::move(predicate)),
              _nestedCounters(nestedCounters),
              _count(0),
              _size(0),
              _maybeRemainderHandler(std::move(maybeRemainderHandler)) {
    }

    bool count(mlir::Operation* op) {
        if (!_predicate(op)) {
            return false;
        }

        ++_count;
        if (mlir::isa_and_nonnull<VPUIP::DMATypeOpInterface>(op)) {
            auto opType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
            _size += opType.getCompactAllocSize().count();
        }
        bool counted = false;
        for (auto& nestedCounter : _nestedCounters) {
            counted |= nestedCounter->count(op);
        }
        if (!counted && _maybeRemainderHandler.has_value()) {
            ++_remainderCounter[_maybeRemainderHandler.value()(op)];
        }
        return true;
    }

    void printStatistics(vpux::Logger log) {
        if (printDMASizes(_category, _size)) {
            log.info("{0} - {1} ops : Size - {2}", _category, _count, convertBytesToReadableSize(_size));
        } else {
            log.info("{0} - {1} ops", _category, _count);
        }
        log = log.nest();
        for (auto& nestedCounter : _nestedCounters) {
            if (nestedCounter->_count) {
                nestedCounter->printStatistics(log);
            }
        }
        for (auto& remainder : _remainderCounter) {
            log.info("{0} - {1} ops", remainder.first, remainder.second);
        }
        log = log.unnest();
    }

private:
    std::string _category;
    IsSpecificOpFunc _predicate;
    CountersVec _nestedCounters;
    size_t _count;
    uint64_t _size;
    std::optional<RemainderHandlerFunc> _maybeRemainderHandler;
    std::map<std::string, size_t> _remainderCounter{};
};

using CountersVec = SpecificCategoryCounter::CountersVec;

//
// Utility Functions
//

bool printDMASizes(const std::string& category, const uint64_t& size) {
    std::vector<std::string> dmaSubStrings = {"DDR", "CMX", "NNDMA"};

    bool anyDmaSubStringFound =
            std::any_of(dmaSubStrings.begin(), dmaSubStrings.end(), [&](const std::string& dmaSubString) {
                return category.find(dmaSubString) != std::string::npos;
            });

    return anyDmaSubStringFound && size;
}

std::string convertBytesToReadableSize(uint64_t bytes) {
    const uint64_t kilobyte = 1024;
    const uint64_t megabyte = kilobyte * 1024;
    const uint64_t gigabyte = megabyte * 1024;

    std::string result;
    if (bytes >= gigabyte) {
        double size = static_cast<double>(bytes) / gigabyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);  // Truncate to two decimal digits
        result += " GB";
    } else if (bytes >= megabyte) {
        double size = static_cast<double>(bytes) / megabyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);
        result += " MB";
    } else if (bytes >= kilobyte) {
        double size = static_cast<double>(bytes) / kilobyte;
        result = std::to_string(size);
        result.resize(result.find('.') + 3);
        result += " KB";
    } else {
        result = std::to_string(bytes) + " bytes";
    }

    return result;
}

bool isNameLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (auto nameLoc = loc.dyn_cast<mlir::NameLoc>()) {
        return nameLoc.getName().str().find(substr) != std::string::npos;
    }
    return false;
}

bool isLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (isNameLocContainsStr(loc, substr)) {
        return true;
    }
    if (auto fusedLoc = loc.dyn_cast<mlir::FusedLoc>()) {
        for (auto& loc : fusedLoc.getLocations()) {
            if (isNameLocContainsStr(loc, substr)) {
                return true;
            }
        }
    }
    return false;
}

CountersVec getSpillCounter(const std::string& category) {
    if (category != "DDR2CMX" && category != "CMX2DDR") {
        return {};
    }
    const auto spillCategory = category == "DDR2CMX" ? "SPILL_READ" : "SPILL_WRITE";
    const auto targetSubstr = category == "DDR2CMX" ? SPILL_READ_OP_NAME_SUFFIX : SPILL_WRITE_OP_NAME_SUFFIX;
    return {std::make_shared<SpecificCategoryCounter>(spillCategory, [=](mlir::Operation* op) {
        if (auto fusedLoc = op->getLoc().dyn_cast<mlir::FusedLoc>()) {
            const auto locations = fusedLoc.getLocations();
            for (auto it = std::rbegin(locations); it != std::rend(locations); ++it) {
                const auto loc = *it;
                // analyzing only spill related locs
                if (isLocContainsStr(loc, SPILL_READ_OP_NAME_SUFFIX) ||
                    isLocContainsStr(loc, SPILL_WRITE_OP_NAME_SUFFIX)) {
                    // return result on first from end matched loc
                    return isLocContainsStr(loc, targetSubstr);
                }
            }
        }
        return false;
    })};
}

std::string getProfSuffix(const std::string& profCategory) {
    return profCategory + profiling::PROFILING_CMX_2_DDR_OP_NAME;
}

void addProfilingCounters(CountersVec& counters, const std::string& category) {
    if (category == "REG2CMX") {
        counters.push_back(std::make_shared<SpecificCategoryCounter>("Profiling Timestamp DMA", [](auto op) {
            if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
                return dmaOp.getProfilingMetadata().has_value();
            }
            return false;
        }));
        return;
    }
    if (category == "REG2DDR") {
        counters.push_back(std::make_shared<SpecificCategoryCounter>("Profiling workpoint", [](auto op) {
            return isLocContainsStr(op->getLoc(), profiling::PROFILING_WORKPOINT_READ_ATTR);
        }));
        return;
    }
    if (category == "CMX2DDR") {
        CountersVec nestedProfCounters;
        const static std::vector<std::pair<std::string, IsSpecificOpFunc>> profCounterCfgs = {
                {"DMA",
                 [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("dma"));
                 }},
                {"DPU",
                 [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("dpu"));
                 }},
                {"ActShave",
                 [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("actshave"));
                 }},
                {"M2I", [](auto op) {
                     return isLocContainsStr(op->getLoc(), getProfSuffix("m2i"));
                 }}};
        for (const auto& p : profCounterCfgs) {
            nestedProfCounters.push_back(std::make_shared<SpecificCategoryCounter>(p.first, p.second));
        }
        counters.push_back(std::make_shared<SpecificCategoryCounter>(
                "Profiling buffer management",
                [](auto op) {
                    return isLocContainsStr(op->getLoc(), profiling::PROFILING_CMX_2_DDR_OP_NAME);
                },
                nestedProfCounters));
    }
    if (category == "DDR2DDR") {
        CountersVec nestedProfCounters;
        const static std::vector<std::pair<std::string, IsSpecificOpFunc>> profCounterCfgs = {
                {"DMA", [](auto op) {
                     return isLocContainsStr(op->getLoc(), std::string("dma") + profiling::PROFILING_DDR_2_DDR_OP_NAME);
                 }}};
        for (const auto& p : profCounterCfgs) {
            nestedProfCounters.push_back(std::make_shared<SpecificCategoryCounter>(p.first, p.second));
        }
        counters.push_back(std::make_shared<SpecificCategoryCounter>(
                "Profiling buffer management",
                [](auto op) {
                    return isLocContainsStr(op->getLoc(), profiling::PROFILING_DDR_2_DDR_OP_NAME);
                },
                nestedProfCounters));
    }
}

CountersVec getInnerCounters(const std::string& category) {
    CountersVec counters = getSpillCounter(category);
    addProfilingCounters(counters, category);
    return counters;
}

template <class DMAType>
CountersVec getDMANestedCounters() {
    using VPU::MemoryKind;

    const auto checkInputOutputMemSpace = [](mlir::Operation* op, VPU::MemoryKind inMemKind,
                                             VPU::MemoryKind outMemKind) {
        VPUX_THROW_WHEN(op == nullptr, "NULL operation provided");

        const auto checkArgMemSpace = [](mlir::Value operand, VPU::MemoryKind memKind) {
            return operand.getType().cast<vpux::NDTypeInterface>().getMemoryKind() == memKind;
        };

        if (auto dmaOp = mlir::dyn_cast<DMAType>(op)) {
            return checkArgMemSpace(dmaOp.getInput(), inMemKind) && checkArgMemSpace(dmaOp.getOutputBuff(), outMemKind);
        }

        VPUX_THROW("Not supported DMA task");
    };

    std::vector<std::pair<std::string, MemoryKind>> configurations = {{
                                                                              "CMX",
                                                                              MemoryKind::CMX_NN,
                                                                      },
                                                                      {"DDR", MemoryKind::DDR},
                                                                      {"REG", MemoryKind::Register}};

    CountersVec counters;
    for (const auto& inputConf : configurations) {
        for (const auto& outputConf : configurations) {
            const auto category = inputConf.first + "2" + outputConf.first;
            const auto nestedCounters = getInnerCounters(category);
            counters.push_back(std::make_shared<SpecificCategoryCounter>(
                    category,
                    [=](auto op) {
                        return checkInputOutputMemSpace(op, inputConf.second, outputConf.second);
                    },
                    nestedCounters));
        }
    }
    return counters;
}

template <class... Args, typename = typename std::enable_if<sizeof...(Args) == 0>::type>
void populateDMACounters(CountersVec&) {
}

template <class DMAType, class... DMATypeArgs>
void populateDMACounters(CountersVec& counters) {
    RemainderHandlerFunc dmaRemainderHandler = [](mlir::Operation*) {
        return "Unknown memory space DMA";
    };
    counters.push_back(std::make_shared<SpecificCategoryCounter>(
            DMAType::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<DMAType>(op);
            },
            getDMANestedCounters<DMAType>(), dmaRemainderHandler));
    populateDMACounters<DMATypeArgs...>(counters);
}

CountersVec getDMACounters() {
    CountersVec dmaCounters;
    using namespace VPUIP;
    populateDMACounters<NNDMAOp, CompressDMAOp, DecompressDMAOp, DepthToSpaceDMAOp, PermuteDMAOp, ExpandDMAOp>(
            dmaCounters);
    return dmaCounters;
}

SpecificCategoryCounter::CounterPtr getSWKernelsCounter() {
    RemainderHandlerFunc swHandler = [](mlir::Operation* op) -> std::string {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            return swKernelOp.getKernelFunction().getLeafReference().str();
        }
        return "Not SwKernel";
    };

    return std::make_shared<SpecificCategoryCounter>(
            VPUIP::SwKernelOp::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::SwKernelOp>(op);
            },
            CountersVec{}, swHandler);
}

SpecificCategoryCounter::CounterPtr getSparsityCounter() {
    RemainderHandlerFunc remainedOpHandler = [](mlir::Operation*) {
        return "Dense";
    };

    auto sparseInputCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse input",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    bool hasSparseInput =
                            nceOp.getInputSparsityMap() != nullptr || nceOp.getInputStorageElementTable() != nullptr;
                    if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
                        hasSparseInput |= nceOp.getWeightsSparsityMap() != nullptr;
                    }
                    return hasSparseInput;
                }
                return false;
            },
            CountersVec{});

    auto sparseWeightsCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse weights",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    return nceOp.getWeightsSparsityMap() != nullptr;
                }
                return false;
            },
            CountersVec{});

    auto sparseOutputCounter = std::make_shared<SpecificCategoryCounter>(
            "Sparse output",
            [](mlir::Operation* op) {
                if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
                    return nceOp.getOutputSparsityMap() != nullptr;
                }
                return false;
            },
            CountersVec{});

    CountersVec counters = {std::move(sparseInputCounter), std::move(sparseWeightsCounter),
                            std::move(sparseOutputCounter)};
    return std::make_shared<SpecificCategoryCounter>(
            "NCETask Operations",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::NCEClusterTaskOp>(op);
            },
            counters, remainedOpHandler);
}

SpecificCategoryCounter::CounterPtr getM2ICounter() {
    return std::make_shared<SpecificCategoryCounter>(VPUIP::M2ITaskOp::getOperationName().str(),
                                                     [](mlir::Operation* op) {
                                                         return mlir::isa<VPUIP::M2ITaskOp>(op);
                                                     });
}

SpecificCategoryCounter populateCounters() {
    RemainderHandlerFunc remainedOpHandler = [](mlir::Operation* op) {
        return op->getName().getIdentifier().str();
    };

    auto counters = getDMACounters();
    counters.push_back(getSWKernelsCounter());
    counters.push_back(getSparsityCounter());
    counters.push_back(getM2ICounter());

    SpecificCategoryCounter topLevelCounter(
            "VPUIP Tasks",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::TaskOpInterface>(op);
            },
            counters, remainedOpHandler);
    return topLevelCounter;
}

class CompressionRateCounter {
public:
    CompressionRateCounter()
            : constantsCounter_(0),
              compressedConstantsCounter_(0),
              compressedF16constantsCounter_(0),
              totalConstantsBeforeCompression_(0),
              totalCompressedConstantsAfterCompression_(0),
              totalCompressedConstantsBeforeCompression_(0),
              compressedF16constantsAfterCompression_(0),
              compressedF16constantsBeforeCompression_(0),
              totalUncompressedConstants_(0) {
    }

    uint64_t getDataSize(mlir::Value buffer) {
        const auto type = buffer.getType().cast<vpux::NDTypeInterface>();
        return type.getShape().totalSize() * type.getElemTypeSize().count() / CHAR_BIT;
    }

    void count(mlir::Operation* op) {
        auto cstOp = mlir::dyn_cast<Const::DeclareOp>(op);
        if (!cstOp) {
            return;
        }
        constantsCounter_++;

        // Handle compressed constants
        for (const auto& user : cstOp.getOutput().getUsers()) {
            if (auto decompressDMAOp = mlir::dyn_cast<VPUIP::DecompressDMAOp>(user)) {
                // Make sure this is not activation decompression
                assert(!decompressDMAOp.getActCompressionSizeEntry());

                compressedConstantsCounter_++;

                const auto inputSize = getDataSize(decompressDMAOp.getInput());
                const auto outputSize = getDataSize(decompressDMAOp.getOutputBuff());
                totalConstantsBeforeCompression_ += outputSize;
                totalCompressedConstantsBeforeCompression_ += outputSize;
                totalCompressedConstantsAfterCompression_ += inputSize;

                auto compressedConstantType =
                        decompressDMAOp.getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();

                if (compressedConstantType.isF16()) {
                    compressedF16constantsCounter_++;
                    compressedF16constantsBeforeCompression_ += outputSize;
                    compressedF16constantsAfterCompression_ += inputSize;
                }

                return;
            }
        }

        // Handle uncompressed constants
        auto size = getDataSize(cstOp);
        totalConstantsBeforeCompression_ += size;
        totalUncompressedConstants_ += size;
    }

    void printStatistics(vpux::Logger log) {
        uint64_t totalConstantsAfterCompression_ =
                totalCompressedConstantsAfterCompression_ + totalUncompressedConstants_;

        auto uncompressedConstantsCount = constantsCounter_ - compressedConstantsCounter_;

        log = log.nest();
        if (compressedConstantsCounter_ == 0) {
            // No compression
            assert(totalConstantsBeforeCompression_ == totalConstantsAfterCompression_);
            assert(totalCompressedConstantsBeforeCompression_ == 0);
            assert(totalCompressedConstantsAfterCompression_ == 0);
            log.info("Total weights - count: {0}, size: {1} (no compression)", constantsCounter_,
                     convertBytesToReadableSize(totalConstantsAfterCompression_));
            return;
        }

        // There has been compression...

        assert(totalConstantsBeforeCompression_ > 0);
        assert(totalCompressedConstantsBeforeCompression_ > 0);

        const double totalCompressionRate =
                (double)totalConstantsAfterCompression_ / totalConstantsBeforeCompression_ * 100;
        log.info("Total weights - count: {0}, size: {1}, compressed size: {2}, ({3}%)", constantsCounter_,
                 convertBytesToReadableSize(totalConstantsBeforeCompression_),
                 convertBytesToReadableSize(totalConstantsAfterCompression_), totalCompressionRate);

        const double compressedConstantsCompressionRate =
                (double)totalCompressedConstantsAfterCompression_ / totalCompressedConstantsBeforeCompression_ * 100;
        log.info("Compressed weights - count: {0}, size: {1}, compressed size: {2}, ({3}%)",
                 compressedConstantsCounter_, convertBytesToReadableSize(totalCompressedConstantsBeforeCompression_),
                 convertBytesToReadableSize(totalCompressedConstantsAfterCompression_),
                 compressedConstantsCompressionRate);
        if (compressedF16constantsCounter_ > 0) {
            log.nest().info(
                    "F16 - count: {0}, size: {1}, compressed size: {2}, ({3}%)", compressedF16constantsCounter_,
                    convertBytesToReadableSize(compressedF16constantsBeforeCompression_),
                    convertBytesToReadableSize(compressedF16constantsAfterCompression_),
                    (double)compressedF16constantsAfterCompression_ / compressedF16constantsBeforeCompression_ * 100);
        }
        if (compressedConstantsCounter_ - compressedF16constantsCounter_ > 0) {
            log.nest().info(
                    "Int8 - count: {0}, size: {1}, compressed size: {2}, ({3}%)",
                    compressedConstantsCounter_ - compressedF16constantsCounter_,
                    convertBytesToReadableSize(totalCompressedConstantsBeforeCompression_ -
                                               compressedF16constantsBeforeCompression_),
                    convertBytesToReadableSize(totalCompressedConstantsAfterCompression_ -
                                               compressedF16constantsAfterCompression_),
                    (double)(totalCompressedConstantsAfterCompression_ - compressedF16constantsAfterCompression_) /
                            (totalCompressedConstantsBeforeCompression_ - compressedF16constantsBeforeCompression_) *
                            100);
        }
        if (uncompressedConstantsCount == 0) {
            return;
        }
        auto uncompressedWeigthsSize = totalConstantsBeforeCompression_ - totalCompressedConstantsBeforeCompression_;
        log.info("Not compressed weights - count: {0}, size: {1}", uncompressedConstantsCount,
                 convertBytesToReadableSize(uncompressedWeigthsSize));
    }

private:
    uint64_t constantsCounter_;
    uint64_t compressedConstantsCounter_;
    uint64_t compressedF16constantsCounter_;
    uint64_t totalConstantsBeforeCompression_;
    uint64_t totalCompressedConstantsAfterCompression_;
    uint64_t totalCompressedConstantsBeforeCompression_;
    uint64_t compressedF16constantsAfterCompression_;
    uint64_t compressedF16constantsBeforeCompression_;
    uint64_t totalUncompressedConstants_;
};

class ConstSwizzlingCounter {
public:
    ConstSwizzlingCounter()
            : numOfSwizzledConsts_{0},
              numOfNotSwizzledConsts_{0},
              totalSizeOfSwizzledConsts_{0},
              totalSizeOfNotSwizzledConsts_{0} {
    }

    void count(mlir::Operation* op) {
        if (auto cstOp = mlir::dyn_cast<Const::DeclareOp>(op)) {
            if (cstOp.getOutput().getUsers().empty()) {
                return;
            }

            auto cstOpType = cstOp.getType().cast<vpux::NDTypeInterface>();
            auto size = cstOpType.getTotalAllocSize().count();

            if (vpux::getSwizzlingSchemeAttr(cstOpType)) {
                numOfSwizzledConsts_++;
                totalSizeOfSwizzledConsts_ += size;
            } else {
                numOfNotSwizzledConsts_++;
                totalSizeOfNotSwizzledConsts_ += size;
            }
        }
    }

    void printStatistics(vpux::Logger log) {
        log = log.nest();
        log.info("Swizzled constants     - count: {0}, size: {1}", numOfSwizzledConsts_,
                 convertBytesToReadableSize(totalSizeOfSwizzledConsts_));
        log.info("Not swizzled constants - count: {0}, size: {1}", numOfNotSwizzledConsts_,
                 convertBytesToReadableSize(totalSizeOfNotSwizzledConsts_));
        log = log.unnest();
    }

private:
    uint64_t numOfSwizzledConsts_;
    uint64_t numOfNotSwizzledConsts_;
    uint64_t totalSizeOfSwizzledConsts_;
    uint64_t totalSizeOfNotSwizzledConsts_;
};

class BarrierCounter {
public:
    BarrierCounter(): numOfBarriers_{0} {
    }

    void count(mlir::Operation* op) {
        if (mlir::dyn_cast<VPURT::ConfigureBarrierOp>(op)) {
            numOfBarriers_++;
        }
    }

    void printStatistics(vpux::Logger log) {
        log.nest().info("VPURT.ConfigureBarrierOp - {0} ops", numOfBarriers_);
    }

private:
    uint64_t numOfBarriers_;
};

std::tuple<uint64_t, uint64_t> getInputOutputSize(mlir::func::FuncOp funcOp) {
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    auto inputs = funcOp.getArgumentTypes();
    auto results = funcOp.getResultTypes();

    for (auto& input : inputs) {
        inputSize += input.cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    }
    for (auto& result : results) {
        outputSize += result.cast<vpux::NDTypeInterface>().getTotalAllocSize().count();
    }
    // For every result we have entry in arguments
    inputSize -= outputSize;
    return {inputSize, outputSize};
}

uint64_t getDDRHeapSize(mlir::func::FuncOp funcOp) {
    uint64_t maxOffset = 0;
    auto calcOffset = [&](VPURT::DeclareBufferOp bufferOp) {
        if (bufferOp.getMemorySpace() == VPURT::BufferSection::DDR) {
            auto currentOffset = bufferOp.getByteOffset() + bufferOp.getBinarySize();
            maxOffset = std::max(maxOffset, currentOffset);
        }
    };
    auto bufferOps = funcOp.getOps<VPURT::DeclareBufferOp>();
    llvm::for_each(bufferOps, calcOffset);

    return maxOffset;
}

//
// DumpStatisticsOfTaskOpsPass
//

class DumpStatisticsOfTaskOpsPass final :
        public VPUIP::impl::DumpStatisticsOfTaskOpsPassBase<DumpStatisticsOfTaskOpsPass> {
public:
    explicit DumpStatisticsOfTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DumpStatisticsOfTaskOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    auto opStatisticsCounter = populateCounters();
    BarrierCounter barrierCounter;
    CompressionRateCounter compressionCounter;
    ConstSwizzlingCounter constSwizzlingCounter;

    auto inputOutputsize = getInputOutputSize(func);
    _log.info("Input size - {0} Output size - {1}", convertBytesToReadableSize(std::get<0>(inputOutputsize)),
              convertBytesToReadableSize(std::get<1>(inputOutputsize)));

    auto ddrHeapSize = getDDRHeapSize(func);
    _log.info("DDR heap size - {0}", convertBytesToReadableSize(ddrHeapSize));

    func->walk([&](mlir::Operation* op) {
        opStatisticsCounter.count(op);
        barrierCounter.count(op);
        compressionCounter.count(op);
        constSwizzlingCounter.count(op);
    });

    _log.info("VPUIP tasks statistics:");
    opStatisticsCounter.printStatistics(_log);
    _log.info("Barrier statistics:");
    barrierCounter.printStatistics(_log);
    _log.info("Weights statistics:");
    compressionCounter.printStatistics(_log);
    _log.info("Const swizzling statistics:");
    constSwizzlingCounter.printStatistics(_log);
}

}  // namespace

//
// createDumpStatisticsOfTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDumpStatisticsOfTaskOpsPass(Logger log, bool forceLogging) {
    // Log level is forced to info by default since the log is checked by a LIT test
    // forceLogging is also used when 'dump-task-stats' is used explicitly
    return std::make_unique<DumpStatisticsOfTaskOpsPass>(forceLogging ? log.nest(0).setLevel(LogLevel::Info) : log);
}
