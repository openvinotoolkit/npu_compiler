//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/counters_category.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/swizzling_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/dialect/net/IR/dialect.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/abstract_tree.hpp"
#include "vpux/compiler/utils/statistics_collection.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/profiling/common.hpp"

#include "vpux/utils/core/range.hpp"

#include <functional>
#include <map>
#include <memory>
#include <optional>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_DUMPSTATISTICSOFTASKOPSPASS
#define GEN_PASS_DEF_DUMPSTATISTICSOFTASKOPSPASS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using namespace vpux::VPUIP;

namespace {

//
// Declare utility functions
//

using CountersNode = vpux::utils::OpCounterTree::Node;

bool isNameLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc)) {
        return nameLoc.getName().str().find(substr) != std::string::npos;
    }
    return false;
}

bool isLocContainsStr(mlir::Location loc, const std::string& substr) {
    if (isNameLocContainsStr(loc, substr)) {
        return true;
    }
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
        for (auto& loc : fusedLoc.getLocations()) {
            if (isNameLocContainsStr(loc, substr)) {
                return true;
            }
        }
    }
    return false;
}

uint64_t getBufferSize(mlir::func::FuncOp funcOp, mlir::Type type, size_t sectionIndex, bool isInput) {
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);

    const auto section = isInput ? VPURT::BufferSection::NetworkInput : VPURT::BufferSection::NetworkOutput;
    return ELF::getBufferBinarySize(ndType, funcOp, section, static_cast<int64_t>(sectionIndex));
}

std::optional<std::tuple<uint64_t, uint64_t>> getInputOutputSize(mlir::func::FuncOp funcOp, Logger log) {
    const auto args = funcOp.getArgumentTypes();
    const auto results = funcOp.getResultTypes();

    if (results.size() > args.size()) {
        log.warning("Skipping input/output size statistics for {0}: function has {1} results but only {2} arguments",
                    funcOp.getName(), results.size(), args.size());
        return std::nullopt;
    }

    const auto inputTypes = args.take_front(args.size() - results.size());

    const auto needsNetworkBounds = [&](auto types) {
        return llvm::any_of(types, [](mlir::Type type) {
            const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
            return ndType.getShape().isDynamic();
        });
    };

    const bool hasDynamicInputs = needsNetworkBounds(args);
    const bool hasDynamicOutputs = needsNetworkBounds(results);

    std::optional<net::NetworkInfoOp> netInfo = std::nullopt;
    if (hasDynamicInputs || hasDynamicOutputs) {
        auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
        if (moduleOp == nullptr) {
            log.warning("Skipping input/output size statistics for {0}: parent module not found", funcOp.getName());
            return std::nullopt;
        }
        auto netOps = to_small_vector(moduleOp.getOps<net::NetworkInfoOp>());
        if (netOps.size() != 1) {
            log.warning(
                    "Skipping input/output size statistics for {0}: expected exactly one net::NetworkInfoOp, found {1}",
                    funcOp.getName(), netOps.size());
            return std::nullopt;
        }
        netInfo = netOps.front();
    }

    const auto netInputs = netInfo.has_value() ? to_small_vector(netInfo->getInputsDataInfo())
                                               : decltype(to_small_vector(netInfo->getInputsDataInfo())){};
    const auto netOutputs = netInfo.has_value() ? to_small_vector(netInfo->getOutputsDataInfo())
                                                : decltype(to_small_vector(netInfo->getOutputsDataInfo())){};

    const auto sumPerBufferSizes = [&](auto types, bool isInput) -> std::optional<uint64_t> {
        uint64_t size = 0;
        for (size_t idx = 0; idx < types.size(); ++idx) {
            const auto type = mlir::cast<vpux::NDTypeInterface>(types[idx]);
            log.trace("Calculating buffer size for {0} buffer {1}", isInput ? "input" : "output", idx);
            if (!type.getShape().isDynamic()) {
                size += type.getTotalAllocSize().count();
                continue;
            }

            const auto& dataInfos = isInput ? netInputs : netOutputs;
            if (idx >= dataInfos.size()) {
                log.warning("Skipping input/output size statistics for {0}: missing NetworkInfo entry for dynamic {1} "
                            "buffer {2}",
                            funcOp.getName(), isInput ? "input" : "output", idx);
                return std::nullopt;
            }
            const auto bounds = getBoundsFromDataInfo(dataInfos, idx);
            if (bounds.empty()) {
                log.warning(
                        "Skipping input/output size statistics for {0}: missing bounds in NetworkInfo for dynamic {1} "
                        "buffer {2}",
                        funcOp.getName(), isInput ? "input" : "output", idx);
                return std::nullopt;
            }
            log.trace("DataInfo[{0}] has bounds: {1}", idx, bounds);
            size += getBufferSize(funcOp, type, idx, isInput);
        }
        return size;
    };

    const auto inputSize = sumPerBufferSizes(inputTypes, /*isInput=*/true);
    const auto outputSize = sumPerBufferSizes(results, /*isInput=*/false);
    if (!inputSize.has_value() || !outputSize.has_value()) {
        return std::nullopt;
    }

    return std::make_tuple(*inputSize, *outputSize);
}

CountersVec getSpillCounter(const std::string& category) {
    CountersVec res;
    if (category != "DDR2CMX" && category != "CMX2DDR") {
        return res;
    }
    const auto spillCategory = category == "DDR2CMX" ? "SPILL_READ" : "SPILL_WRITE";
    const auto targetSubstr = category == "DDR2CMX" ? SPILL_READ_OP_NAME_SUFFIX : SPILL_WRITE_OP_NAME_SUFFIX;
    res.push_back(makeCounterNode(spillCategory, [=](mlir::Operation* op) {
        if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(op->getLoc())) {
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
    }));
    return res;
}

std::string getProfSuffix(const std::string& profCategory) {
    return profCategory + profiling::PROFILING_CMX_2_DDR_OP_NAME;
}

void addProfilingCounters(CountersVec& counters, const std::string& category) {
    if (category == "REG2CMX") {
        counters.push_back(makeCounterNode("Profiling Timestamp DMA", [](auto op) {
            if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
                return dmaOp.getProfilingMetadata().has_value();
            }
            return false;
        }));
        return;
    }
    if (category == "REG2DDR") {
        counters.push_back(makeCounterNode("Profiling workpoint", [](auto op) {
            return isLocContainsStr(op->getLoc(), profiling::PROFILING_WORKPOINT_READ_ATTR);
        }));
        return;
    }
    if (category == "CMX2DDR") {
        CountersVec nestedProfCounters;
        const static std::vector<std::pair<std::string, utils::OpCounter::IsOperationSuitable>> profCounterCfgs = {
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
            nestedProfCounters.push_back(makeCounterNode(p.first, p.second));
        }
        counters.push_back(makeCounterNode(
                "Profiling buffer management",
                [](auto op) {
                    return isLocContainsStr(op->getLoc(), profiling::PROFILING_CMX_2_DDR_OP_NAME);
                },
                std::move(nestedProfCounters)));
    }
    if (category == "DDR2DDR") {
        CountersVec nestedProfCounters;
        const static std::vector<std::pair<std::string, utils::OpCounter::IsOperationSuitable>> profCounterCfgs = {
                {"DMA", [](auto op) {
                     return isLocContainsStr(op->getLoc(), std::string("dma") + profiling::PROFILING_DDR_2_DDR_OP_NAME);
                 }}};
        for (const auto& p : profCounterCfgs) {
            nestedProfCounters.push_back(makeCounterNode(p.first, p.second));
        }
        counters.push_back(makeCounterNode(
                "Profiling buffer management",
                [](auto op) {
                    return isLocContainsStr(op->getLoc(), profiling::PROFILING_DDR_2_DDR_OP_NAME);
                },
                std::move(nestedProfCounters)));
    }
}

CountersVec getInnerCounters(const std::string& category) {
    auto counters = getSpillCounter(category);
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
            return mlir::cast<vpux::NDTypeInterface>(operand.getType()).getMemoryKind() == memKind;
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
            auto nestedCounters = getInnerCounters(category);
            counters.push_back(makeCounterNode(
                    category,
                    [=](auto op) {
                        return checkInputOutputMemSpace(op, inputConf.second, outputConf.second);
                    },
                    std::move(nestedCounters)));
        }
    }
    return counters;
}

template <class... Args, typename = typename std::enable_if<sizeof...(Args) == 0>::type>
void populateDMACounters(CountersVec&) {
}

template <class DMAType, class... DMATypeArgs>
void populateDMACounters(CountersVec& counters) {
    utils::OpCounter::HandleUnrecognizedCounter dmaRemainderHandler = [](mlir::Operation*) {
        return "Unknown memory space DMA";
    };
    counters.push_back(makeCounterNode(
            DMAType::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<DMAType>(op);
            },
            getDMANestedCounters<DMAType>(), std::move(dmaRemainderHandler)));
    populateDMACounters<DMATypeArgs...>(counters);
}

CountersVec getDMACounters() {
    CountersVec dmaCounters;
    using namespace VPUIP;
    populateDMACounters<NNDMAOp, CompressDMAOp, DecompressDMAOp, DepthToSpaceDMAOp, PermuteDMAOp, ExpandDMAOp>(
            dmaCounters);
    return dmaCounters;
}

CountersNode getSWKernelsCounter() {
    utils::OpCounter::HandleUnrecognizedCounter swHandler = [](mlir::Operation* op) -> std::string {
        if (auto swKernelOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            return swKernelOp.getKernelFunction().getLeafReference().str();
        }
        return "Not SwKernel";
    };

    return makeCounterNode(
            VPUIP::SwKernelOp::getOperationName().str(),
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::SwKernelOp>(op);
            },
            CountersVec{}, std::move(swHandler));
}

CountersNode getSparsityCounter() {
    utils::OpCounter::HandleUnrecognizedCounter remainedOpHandler = [](mlir::Operation*) {
        return "Dense";
    };

    auto sparseInputCounter = makeCounterNode("Sparse input", [](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            bool hasSparseInput =
                    nceOp.getInputSparsityMap() != nullptr || nceOp.getInputStorageElementTable() != nullptr;
            if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE) {
                hasSparseInput |= nceOp.getWeightsSparsityMap() != nullptr;
            }
            return hasSparseInput;
        }
        return false;
    });

    auto sparseWeightsCounter = makeCounterNode("Sparse weights", [](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            return nceOp.getWeightsSparsityMap() != nullptr;
        }
        return false;
    });

    auto sparseOutputCounter = makeCounterNode("Sparse output", [](mlir::Operation* op) {
        if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(op)) {
            return nceOp.getOutputSparsityMap() != nullptr;
        }
        return false;
    });

    CountersVec counters;
    counters.push_back(std::move(sparseInputCounter));
    counters.push_back(std::move(sparseWeightsCounter));
    counters.push_back(std::move(sparseOutputCounter));
    return makeCounterNode(
            "NCETask Operations",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::NCEClusterTaskOp>(op);
            },
            std::move(counters), std::move(remainedOpHandler));
}

CountersNode getM2ICounter() {
    return makeCounterNode(VPUIP::M2ITaskOp::getOperationName().str(), [](mlir::Operation* op) {
        return mlir::isa<VPUIP::M2ITaskOp>(op);
    });
}

utils::OpCounterTree populateCounters(mlir::MLIRContext* ctx) {
    utils::OpCounter::HandleUnrecognizedCounter remainedOpHandler = [](mlir::Operation* op) {
        return op->getName().getIdentifier().str();
    };

    auto counters = getDMACounters();
    counters.push_back(getSWKernelsCounter());
    counters.push_back(getSparsityCounter());
    counters.push_back(getM2ICounter());

    auto mpeModeCountersStrategy = VPUIP::getVPUIPStrategyFactory(ctx)->getCountersStrategy();
    mpeModeCountersStrategy->appendCounters(counters);

    CountersVec roots;
    roots.push_back(makeCounterNode(
            "VPUIP Tasks",
            [](mlir::Operation* op) {
                return mlir::isa<VPUIP::TaskOpInterface>(op);
            },
            std::move(counters), std::move(remainedOpHandler)));
    return utils::OpCounterTree(std::move(roots));
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
        const auto type = mlir::cast<vpux::NDTypeInterface>(buffer.getType());
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
                        mlir::cast<vpux::NDTypeInterface>(decompressDMAOp.getOperand(0).getType()).getElementType();

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
        if (compressedConstantsCounter_ > compressedF16constantsCounter_) {
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

            auto cstOpType = mlir::cast<vpux::NDTypeInterface>(cstOp.getType());
            auto size = cstOpType.getTotalAllocSize().count();

            if (VPUIP::getSwizzlingSchemeAttr(cstOpType)) {
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

    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<vpux::VPUIP::VPUIPDialect, vpux::VPURT::VPURTDialect, vpux::Const::ConstDialect,
                        vpux::net::NetDialect>();
    }

private:
    void safeRunOnFunc() final;
};

void DumpStatisticsOfTaskOpsPass::safeRunOnFunc() {
    auto func = getOperation();

    auto opStatisticsCounter = populateCounters(&getContext());
    BarrierCounter barrierCounter;
    CompressionRateCounter compressionCounter;
    ConstSwizzlingCounter constSwizzlingCounter;

    if (auto inputOutputsize = getInputOutputSize(func, _log); inputOutputsize.has_value()) {
        _log.info("Input size - {0} Output size - {1}", convertBytesToReadableSize(std::get<0>(*inputOutputsize)),
                  convertBytesToReadableSize(std::get<1>(*inputOutputsize)));
    }

    auto ddrHeapSize = getDDRHeapSize(func);
    _log.info("DDR heap size - {0}", convertBytesToReadableSize(ddrHeapSize));

    func->walk([&](mlir::Operation* op) {
        utils::AddOpRecordVisitor collector(op);
        opStatisticsCounter.apply(collector);

        barrierCounter.count(op);
        compressionCounter.count(op);
        constSwizzlingCounter.count(op);
    });

    _log.info("VPUIP tasks statistics:");
    utils::PrintOpRecordVisitor printer(_log);
    opStatisticsCounter.apply(printer);
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
