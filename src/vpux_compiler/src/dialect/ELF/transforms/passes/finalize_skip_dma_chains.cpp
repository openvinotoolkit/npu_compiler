//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/ELF/IR/dialect.hpp"
#include "vpux/compiler/dialect/ELF/IR/ops.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"

namespace vpux::ELF {
#define GEN_PASS_DECL_FINALIZESKIPDMACHAINS
#define GEN_PASS_DEF_FINALIZESKIPDMACHAINS
#include "vpux/compiler/dialect/ELF/passes.hpp.inc"
}  // namespace vpux::ELF

using namespace vpux;

// (section block, logicalTask)
using SkipGroupKey = std::pair<mlir::Block*, int64_t>;
namespace {
class FinalizeSkipDmaChainsPass : public ELF::impl::FinalizeSkipDmaChainsBase<FinalizeSkipDmaChainsPass> {
public:
    explicit FinalizeSkipDmaChainsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

VPUASM::NNDMAOp resolveReleaseDMA(VPUASM::NNDMAOp startDma, ELF::SymbolReferenceMap& symRefMap,
                                  llvm::DenseMap<mlir::SymbolRefAttr, VPUASM::NNDMAOp>& taskLocToDMA) {
    auto resolveNextDMA = [&](VPUASM::NNDMAOp dma) -> VPUASM::NNDMAOp {
        if (!dma || !dma.getNextLinkAttr()) {
            return nullptr;
        }

        auto target = symRefMap.lookupSymbol(dma.getNextLinkAttr());
        if (!target) {
            return nullptr;
        }

        // Case 1: DMA -> DMA
        if (auto nextDMA = mlir::dyn_cast<VPUASM::NNDMAOp>(target)) {
            return nextDMA;
        }

        // Case 2: DMA -> TaskBuffer -> DMA via taskLocation
        if (auto taskBuf = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(target)) {
            mlir::MLIRContext* ctx = dma.getContext();
            auto rootAttr = mlir::StringAttr::get(ctx, "program.metadata.cmx");
            auto symRef = mlir::SymbolRefAttr::get(rootAttr, {mlir::SymbolRefAttr::get(taskBuf.getSymNameAttr())});

            auto it = taskLocToDMA.find(symRef);
            if (it != taskLocToDMA.end()) {
                return it->second;
            }
        }

        return nullptr;
    };

    VPUASM::NNDMAOp current = startDma;

    while (current) {
        auto next = resolveNextDMA(current);
        if (!next) {
            break;
        }

        // First DMA without taskLocation is the release DMA
        if (!next.getTaskLocationAttr()) {
            return next;
        }
        current = next;
    }
    return nullptr;
}

// Forms a circular execution chain among Skip DMA operations grouped by
// (section block, logical task).
//
// Each group contains Skip DMAs in IR order for a given logical task within
// a section. This function links every DMA to the next one in the group via
// `nextLinkAttr`, with the last DMA wrapping around to the first, creating
// a closed loop.
//
void formSkipDMALoopPerLogicalTask(llvm::DenseMap<SkipGroupKey, SmallVector<VPUASM::NNDMAOp>>& skipGroups) {
    for (auto& [key, group] : skipGroups) {
        if (group.empty()) {
            continue;
        }

        for (size_t i = 0; i < group.size(); ++i) {
            auto current = group[i];
            auto next = group[(i + 1) % group.size()];

            auto nextTaskLoc = next.getTaskLocationAttr();
            VPUX_THROW_UNLESS(nextTaskLoc != nullptr,
                              "Skip DMA in group for logical task '{0}' is missing taskLocationAttr", key.second);
            current.setNextLinkAttr(nextTaskLoc);
        }
    }
}

// Resolves and assigns release descriptors for all Skip DMAs and propagates
// them to the corresponding KernelParamsOps.
//
// For each Skip DMA (keyed by descId):
// 1. Resolve its associated "release" DMA via symbol references.
// 2. Construct a SymbolRefAttr pointing to the release DMA
// 3. Map the resolved release descriptor to the corresponding KernelParamsOp
//    using descId -> KernelParamsOp association.
//
// After collection:
// - Each KernelParamsOp is assigned an ordered list of release descriptors
// - KernelParams buffer is resized to accommodate space for release descriptor
//   addresses (8 bytes per descriptor, representing DDR pointers).
//
// Note: This function appends kernel_params with 0s to make space for release descriptors, and relies on the convention
// that release descriptor addresses will be resolved with relocation
void resolveAndSetReleaseDescriptorForSkipDMAs(ELF::SymbolReferenceMap& symRefMap,
                                               llvm::DenseMap<int64_t, VPUASM::NNDMAOp>& skipMap,
                                               llvm::DenseMap<mlir::SymbolRefAttr, VPUASM::NNDMAOp>& taskLocToDMA,
                                               llvm::DenseMap<llvm::StringRef, llvm::StringRef>& dmaToSection,
                                               llvm::MapVector<int64_t, VPUASM::KernelParamsOp>& descIdToKernelParams) {
    // KernelParamsOp -> list of release descriptors
    llvm::DenseMap<VPUASM::KernelParamsOp, SmallVector<std::pair<int64_t, mlir::SymbolRefAttr>>> kernelToReleaseDescs;
    for (auto& [key, skipDma] : skipMap) {
        auto releaseDma = resolveReleaseDMA(skipDma, symRefMap, taskLocToDMA);
        VPUX_THROW_UNLESS(releaseDma != nullptr, "Failed to resolve release DMA for skip DMA '{0}' (descId={1})",
                          skipDma.getSymName(), key);

        auto symNameAttr = releaseDma.getSymNameAttr();
        VPUX_THROW_UNLESS(symNameAttr != nullptr, "Release DMA must have symbol");

        auto secIt = dmaToSection.find(symNameAttr.getValue());
        VPUX_THROW_UNLESS(secIt != dmaToSection.end(), "Release DMA must have root section reference");

        auto rootAttr = mlir::StringAttr::get(skipDma.getContext(), secIt->second);
        auto symRef = mlir::SymbolRefAttr::get(rootAttr, {mlir::FlatSymbolRefAttr::get(symNameAttr)});

        // Collect for KernelParams (with key for ordering)
        auto kernelParamIt = descIdToKernelParams.find(key);
        VPUX_THROW_UNLESS(kernelParamIt != descIdToKernelParams.end(),
                          "No KernelParamsOp found for Skip DMA with descId {0}", key);
        auto kernelParamOp = kernelParamIt->second;
        kernelToReleaseDescs[kernelParamOp].emplace_back(key, symRef);
    }

    // Set aggregated releaseDesc on KernelParamsOp
    //
    // kernel_params is an array ref of uint8_t
    // <---------------attrs-------------------> <-----8 byte x maxNumReleaseDesc release descriptor addr------>
    // [----------------------------------------|------------|-----------|-----------|----------]
    //
    for (auto& [kernelOp, releaseVec] : kernelToReleaseDescs) {
        VPUX_THROW_UNLESS(releaseVec.size() <= VPUASM::maxNumReleaseDesc,
                          "KernelOp has {0} release descriptors, but maximum supported is {1}", releaseVec.size(),
                          VPUASM::maxNumReleaseDesc);

        std::sort(releaseVec.begin(), releaseVec.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        mlir::Builder builder(kernelOp.getContext());
        auto paramsRef = kernelOp.getKernelParams();
        SmallVector<uint8_t> params(paramsRef.begin(), paramsRef.end());
        // Reserve a fixed trailing area for up to VPUASM::maxNumReleaseDesc release descriptors. Each address uses 8
        // bytes (uint64_t). Fixed size reservation keeps the kernel_params stable even when fewer release descriptors
        // are resolved for KernelParamsOp.
        params.resize(params.size() + VPUASM::maxNumReleaseDesc * sizeof(uint64_t), 0);
        kernelOp.setKernelParams(params);

        // Create ArrayAttr for releaseDesc
        SmallVector<mlir::Attribute> releaseAttrs;
        releaseAttrs.reserve(VPUASM::maxNumReleaseDesc);
        for (auto& [_, symRef] : releaseVec) {
            releaseAttrs.push_back(symRef);
        }
        auto arrayAttr = builder.getArrayAttr(releaseAttrs);
        kernelOp.setReleaseDescAttr(arrayAttr);
    }
}

//
// Patches fetch DMAs to point to their corresponding skip DMAs.
//
// For each fetch DMA in the `fetchMap`, this function finds the matching skip DMA
// in `skipMap` based on descriptor ID. It then sets the `input` attribute of the
// fetch DMA to reference the skip DMA symbol. If a fetch DMA does not
// have a corresponding skip DMA, an exception is thrown.
//
void patchFetchDMAsForSkipDmas(llvm::DenseMap<int64_t, VPUASM::NNDMAOp>& fetchMap,
                               llvm::DenseMap<int64_t, VPUASM::NNDMAOp>& skipMap,
                               llvm::DenseMap<llvm::StringRef, llvm::StringRef>& dmaToSection) {
    for (auto it : llvm::make_early_inc_range(fetchMap)) {
        auto fetchDmaOp = it.second;
        auto skipIt = skipMap.find(it.first);
        VPUX_THROW_UNLESS(skipIt != skipMap.end(), "Fetch DMA {0} has no corresponding Skip DMA",
                          fetchDmaOp.getSymName());

        auto skipDmaOp = skipIt->second;
        const auto sectionIt = dmaToSection.find(skipDmaOp.getSymName());
        VPUX_THROW_UNLESS(sectionIt != dmaToSection.end(), "DMA '{0}' has no associated ELF section",
                          skipDmaOp.getSymName());
        auto rootAttr = mlir::StringAttr::get(skipDmaOp.getContext(), sectionIt->second);
        auto symName = mlir::SymbolRefAttr::get(rootAttr, {mlir::FlatSymbolRefAttr::get(skipDmaOp.getSymNameAttr())});
        fetchDmaOp.setInputAttr(symName);
    }
}

void FinalizeSkipDmaChainsPass::safeRunOnFunc() {
    auto netFunc = getOperation();

    VPUX_THROW_UNLESS(llvm::hasSingleElement(netFunc.getOps<ELF::MainOp>()), "Expected exactly one ELF mainOp. Got {0}",
                      llvm::range_size(netFunc.getOps<ELF::MainOp>()));
    auto elfMain = *netFunc.getOps<ELF::MainOp>().begin();
    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    llvm::DenseMap<llvm::StringRef, llvm::StringRef> dmaToSection;
    llvm::DenseMap<mlir::SymbolRefAttr, VPUASM::NNDMAOp> taskLocToDMA;
    llvm::DenseMap<int64_t, VPUASM::NNDMAOp> fetchMap;
    llvm::DenseMap<int64_t, VPUASM::NNDMAOp> skipMap;

    // This is intentionally a MapVector to preserve the order of descIds as they appear in the IR
    llvm::MapVector<int64_t, VPUASM::KernelParamsOp> descIdToKernelParams;
    llvm::DenseMap<SkipGroupKey, SmallVector<VPUASM::NNDMAOp>> skipGroups;

    // Collect all DMAs, taskLocations and section name mapping
    _log.trace("Collecting DMA and KernelParams operations and building reference maps");
    for (auto section : elfMain.getOps<ELF::ElfSectionInterface>()) {
        auto* block = section.getBlock();
        if (!block || block->empty()) {
            continue;
        }

        auto sectionName = section.getSectionName();
        _log.trace("Processing section '{0}' for DMA collection", sectionName);
        for (auto& op : block->getOperations()) {
            if (auto dmaOp = mlir::dyn_cast<VPUASM::NNDMAOp>(&op)) {
                // DMA symbol -> section mapping
                dmaToSection[dmaOp.getSymName()] = sectionName;
                if (auto taskLoc = dmaOp.getTaskLocationAttr()) {
                    taskLocToDMA[taskLoc] = dmaOp;
                }

                // Fetch map
                if (auto fetchAttr = dmaOp.getFetchDmaAttr()) {
                    VPUX_THROW_UNLESS(fetchAttr.getDescId(), "Fetch DMA {0} must have descId", dmaOp.getSymName());
                    auto key = fetchAttr.getDescId().getValue().getSExtValue();
                    fetchMap[key] = dmaOp;
                }

                // Skip map and grouping
                if (auto skipAttr = dmaOp.getSkipDmaAttr()) {
                    // Skip Group
                    auto logicalTask = skipAttr.getAssociatedLogicalTaskIdx().getValue().getSExtValue();
                    SkipGroupKey groupKey{block, logicalTask};
                    skipGroups[groupKey].push_back(dmaOp);

                    // skipMap
                    auto key = skipAttr.getDescId().getValue().getSExtValue();
                    skipMap[key] = dmaOp;
                }
            }
            if (auto kernelParamsOp = mlir::dyn_cast<VPUASM::KernelParamsOp>(&op)) {
                auto skipDescIds = kernelParamsOp.getSkipDescIds();
                if (!skipDescIds) {
                    continue;
                }

                for (auto attr : skipDescIds.value()) {
                    auto descId = mlir::cast<mlir::IntegerAttr>(attr);
                    int64_t key = descId.getValue().getSExtValue();
                    VPUX_THROW_UNLESS(descIdToKernelParams.count(key) == 0, "Duplicate descId {0} in KernelParamsOp",
                                      key);

                    descIdToKernelParams[key] = kernelParamsOp;
                }
            }
        }
    }

    if (skipMap.empty()) {
        _log.trace("No Skip DMAs found, skipping pass");
        return;
    }

    _log.trace("Collected {0} Skip DMAs and {1} Fetch DMAs", skipMap.size(), fetchMap.size());

    _log.trace("Resolving and setting release descriptors for skip DMAs");
    resolveAndSetReleaseDescriptorForSkipDMAs(symRefMap, skipMap, taskLocToDMA, dmaToSection, descIdToKernelParams);
    _log.trace("Forming circular skip DMA loops");
    formSkipDMALoopPerLogicalTask(skipGroups);
    _log.trace("Patching fetch DMAs to point to skip DMAs");
    patchFetchDMAsForSkipDmas(fetchMap, skipMap, dmaToSection);
}
}  // namespace

//
// createFinalizeSkipDmaChainsPass
//

std::unique_ptr<mlir::Pass> ELF::createFinalizeSkipDmaChainsPass(Logger log) {
    return std::make_unique<FinalizeSkipDmaChainsPass>(log);
}
