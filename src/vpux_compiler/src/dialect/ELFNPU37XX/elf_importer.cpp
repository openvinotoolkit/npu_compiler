//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELFNPU37XX/elf_importer.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <vpux_elf/accessor.hpp>
#include "vpux/compiler/dialect/ELFNPU37XX/attributes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/dialect.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <vector>

constexpr size_t INVALID_SEC_IDX = 0;

using namespace vpux;

vpux::ELFNPU37XX::ElfImporter::ElfImporter(mlir::MLIRContext* ctx, const std::string& elfFileName, Logger log)
        : _accessor(new elf::FSAccessManager(elfFileName)),
          _elfReader(elf::Reader<elf::ELF_Bitness::Elf64>(_accessor)),
          _ctx(ctx),
          _log(log) {
    _log.setName("ELFNPU37XX::ElfImporter");

    _log.trace("Load VPUX::VPUELF dependent Dialects");
    _ctx->loadDialect<IE::IEDialect>();
    _ctx->loadDialect<VPUMI37XX::VPUMI37XXDialect>();
    _ctx->loadDialect<VPURegMapped::VPURegMappedDialect>();
    _ctx->loadDialect<VPURT::VPURTDialect>();
    _ctx->loadDialect<ELFNPU37XX::ELFNPU37XXDialect>();

    _mainFuncName = mlir::FlatSymbolRefAttr::get(_ctx, "main");
    _noOfSections = _elfReader.getSectionsNum();
}

vpux::ELFNPU37XX::ElfImporter::~ElfImporter() {
    delete _accessor;
}

void vpux::ELFNPU37XX::ElfImporter::buildNetworkInfoOp() {
    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockEnd(_module.getBody(), &builderLog);

    auto netInfo = builder.create<net::NetworkInfoOp>(mlir::UnknownLoc::get(_ctx), _mainFuncName, false);

    parseUserInputsOutputs(builderLog, netInfo);
}

void vpux::ELFNPU37XX::ElfImporter::parseUserInputsOutputs(OpBuilderLogger& builderLog, net::NetworkInfoOp& netInfo) {
    netInfo.getInputsInfo().emplaceBlock();
    netInfo.getOutputsInfo().emplaceBlock();

    const auto processUserIO = [this](const elf::Reader<elf::ELF_Bitness::Elf64>::Section& section,
                                      mlir::OpBuilder& builder, SmallVector<mlir::Type>& paramTypes) {
        const auto& symStrTab = _elfReader.getSection(section.getHeader()->sh_link);
        const auto* symbols = section.getData<elf::SymbolEntry>();
        const auto symbolsNum = section.getEntriesNum();

        for (unsigned int idx = 1; idx < symbolsNum; ++idx) {
            const auto inputName = std::string(symStrTab.getData<char>() + symbols[idx].st_name);
            const int64_t size = checked_cast<int64_t>(symbols[idx].st_size);
            ArrayRef<int64_t> shapeType{&size, 1};
            const auto rankedTensor = mlir::RankedTensorType::get(shapeType, getUInt8Type(_ctx));
            const auto memRefRankedTensor = mlir::MemRefType::get(shapeType, getUInt8Type(_ctx));

            paramTypes.push_back(memRefRankedTensor);
            builder.create<net::DataInfoOp>(mlir::UnknownLoc::get(_ctx), inputName, rankedTensor);
        }
    };

    for (size_t sectionCtr = 0; sectionCtr < _noOfSections; ++sectionCtr) {
        const auto& section = _elfReader.getSection(sectionCtr);
        const auto sectionHeader = section.getHeader();

        if (elf::SHT_SYMTAB == sectionHeader->sh_type) {
            auto sectionFlags = sectionHeader->sh_flags;
            if (sectionFlags & elf::VPU_SHF_USERINPUT) {
                auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&netInfo.getInputsInfo().front(), &builderLog);
                processUserIO(section, inputsInfoBuilder, _inputTypes);
            } else if (sectionFlags & elf::VPU_SHF_USEROUTPUT) {
                auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&netInfo.getOutputsInfo().front(), &builderLog);
                processUserIO(section, outputsInfoBuilder, _outputTypes);
            }
        }
    }
}

void vpux::ELFNPU37XX::ElfImporter::createConfigureBarrierOp(mlir::OpBuilder& opsBuilder,
                                                             const size_t noOfBarrierConfigs) {
    auto barrierOffest =
            offsetof(npu37xx::nn_public::VpuMappedInference, barrier_configs) +
            offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuBarrierCountConfig>, address);
    auto barrierSectionIdx = getSectionIndexBasedOnRelocAOffset(barrierOffest, _mappedInferSectionIdx);

    _log.debug("barrierSectionIdx {0}", barrierSectionIdx);
    if (barrierSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    const auto& section = _elfReader.getSection(barrierSectionIdx);
    const auto* barrierCountConfig = section.getData<npu37xx::nn_public::VpuBarrierCountConfig>();
    VPUX_THROW_UNLESS(barrierCountConfig != nullptr, "Got null barrierCountConfig");

    std::vector<mlir::Value> barrierConfigOps;
    mlir::Type type64Attr = mlir::IntegerType::get(_ctx, 64, mlir::IntegerType::Signed);
    mlir::Type type8Attr = mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::Unsigned);
    mlir::Type typeU8Attr = mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::Unsigned);
    for (size_t idx = 0; idx < noOfBarrierConfigs; ++idx) {
        mlir::IntegerAttr realId = mlir::IntegerAttr::get(type8Attr, barrierCountConfig->real_id_);
        mlir::IntegerAttr nextSameId = mlir::IntegerAttr::get(type64Attr, barrierCountConfig->next_same_id_);
        mlir::IntegerAttr producerCount = mlir::IntegerAttr::get(typeU8Attr, barrierCountConfig->producer_count_);
        mlir::IntegerAttr consumerCount = mlir::IntegerAttr::get(typeU8Attr, barrierCountConfig->consumer_count_);

        auto barrierOpValue = opsBuilder.create<VPUMI37XX::ConfigureBarrierOp>(
                mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, idx), realId, nextSameId,
                /*previousSameId*/ nullptr, producerCount, consumerCount);
        _barrierConfigsByRealId.push_back(std::make_pair(barrierCountConfig->real_id_, barrierOpValue));
        barrierConfigOps.push_back(barrierOpValue);
        barrierCountConfig++;
    }

    createSectionOp(opsBuilder, barrierSectionIdx, barrierConfigOps);
}

std::vector<std::pair<unsigned int, ELFNPU37XX::SymbolOp>> vpux::ELFNPU37XX::ElfImporter::createSymbolOp(
        mlir::func::FuncOp& func, mlir::OpBuilder& opsBuilder,
        const elf::Reader<elf::ELF_Bitness::Elf64>::Section& section) {
    const auto sectionHeader = section.getHeader();
    const auto& symStrTab = _elfReader.getSection(sectionHeader->sh_link);
    const auto* symbols = section.getData<elf::SymbolEntry>();
    const auto symbolsNum = section.getEntriesNum();

    std::vector<std::pair<unsigned int, ELFNPU37XX::SymbolOp>> elfSymbolsOp;

    size_t argsIdx = 0;
    for (size_t idx = 1; idx < symbolsNum; ++idx) {
        const auto symName = std::string(symStrTab.getData<char>() + symbols[idx].st_name);
        mlir::Type typeAttr = mlir::IntegerType::get(_ctx, 64, mlir::IntegerType::Unsigned);
        mlir::IntegerAttr symSizeAttr = mlir::IntegerAttr::get(typeAttr, symbols[idx].st_size);
        mlir::IntegerAttr symValAttr = mlir::IntegerAttr::get(typeAttr, symbols[idx].st_value);
        auto symAttr =
                symbolizeSymbolTypeEnum(symbols[idx].st_info).value_or(vpux::ELFNPU37XX::SymbolTypeEnum::STT_NOTYPE);

        mlir::Value inputArg;
        if (sectionHeader->sh_flags & elf::VPU_SHF_USERINPUT) {
            inputArg = func.getArgument(checked_cast<unsigned>(argsIdx));
            argsIdx += argsIdx < checked_cast<size_t>(func.getNumArguments()) ? 1 : 0;
        } else if (sectionHeader->sh_flags & elf::VPU_SHF_USEROUTPUT) {
            argsIdx = !argsIdx ? _inputTypes.size() : argsIdx;
            inputArg = func.getArgument(checked_cast<unsigned>(argsIdx));
            argsIdx += argsIdx < checked_cast<size_t>(func.getNumArguments()) ? 1 : 0;
        } else {
            if (_sectionOpByValue.find(symbols[idx].st_shndx) == _sectionOpByValue.end()) {
                _log.debug("sectionOp for symbols[idx].st_shndx {0} doesn't exist -> continue", symbols[idx].st_shndx);
                continue;
            }
            inputArg = _sectionOpByValue[symbols[idx].st_shndx];
        }

        elfSymbolsOp.push_back(std::make_pair(
                idx, opsBuilder.create<ELFNPU37XX::SymbolOp>(
                             mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SymbolType::get(_ctx), inputArg, nullptr,
                             mlir::StringAttr::get(_ctx, symName),
                             vpux::ELFNPU37XX::SymbolTypeEnumAttr::get(_ctx, symAttr), symSizeAttr, symValAttr)));
    }

    return elfSymbolsOp;
}

size_t vpux::ELFNPU37XX::ElfImporter::getMappedInferenceSectionIndex() {
    for (size_t sectionCtr = 0; sectionCtr < _noOfSections; ++sectionCtr) {
        const auto& section = _elfReader.getSection(sectionCtr);
        const auto sectionHeader = section.getHeader();
        if (elf::SHT_SYMTAB == sectionHeader->sh_type) {
            const auto* symbols = section.getData<elf::SymbolEntry>();
            const auto symbolsNum = section.getEntriesNum();
            for (size_t idx = 1; idx < symbolsNum; ++idx) {
                if (symbols[idx].st_info & elf::VPU_STT_ENTRY) {
                    _log.info("flag for VPU_STT_ENTRY set in section with index {0}", symbols[idx].st_shndx);
                    return symbols[idx].st_shndx;
                }
            }
        }
    }

    return INVALID_SEC_IDX;
}

size_t vpux::ELFNPU37XX::ElfImporter::getSectionIndexBasedOnRelocAOffset(const size_t offset, const size_t shInfo) {
    for (size_t sectionCtr = 0; sectionCtr < _noOfSections; ++sectionCtr) {
        const auto& section = _elfReader.getSection(sectionCtr);
        if (const auto sectionHeader = section.getHeader()) {
            if (elf::SHT_RELA == sectionHeader->sh_type) {
                const auto* entries = section.getData<elf::RelocationAEntry>();
                const auto entriesNum = section.getEntriesNum();
                for (size_t idx = 0; idx < entriesNum; ++idx) {
                    const auto symbolIdx = elf::elf64RSym(entries[idx].r_info);
                    if (offset == entries[idx].r_offset && elf::VPU_RT_SYMTAB != sectionHeader->sh_link &&
                        shInfo == sectionHeader->sh_info) {
                        const auto& symtabSection = _elfReader.getSection(sectionHeader->sh_link);
                        const auto* symbols = symtabSection.getData<elf::SymbolEntry>();
                        return symbols[symbolIdx].st_shndx;
                    }
                }
            }
        }
    }

    return INVALID_SEC_IDX;
}

mlir::Value vpux::ELFNPU37XX::ElfImporter::getInputOrOutputValueForDmaTask(
        mlir::OpBuilder& opsBuilder, const size_t dmaSectionIdx, const uint64_t& offset, const size_t bufferSize,
        const elf::Elf_Xword& flag, const mlir::BlockArgument& funcArg) {
    for (size_t sectionCtr = 0; sectionCtr < _noOfSections; ++sectionCtr) {
        const auto& section = _elfReader.getSection(sectionCtr);
        if (const auto sectionHeader = section.getHeader()) {
            if (elf::SHT_RELA == sectionHeader->sh_type) {
                const auto* entries = section.getData<elf::RelocationAEntry>();
                const auto entriesNum = section.getEntriesNum();
                for (size_t idx = 0; idx < entriesNum; ++idx) {
                    if (offset == entries[idx].r_offset && dmaSectionIdx == sectionHeader->sh_info) {
                        auto isTypeDDR = elf::VPU_RT_SYMTAB != sectionHeader->sh_link;
                        mlir::Value value =
                                (flag & sectionHeader->sh_flags)
                                        ? static_cast<mlir::Value>(funcArg)
                                        : static_cast<mlir::Value>(createDeclareBufferOp(
                                                  opsBuilder, bufferSize, isTypeDDR, entries[idx].r_addend));
                        return value;
                    }
                }
            }
        }
    }

    return nullptr;
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActKernalRange(mlir::OpBuilder& opsBuilder,
                                                                     const size_t noOfActKRangeTasks) {
    auto actKRangesOffest =
            offsetof(npu37xx::nn_public::VpuMappedInference, act_kernel_ranges) +
            offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuActKernelRange>, address);
    auto actKRangeSectionIdx = getSectionIndexBasedOnRelocAOffset(actKRangesOffest, _mappedInferSectionIdx);
    _log.debug("actKRangeSectionIdx {0}", actKRangeSectionIdx);
    if (actKRangeSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    const auto& section = _elfReader.getSection(actKRangeSectionIdx);
    const auto* actKernelRange = section.getData<npu37xx::nn_public::VpuActKernelRange>();

    if (actKernelRange == nullptr) {
        _log.debug("actKernelRange is null");
        return;
    }

    const auto kernelTypeElfDummy = std::string("softmax") + std::string(".3720xx") + std::string(".elf");

    std::vector<mlir::Value> actKRangeOps;
    std::vector<mlir::Value> actKTextOps;
    std::vector<mlir::Value> actKDataOps;
    for (size_t idx = 0; idx < noOfActKRangeTasks; ++idx) {
        bool isCacheOp = true;
        mlir::SymbolRefAttr taskType;
        switch (actKernelRange->type) {
        case npu37xx::nn_public::VpuActWLType::WL_CACHE_OP_FLUSH:
            taskType =
                    mlir::SymbolRefAttr::get(_ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::CACHE_FLUSH));
            break;
        case npu37xx::nn_public::VpuActWLType::WL_CACHE_OP_INVALIDATE:
            taskType = mlir::SymbolRefAttr::get(
                    _ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::CACHE_INVALIDATE));
            break;
        case npu37xx::nn_public::VpuActWLType::WL_CACHE_OP_FLUSHINV:
            taskType = mlir::SymbolRefAttr::get(
                    _ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::CACHE_FLUSH_INVALIDATE));
            break;
        default:
            taskType = mlir::SymbolRefAttr::get(_ctx, VPU::stringifyActShaveTaskType(VPU::ActShaveTaskType::COMPUTE));
            isCacheOp = false;
            break;
        }

        VPUMI37XX::ActKernelRangeOp actKernalRangeOp;
        if (!isCacheOp) {
            auto kernelTextOp = opsBuilder.create<VPUMI37XX::DeclareKernelTextOp>(
                    opsBuilder.getUnknownLoc(), VPURegMapped::IndexType::get(_ctx, idx),
                    mlir::StringAttr::get(_ctx, kernelTypeElfDummy));
            actKTextOps.push_back(kernelTextOp.getResult());

            auto kernelArgsOp = opsBuilder.create<VPUMI37XX::DeclareKernelArgsOp>(
                    opsBuilder.getUnknownLoc(), VPURegMapped::IndexType::get(_ctx, idx),
                    mlir::StringAttr::get(_ctx, kernelTypeElfDummy));
            actKDataOps.push_back(kernelArgsOp.getResult());

            auto kernelEntryOp = opsBuilder.create<VPUMI37XX::DeclareKernelEntryOp>(
                    opsBuilder.getUnknownLoc(), VPURegMapped::IndexType::get(_ctx, idx),
                    mlir::StringAttr::get(_ctx, kernelTypeElfDummy));

            actKernalRangeOp = opsBuilder.create<VPUMI37XX::ActKernelRangeOp>(
                    mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, idx), /*taskLocation*/ nullptr,
                    kernelTextOp, kernelArgsOp, kernelEntryOp, taskType);
        } else {
            actKernalRangeOp = opsBuilder.create<VPUMI37XX::ActKernelRangeOp>(
                    mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, idx), /*taskLocation*/ nullptr,
                    nullptr, nullptr, nullptr, taskType);
        }

        _actKRangeOps.push_back(actKernalRangeOp);
        actKRangeOps.push_back(actKernalRangeOp.getResult());

        actKernelRange++;
    }

    createSectionOp(opsBuilder, actKRangeSectionIdx, actKRangeOps);

    createSectionOpForActKernelText(opsBuilder, actKTextOps, actKRangeSectionIdx);
    createSectionOpForActKernelData(opsBuilder, actKDataOps);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActKernelInvocation(mlir::OpBuilder& opsBuilder,
                                                                          const size_t noOfActKInvocationTasks) {
    auto actKInvocationsOffest =
            offsetof(npu37xx::nn_public::VpuMappedInference, act_kernel_invocations) +
            offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuActKernelInvocation>, address);
    auto actKInvocationSectionIdx = getSectionIndexBasedOnRelocAOffset(actKInvocationsOffest, _mappedInferSectionIdx);

    _log.debug("actKInvocationSectionIdx {0}", actKInvocationSectionIdx);
    if (actKInvocationSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    const auto& section = _elfReader.getSection(actKInvocationSectionIdx);
    const auto* actKernelInvocation = section.getData<npu37xx::nn_public::VpuActKernelInvocation>();
    if (actKernelInvocation == nullptr) {
        _log.debug("actKernelInvocation is null");
        return;
    }

    std::vector<mlir::Value> actKInvocationOps;
    mlir::Type type64UAttr = mlir::IntegerType::get(_ctx, 64, mlir::IntegerType::Unsigned);
    for (size_t idx = 0; idx < noOfActKInvocationTasks; ++idx) {
        mlir::ValueRange waitBarriers, updateBarriers;
        fillValueForWaitAndUpdateBarrierConfigs(actKernelInvocation->barriers.post_mask_,
                                                actKernelInvocation->barriers.wait_mask_, updateBarriers, waitBarriers);
        auto actKInvOp = opsBuilder.create<VPUMI37XX::ActKernelInvocationOp>(
                mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, idx), /*taskLocation*/ nullptr,
                mlir::ValueRange(waitBarriers), mlir::ValueRange(updateBarriers),
                _actKRangeOps.at(actKernelInvocation->kernel_range_index).getResult(), nullptr,
                /*profiling_data*/ nullptr, mlir::IntegerAttr::get(type64UAttr, actKernelInvocation->invo_tile),
                mlir::IntegerAttr::get(type64UAttr, actKernelInvocation->barriers_sched.start_after_),
                mlir::IntegerAttr::get(type64UAttr, actKernelInvocation->barriers_sched.clean_after_));

        _actKInvocationOps.push_back(actKInvOp);
        actKInvocationOps.push_back(actKInvOp.getResult());

        actKernelInvocation++;
    }
    createSectionOp(opsBuilder, actKInvocationSectionIdx, actKInvocationOps);

    createSectionOpForActKernelParams(opsBuilder, actKInvocationSectionIdx);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActKernelParams(mlir::OpBuilder& opsBuilder,
                                                                      const size_t actKInvocationSectionIdx) {
    auto actKernelParamsOffest = offsetof(npu37xx::nn_public::VpuActKernelInvocation, kernel_args);
    auto actKernelParamsSectionIdx =
            getSectionIndexBasedOnRelocAOffset(actKernelParamsOffest, actKInvocationSectionIdx);

    _log.debug("actKernelParamsSectionIdx {0}", actKernelParamsSectionIdx);
    if (actKernelParamsSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    SmallVector<uint8_t> params_vector_dummy = {
            // input
            0x00, 0x00, 0x00, 0x00,                          // dataAddr
            0x01, 0x00, 0x00, 0x00,                          // isStatic
            0x04, 0x00, 0x00, 0x00,                          // numDims
            0x00, 0x00, 0x00, 0x00,                          // dimsAddr
            0x00, 0x00, 0x00, 0x00,                          // stridesAddr
            0x00, 0x00, 0x00, 0x00,                          // dataType
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // dimsOrder
            0x02, 0x00, 0x00, 0x00,                          // location

            // output
            0x00, 0x00, 0x00, 0x00,                          // dataAddr
            0x01, 0x00, 0x00, 0x00,                          // isStatic
            0x04, 0x00, 0x00, 0x00,                          // numDims
            0x00, 0x00, 0x00, 0x00,                          // dimsAddr
            0x00, 0x00, 0x00, 0x00,                          // stridesAddr
            0x00, 0x00, 0x00, 0x00,                          // dataType
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // dimsOrders
            0x02, 0x00, 0x00, 0x00,                          // location

            0x01, 0x00, 0x00, 0x00  // axis
    };

    mlir::Type type8UIntAttr = mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
    auto params_size = static_cast<long int>(params_vector_dummy.size());
    SmallVector<mlir::ValueRange> dynInputShapes(1, mlir::ValueRange());
    SmallVector<mlir::ValueRange> dynOutputShapes(1, mlir::ValueRange());
    auto kernalParamsOp = opsBuilder.create<VPUMI37XX::KernelParamsOp>(
            mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, 0),
            mlir::ValueRange(_buffers.front().second.getResult()), mlir::ValueRange(_buffers.back().second.getResult()),
            dynInputShapes, dynOutputShapes, mlir::StringAttr::get(_ctx, "Softmax"),
            mlir::DenseIntElementsAttr::get(mlir::VectorType::get({params_size}, type8UIntAttr), params_vector_dummy),
            mlir::BoolAttr::get(_ctx, true)  // TODO: make it work in general case
    );

    createSectionOp(opsBuilder, actKernelParamsSectionIdx, kernalParamsOp.getResult());
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActKernelText(mlir::OpBuilder& opsBuilder,
                                                                    const std::vector<mlir::Value>& actKTextOps,
                                                                    const size_t actKRangeSectionIdx) {
    auto actKernelTextOffest = offsetof(npu37xx::nn_public::VpuActKernelInvocation, range) +
                               offsetof(npu37xx::nn_public::VpuActKernelRange, text_window_base);
    auto actKernelTextSectionIdx = getSectionIndexBasedOnRelocAOffset(actKernelTextOffest, actKRangeSectionIdx);

    _log.debug("actKernelTextSectionIdx {0}", actKernelTextSectionIdx);
    if (actKernelTextSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    createSectionOp(opsBuilder, actKernelTextSectionIdx, actKTextOps);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActKernelData(mlir::OpBuilder& opsBuilder,
                                                                    const std::vector<mlir::Value>& actKDataOps) {
    auto actKInvocationsOffest =
            offsetof(npu37xx::nn_public::VpuMappedInference, act_kernel_invocations) +
            offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuActKernelInvocation>, address);
    auto actKInvocationSectionIdx = getSectionIndexBasedOnRelocAOffset(actKInvocationsOffest, _mappedInferSectionIdx);

    auto actKernelDataOffest = offsetof(npu37xx::nn_public::VpuActKernelInvocation, data_window_base);

    auto actKernelDataSectionIdx = getSectionIndexBasedOnRelocAOffset(actKernelDataOffest, actKInvocationSectionIdx);

    _log.debug("actKernelDataSectionIdx {0}", actKernelDataSectionIdx);
    if (actKernelDataSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    createSectionOp(opsBuilder, actKernelDataSectionIdx, actKDataOps);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForShaveRtConfigs(mlir::OpBuilder& opsBuilder,
                                                                     const bool isScheduleEmbeddedRtUsed,
                                                                     mlir::Value& actRtTextOpValue) {
    if (!isScheduleEmbeddedRtUsed) {
        _log.debug("schedule embedded RT is not used");
        return;
    }

    auto actRtConfigsOffest = offsetof(npu37xx::nn_public::VpuMappedInference, shv_rt_configs) +
                              offsetof(npu37xx::nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base);
    auto actRtConfigsSectionIdx = getSectionIndexBasedOnRelocAOffset(actRtConfigsOffest, _mappedInferSectionIdx);

    _log.debug("actRtConfigsSectionIdx {0}", actRtConfigsSectionIdx);
    if (actRtConfigsSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    auto actShaveRtOpValue = opsBuilder.create<VPUMI37XX::ActShaveRtOp>(
            mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, 0), mlir::StringRef("nnActEntry"));
    actRtTextOpValue = actShaveRtOpValue.getResult();
    createSectionOp(opsBuilder, actRtConfigsSectionIdx, actShaveRtOpValue.getResult());
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForActShaveStacks(mlir::OpBuilder& opsBuilder) {
    for (size_t idx = 0; idx < npu37xx::nn_public::VPU_AS_TOTAL; ++idx) {
        auto actRtConfigsShaveStacksOffest =
                offsetof(npu37xx::nn_public::VpuMappedInference, shv_rt_configs) +
                offsetof(npu37xx::nn_public::VpuNNShaveRuntimeConfigs, stack_frames) +
                idx * sizeof(npu37xx::nn_public::VpuNNShaveRuntimeConfigs::stack_frames[0]);
        auto actRtConfigsShaveStacksIdx =
                getSectionIndexBasedOnRelocAOffset(actRtConfigsShaveStacksOffest, _mappedInferSectionIdx);

        _log.debug("actRtConfigsShaveStacksIdx {0}", actRtConfigsShaveStacksIdx);
        if (actRtConfigsShaveStacksIdx == INVALID_SEC_IDX) {
            continue;
        }

        const auto& section = _elfReader.getSection(actRtConfigsShaveStacksIdx);
        const auto sectionHeader = section.getHeader();
        int64_t stackSize = sectionHeader->sh_size;
        const auto bufferMemrefShape = SmallVector<int64_t>{stackSize};
        auto nameAttr = mlir::FlatSymbolRefAttr::get(_ctx, stringifyEnum(VPU::MemoryKind::DDR));
        const auto symbolAttr = vpux::IndexedSymbolAttr::get(_ctx, nameAttr);
        unsigned int perm[1] = {0};
        auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), _ctx);
        auto memrefType = mlir::MemRefType::get(
                bufferMemrefShape, mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::Unsigned), map, symbolAttr);
        auto loc = mlir::UnknownLoc::get(_ctx);

        auto bufferOpValue =
                opsBuilder.create<VPURT::DeclareBufferOp>(loc, memrefType, vpux::VPURT::BufferSection::DDR, 0);

        createLogicalSectionOp(opsBuilder, actRtConfigsShaveStacksIdx, bufferOpValue.getResult());
        _shaveStacks.push_back(bufferOpValue.getResult());
    }
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForInvariants(mlir::OpBuilder& opsBuilder,
                                                                 const size_t noOfInvariantsTasks) {
    auto invariantsOffest =
            offsetof(npu37xx::nn_public::VpuMappedInference, invariants) +
            offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuDPUInvariant>, address);
    auto invariantsSectionIdx = getSectionIndexBasedOnRelocAOffset(invariantsOffest, _mappedInferSectionIdx);

    _log.debug("invariantsSectionIdx {0}", invariantsSectionIdx);
    if (invariantsSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    std::vector<mlir::Value> invariantsOps;
    for (size_t idx = 0; idx < noOfInvariantsTasks; ++idx) {
        auto invariantIndex = VPURegMapped::IndexType::get(_ctx, idx);
        VPUX_UNUSED(invariantIndex);
        // TBI - create <VPUMI37XX::DPUInvariantOp>
    }
    createSectionOp(opsBuilder, invariantsSectionIdx, invariantsOps);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForVariants(mlir::OpBuilder& opsBuilder,
                                                               const size_t noOfVariantsTasks) {
    auto variantsOffest = offsetof(npu37xx::nn_public::VpuMappedInference, variants) +
                          offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuDPUVariant>, address);
    auto variantsSectionIdx = getSectionIndexBasedOnRelocAOffset(variantsOffest, _mappedInferSectionIdx);

    _log.debug("variantsSectionIdx {0}", variantsSectionIdx);
    if (variantsSectionIdx == INVALID_SEC_IDX) {
        return;
    }

    const auto& section = _elfReader.getSection(variantsSectionIdx);
    const auto* dpuVariant = section.getData<npu37xx::nn_public::VpuDPUVariant>();
    VPUX_THROW_UNLESS(dpuVariant != nullptr, "Got null dpuVariant");

    std::vector<mlir::Value> variantsOps;
    for (size_t idx = 0; idx < noOfVariantsTasks; ++idx) {
        auto variantIndex = VPURegMapped::IndexType::get(_ctx, idx);
        VPUX_UNUSED(variantIndex);
        // TBI - create <VPUMI37XX::DPUVariantOp>
    }
    createSectionOp(opsBuilder, variantsSectionIdx, variantsOps);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForMappedInferece(mlir::func::FuncOp& func,
                                                                     mlir::OpBuilder& opsBuilder) {
    _mappedInferSectionIdx = getMappedInferenceSectionIndex();
    _log.debug("mappedInference section index {0}", _mappedInferSectionIdx);
    VPUX_THROW_UNLESS(_mappedInferSectionIdx != INVALID_SEC_IDX, "Got invalid mappedInference section index");

    const auto& section = _elfReader.getSection(_mappedInferSectionIdx);
    const auto* mappedInference = section.getData<npu37xx::nn_public::VpuMappedInference>();
    VPUX_THROW_UNLESS(mappedInference != nullptr, "Got null mappedInference");

    llvm::SmallVector<mlir::Value> dmaListHeads;
    llvm::SmallVector<int64_t, npu37xx::nn_public::VPU_MAX_DMA_ENGINES> dmaCountVec;
    mlir::Value invariantList;
    mlir::Value variantList;
    mlir::Value actKernelInvocations;
    mlir::Value actKernelRanges;
    mlir::Value barrierList;
    mlir::Value actRtText;

    _log.debug("mappedInference->dma_tasks[0].count {0}", mappedInference->dma_tasks[0].count);
    _log.debug("mappedInference->barrier_configs.count {0}", mappedInference->barrier_configs.count);
    _log.debug("mappedInference->act_kernel_ranges.count {0}", mappedInference->act_kernel_ranges.count);
    _log.debug("mappedInference->act_kernel_invocations.count {0}", mappedInference->act_kernel_invocations.count);

    createConfigureBarrierOp(opsBuilder, mappedInference->barrier_configs.count);
    createSectionOpForDMA(func, opsBuilder, mappedInference);
    createSectionOpForActKernalRange(opsBuilder, mappedInference->act_kernel_ranges.count);
    createSectionOpForActKernelInvocation(opsBuilder, mappedInference->act_kernel_invocations.count);
    createSectionOpForActShaveStacks(opsBuilder);
    createSectionOpForShaveRtConfigs(opsBuilder, mappedInference->shv_rt_configs.use_schedule_embedded_rt, actRtText);
    createSectionOpForInvariants(opsBuilder, mappedInference->invariants.count);
    createSectionOpForVariants(opsBuilder, mappedInference->variants.count);

    for (uint8_t dmaEnginesIdx = 0; dmaEnginesIdx < npu37xx::nn_public::VPU_MAX_DMA_ENGINES; ++dmaEnginesIdx) {
        if (mappedInference->dma_tasks[dmaEnginesIdx].count >= 1 && !_nndmaOps[dmaEnginesIdx].empty()) {
            dmaListHeads.push_back(_nndmaOps[dmaEnginesIdx].front());
            dmaCountVec.push_back(static_cast<int64_t>(mappedInference->dma_tasks[dmaEnginesIdx].count));
        }
    }

    if (mappedInference->barrier_configs.count >= 1 && !_barrierConfigsByRealId.empty()) {
        barrierList = _barrierConfigsByRealId.front().second;
    }
    if (mappedInference->act_kernel_ranges.count >= 1 && !_actKRangeOps.empty()) {
        actKernelRanges = _actKRangeOps.front();
    }
    if (mappedInference->act_kernel_invocations.count >= 1 && !_actKInvocationOps.empty()) {
        actKernelInvocations = _actKInvocationOps.front();
    }

    auto mappedInferenceOp = opsBuilder.create<VPUMI37XX::MappedInferenceOp>(
            mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, 0), mlir::ValueRange{dmaListHeads},
            invariantList, variantList, actKernelRanges, actKernelInvocations, barrierList, actRtText,
            mlir::ValueRange{_shaveStacks}, opsBuilder.getI64ArrayAttr(dmaCountVec), mappedInference->invariants.count,
            mappedInference->variants.count, mappedInference->act_kernel_ranges.count,
            mappedInference->act_kernel_invocations.count, mappedInference->barrier_configs.count);

    createSectionOp(opsBuilder, _mappedInferSectionIdx, mappedInferenceOp.getResult());
}

void vpux::ELFNPU37XX::ElfImporter::fillValueForWaitAndUpdateBarrierConfigs(const uint64_t& prodMask,
                                                                            const uint64_t& consMask,
                                                                            mlir::ValueRange& updateBarriers,
                                                                            mlir::ValueRange& waitBarriers) {
    for (auto& barrier : _barrierConfigsByRealId) {
        const auto barrierRealId = ((uint64_t)1) << barrier.first;
        if (barrierRealId & prodMask) {
            _log.info("barrier to be updated {0} ", barrier.second);
            updateBarriers = barrier.second;
        }
        if (barrierRealId & consMask) {
            _log.info("barrier to wait for {0} ", barrier.second);
            waitBarriers = barrier.second;
        }
    }
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOpForDMA(
        mlir::func::FuncOp& func, mlir::OpBuilder& opsBuilder,
        const npu37xx::nn_public::VpuMappedInference* mappedInference) {
    for (size_t dmaTaskIdx = 0; dmaTaskIdx < npu37xx::nn_public::VPU_MAX_DMA_ENGINES; dmaTaskIdx++) {
        auto dmaOffest = offsetof(npu37xx::nn_public::VpuMappedInference, dma_tasks) +
                         dmaTaskIdx * sizeof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuDMATask>) +
                         offsetof(npu37xx::nn_public::VpuTaskReference<npu37xx::nn_public::VpuDMATask>, address);
        auto dmaSectionIdx = getSectionIndexBasedOnRelocAOffset(dmaOffest, _mappedInferSectionIdx);

        _log.debug("dmaSectionIdx {0}", dmaSectionIdx);
        if (dmaSectionIdx == INVALID_SEC_IDX) {
            continue;
        }

        const auto& section = _elfReader.getSection(dmaSectionIdx);
        const auto* dmaTasks = section.getData<npu37xx::nn_public::VpuDMATask>();
        VPUX_THROW_UNLESS(dmaTasks != nullptr, "Got null dmaTasks");

        size_t dmaSectionOffsetTracker = 0;
        mlir::Type type64UAttr = mlir::IntegerType::get(_ctx, 64, mlir::IntegerType::Unsigned);
        mlir::Type type64SAttr = mlir::IntegerType::get(_ctx, 64, mlir::IntegerType::Signed);
        std::optional<VPUMI37XX::NNDMAOp> previousDMA;
        std::optional<mlir::OpBuilder::InsertPoint> afterDMAInsertPoint;
        for (size_t idx = 0; idx < mappedInference->dma_tasks[dmaTaskIdx].count; ++idx) {
            auto compression = vpux::VPUIP::DMAAccModeAttr::get(_ctx, dmaTasks->transaction_.cfg_link.cfg_bits.dec_en
                                                                              ? vpux::VPUIP::DMAAccMode::DECOMPRESSION
                                                                              : vpux::VPUIP::DMAAccMode::DISABLE);
            mlir::IntegerAttr port = mlir::IntegerAttr::get(type64SAttr, 0);
            mlir::IntegerAttr startAfter = mlir::IntegerAttr::get(type64UAttr, dmaTasks->barriers_sched_.start_after_);
            mlir::IntegerAttr cleanAfter = mlir::IntegerAttr::get(type64UAttr, dmaTasks->barriers_sched_.clean_after_);
            const auto isCritical =
                    dmaTasks->transaction_.cfg_link.cfg_bits.critical ? mlir::UnitAttr::get(_ctx) : nullptr;
            const auto isOutOfOrder =
                    dmaTasks->transaction_.cfg_link.cfg_bits.order_forced ? mlir::UnitAttr::get(_ctx) : nullptr;
            const auto dmaDescriptor =
                    VPUIP::DMADescriptorAttr::get(_ctx, vpux::getIntAttr(_ctx, dmaTasks->transaction_.num_planes),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.length),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.attr2d.src_width),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.attr2d.src_stride),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.src_plane_stride),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.attr2d.dst_width),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.attr2d.dst_stride),
                                                  vpux::getIntAttr(_ctx, dmaTasks->transaction_.dst_plane_stride));

            mlir::ValueRange waitBarriers, updateBarriers;
            const auto prodMask = dmaTasks->transaction_.cfg_link.cfg_bits.type
                                          ? dmaTasks->transaction_.barriers.prod_mask
                                          : dmaTasks->transaction_.barriers1d.prod_mask;
            const auto consMask = dmaTasks->transaction_.cfg_link.cfg_bits.type
                                          ? dmaTasks->transaction_.barriers.cons_mask
                                          : dmaTasks->transaction_.barriers1d.cons_mask;
            fillValueForWaitAndUpdateBarrierConfigs(prodMask, consMask, updateBarriers, waitBarriers);

            auto transactionSrcOffset = dmaSectionOffsetTracker +
                                        offsetof(npu37xx::nn_public::VpuDMATask, transaction_) +
                                        offsetof(npu37xx::vpu_dma_descriptor_t, src);
            auto transactionDstOffset = dmaSectionOffsetTracker +
                                        offsetof(npu37xx::nn_public::VpuDMATask, transaction_) +
                                        offsetof(npu37xx::vpu_dma_descriptor_t, dst);

            mlir::Value inputBuf = getInputOrOutputValueForDmaTask(
                    opsBuilder, dmaSectionIdx, transactionSrcOffset, dmaTasks->transaction_.attr2d.src_width,
                    elf::VPU_SHF_USERINPUT, func.getArgument(dmaTaskIdx));
            mlir::Value outputBuf = getInputOrOutputValueForDmaTask(
                    opsBuilder, dmaSectionIdx, transactionDstOffset, dmaTasks->transaction_.attr2d.dst_width,
                    elf::VPU_SHF_USEROUTPUT, func.getArgument(dmaTaskIdx + 1));

            auto nndmaOp = opsBuilder.create<VPUMI37XX::NNDMAOp>(
                    mlir::UnknownLoc::get(_ctx), VPURegMapped::IndexType::get(_ctx, idx), /*taskLocation*/ nullptr,
                    inputBuf, outputBuf, nullptr, waitBarriers, updateBarriers, startAfter, cleanAfter, isOutOfOrder,
                    isCritical, port, compression, dmaDescriptor);
            if (!afterDMAInsertPoint.has_value()) {
                afterDMAInsertPoint = opsBuilder.saveInsertionPoint();
            }
            opsBuilder.setInsertionPoint(nndmaOp);

            if (previousDMA.has_value()) {
                previousDMA.value().getNextDMAIdxMutable().assign(nndmaOp.getIndex());
            }
            previousDMA = nndmaOp;

            _nndmaOps[dmaTaskIdx].push_back(nndmaOp.getIndex());

            dmaSectionOffsetTracker += sizeof(npu37xx::nn_public::VpuDMATask);
            dmaTasks++;
        }

        if (afterDMAInsertPoint.has_value()) {
            opsBuilder.restoreInsertionPoint(afterDMAInsertPoint.value());
        }
        createSectionOp(opsBuilder, dmaSectionIdx, _nndmaOps[dmaTaskIdx]);
    }
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                                                    const mlir::Value& inputArg) {
    std::vector<mlir::Value> inputArgs{inputArg};
    createSectionOp(opsBuilder, sectionIdx, inputArgs);
}

void vpux::ELFNPU37XX::ElfImporter::createSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                                                    const std::vector<mlir::Value>& inputArgs) {
    const auto& section = _elfReader.getSection(sectionIdx);
    const auto sectionHeader = section.getHeader();
    auto elfCreateSectionOp = opsBuilder.create<ELFNPU37XX::CreateSectionOp>(
            mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx), mlir::StringRef(section.getName()),
            symbolizeSectionTypeAttr(sectionHeader->sh_type).value_or(vpux::ELFNPU37XX::SectionTypeAttr::SHT_NULL),
            symbolizeSectionFlagsAttr(sectionHeader->sh_flags).value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),
            static_cast<int64_t>(sectionHeader->sh_info), static_cast<int64_t>(sectionHeader->sh_addralign));

    mlir::Region& region = elfCreateSectionOp.getOperation()->getRegion(0);
    mlir::Block* block = new mlir::Block();
    region.push_back(block);
    mlir::OpBuilder builderElfSectionOpReg(block, block->begin());

    for (const auto& inputArg : inputArgs) {
        builderElfSectionOpReg.create<ELFNPU37XX::PutOpInSectionOp>(builderElfSectionOpReg.getUnknownLoc(), inputArg);
    }

    _sectionOpByValue[sectionIdx] = elfCreateSectionOp.getResult();
}

void vpux::ELFNPU37XX::ElfImporter::createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                                                           const mlir::Value& inputArg) {
    std::vector<mlir::Value> inputArgs{inputArg};
    createLogicalSectionOp(opsBuilder, sectionIdx, inputArgs);
}

void vpux::ELFNPU37XX::ElfImporter::createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                                                           const std::vector<mlir::Value>& inputArgs) {
    const auto& section = _elfReader.getSection(sectionIdx);
    const auto sectionHeader = section.getHeader();
    auto elfLogicalCreateSectionOp = opsBuilder.create<ELFNPU37XX::CreateLogicalSectionOp>(
            mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx), mlir::StringRef(section.getName()),
            symbolizeSectionTypeAttr(sectionHeader->sh_type).value_or(vpux::ELFNPU37XX::SectionTypeAttr::SHT_NULL),
            symbolizeSectionFlagsAttr(sectionHeader->sh_flags).value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),
            static_cast<int64_t>(sectionHeader->sh_info), static_cast<int64_t>(sectionHeader->sh_addralign));

    mlir::Region& region = elfLogicalCreateSectionOp.getOperation()->getRegion(0);
    mlir::Block* block = new mlir::Block();
    region.push_back(block);
    mlir::OpBuilder builderElfSectionOpReg(block, block->begin());

    for (const auto& inputArg : inputArgs) {
        builderElfSectionOpReg.create<ELFNPU37XX::PutOpInSectionOp>(builderElfSectionOpReg.getUnknownLoc(), inputArg);
    }

    _sectionOpByValue[sectionIdx] = elfLogicalCreateSectionOp.getResult();
}

void vpux::ELFNPU37XX::ElfImporter::createGenericBuiltInRegion(mlir::OpBuilder& opsBuilder) {
    if (_sectionOpByValue.find(elf::VPU_RT_SYMTAB) != _sectionOpByValue.end()) {
        _log.debug("generic arith const section already created");
        return;
    }

    std::vector<std::pair<unsigned int, ELFNPU37XX::SymbolOp>> elfSymbolsConstOp;

    for (uint8_t val = 0; val <= vpux::ELFNPU37XX::getMaxEnumValForCMXMappingSymbol(); ++val) {
        auto arithOperation = opsBuilder.create<mlir::arith::ConstantIntOp>(mlir::UnknownLoc::get(_ctx), val, 8);
        const auto specialSym = symbolizeCMXMappingSymbol(val).value_or(
                vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR);
        const auto specialSymAttr = mlir::StringAttr::get(_ctx, stringifyCMXMappingSymbol(specialSym));
        auto symbolValue = opsBuilder.create<ELFNPU37XX::SymbolOp>(
                mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SymbolType::get(_ctx), arithOperation.getResult(),
                mlir::UnitAttr::get(_ctx), specialSymAttr, vpux::ELFNPU37XX::SymbolTypeEnumAttr{}, mlir::IntegerAttr{},
                mlir::IntegerAttr{});
        elfSymbolsConstOp.push_back(std::make_pair(val, symbolValue));
    }

    _symbolsOpByValue[elf::VPU_RT_SYMTAB] = elfSymbolsConstOp;

    constexpr bool isBuiltin = true;
    auto elfSymTabSectionOp = opsBuilder.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx), mlir::StringRef("VPU_RT_SYMTAB"),
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE, isBuiltin);

    _sectionOpByValue[elf::VPU_RT_SYMTAB] = elfSymTabSectionOp.getResult();

    mlir::Region& regionSymTabSec = elfSymTabSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkSym = new mlir::Block();
    regionSymTabSec.push_back(blkSym);
    mlir::OpBuilder builderSymTabSec(blkSym, blkSym->begin());

    for (auto& elfSymbolOp : elfSymbolsConstOp) {
        builderSymTabSec.create<ELFNPU37XX::PutOpInSectionOp>(builderSymTabSec.getUnknownLoc(),
                                                              elfSymbolOp.second.getResult());
    }
}

VPURT::DeclareBufferOp vpux::ELFNPU37XX::ElfImporter::createDeclareBufferOp(mlir::OpBuilder& opsBuilder,
                                                                            const int64_t& bufferSize,
                                                                            const bool isTypeDDR,
                                                                            const int64_t& byteOffset) {
    for (auto& buffer : _buffers) {
        mlir::Value bufferVal = buffer.second.getResult();
        auto type = mlir::cast<vpux::NDTypeInterface>(bufferVal.getType());
        auto bufSize = static_cast<int64_t>(type.getCompactAllocSize().count());
        if (bufSize == bufferSize && buffer.first == isTypeDDR && buffer.second.getByteOffset() == byteOffset) {
            return buffer.second;
        }
    }

    const auto bufferMemrefShape = SmallVector<int64_t>{bufferSize};
    auto memoryKind = isTypeDDR ? VPU::MemoryKind::DDR : VPU::MemoryKind::CMX_NN;
    auto nameAttr = mlir::FlatSymbolRefAttr::get(_ctx, stringifyEnum(memoryKind));
    const auto symbolAttr = isTypeDDR ? vpux::IndexedSymbolAttr::get(_ctx, nameAttr)
                                      : vpux::IndexedSymbolAttr::get(_ctx, {nameAttr, vpux::getIntAttr(_ctx, 0)});

    unsigned int perm[1] = {0};
    auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), _ctx);
    auto memrefType = mlir::MemRefType::get(
            bufferMemrefShape, mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::Unsigned), map, symbolAttr);
    auto loc = mlir::UnknownLoc::get(_ctx);

    VPURT::DeclareBufferOp bufferOpValue;
    if (!isTypeDDR) {
        bufferOpValue =
                opsBuilder.create<VPURT::DeclareBufferOp>(loc, memrefType, VPURT::BufferSection::CMX_NN, 0, byteOffset);
    } else {
        bufferOpValue =
                opsBuilder.create<VPURT::DeclareBufferOp>(loc, memrefType, vpux::VPURT::BufferSection::DDR, byteOffset);
    }

    _buffers.push_back(std::make_pair(isTypeDDR, bufferOpValue));

    return bufferOpValue;
}

mlir::Value vpux::ELFNPU37XX::ElfImporter::getSymbolValueBySecHeaderAndSymbolIdx(const size_t secHeaderIdx,
                                                                                 const size_t symbolIndex) {
    mlir::Value symbolForRelocation = nullptr;

    if (_symbolsOpByValue.find(secHeaderIdx) == _symbolsOpByValue.end()) {
        _log.error("sectionHeader index not valid : {0}\n", secHeaderIdx);
        return symbolForRelocation;
    }

    for (auto& symbol : _symbolsOpByValue[secHeaderIdx]) {
        if (symbol.first == symbolIndex) {
            symbolForRelocation = symbol.second.getResult();
            _log.info("symbol Index is : {0}, symbolForRelocation is : {1}\n", symbolIndex, symbolForRelocation);
            break;
        }
    }

    return symbolForRelocation;
}

void vpux::ELFNPU37XX::ElfImporter::buildMainFunc() {
    OpBuilderLogger builderLog(_log.nest());
    auto builder = mlir::OpBuilder::atBlockEnd(_module.getBody(), &builderLog);

    auto funcArguments = _inputTypes;
    funcArguments.insert(funcArguments.end(), _outputTypes.begin(), _outputTypes.end());
    const auto funcType = mlir::FunctionType::get(_ctx, ArrayRef(funcArguments), ArrayRef(_outputTypes));
    auto func = builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(_ctx), _mainFuncName.getValue(), funcType);

    auto opsBuilder = mlir::OpBuilder::atBlockBegin(func.addEntryBlock(), &builderLog);

    createSectionOpForMappedInferece(func, opsBuilder);

    for (size_t sectionCtr = 0; sectionCtr < _noOfSections; ++sectionCtr) {
        const auto& section = _elfReader.getSection(sectionCtr);
        const auto sectionHeader = section.getHeader();

        _log.debug("section no {0} ", sectionCtr);

        if (elf::SHT_PROGBITS == sectionHeader->sh_type) {
            if (_sectionOpByValue.find(sectionCtr) != _sectionOpByValue.end()) {
                _log.debug("SHT_PROGBITS section {0} handled via mappedInferece", sectionCtr);
                continue;
            }

            std::vector<mlir::Value> inputArgs = {};
            createSectionOp(opsBuilder, sectionCtr, inputArgs);
        } else if (elf::SHT_NULL == sectionHeader->sh_type) {
            _log.debug("SHT_NULL section");
        } else if (elf::SHT_NOBITS == sectionHeader->sh_type) {
            if (_sectionOpByValue.find(sectionCtr) != _sectionOpByValue.end()) {
                _log.debug("SHT_NOBITS section {0} already created", sectionCtr);
                continue;
            }

            std::vector<mlir::Value> ddrBuffers;
            for (auto& buffer : _buffers) {
                if (buffer.first) {
                    ddrBuffers.push_back(buffer.second);
                }
            }
            createLogicalSectionOp(opsBuilder, sectionCtr, ddrBuffers);
        } else if (elf::SHT_SYMTAB == sectionHeader->sh_type) {
            auto elfSymbolsOp = createSymbolOp(func, opsBuilder, section);
            _symbolsOpByValue[sectionCtr] = elfSymbolsOp;

            constexpr bool isBuiltin = false;
            auto elfSymTabSectionOp = opsBuilder.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
                    mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx),
                    mlir::StringRef(section.getName()),
                    symbolizeSectionFlagsAttr(sectionHeader->sh_flags)
                            .value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),
                    isBuiltin);
            _sectionOpByValue[sectionCtr] = elfSymTabSectionOp.getResult();

            mlir::Region& regionSymTabSec = elfSymTabSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkSym = new mlir::Block();
            regionSymTabSec.push_back(blkSym);
            mlir::OpBuilder builderSymTabSec(blkSym, blkSym->begin());

            for (auto& elfSymbolOp : elfSymbolsOp) {
                builderSymTabSec.create<ELFNPU37XX::PutOpInSectionOp>(builderSymTabSec.getUnknownLoc(),
                                                                      elfSymbolOp.second.getResult());
            }
        } else if (elf::SHT_RELA == sectionHeader->sh_type) {
            createGenericBuiltInRegion(opsBuilder);

            if (_sectionOpByValue.find(sectionHeader->sh_link) == _sectionOpByValue.end() ||
                _sectionOpByValue.find(sectionHeader->sh_info) == _sectionOpByValue.end()) {
                _log.debug("continue, doing nothing for section with index {0} ", sectionCtr);
                continue;
            }

            auto elfRelocateSectionOp = opsBuilder.create<ELFNPU37XX::CreateRelocationSectionOp>(
                    mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx),
                    mlir::StringRef(section.getName()), _sectionOpByValue[sectionHeader->sh_link],
                    _sectionOpByValue[sectionHeader->sh_info],
                    symbolizeSectionFlagsAttr(sectionHeader->sh_flags)
                            .value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE));
            _sectionOpByValue[sectionCtr] = elfRelocateSectionOp.getResult();

            mlir::Region& regionSec = elfRelocateSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkSym = new mlir::Block();
            regionSec.push_back(blkSym);
            mlir::OpBuilder builderSec(blkSym, blkSym->begin());

            const auto* entries = section.getData<elf::RelocationAEntry>();
            const auto entriesNum = section.getEntriesNum();

            for (size_t idx = 0; idx < entriesNum; ++idx) {
                const auto rType = vpux::ELFNPU37XX::symbolizeRelocationType(elf::elf64RType(entries[idx].r_info));
                const auto relocationType = rType.value_or(vpux::ELFNPU37XX::RelocationType::R_VPU_64);
                const auto symbolIdx = elf::elf64RSym(entries[idx].r_info);
                auto symbolForRelocation = getSymbolValueBySecHeaderAndSymbolIdx(sectionHeader->sh_link, symbolIdx);
                if (symbolForRelocation == nullptr) {
                    continue;
                }

                builderSec.create<ELFNPU37XX::RelocOp>(mlir::UnknownLoc::get(_ctx), nullptr, entries[idx].r_offset,
                                                       relocationType, symbolForRelocation, entries[idx].r_addend,
                                                       "Main func entry (at idx " + std::to_string(idx) + ") reloc");
            }
        } else if (elf::VPU_SHT_NETDESC == sectionHeader->sh_type) {
            auto elfMetadataSectionOp = opsBuilder.create<ELFNPU37XX::CreateMetadataSectionOp>(
                    mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx),
                    mlir::StringRef(section.getName()),
                    symbolizeSectionFlagsAttr(sectionHeader->sh_flags)
                            .value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),
                    static_cast<int64_t>(sectionHeader->sh_info), static_cast<int64_t>(sectionHeader->sh_addralign));

            mlir::Region& regionSymTabSec = elfMetadataSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkSym = new mlir::Block();
            regionSymTabSec.push_back(blkSym);
            mlir::OpBuilder builderSymTabSec(blkSym, blkSym->begin());

            builderSymTabSec.create<VPUMI37XX::NetworkMetadataOp>(mlir::UnknownLoc::get(_ctx),
                                                                  VPURegMapped::IndexType::get(_ctx, 0));

            _sectionOpByValue[sectionCtr] = elfMetadataSectionOp.getResult();
        } else if (elf::VPU_SHT_PROF == sectionHeader->sh_type) {
            auto elfProfSectionOp = opsBuilder.create<ELFNPU37XX::CreateProfilingSectionOp>(
                    mlir::UnknownLoc::get(_ctx), vpux::ELFNPU37XX::SectionType::get(_ctx),
                    mlir::StringRef(section.getName()),
                    symbolizeSectionFlagsAttr(sectionHeader->sh_flags)
                            .value_or(vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),
                    static_cast<int64_t>(sectionHeader->sh_info), static_cast<int64_t>(sectionHeader->sh_addralign));

            mlir::Region& regionSymTabSec = elfProfSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkSym = new mlir::Block();
            regionSymTabSec.push_back(blkSym);
            mlir::OpBuilder builderSymTabSec(blkSym, blkSym->begin());

            const char* profMetaSection = section.getData<char>();
            const size_t metaSize = section.getEntriesNum();
            const llvm::ArrayRef<char> rawMetadata{profMetaSection, metaSize};

            auto uint8Type = mlir::IntegerType::get(_ctx, 8, mlir::IntegerType::SignednessSemantics::Unsigned);
            auto vectorType = mlir::VectorType::get({checked_cast<long int>(metaSize)}, uint8Type);
            const auto elemAttr = mlir::DenseElementsAttr::getFromRawBuffer(vectorType, rawMetadata);

            builderSymTabSec.create<VPUMI37XX::ProfilingMetadataOp>(mlir::UnknownLoc::get(_ctx),
                                                                    VPURegMapped::IndexType::get(_ctx, 0), elemAttr);

            _sectionOpByValue[sectionCtr] = elfProfSectionOp.getResult();
        } else {
            auto secHeaderTypeAttr = symbolizeSectionTypeAttr(sectionHeader->sh_type)
                                             .value_or(vpux::ELFNPU37XX::SectionTypeAttr::SHT_NULL);
            auto sectionHeaderTypeStr = stringifySectionTypeAttr(secHeaderTypeAttr).str();
            _log.debug("unsupported section header type {0} ", sectionHeaderTypeStr);
        }
    }

    const auto functionOutArguments = mlir::ValueRange{func.getArguments().begin() + _inputTypes.size(),
                                                       static_cast<ptrdiff_t>(_outputTypes.size())};
    opsBuilder.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(_ctx), functionOutArguments);
}

mlir::OwningOpRef<mlir::ModuleOp> vpux::ELFNPU37XX::ElfImporter::read() {
    const auto* header = _elfReader.getHeader();
    VPUX_THROW_UNLESS(header != nullptr, "Got NULL header");

    const auto moduleName = "Test";
    _module = mlir::ModuleOp::create(mlir::UnknownLoc::get(_ctx), StringRef(moduleName));

    // buildRunTimeResourcesOp(); - not supported at the moment
    buildNetworkInfoOp();
    buildMainFunc();

    return _module;
}
