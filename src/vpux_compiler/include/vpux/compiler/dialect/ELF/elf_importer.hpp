//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/nn_public/vpu_nnrt_api.h"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/preprocessing.hpp"
#include "vpux_compiler.hpp"

#include <flatbuffers/flatbuffers.h>

#include <vpux_elf/reader.hpp>
#include <vpux_elf/types/section_header.hpp>

#include <vpux_loader/vpux_loader.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <map>

namespace vpux {
namespace ELF {

class ImporterBufferManager : public elf::BufferManager {
public:
    elf::DeviceBuffer allocate(const elf::BufferSpecs& buffSpecs) override;
    void deallocate(elf::DeviceBuffer& devBuffer) override;
    void lock(elf::DeviceBuffer&) override;
    void unlock(elf::DeviceBuffer&) override;
    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) override;

    ~ImporterBufferManager();

private:
    std::vector<elf::DeviceBuffer> m_allocatedZones;
};

class ElfImporter final {
public:
    ElfImporter(mlir::MLIRContext* ctx, const std::string& elfFileName, Logger log);
    ElfImporter(const ElfImporter&) = delete;
    ElfImporter(ElfImporter&&) = delete;
    mlir::OwningOpRef<mlir::ModuleOp> read();
    ~ElfImporter();
    ElfImporter& operator=(const ElfImporter&) = delete;
    ElfImporter& operator=(ElfImporter&&) = delete;

private:
    void buildCNNNetworkOp();
    void parseUserInputsOutputs(OpBuilderLogger& builderLog, IE::CNNNetworkOp& cnnOp);
    void buildMainFunc();

    void createSectionOp(mlir::OpBuilder& opsBuilder, const uint32_t sectionIdx, const mlir::Value& inputArg);
    void createSectionOp(mlir::OpBuilder& opsBuilder, const uint32_t sectionIdx,
                         const std::vector<mlir::Value>& inputArgs);
    void createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const uint32_t sectionIdx, const mlir::Value& inputArgs);
    void createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const uint32_t sectionIdx,
                                const std::vector<mlir::Value>& inputArgs);
    void createConfigureBarrierOp(mlir::OpBuilder& opsBuilder, const uint32_t noOfBarrierConfigs);
    void createSectionOpForMappedInferece(mlir::FuncOp& func, mlir::OpBuilder& opsBuilder);
    void createSectionOpForDMA(mlir::FuncOp& func, mlir::OpBuilder& opsBuilder,
                               const nn_public::VpuMappedInference* mappedInference);
    void createSectionOpForActKernalRange(mlir::OpBuilder& opsBuilder, const uint32_t noOfActKRangeTasks);
    void createSectionOpForActKernelInvocation(mlir::OpBuilder& opsBuilder, const uint32_t noOfActKInvocationTasks);
    void createSectionOpForActKernelParams(mlir::OpBuilder& opsBuilder, const uint32_t actKInvocationSectionIdx);
    void createSectionOpForActKernelText(mlir::OpBuilder& opsBuilder, const std::vector<mlir::Value>& actKTextOps,
                                         const uint32_t actKRangeSectionIdx);
    void createSectionOpForActKernelData(mlir::OpBuilder& opsBuilder, const std::vector<mlir::Value>& actKDataOps);
    void createSectionOpForShaveRtConfigs(mlir::OpBuilder& opsBuilder, const bool isScheduleEmbeddedRtUsed,
                                          mlir::Value& actRtTextOpValue);
    void createSectionOpForActShaveStacks(mlir::OpBuilder& opsBuilder);
    void createSectionOpForInvariants(mlir::OpBuilder& opsBuilder, const uint32_t noOfInvariantsTasks);
    void createSectionOpForVariants(mlir::OpBuilder& opsBuilder, const uint32_t noOfVariantsTasks);
    void createGenericBuiltInRegion(mlir::OpBuilder& opsBuilder);
    std::vector<std::pair<unsigned int, ELF::SymbolOp>> createSymbolOp(
            mlir::FuncOp& func, mlir::OpBuilder& opsBuilder,
            const elf::Reader<elf::ELF_Bitness::Elf64>::Section& section);
    VPURT::DeclareBufferOp createDeclareBufferOp(mlir::OpBuilder& opsBuilder, const int64_t& bufferSize,
                                                 const bool isTypeDDR, const int64_t& byteOffset);
    mlir::Value getSymbolValueBySecHeaderAndSymbolIdx(const uint32_t secHeaderIdx, const uint32_t symbolIndex);
    void fillValueForWaitAndUpdateBarrierConfigs(const uint64_t& prodMask, const uint64_t& consMask,
                                                 mlir::ValueRange& updateBarriers, mlir::ValueRange& waitBarriers);
    uint32_t getMappedInferenceSectionIndex();
    uint32_t getSectionIndexBasedOnRelocAOffset(const uint32_t offset, const uint32_t shInfo);
    mlir::Value getInputOrOutputValueForDmaTask(mlir::OpBuilder& opsBuilder, const uint32_t dmaSectionIdx,
                                                const uint64_t& offset, const uint32_t bufferSize,
                                                const elf::Elf_Xword& flag, const mlir::BlockArgument& funcArg);

    elf::BufferManager* _importerBufMgr = nullptr;
    elf::AccessManager* _accessor = nullptr;
    elf::Reader<elf::ELF_Bitness::Elf64> _elfReader;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
    mlir::FlatSymbolRefAttr _mainFuncName;
    size_t _noOfSections;
    Logger _log;

    SmallVector<mlir::Type> _inputTypes;
    SmallVector<mlir::Type> _outputTypes;
    uint32_t _mappedInferSectionIdx = 0;

    std::map<size_t, mlir::Value> _sectionOpByValue;
    std::map<size_t, std::vector<std::pair<unsigned int, ELF::SymbolOp>>> _symbolsOpByValue;
    std::vector<std::pair<uint32_t, mlir::Value>> _barrierConfigsByRealId;
    std::map<size_t, std::vector<mlir::Value>> _nndmaOps;
    std::vector<VPUIPRegMapped::ActKernelRangeOp> _actKRangeOps;
    std::vector<VPUIPRegMapped::ActKernelInvocationOp> _actKInvocationOps;
    std::vector<std::pair<bool, VPURT::DeclareBufferOp>> _buffers;
    llvm::SmallVector<mlir::Value> _shaveStacks;
};

}  // namespace ELF
}  // namespace vpux
