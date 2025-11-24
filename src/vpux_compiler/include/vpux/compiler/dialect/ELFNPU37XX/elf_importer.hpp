//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <npu_37xx_nnrt.hpp>

#include <vpux_elf/reader.hpp>
#include <vpux_elf/types/section_header.hpp>

#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_headers/buffer_specs.hpp>
#include <vpux_headers/device_buffer.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

#include <map>
#include <vector>

namespace vpux {
namespace ELFNPU37XX {

class ImporterBufferManager : public elf::BufferManager {
public:
    elf::DeviceBuffer allocate(const elf::BufferSpecs& buffSpecs) override;
    void deallocate(elf::DeviceBuffer& devBuffer) override;
    void lock(elf::DeviceBuffer&) override;
    void unlock(elf::DeviceBuffer&) override;
    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) override;

    ~ImporterBufferManager() override;

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
    void buildNetworkInfoOp();
    void parseUserInputsOutputs(OpBuilderLogger& builderLog, net::NetworkInfoOp& cnnOp);
    void buildMainFunc();

    void createSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx, const mlir::Value& inputArg);
    void createSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                         const std::vector<mlir::Value>& inputArgs);
    void createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx, const mlir::Value& inputArgs);
    void createLogicalSectionOp(mlir::OpBuilder& opsBuilder, const size_t sectionIdx,
                                const std::vector<mlir::Value>& inputArgs);
    void createConfigureBarrierOp(mlir::OpBuilder& opsBuilder, const size_t noOfBarrierConfigs);
    void createSectionOpForMappedInferece(mlir::func::FuncOp& func, mlir::OpBuilder& opsBuilder);
    void createSectionOpForDMA(mlir::func::FuncOp& func, mlir::OpBuilder& opsBuilder,
                               const npu37xx::nn_public::VpuMappedInference* mappedInference);
    void createSectionOpForActKernalRange(mlir::OpBuilder& opsBuilder, const size_t noOfActKRangeTasks);
    void createSectionOpForActKernelInvocation(mlir::OpBuilder& opsBuilder, const size_t noOfActKInvocationTasks);
    void createSectionOpForActKernelParams(mlir::OpBuilder& opsBuilder, const size_t actKInvocationSectionIdx);
    void createSectionOpForActKernelText(mlir::OpBuilder& opsBuilder, const std::vector<mlir::Value>& actKTextOps,
                                         const size_t actKRangeSectionIdx);
    void createSectionOpForActKernelData(mlir::OpBuilder& opsBuilder, const std::vector<mlir::Value>& actKDataOps);
    void createSectionOpForShaveRtConfigs(mlir::OpBuilder& opsBuilder, const bool isScheduleEmbeddedRtUsed,
                                          mlir::Value& actRtTextOpValue);
    void createSectionOpForActShaveStacks(mlir::OpBuilder& opsBuilder);
    void createSectionOpForInvariants(mlir::OpBuilder& opsBuilder, const size_t noOfInvariantsTasks);
    void createSectionOpForVariants(mlir::OpBuilder& opsBuilder, const size_t noOfVariantsTasks);
    void createGenericBuiltInRegion(mlir::OpBuilder& opsBuilder);
    std::vector<std::pair<unsigned int, ELFNPU37XX::SymbolOp>> createSymbolOp(
            mlir::func::FuncOp& func, mlir::OpBuilder& opsBuilder,
            const elf::Reader<elf::ELF_Bitness::Elf64>::Section& section);
    VPURT::DeclareBufferOp createDeclareBufferOp(mlir::OpBuilder& opsBuilder, const int64_t& bufferSize,
                                                 const bool isTypeDDR, const int64_t& byteOffset);
    mlir::Value getSymbolValueBySecHeaderAndSymbolIdx(const size_t secHeaderIdx, const size_t symbolIndex);
    void fillValueForWaitAndUpdateBarrierConfigs(const uint64_t& prodMask, const uint64_t& consMask,
                                                 mlir::ValueRange& updateBarriers, mlir::ValueRange& waitBarriers);
    size_t getMappedInferenceSectionIndex();
    size_t getSectionIndexBasedOnRelocAOffset(const size_t offset, const size_t shInfo);
    mlir::Value getInputOrOutputValueForDmaTask(mlir::OpBuilder& opsBuilder, const size_t dmaSectionIdx,
                                                const uint64_t& offset, const size_t bufferSize,
                                                const elf::Elf_Xword& flag, const mlir::BlockArgument& funcArg);

    elf::AccessManager* _accessor = nullptr;
    elf::Reader<elf::ELF_Bitness::Elf64> _elfReader;
    mlir::MLIRContext* _ctx;
    mlir::ModuleOp _module;
    mlir::FlatSymbolRefAttr _mainFuncName;
    size_t _noOfSections;
    Logger _log;

    SmallVector<mlir::Type> _inputTypes;
    SmallVector<mlir::Type> _outputTypes;
    size_t _mappedInferSectionIdx = 0;

    std::map<size_t, mlir::Value> _sectionOpByValue;
    std::map<size_t, std::vector<std::pair<unsigned int, ELFNPU37XX::SymbolOp>>> _symbolsOpByValue;
    std::vector<std::pair<size_t, mlir::Value>> _barrierConfigsByRealId;
    std::map<size_t, std::vector<mlir::Value>> _nndmaOps;
    std::vector<VPUMI37XX::ActKernelRangeOp> _actKRangeOps;
    std::vector<VPUMI37XX::ActKernelInvocationOp> _actKInvocationOps;
    std::vector<std::pair<bool, VPURT::DeclareBufferOp>> _buffers;
    llvm::SmallVector<mlir::Value> _shaveStacks;
};

}  // namespace ELFNPU37XX
}  // namespace vpux
