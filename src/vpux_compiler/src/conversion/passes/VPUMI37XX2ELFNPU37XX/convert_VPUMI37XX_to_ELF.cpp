//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/ops.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <npu_37xx_nnrt.hpp>

#include <kernels/inc/common_types.h>

#include <vpux_elf/types/vpu_extensions.hpp>

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/Hashing.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUMI37XX2ELF
#define GEN_PASS_DEF_CONVERTVPUMI37XX2ELF
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace npu37xx;

constexpr auto NNCMX_SLICE_SIZE = vpux::ELFNPU37XX::CMX_SLICE_SIZE;
constexpr auto ACT_RT_CODE_BUFFER_SIZE = (1_MB).to<vpux::Byte>().count();

namespace {

constexpr auto VPUX_SHAVE_KERNEL_PREFETCH_PAD = static_cast<size_t>((128_Byte).count());

//
// ConvertVPUMI37XX2ELFPass
//

class ConvertVPUMI37XX2ELFPass final : public impl::ConvertVPUMI37XX2ELFBase<ConvertVPUMI37XX2ELFPass> {
public:
    explicit ConvertVPUMI37XX2ELFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    template <typename DerivedOpType, typename CreateSectionOpType>
    CreateSectionOpType createSection(mlir::func::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr,
                                      vpux::ELFNPU37XX::SectionTypeAttr secType,
                                      vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                                      elf::Elf_Word secAlign = elf::VPU_SH_ADDR_ALIGN_FOR_VPU);
    mlir::SmallVector<ELFNPU37XX::CreateSectionOp> createDMASections(mlir::func::FuncOp& func, mlir::MLIRContext* ctx,
                                                                     mlir::SmallVector<int64_t>& dmaCount,
                                                                     mlir::Operation::operand_range dmaTasks);

    mlir::Value createCMXMappingSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx);
    mlir::Value lookupELFSymbol(mlir::Value symtabValue, mlir::Value sym_input_value);
    mlir::Value createBuffersSecAndSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx);
    void createNetworkIOSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx, net::NetworkInfoOp netInfo);
    void createDMARelocs(mlir::func::FuncOp& func, mlir::MLIRContext* ctx, mlir::SmallVector<int64_t>& dmaCount,
                         mlir::Operation::operand_range dmaTasks,
                         mlir::SmallVector<ELFNPU37XX::CreateSectionOp>& dmaSectionValues);
    void createKernelParamsRelocs(mlir::func::FuncOp func, bool& shaveScratchAccess, bool& shaveConstAccess);
    void createActKernelRelocs(mlir::func::FuncOp func, ELFNPU37XX::CreateSectionOp actKernelRangeSection,
                               ELFNPU37XX::CreateSectionOp kernelTextSection,
                               ELFNPU37XX::CreateSectionOp actKernelInvocationSection,
                               ELFNPU37XX::CreateSectionOp kernelDataSection,
                               ELFNPU37XX::CreateSectionOp kernelParamSection);
    void setupActKernelRtConfigs(mlir::func::FuncOp func, mlir::ModuleOp moduleOp, mlir::MLIRContext* ctx);
    void createDPURelocs(mlir::func::FuncOp func);
    void createBarriersConfigRelocs(mlir::func::FuncOp func);
    void createAKISingleTileConfigRelocs(mlir::func::FuncOp func,
                                         ELFNPU37XX::CreateSectionOp actKernelInvocationSection);
    template <class T>
    void createBlockArgReloc(T op, mlir::OpBuilder inBuilder, mlir::OpBuilder outBuilder, size_t offset,
                             vpux::ELFNPU37XX::RelocationType relocationType, mlir::BlockArgument blockArg);

    void safeRunOnModule() final;

private:
    vpux::ELFNPU37XX::RelocationManager relocationManager;

    ELFNPU37XX::CreateLogicalSectionOp scratchBufferSectionOp;
    ELFNPU37XX::CreateSectionOp constSectionOp;

    mlir::Value networkInputSymTabValue, networkOutputSymTabValue, profOutputSymTabValue;

    mlir::Value bufferSymTabValue, CMXMappingSymtabValue;

    ELFNPU37XX::CreateSectionOp mappedInferenceSectionOp;

    std::map<std::string, mlir::Value> symbolMap;

    // map that correlates between Const::DeclareOp values and their ELFNPU37XX::SymbolOp value
    llvm::MapVector<mlir::Value, mlir::Value> constSymMap;

    // map that correlates between Const::DeclareOp values and their offset in the .data.const section
    llvm::MapVector<mlir::Value, size_t> constOffsetMap;

    std::vector<ELFNPU37XX::SymbolOp> elfCMXMappingSyms;

    using SymbolCache = mlir::DenseMap<mlir::Value, mlir::DenseMap<mlir::Value, mlir::Value>>;
    SymbolCache symbolCache;
    ELFNPU37XX::OffsetCache offsetCache;
    ELFNPU37XX::CreateSymbolTableSectionOp tasksSymbolTable;
    ELFNPU37XX::CreateSymbolTableSectionOp ddrSymbolTable;
};

// createSection() creates an ELFNPU37XX::CreateSectionOp and puts into its body
//   an ELF.PutOpInSectionOp instruction for each object of type DerivedOpType
//   from func (a FuncOp).
template <typename DerivedOpType, typename CreateSectionOpType>
CreateSectionOpType ConvertVPUMI37XX2ELFPass::createSection(mlir::func::FuncOp func, mlir::MLIRContext* ctx,
                                                            std::string secNameStr,
                                                            vpux::ELFNPU37XX::SectionTypeAttr secType,
                                                            vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                                                            elf::Elf_Word secAlign) {
    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    vpux::ELFNPU37XX::SectionType sectionType = vpux::ELFNPU37XX::SectionType::get(ctx);

    size_t opAlignmentRequirements = DerivedOpType::getAlignmentRequirements();
    size_t secAlignReq = vpux::ELFNPU37XX::math::lcm(secAlign, opAlignmentRequirements);

    auto elfCreateSectionOp =
            builderFunc.create<CreateSectionOpType>(mlir::UnknownLoc::get(ctx),
                                                    sectionType,  // mlir::Type
                                                    secNameStr,   // llvm::StringRef secName,
                                                    secType,      // vpux::ELFNPU37XX::SectionTypeAttr secType,
                                                    secFlags,     // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                                                    elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                    secAlignReq                // int64_t secAddrAlign
            );

    auto builder = mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());

    size_t offsetTracker = secAlignReq;

    auto ops = func.getOps<DerivedOpType>();
    if (ops.empty()) {
        return elfCreateSectionOp;
    }

    auto requiresEndPadding =
            ELFNPU37XX::bitEnumContainsAll(secFlags, ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);

    for (DerivedOpType op : ops) {
        if (auto declareBufferOp = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(&op)) {
            if (declareBufferOp->getSection() != vpux::VPURT::BufferSection::DDR) {
                continue;
            }
        }

        auto binaryOp = mlir::cast<vpux::ELFNPU37XX::BinaryOpInterface>(op.getOperation());
        size_t paddingRequired = offsetTracker % binaryOp.getAlignmentRequirements();
        if (paddingRequired) {
            auto off = secAlignReq - paddingRequired;
            builder.template create<ELFNPU37XX::PadOp>(builder.getUnknownLoc(), off, nullptr);
            offsetTracker += off;
        }

        builder.template create<ELFNPU37XX::PutOpInSectionOp>(builder.getUnknownLoc(), op.getResult());
        offsetTracker += binaryOp.getBinarySize();
    }

    if (requiresEndPadding && secType != vpux::ELFNPU37XX::SectionTypeAttr::SHT_NOBITS) {
        builder.template create<ELFNPU37XX::PadOp>(builder.getUnknownLoc(), VPUX_SHAVE_KERNEL_PREFETCH_PAD, nullptr);
        offsetTracker += VPUX_SHAVE_KERNEL_PREFETCH_PAD;
    }

    return elfCreateSectionOp;
}

mlir::SmallVector<ELFNPU37XX::CreateSectionOp> ConvertVPUMI37XX2ELFPass::createDMASections(
        mlir::func::FuncOp& func, mlir::MLIRContext* ctx, mlir::SmallVector<int64_t>& dmaCount,
        mlir::Operation::operand_range dmaTasks) {
    mlir::SmallVector<ELFNPU37XX::CreateSectionOp> returnValues;

    if (dmaTasks.empty()) {
        return returnValues;
    }

    std::string secNameBaseStr = ".text.dmaTasks";
    vpux::ELFNPU37XX::SectionTypeAttr secType = vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS;
    vpux::ELFNPU37XX::SectionFlagsAttr secFlags = vpux::ELFNPU37XX::SectionFlagsAttr::SHF_ALLOC;

    vpux::ELFNPU37XX::SectionType sectionType = vpux::ELFNPU37XX::SectionType::get(ctx);
    size_t opAlignmentRequirements = VPUMI37XX::NNDMAOp::getAlignmentRequirements();
    size_t secAlignReq = vpux::ELFNPU37XX::math::lcm(elf::VPU_SH_ADDR_ALIGN_FOR_VPU, opAlignmentRequirements);

    // Firstly, create sections for all DMA ports to keep reloc logic simple
    // Empty sections will be removed by cleanup pass
    for (size_t listIdx = 0; listIdx < dmaCount.size(); ++listIdx) {
        auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());
        auto elfCreateSectionOp = builderFunc.create<ELFNPU37XX::CreateSectionOp>(
                mlir::UnknownLoc::get(ctx),
                sectionType,                               // mlir::Type
                secNameBaseStr + std::to_string(listIdx),  // llvm::StringRef secName,
                secType,                                   // vpux::ELFNPU37XX::SectionTypeAttr secType,
                secFlags,                                  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                  // int64_t secInfo,
                secAlignReq                                // int64_t secAddrAlign
        );

        mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());
        returnValues.push_back(elfCreateSectionOp);
    }

    // Secondly, populate sections with the corresponding DMA tasks
    for (auto listHead : dmaTasks) {
        size_t offsetTracker = secAlignReq;
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getPort();
        auto listElemCount = dmaCount[listIdx];
        auto builder = mlir::OpBuilder::atBlockEnd(returnValues[listIdx].getBlock());

        for (auto dmaTaskIdx = 0; dmaTaskIdx < listElemCount; ++dmaTaskIdx) {
            auto nndmaOp = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp());
            auto binaryOp = mlir::cast<vpux::ELFNPU37XX::BinaryOpInterface>(nndmaOp.getOperation());
            size_t paddingRequired = offsetTracker % binaryOp.getAlignmentRequirements();
            if (paddingRequired) {
                auto off = secAlignReq - paddingRequired;
                builder.template create<ELFNPU37XX::PadOp>(builder.getUnknownLoc(), off, nullptr);
                offsetTracker += off;
            }

            builder.template create<ELFNPU37XX::PutOpInSectionOp>(builder.getUnknownLoc(), nndmaOp.getResult());
            offsetTracker += binaryOp.getBinarySize();

            listHead = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getNextDMAIdx();
        }
    }

    return returnValues;
}  // namespace

template <>
ELFNPU37XX::CreateSectionOp ConvertVPUMI37XX2ELFPass::createSection<Const::DeclareOp, ELFNPU37XX::CreateSectionOp>(
        mlir::func::FuncOp func, mlir::MLIRContext* ctx, std::string secNameStr,
        vpux::ELFNPU37XX::SectionTypeAttr secType, vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
        elf::Elf_Word secAlign) {
    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    vpux::ELFNPU37XX::SectionType sectionType = vpux::ELFNPU37XX::SectionType::get(ctx);

    auto elfCreateSectionOp =
            builderFunc.create<ELFNPU37XX::CreateSectionOp>(mlir::UnknownLoc::get(ctx),
                                                            sectionType,  // mlir::Type
                                                            secNameStr,   // llvm::StringRef secName,
                                                            secType,      // vpux::ELFNPU37XX::SectionTypeAttr secType,
                                                            secFlags,  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                                                            elf::VPU_SH_INFO_FOR_VPU,  // int64_t secInfo,
                                                            secAlign                   // int64_t secAddrAlign
            );

    vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    auto constSecValue = elfCreateSectionOp.getResult();

    symbolMap["sym_constSection"] =
            builderFunc
                    .create<ELFNPU37XX::SymbolOp>(builderFunc.getUnknownLoc(),
                                                  vpux::ELFNPU37XX::SymbolType::get(ctx),  // mlir::Type
                                                  constSecValue,                           // mlir::Value inputArg
                                                  isBuiltin,                               // mlir::UnitAttr
                                                  mlir::StringAttr::get(ctx, "sym_constSection"),  // mlir::StringAttr
                                                  typeSym,  // vpux::ELFNPU37XX::SymbolTypeEnumAttr
                                                  sizeSym,  // size
                                                  valueSym  // value
                                                  )
                    .getResult();

    auto builder = mlir::OpBuilder::atBlockEnd(elfCreateSectionOp.getBlock());

    for (Const::DeclareOp op : func.getOps<Const::DeclareOp>()) {
        builder.create<ELFNPU37XX::PutOpInSectionOp>(builder.getUnknownLoc(),  // endOp->getLoc(),
                                                     op.getResult()            // mlir::Value inputArg
        );
    }

    return elfCreateSectionOp;
}

mlir::Value ConvertVPUMI37XX2ELFPass::createBuffersSecAndSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx) {
    scratchBufferSectionOp = createSection<vpux::VPURT::DeclareBufferOp, ELFNPU37XX::CreateLogicalSectionOp>(
            func, ctx, ".data.BuffersIO", vpux::ELFNPU37XX::SectionTypeAttr::SHT_NOBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_WRITE | vpux::ELFNPU37XX::SectionFlagsAttr::SHF_ALLOC);

    constSectionOp = createSection<vpux::Const::DeclareOp, ELFNPU37XX::CreateSectionOp>(
            func, ctx, ".data.ConstIO", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_ALLOC);

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    auto bufferSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),           // mlir::Type
            scratchBufferSectionOp.getResult(),               // mlir::Value inputArg
            isBuiltin,                                        // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_bufferSection"),  // mlir::StringAttr
            typeSym,                                          // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                          // size
            valueSym                                          // value
    );

    symbolMap["sym_bufferSection"] = bufferSectionSym.getResult();

    ddrSymbolTable = builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SectionType::get(ctx),        // mlir::Type
            mlir::StringAttr::get(ctx, ".symtab.buffers"),  // mlir::StringAttr secName,
            vpux::ELFNPU37XX::SectionFlagsAttrAttr::get(
                    ctx, vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags
            isBuiltin                                                    // mlir::UnitAttr
    );

    mlir::Region& regBufferSymTabOp = ddrSymbolTable.getOperation()->getRegion(0);
    mlir::Block* blkBufferSymTabOp = new mlir::Block();
    regBufferSymTabOp.push_back(blkBufferSymTabOp);
    mlir::OpBuilder builderBufferSymTab(blkBufferSymTabOp, blkBufferSymTabOp->begin());

    mlir::Value bufferSectionSymValue = bufferSectionSym.getResult();
    mlir::Value bufferSymTabValue = ddrSymbolTable.getResult();

    builderBufferSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderBufferSymTab.getUnknownLoc(),
                                                             bufferSectionSymValue);
    builderBufferSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderBufferSymTab.getUnknownLoc(),
                                                             symbolMap["sym_constSection"]);

    return bufferSymTabValue;
}

mlir::Value ConvertVPUMI37XX2ELFPass::createCMXMappingSymtab(mlir::func::FuncOp funcOp, mlir::MLIRContext* ctx) {
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    std::vector<mlir::arith::ConstantOp> symVals;

    vpux::ELFNPU37XX::SymbolType symbolType = vpux::ELFNPU37XX::SymbolType::get(ctx);
    vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = mlir::UnitAttr::get(ctx);

    for (unsigned i = 0; i <= vpux::ELFNPU37XX::getMaxEnumValForCMXMappingSymbol(); ++i) {
        auto optionalCMXMappingSymValue = vpux::ELFNPU37XX::symbolizeCMXMappingSymbol(i);
        if (!optionalCMXMappingSymValue.has_value()) {
            continue;
        }

        auto CMXMappingSymValue = optionalCMXMappingSymValue.value();
        auto CMXMappingSymStringRef = vpux::ELFNPU37XX::stringifyCMXMappingSymbol(CMXMappingSymValue);

        symVals.push_back(builderFunc.create<mlir::arith::ConstantIntOp>(mlir::UnknownLoc::get(ctx), i, 8));
        elfCMXMappingSyms.push_back(builderFunc.create<ELFNPU37XX::SymbolOp>(
                mlir::UnknownLoc::get(ctx),
                symbolType,                                                            // mlir::Type
                symVals[i],                                                            // mlir::Value inputArg
                isBuiltin, mlir::StringAttr::get(ctx, CMXMappingSymStringRef.data()),  // mlir::StringAttr
                typeSym,  // vpux::ELFNPU37XX::SymbolTypeEnumAttr
                sizeSym,  // size
                valueSym  // value
                ));
    }

    vpux::ELFNPU37XX::SectionType secType = vpux::ELFNPU37XX::SectionType::get(ctx);

    ELFNPU37XX::CreateSymbolTableSectionOp createCMXMappingSymtabOp =
            builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
                    mlir::UnknownLoc::get(ctx),
                    secType,                                      // mlir::Type
                    mlir::StringAttr::get(ctx, "VPU_RT_SYMTAB"),  // mlir::StringAttr secName,
                    vpux::ELFNPU37XX::SectionFlagsAttrAttr::get(
                            ctx, vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                                 // secFlags,
                    isBuiltin                                                    // mlir::UnitAttr
            );

    mlir::Region& regCMXMappingSymtab = createCMXMappingSymtabOp.getOperation()->getRegion(0);
    mlir::Block* blkCMXMappingSymtab = new mlir::Block();

    regCMXMappingSymtab.push_back(blkCMXMappingSymtab);

    mlir::OpBuilder builderCMXMappingSymtab(blkCMXMappingSymtab, blkCMXMappingSymtab->begin());

    for (auto elfCMXMappingSym : elfCMXMappingSyms) {
        builderCMXMappingSymtab.create<ELFNPU37XX::PutOpInSectionOp>(
                builderCMXMappingSymtab.getUnknownLoc(),  // endOp->getLoc(),
                elfCMXMappingSym.getResult()              // mlir::Value inputArg
        );
    }

    return createCMXMappingSymtabOp.getResult();
}

void ConvertVPUMI37XX2ELFPass::createNetworkIOSymtab(mlir::func::FuncOp func, mlir::MLIRContext* ctx,
                                                     net::NetworkInfoOp netInfo) {
    auto dataInfoOpInVec = netInfo.getInputsDataInfo();
    auto dataInfoOpOutVec = netInfo.getOutputsDataInfo();
    auto dataInfoOpProfilingOutVec = netInfo.getProfilingOutputsDataInfo();

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());

    std::vector<mlir::Value> inputSyms;
    std::vector<mlir::Value> outputSyms;
    std::vector<mlir::Value> profOutputSyms;

    mlir::IntegerType uint64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    for (auto funcArg : func.getArguments()) {
        vpux::ELFNPU37XX::SymbolType symbolType = vpux::ELFNPU37XX::SymbolType::get(ctx);
        vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
        mlir::IntegerAttr valueSym;
        mlir::UnitAttr isBuiltin = nullptr;

        auto argNDType = mlir::cast<vpux::NDTypeInterface>(funcArg.getType());
        mlir::IntegerAttr sizeSym = mlir::IntegerAttr::get(uint64Type, argNDType.getTotalAllocSize().count());

        mlir::StringAttr nameSym;
        std::vector<mlir::Value>* symsVecPtr = nullptr;
        auto index = funcArg.getArgNumber();
        if (index < dataInfoOpInVec.size()) {
            symsVecPtr = &inputSyms;
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpInVec[index].getName());
        } else if (index < (dataInfoOpInVec.size() + dataInfoOpOutVec.size())) {
            symsVecPtr = &outputSyms;
            index -= dataInfoOpInVec.size();
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpOutVec[index].getName());
        } else {
            symsVecPtr = &profOutputSyms;
            index -= dataInfoOpInVec.size() + dataInfoOpOutVec.size();
            nameSym = mlir::StringAttr::get(ctx, dataInfoOpProfilingOutVec[index].getName());
        }

        auto netIOSym = builderFunc.create<ELFNPU37XX::SymbolOp>(builderFunc.getUnknownLoc(),
                                                                 symbolType,  // mlir::Type
                                                                 funcArg,     // mlir::Value inputArg
                                                                 isBuiltin,   // mlir::UnitAttr
                                                                 nameSym,     // mlir::StringAttr
                                                                 typeSym,     // vpux::ELFNPU37XX::SymbolTypeEnumAttr
                                                                 sizeSym,     // size
                                                                 valueSym     // value
        );

        symsVecPtr->push_back(netIOSym.getResult());
    }

    // Secondly we create the symbol table for the input symbols
    ELFNPU37XX::CreateSymbolTableSectionOp createInputSymTableSectionOp =
            builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
                    mlir::UnknownLoc::get(ctx),
                    vpux::ELFNPU37XX::SectionType::get(ctx),                // mlir::Type
                    ".symtab.input",                                        // llvm::StringRef secName,
                    vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_USERINPUT,  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                            // secFlags,
                    false                                                   // bool isBuiltin
            );
    //
    mlir::Region& regInputSymTabSec = createInputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkInputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regInputSymTabSec.push_back(blkInputSymTabSec);
    //
    mlir::OpBuilder builderInputSymTabSec(blkInputSymTabSec, blkInputSymTabSec->begin());
    networkInputSymTabValue = createInputSymTableSectionOp.getResult();

    // Thirdly we create the symbol table for the output symbols
    ELFNPU37XX::CreateSymbolTableSectionOp createOutputSymTableSectionOp =
            builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
                    mlir::UnknownLoc::get(ctx),
                    vpux::ELFNPU37XX::SectionType::get(ctx),                 // mlir::Type
                    ".symtab.output",                                        // llvm::StringRef secName,
                    vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_USEROUTPUT,  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                             // secFlags,
                    false                                                    // bool isBuiltin
            );
    //
    mlir::Region& regOutputSymTabSec = createOutputSymTableSectionOp.getOperation()->getRegion(0);
    mlir::Block* blkOutputSymTabSec = new mlir::Block();
    //
    // This instruction has to be before defining builderSymTabSec to avoid SegFault
    regOutputSymTabSec.push_back(blkOutputSymTabSec);
    //
    mlir::OpBuilder builderOutputSymTabSec(blkOutputSymTabSec, blkOutputSymTabSec->begin());
    networkOutputSymTabValue = createOutputSymTableSectionOp.getResult();

    for (auto inputSym : inputSyms) {
        builderInputSymTabSec.create<ELFNPU37XX::PutOpInSectionOp>(
                builderInputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                inputSym                                // mlir::Value inputArg
        );
    }

    for (auto outputSym : outputSyms) {
        builderOutputSymTabSec.create<ELFNPU37XX::PutOpInSectionOp>(
                builderOutputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                outputSym                                // mlir::Value inputArg
        );
    }

    // If profiling is enabled add also profiling output symbol table
    if (!dataInfoOpProfilingOutVec.empty()) {
        ELFNPU37XX::CreateSymbolTableSectionOp createProfOutputSymTableSectionOp =
                builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
                        mlir::UnknownLoc::get(ctx),
                        vpux::ELFNPU37XX::SectionType::get(ctx),                 // mlir::Type
                        ".symtab.prof_output",                                   // llvm::StringRef secName,
                        vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROFOUTPUT,  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                                 // secFlags,
                        false                                                    // bool isBuiltin
                );
        //
        mlir::Region& regProfOutputSymTabSec = createProfOutputSymTableSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkProfOutputSymTabSec = new mlir::Block();
        //
        // This instruction has to be before defining builderSymTabSec to avoid SegFault
        regProfOutputSymTabSec.push_back(blkProfOutputSymTabSec);
        //
        mlir::OpBuilder builderProfOutputSymTabSec(blkProfOutputSymTabSec, blkProfOutputSymTabSec->begin());
        profOutputSymTabValue = createProfOutputSymTableSectionOp.getResult();

        for (auto profOutputSym : profOutputSyms) {
            builderProfOutputSymTabSec.create<ELFNPU37XX::PutOpInSectionOp>(
                    builderProfOutputSymTabSec.getUnknownLoc(),  // endOp->getLoc(),
                    profOutputSym                                // mlir::Value inputArg
            );
        }
    }
}

template <typename T>
void ConvertVPUMI37XX2ELFPass::createBlockArgReloc(T op, mlir::OpBuilder builderInputRelocSec,
                                                   mlir::OpBuilder builderOutputRelocSec, size_t offset,
                                                   vpux::ELFNPU37XX::RelocationType relocationType,
                                                   mlir::BlockArgument blockArg) {
    if (mlir::Value netSymValue = lookupELFSymbol(networkOutputSymTabValue, blockArg)) {
        builderOutputRelocSec.create<ELFNPU37XX::RelocOp>(builderOutputRelocSec.getUnknownLoc(), op.getResult(), offset,
                                                          relocationType,  // relocationType
                                                          netSymValue,     // ::mlir::Value sourceSymbol
                                                          0,               // int64_t addend
                                                          op.getOperationName().str() + " input block argument reloc");
    } else if (mlir::Value netSymValue = lookupELFSymbol(networkInputSymTabValue, blockArg)) {
        builderInputRelocSec.create<ELFNPU37XX::RelocOp>(builderInputRelocSec.getUnknownLoc(), op.getResult(), offset,
                                                         relocationType,  // relocationType
                                                         netSymValue,     // ::mlir::Value sourceSymbol
                                                         0,               // int64_t addend
                                                         op.getOperationName().str() + " output block argument reloc");
    }
}

void ConvertVPUMI37XX2ELFPass::createKernelParamsRelocs(mlir::func::FuncOp func, bool& shaveScratchAccess,
                                                        bool& shaveConstAccess) {
    auto kernelParamsOps = func.getOps<vpux::VPUMI37XX::KernelParamsOp>();

    if (kernelParamsOps.empty()) {
        return;
    }

    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELFNPU37XX::ElfSectionInterface targetSection;
    ELFNPU37XX::CreateSymbolTableSectionOp symTab;
    ELFNPU37XX::CreateRelocationSectionOp relocSection;
    ELFNPU37XX::SymbolOp sourceSym;
    mlir::Value kernelParamsSectionSym = symbolMap["sym_kernelParamsSection"];

    // All the Kenel Params stuctures are serialized in a single section, in a continuous manner
    // All the relocations (excluding I/O ones), relocated addresses belonging to the same section as the target section
    targetSection = relocationManager.getSection((*kernelParamsOps.begin()).getResult());
    symTab = relocationManager.getSymTab((*kernelParamsOps.begin()));
    relocSection = relocationManager.getRelocSection(targetSection, symTab);

    mlir::Value kernelParamsSectionValue = targetSection.getOperation()->getResult(0);
    auto paramsAutoRelocBuilder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    for (auto kernelParamsOpIt : kernelParamsOps | indexed) {
        auto kernelParamsOp = kernelParamsOpIt.value();
        auto kernelParamsOpName = kernelParamsOp.getOperationName().str();

        auto kernelParamsOpVal = kernelParamsOp.getResult();
        auto partial_addend =
                ELFNPU37XX::getOffsetOfOpInSection(kernelParamsOpVal, kernelParamsSectionValue, offsetCache) +
                kernelParamsOp.getParamsStructSize();

        auto kernelInputs = kernelParamsOp.getInputs();

        // input addr
        size_t inputRelocOffset = 0, shaveCodeOffsetTracker = 0;
        for (auto kernelInputIt : kernelInputs | indexed) {
            auto kernelInput = kernelInputIt.value();
            if (kernelParamsOp.getIsCompiled()) {
                const auto shape = getShape(kernelInput);
                int64_t rankMemref = checked_cast<uint32_t>(shape.size());
                inputRelocOffset = shaveCodeOffsetTracker +
                                   sizeof(uint32_t); /* offset of allignedPointer in the struct - since this field comes
                                                        after 1st field of struct, of uint32_t type */
                shaveCodeOffsetTracker +=
                        sizeof(uint32_t) * 2 + sizeof(int32_t) +
                        rankMemref * sizeof(int32_t) * 2 /* sizeof MemRef LLVM struct at inputsItValue */;
            } else {
                inputRelocOffset = kernelInputIt.index() * sizeof(sw_params::MemRefData) +
                                   offsetof(sw_params::MemRefData, dataAddr);
            }

            if (kernelInput.getDefiningOp<Const::DeclareOp>() != nullptr) {
                shaveConstAccess = true;
            }

            auto kernelInputBuff = kernelInput.getDefiningOp<VPURT::DeclareBufferOp>();
            if (kernelInputBuff && (kernelInputBuff.getMemorySpace() == VPURT::BufferSection::DDR)) {
                shaveScratchAccess = true;
            }

            symTab = relocationManager.getSymTab(kernelInput);

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            auto kernelInputBinaryOp = mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(kernelInput.getDefiningOp());

            size_t addend = 0;

            if (kernelInputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELFNPU37XX::getOffsetOfOpInSection(kernelInput);
            } else {
                auto kernelInputSection = relocationManager.getSection(kernelInput);
                sourceSym = ELFNPU37XX::RelocationManager::getSymbol(kernelInputSection);
                mlir::Value kernelInputSectionValue = kernelInputSection.getOperation()->getResult(0);
                addend = ELFNPU37XX::getOffsetOfOpInSection(kernelInput, kernelInputSectionValue, offsetCache);
            }

            builder.create<ELFNPU37XX::RelocOp>(kernelInput.getLoc(), kernelParamsOp.getResult(), inputRelocOffset,
                                                vpux::ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                                                kernelParamsOpName + " input kernel params reloc");
        }

        auto addDynamicShapeRelocation = [&](mlir::Value dims, uint64_t offset) {
            symTab = relocationManager.getSymTab(dims);

            auto dimsBinaryOp = mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(dims.getDefiningOp());
            size_t addend = 0;
            if (dimsBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELFNPU37XX::getOffsetOfOpInSection(dims);
            } else {
                auto dimsSection = relocationManager.getSection(dims);
                sourceSym = ELFNPU37XX::RelocationManager::getSymbol(dimsSection);
                mlir::Value dimsSectionValue = dimsSection.getOperation()->getResult(0);
                addend = ELFNPU37XX::getOffsetOfOpInSection(dims, dimsSectionValue, offsetCache);
            }

            relocSection = relocationManager.getRelocSection(targetSection, symTab);
            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());
            builder.create<ELFNPU37XX::RelocOp>(kernelParamsOp.getLoc(), kernelParamsOp.getResult(), offset,
                                                vpux::ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                                                kernelParamsOpName + " dynamic shape I/O dims kernel params reloc");
        };

        const auto& dynInputShapes = kernelParamsOp.getDynamicInputShapes();
        if (!dynInputShapes.empty()) {
            VPUX_THROW_UNLESS(
                    dynInputShapes.size() == kernelInputs.size(),
                    "The number of dynamic shapes '{0}' must be equal to zero (means all input shapes are static) or "
                    "match the number of inputs '{1}'",
                    dynInputShapes.size(), kernelInputs.size());
        }

        // input Dims addr
        if (!kernelParamsOp.getIsCompiled()) {
            for (const auto& kernelInputsIt : kernelInputs | indexed) {
                if (!dynInputShapes.empty() && !dynInputShapes[kernelInputsIt.index()].empty()) {
                    const auto& inputShape = dynInputShapes[kernelInputsIt.index()];
                    VPUX_THROW_WHEN(inputShape.size() > 1,
                                    "The number of dynamic shapes per input must be either 0 or 1");
                    addDynamicShapeRelocation(inputShape[0], kernelInputsIt.index() * sizeof(sw_params::MemRefData) +
                                                                     offsetof(sw_params::MemRefData, dimsAddr));
                } else {
                    paramsAutoRelocBuilder.create<ELFNPU37XX::RelocOp>(
                            kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                            kernelInputsIt.index() * sizeof(sw_params::MemRefData) +
                                    offsetof(sw_params::MemRefData, dimsAddr),
                            vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelParamsSectionSym, partial_addend,
                            kernelParamsOpName + " static input dims kernel params reloc");
                }
                partial_addend += sizeof(int32_t) * getShape(kernelInputsIt.value()).size();
            }

            // input Strides addr
            for (auto kernelInputsIt : kernelInputs | indexed) {
                paramsAutoRelocBuilder.create<ELFNPU37XX::RelocOp>(
                        kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                        kernelInputsIt.index() * sizeof(sw_params::MemRefData) +
                                offsetof(sw_params::MemRefData, stridesAddr),
                        vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelParamsSectionSym, partial_addend,
                        kernelParamsOpName + " static input strides kernel params reloc");

                partial_addend += sizeof(int64_t) * getMemStrides(kernelInputsIt.value()).size();
            }
        }

        auto kernelOutputs = kernelParamsOp.getOutputs();
        auto kernelInputsSize = kernelInputs.size();

        // output addr
        size_t outputRelocOffset = 0;
        for (auto kernelOutputIt : kernelOutputs | indexed) {
            auto kernelOutput = kernelOutputIt.value();
            if (kernelParamsOp.getIsCompiled()) {
                const auto shape = getShape(kernelOutput);
                int64_t rankMemref = checked_cast<uint32_t>(shape.size());
                outputRelocOffset = shaveCodeOffsetTracker +
                                    sizeof(uint32_t); /* offset of allignedPointer in the struct - since this field
                                                        comes after 1st field of struct, of uint32_t type */
                shaveCodeOffsetTracker +=
                        sizeof(uint32_t) * 2 + sizeof(int32_t) +
                        rankMemref * sizeof(int32_t) * 2 /* sizeof MemRef LLVM struct at inputsItValue */;
            } else {
                outputRelocOffset = (kernelInputsSize + kernelOutputIt.index()) * sizeof(sw_params::MemRefData) +
                                    offsetof(sw_params::MemRefData, dataAddr);
            }

            auto kernelOutputBuff = kernelOutput.getDefiningOp<VPURT::DeclareBufferOp>();
            if (kernelOutputBuff && (kernelOutputBuff.getMemorySpace() == VPURT::BufferSection::DDR)) {
                shaveScratchAccess = true;
            }

            symTab = relocationManager.getSymTab(kernelOutput);

            relocSection = relocationManager.getRelocSection(targetSection, symTab);

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            auto kernelOutputBinaryOp = mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(kernelOutput.getDefiningOp());

            size_t addend = 0;

            if (kernelOutputBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                addend = ELFNPU37XX::getOffsetOfOpInSection(kernelOutput);
            } else {
                auto kernelOutputSection = relocationManager.getSection(kernelOutput);
                sourceSym = ELFNPU37XX::RelocationManager::getSymbol(kernelOutputSection);
                mlir::Value kernelOutputSectionValue = kernelOutputSection.getOperation()->getResult(0);
                addend = ELFNPU37XX::getOffsetOfOpInSection(kernelOutput, kernelOutputSectionValue, offsetCache);
            }

            builder.create<ELFNPU37XX::RelocOp>(kernelOutput.getLoc(), kernelParamsOp, outputRelocOffset,
                                                vpux::ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                                                kernelParamsOpName + " output kernel params reloc");
        }

        const auto& dynamicOutputShapes = kernelParamsOp.getDynamicOutputShapes();
        if (!dynamicOutputShapes.empty()) {
            VPUX_THROW_UNLESS(
                    dynamicOutputShapes.size() == kernelOutputs.size(),
                    "The number of dynamic shapes '{0}' must be equal to zero (means all output shapes are static) or "
                    "match the number of outputs '{1}'",
                    dynamicOutputShapes.size(), kernelOutputs.size());
        }

        if (!kernelParamsOp.getIsCompiled()) {
            // output Dims addr
            for (const auto& kernelOutputsIt : kernelOutputs | indexed) {
                if (!dynamicOutputShapes.empty() && !dynamicOutputShapes[kernelOutputsIt.index()].empty()) {
                    const auto& outputShape = dynamicOutputShapes[kernelOutputsIt.index()];
                    VPUX_THROW_WHEN(outputShape.size() > 1,
                                    "The number of dynamic shapes per output must be either 0 or 1");
                    addDynamicShapeRelocation(outputShape[0], (kernelInputsSize + kernelOutputsIt.index()) *
                                                                              sizeof(sw_params::MemRefData) +
                                                                      offsetof(sw_params::MemRefData, dimsAddr));
                } else {
                    paramsAutoRelocBuilder.create<ELFNPU37XX::RelocOp>(
                            kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                            (kernelInputsSize + kernelOutputsIt.index()) * sizeof(sw_params::MemRefData) +
                                    offsetof(sw_params::MemRefData, dimsAddr),
                            vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelParamsSectionSym, partial_addend,
                            kernelParamsOpName + " static output dims kernel params reloc");
                }

                partial_addend += sizeof(int32_t) * getShape(kernelOutputsIt.value()).size();
            }

            // output Strides addr
            for (auto kernelOutputsIt : kernelOutputs | indexed) {
                paramsAutoRelocBuilder.create<ELFNPU37XX::RelocOp>(
                        kernelParamsOp.getLoc(), kernelParamsOp.getResult(),
                        (kernelInputsSize + kernelOutputsIt.index()) * sizeof(sw_params::MemRefData) +
                                offsetof(sw_params::MemRefData, stridesAddr),
                        vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelParamsSectionSym, partial_addend,
                        kernelParamsOpName + " static output strides kernel params reloc");

                partial_addend += sizeof(int64_t) * getMemStrides(kernelOutputsIt.value()).size();
            }
        }
    }
}

mlir::Value ConvertVPUMI37XX2ELFPass::lookupELFSymbol(mlir::Value symtabValue, mlir::Value sym_input_value) {
    auto& symbolTableCache = symbolCache[symtabValue];
    if (symbolTableCache.empty()) {
        auto symtabOp = llvm::dyn_cast<vpux::ELFNPU37XX::CreateSymbolTableSectionOp>(symtabValue.getDefiningOp());

        auto symtabBlk = symtabOp.getBody();
        for (auto& op : symtabBlk->getOperations()) {
            if (auto symOp = llvm::dyn_cast<vpux::ELFNPU37XX::SymbolOp>(op)) {
                symbolTableCache[symOp.getInputArg()] = symOp.getResult();
            } else if (auto placeholder = llvm::dyn_cast<vpux::ELFNPU37XX::PutOpInSectionOp>(op)) {
                auto actualOp = placeholder.getInputArg().getDefiningOp();
                auto symOp = llvm::dyn_cast<vpux::ELFNPU37XX::SymbolOp>(actualOp);
                symbolTableCache[symOp.getInputArg()] = symOp.getResult();
            }
        }
    }

    const auto symbolIt = symbolTableCache.find(sym_input_value);
    return symbolIt == symbolTableCache.end() ? mlir::Value{} : symbolIt->second;
}

void ConvertVPUMI37XX2ELFPass::createActKernelRelocs(mlir::func::FuncOp func,
                                                     ELFNPU37XX::CreateSectionOp actKernelRangeSection,
                                                     ELFNPU37XX::CreateSectionOp kernelTextSection,
                                                     ELFNPU37XX::CreateSectionOp actKernelInvocationSection,
                                                     ELFNPU37XX::CreateSectionOp kernelDataSection,
                                                     ELFNPU37XX::CreateSectionOp kernelParamsSection) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    auto cmxSymbolTable = relocationManager.getCMXSymTab();

    {
        auto kernelTextSectionSymbol =
                mlir::dyn_cast<ELFNPU37XX::SymbolOp>(symbolMap["sym_kernelTextSection"].getDefiningOp());
        auto actKernelRangeRelocSection = relocationManager.getRelocSection(actKernelRangeSection, tasksSymbolTable);
        auto actKernelRangeRelocSectionBuilder = mlir::OpBuilder::atBlockEnd(actKernelRangeRelocSection.getBlock());
        for (auto actKernelRangeOp : func.getOps<vpux::VPUMI37XX::ActKernelRangeOp>()) {
            if (VPUMI37XX::isSwKernelCacheOp(actKernelRangeOp)) {
                continue;
            }

            auto kernelText = actKernelRangeOp.getKernelTextIndex();
            actKernelRangeRelocSectionBuilder.create<ELFNPU37XX::RelocOp>(
                    kernelText.getLoc(), actKernelRangeOp, actKernelRangeOp.getOffsetOfWithinOperation(kernelText),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelTextSectionSymbol,
                    ELFNPU37XX::getOffsetOfOpInSection(kernelText, kernelTextSection->getResult(0), offsetCache),
                    actKernelRangeOp.getOperationName().str() + " kernelText act kernel reloc");
        }
    }

    {
        auto kernelDataSectionSymbol =
                mlir::dyn_cast<ELFNPU37XX::SymbolOp>(symbolMap["sym_kernelDataSection"].getDefiningOp());
        auto kernelParamsSectionSymbol =
                mlir::dyn_cast<ELFNPU37XX::SymbolOp>(symbolMap["sym_kernelParamsSection"].getDefiningOp());
        auto actKernelInvocationCMXRelocSection =
                relocationManager.getRelocSection(actKernelInvocationSection, cmxSymbolTable);
        auto actKernelInovcationCMXRelocSectionBuilder =
                mlir::OpBuilder::atBlockEnd(actKernelInvocationCMXRelocSection.getBlock());
        auto actKernelInvocationRelocSection =
                relocationManager.getRelocSection(actKernelInvocationSection, tasksSymbolTable);
        auto actKernelInovcationRelocSectionBuilder =
                mlir::OpBuilder::atBlockEnd(actKernelInvocationRelocSection.getBlock());
        for (auto actKernelInvoOp : func.getOps<vpux::VPUMI37XX::ActKernelInvocationOp>()) {
            auto associatedRangeOp =
                    llvm::dyn_cast<vpux::VPUMI37XX::ActKernelRangeOp>(actKernelInvoOp.getRangeIndex().getDefiningOp());

            actKernelInovcationCMXRelocSectionBuilder.create<ELFNPU37XX::RelocOp>(
                    actKernelInvocationCMXRelocSection.getLoc(), actKernelInvoOp.getResult(),
                    actKernelInvoOp.getOffsetOfWithinOperation(actKernelInvoOp.getRangeIndex()),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_32_RTM,
                    elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_RTM_ACT)]
                            .getResult(),
                    sizeof(nn_public::VpuActKernelRange), "Range in act kernel invocation reloc");

            if (!VPUMI37XX::isSwKernelCacheOp(associatedRangeOp)) {
                auto kernelData = associatedRangeOp.getKernelArgsIndex();
                actKernelInovcationRelocSectionBuilder.create<ELFNPU37XX::RelocOp>(
                        actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                        offsetof(nn_public::VpuActKernelInvocation, data_window_base),
                        vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelDataSectionSymbol,
                        ELFNPU37XX::getOffsetOfOpInSection(kernelData, kernelDataSection->getResult(0), offsetCache),
                        "Kernel data (data_window_base) in act kernel invocation reloc");
            }

            actKernelInovcationRelocSectionBuilder.create<ELFNPU37XX::RelocOp>(
                    actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                    offsetof(nn_public::VpuActKernelInvocation, kernel_args),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_32, kernelParamsSectionSymbol,
                    ELFNPU37XX::getOffsetOfOpInSection(actKernelInvoOp.getParamsIndex(),
                                                       kernelParamsSection->getResult(0), offsetCache),
                    "kernel_args in act kernel invocation reloc");

            // profiling reloc
            // perf_packet_out field from ActKernelInvocation structure needs to point to
            // profiling buffer allocated by compiler
            if (auto profBuffer = actKernelInvoOp.getProfilingData()) {
                auto kernelProfilingBinaryOp =
                        mlir::dyn_cast<ELFNPU37XX::BinaryOpInterface>(profBuffer.getDefiningOp());

                size_t addend = 0;

                ELFNPU37XX::SymbolOp sourceSym;
                if (kernelProfilingBinaryOp.getMemorySpace() == VPURT::BufferSection::CMX_NN) {
                    sourceSym = elfCMXMappingSyms[static_cast<int>(
                            vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                    addend = ELFNPU37XX::getOffsetOfOpInSection(profBuffer);
                } else {
                    auto kernelProfilingSection = relocationManager.getSection(profBuffer);
                    sourceSym = ELFNPU37XX::RelocationManager::getSymbol(kernelProfilingSection);
                    mlir::Value kernelProfilingSectionValue = kernelProfilingSection.getOperation()->getResult(0);
                    addend = ELFNPU37XX::getOffsetOfOpInSection(profBuffer, kernelProfilingSectionValue, offsetCache);
                }

                actKernelInovcationCMXRelocSectionBuilder.create<ELFNPU37XX::RelocOp>(
                        actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                        offsetof(nn_public::VpuActKernelInvocation, perf_packet_out),
                        vpux::ELFNPU37XX::RelocationType::R_VPU_32_SUM, sourceSym, addend,
                        "Profiling (perf_packet_out) in act kernel invocation reloc");
            }
        }
    }
}
void ConvertVPUMI37XX2ELFPass::createAKISingleTileConfigRelocs(mlir::func::FuncOp func,
                                                               ELFNPU37XX::CreateSectionOp actKernelInvocationSection) {
    auto cmxSymbolTable = relocationManager.getCMXSymTab();

    for (auto actKernelInvoOp : func.getOps<vpux::VPUMI37XX::ActKernelInvocationOp>()) {
        auto relocSectionSymTab = relocationManager.getRelocSection(actKernelInvocationSection, cmxSymbolTable);
        auto builderRtSymTab = mlir::OpBuilder::atBlockEnd(relocSectionSymTab.getBlock());

        builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                offsetof(struct nn_public::VpuActKernelInvocation, barriers.wait_mask_),
                vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT,
                elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_BARRIERS_START)],
                0);

        builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                offsetof(struct nn_public::VpuActKernelInvocation, barriers.post_mask_),
                vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT,
                elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_BARRIERS_START)],
                0);

        // AKI cluster index relocation
        // It is set by the symbol vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_FIFO_BASE.
        // 0x0 - tile0
        // 0x1 - tile1
        const uint32_t clusterIdx = 0;
        builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                actKernelInvoOp.getLoc(), actKernelInvoOp.getResult(),
                offsetof(struct nn_public::VpuActKernelInvocation, invo_tile),
                vpux::ELFNPU37XX::RelocationType::R_VPU_32,
                elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_FIFO_BASE)],
                clusterIdx);
    }
}

void ConvertVPUMI37XX2ELFPass::setupActKernelRtConfigs(mlir::func::FuncOp func, mlir::ModuleOp moduleOp,
                                                       mlir::MLIRContext* ctx) {
    auto mappedInferenceOps = func.getOps<VPUMI37XX::MappedInferenceOp>();

    VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");

    auto mappedInferenceOp = *(mappedInferenceOps.begin());

    if (mappedInferenceOp.getActKernelInvocationsCount() == 0) {
        return;
    }

    auto builderFunc = mlir::OpBuilder::atBlockTerminator(&func.getBody().front());
    builderFunc.setInsertionPoint(mappedInferenceOp.getOperation());

    auto vpuSwModuleOp = moduleOp.lookupSymbol<mlir::ModuleOp>("VPU.SW");

    VPUX_THROW_UNLESS(vpuSwModuleOp != nullptr, "setupActKernelConfig: @VPU.SW module missing.");

    auto runtimeKernelFunction = vpuSwModuleOp.lookupSymbol<mlir::func::FuncOp>("runtime");

    mlir::Value actShaveRt;
    ELFNPU37XX::ElfSectionInterface actKRtConfigSec;

    vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;

    if (runtimeKernelFunction) {
        const auto kernelElf =
                std::string(runtimeKernelFunction->getAttrOfType<mlir::StringAttr>("VPU.kernel_code").getValue());

        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

        auto actShvRtOp = builderFunc.create<VPUMI37XX::ActShaveRtOp>(builderFunc.getUnknownLoc(), trivialIndexType,
                                                                      mlir::StringAttr::get(ctx, kernelElf));

        actShaveRt = actShvRtOp.getResult();

        actKRtConfigSec = builderFunc.create<ELFNPU37XX::CreateSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELFNPU37XX::SectionType::get(ctx),          // mlir::Type
                ".text.actKernelRtConfigSec",                     // llvm::StringRef secName,
                vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,  // vpux::ELFNPU37XX::SectionTypeAttr secType,
                vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE,     // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                         // int64_t secInfo,
                1024                                              // int64_t secAddrAlign
        );

        mappedInferenceOp.getActShaveRtMutable().assign(actShaveRt);

    } else {
        auto actRtCodeBufferMemrefType = vpux::ELF::getLinearMemrefType(ctx, ACT_RT_CODE_BUFFER_SIZE,
                                                                        vpux::getInt8Type(ctx), VPU::MemoryKind::DDR);

        auto declareBufferOp = builderFunc.create<VPURT::DeclareBufferOp>(builderFunc.getUnknownLoc(),
                                                                          actRtCodeBufferMemrefType,  // Type
                                                                          VPURT::BufferSection::DDR,  // Buffer Type
                                                                          0                           // byteOffset
        );

        actShaveRt = declareBufferOp.getResult();

        actKRtConfigSec = builderFunc.create<ELFNPU37XX::CreateLogicalSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELFNPU37XX::SectionType::get(ctx),        // mlir::Type
                ".bss.actKernelRtConfigSec",                    // llvm::StringRef secName,
                vpux::ELFNPU37XX::SectionTypeAttr::SHT_NOBITS,  // vpux::ELFNPU37XX::SectionTypeAttr secType,
                vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE,   // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                       // int64_t secInfo,
                ELFNPU37XX::VPUX_SHAVE_ALIGNMENT                // int64_t secAddrAlign
        );
    }

    // Depending on the case, the Section must be binary or logical
    // Refactor such that it comprises both logic

    auto builderElfSectionOpReg = mlir::OpBuilder::atBlockEnd(actKRtConfigSec.getBlock());

    builderElfSectionOpReg.create<ELFNPU37XX::PutOpInSectionOp>(
            builderElfSectionOpReg.getUnknownLoc(),  // endOp->getLoc(),
            actShaveRt                               // mlir::Value inputArg
    );

    mlir::Value actKRtConfigSecValue = actKRtConfigSec.getOperation()->getResult(0);

    auto actKRtConfigSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELFNPU37XX::SymbolType::get(ctx),                   // mlir::Type
            actKRtConfigSecValue,                                     // mlir::Value inputArg
            isBuiltin,                                                // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_actKernelRtConfigsSec"),  // mlir::StringAttr
            typeSym,                                                  // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                                  // size
            valueSym                                                  // value
    );
    symbolMap["sym_actKernelRtConfigsSec"] = actKRtConfigSym.getResult();

    auto actKRtConfigSymTab = builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELFNPU37XX::SectionType::get(ctx),                  // mlir::Type
            mlir::StringAttr::get(ctx, ".symtab.actKernelRtConfig"),  // mlir::StringAttr secName,
            vpux::ELFNPU37XX::SectionFlagsAttrAttr::get(
                    ctx, vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
            isBuiltin                                                    // mlir::UnitAttr
    );

    auto builderActKRtConfigSymTab = mlir::OpBuilder::atBlockEnd(actKRtConfigSymTab.getBlock());

    builderActKRtConfigSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderActKRtConfigSymTab.getUnknownLoc(),
                                                                   actKRtConfigSym.getResult());

    mlir::Value actKRtConfigSymValue = actKRtConfigSym.getResult();

    VPUX_THROW_UNLESS(mappedInferenceSectionOp != nullptr, "CreateActKernelConfig: MappedInference section is null");

    auto relocSection = relocationManager.getRelocSection(mappedInferenceSectionOp, actKRtConfigSymTab);

    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    for (auto mappedInferenceOp : mappedInferenceOps) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                offsetof(nn_public::VpuMappedInference, shv_rt_configs) +
                        offsetof(nn_public::VpuNNShaveRuntimeConfigs, act_rt_window_base),
                vpux::ELFNPU37XX::RelocationType::R_VPU_32, actKRtConfigSymValue, 0,
                "Mapped inference op (act_rt_window_base) in act kernel rt configs reloc");
    }
}

void ConvertVPUMI37XX2ELFPass::createBarriersConfigRelocs(mlir::func::FuncOp func) {
    mlir::OpBuilder builderFunc(&(func.getBody().front().back()));

    ELFNPU37XX::ElfSectionInterface targetSection;
    ELFNPU37XX::CreateSymbolTableSectionOp symTab;
    ELFNPU37XX::CreateRelocationSectionOp relocSection;
    ELFNPU37XX::SymbolOp sourceSym;

    // VpuBarrierCountConfig relocs
    auto barriersCountConfigOps = func.getOps<vpux::VPUMI37XX::ConfigureBarrierOp>();
    for (auto barriersCountConfig : barriersCountConfigOps) {
        targetSection = relocationManager.getSection(barriersCountConfig.getResult());

        auto rtSymTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
        auto relocSectionSymTab = relocationManager.getRelocSection(targetSection, rtSymTab);
        auto builderRtSymTab = mlir::OpBuilder::atBlockEnd(relocSectionSymTab.getBlock());

        sourceSym =
                elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_BARRIERS_START)];
        builderRtSymTab.create<ELFNPU37XX::RelocOp>(barriersCountConfig.getLoc(), barriersCountConfig.getResult(),
                                                    offsetof(nn_public::VpuBarrierCountConfig, real_id_),
                                                    vpux::ELFNPU37XX::RelocationType::R_VPU_16_SUM, sourceSym, 0);
    }
}

void ConvertVPUMI37XX2ELFPass::createDPURelocs(mlir::func::FuncOp func) {
    auto invariants = func.getOps<VPUMI37XX::DPUInvariantOp>();

    ELFNPU37XX::ElfSectionInterface targetSection;
    ELFNPU37XX::CreateSymbolTableSectionOp symTabOfInput;
    ELFNPU37XX::CreateRelocationSectionOp relocSection;
    ELFNPU37XX::SymbolOp sourceSym;
    VPURT::DeclareBufferOp declarator;

    ELFNPU37XX::SymbolOp weightTableStartSym;
    uint64_t weightTableStartAddend = 0;

    vpux::IndexedSymbolAttr bufferMemSpace;
    std::optional<VPURT::BufferSection> bufferSection;

    // TODO: E#54007 currently ignoring sparsity and SOH/SOK.
    for (auto invariant : invariants) {
        int64_t addend = 0;

        auto opType = invariant.getNceTaskType();

        auto result = invariant.getIndex();
        targetSection = relocationManager.getSection(result);

        // inputs resolution
        auto input = invariant.getInput();
        declarator = mlir::cast<VPURT::DeclareBufferOp>(input.getDefiningOp());

        symTabOfInput =
                declarator.getSection() == VPURT::BufferSection::CMX_NN
                        ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                        : relocationManager.getSymTab(input);

        relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

        addend = declarator.getByteOffset();

        bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(input.getType()).getMemSpace();
        bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
        VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

        if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
            sourceSym = elfCMXMappingSyms[static_cast<int>(
                    vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
            // if we DO NOT have segmentation we relocate to Start_of_CMX + (sectionIdx * SLICE_LENGHT) + local_offset
            // if we DO have segmentation we relocate to start_of_CMX + local_offset. Distributed buffers always
            // assume we start from cluster 0
            if (!invariant.getIsSegmented() || invariant.getNceTaskType() == VPUIP::NCETaskType::ELTWISE) {
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            }
        } else {
            sourceSym = relocationManager.getSymbol(targetSection);
        }

        auto regsOffset = offsetof(nn_public::VpuDPUInvariant, registers_);

        // Input relocations, relocating act_offset registers
        builder.create<ELFNPU37XX::RelocOp>(input.getLoc(), invariant,
                                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[0]),
                                            ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                                            "Input (act_offset[0]) in DPU Invariant reloc");

        // barriers mask relocation
        {
            auto rtSymTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
            auto relocSectionRtSymTab = relocationManager.getRelocSection(targetSection, rtSymTab);
            auto builderRtSymTab = mlir::OpBuilder::atBlockEnd(relocSectionRtSymTab.getBlock());

            auto barriersSym = elfCMXMappingSyms[static_cast<int>(
                    vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_BARRIERS_START)];

            builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                    invariant.getLoc(), invariant, offsetof(struct nn_public::VpuDPUInvariant, barriers_.wait_mask_),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT, barriersSym, 0);

            builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                    invariant.getLoc(), invariant, offsetof(struct nn_public::VpuDPUInvariant, barriers_.post_mask_),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT, barriersSym, 0);
        }

        // For dense_se=0 (i.e. explicit SE table), the act_offset registers act as a base address over which the
        // SE pointers offsets are added. As such, they have to correspond to the address where the data is found
        // in each cluster
        if (invariant.getInputStorageElementTable() != nullptr) {
            addend += NNCMX_SLICE_SIZE;
        }
        builder.create<ELFNPU37XX::RelocOp>(input.getLoc(), invariant,
                                            regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[1]),
                                            ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                                            "Input (act_offset[1]) in DPU invariant registers reloc");

        bool isSegmented = invariant.getIsSegmented();

        if (auto inputSparsityMap = invariant.getInputSparsityMap()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(inputSparsityMap.getDefiningOp());

            symTabOfInput =
                    declarator.getSection() == VPURT::BufferSection::CMX_NN
                            ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                            : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.getByteOffset();

            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(inputSparsityMap.getType()).getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

            if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            bool isEltwise = invariant.getNceTaskType() == VPUIP::NCETaskType::ELTWISE;

            auto secIdx = bufferMemSpace.getIndex().value_or(0);
            addend += (isSegmented && !isEltwise) ? 0 : secIdx * NNCMX_SLICE_SIZE;

            builder.create<ELFNPU37XX::RelocOp>(
                    invariant.getInputSparsityMap().getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr) + sizeof(uint32_t),
                    ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                    "Input sparsity map (sparsity_addr in se_sp_addr[0]) in DPU invariant registers for input sparsity "
                    "map reloc");

            if (isEltwise) {
                auto elop_addend = addend;
                if (auto tensorBSparsityMapVal = invariant.getWeightsSparsityMap()) {
                    auto tensorBSparsity = mlir::cast<VPURT::DeclareBufferOp>(tensorBSparsityMapVal.getDefiningOp());
                    elop_addend = tensorBSparsity.getByteOffset();
                }
                builder.create<ELFNPU37XX::RelocOp>(
                        invariant.getInputSparsityMap().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, elop_sparsity_addr),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, elop_addend,
                        "Sparsity map (elop_sparsity_addr, for ELTWISE) in DPU invariant registers for input sparsity "
                        "map reloc");
            } else if (isSegmented) {
                addend += NNCMX_SLICE_SIZE;
                builder.create<ELFNPU37XX::RelocOp>(
                        invariant.getInputSparsityMap().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr[1]) + sizeof(uint32_t),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                        "Segmented sparsity map (sparsity_addr in se_sp_addr[1]) (for ELTWISE) in DPU invariant "
                        "registers for input sparsity map reloc");
            }
        }

        if (auto inputSETable = invariant.getInputStorageElementTable()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(inputSETable.getDefiningOp());

            symTabOfInput =
                    declarator.getSection() == VPURT::BufferSection::CMX_NN
                            ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                            : relocationManager.getSymTab(input);

            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.getByteOffset();

            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(inputSETable.getType()).getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

            if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            auto secIdx = bufferMemSpace.getIndex().value_or(0);
            addend += isSegmented ? 0 : secIdx * NNCMX_SLICE_SIZE;
            builder.create<ELFNPU37XX::RelocOp>(
                    invariant.getInputStorageElementTable().getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr),
                    ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                    "Input se table (se_sp_addr[0]) in DPU invariant registers for input storage element table reloc");

            if (isSegmented) {
                addend += NNCMX_SLICE_SIZE;
                builder.create<ELFNPU37XX::RelocOp>(
                        invariant.getInputStorageElementTable().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, se_sp_addr[1]),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                        "Segmented input se table (se_sp_addr[1]) in DPU invariant registers for input storage element "
                        "table reloc");
            }
        }

        // weights
        if (auto weights = invariant.getWeights()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp());

            symTabOfInput =
                    declarator.getSection() == VPURT::BufferSection::CMX_NN
                            ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                            : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            // for weights, we only need to set the start of CMX as base offset. Actual slice_start based offsets of
            // actual weighs are in the weights table
            addend = 0;

            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(weights.getType()).getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

            if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            if (opType != VPUIP::NCETaskType::ELTWISE) {
                builder.create<ELFNPU37XX::RelocOp>(
                        weights.getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                        "Weights (wt_offset, for ELTWISE) in DPU invariant registers reloc");
            } else {
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                auto weightsOffs = mlir::cast<VPURT::DeclareBufferOp>(weights.getDefiningOp()).getByteOffset() +
                                   (secIdx * NNCMX_SLICE_SIZE);

                auto actSecIdx = mlir::cast<vpux::NDTypeInterface>(invariant.getInput().getType())
                                         .getMemSpace()
                                         .getIndex()
                                         .value_or(0);
                auto actOffs =
                        mlir::cast<VPURT::DeclareBufferOp>(invariant.getInput().getDefiningOp()).getByteOffset() +
                        (actSecIdx * NNCMX_SLICE_SIZE);

                // correlated with serializer, where rest of the offsets are expected to be directly filled, in
                // accordance with this if-then-else
                builder.create<ELFNPU37XX::RelocOp>(
                        invariant.getInput().getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, act_offset[0]),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, std::min(actOffs, weightsOffs),
                        "Weights (act_offset[0]) in DPU invariant registers reloc");
            }
        }

        // no output in case of continued convolution
        if (!invariant.getIsContinued()) {
            const auto outputBuffs = invariant.getOutputBuffs();
            auto minOutIt =
                    std::min_element(outputBuffs.begin(), outputBuffs.end(), [](mlir::Value lhs, mlir::Value rhs) {
                        auto lhsSliceIdx =
                                mlir::cast<vpux::NDTypeInterface>(lhs.getType()).getMemSpace().getIndex().value_or(0);
                        auto rhsSliceIdx =
                                mlir::cast<vpux::NDTypeInterface>(rhs.getType()).getMemSpace().getIndex().value_or(0);
                        return lhsSliceIdx < rhsSliceIdx;
                    });
            auto output = *minOutIt;

            declarator = mlir::cast<VPURT::DeclareBufferOp>(output.getDefiningOp());

            symTabOfInput =
                    declarator.getSection() == VPURT::BufferSection::CMX_NN
                            ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                            : relocationManager.getSymTab(input);

            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = declarator.getByteOffset();

            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(output.getType()).getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

            if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            // in case of output the BASE_ADR_0 register is the only one we need to statically relocate. The actual
            // register that the DPU will get is ODU_ACT_BASE. By current contract with shaveNN, it will read BASE_ADR_0
            // reg and write it in ODU_ACT_BASE.  In case of Broadcasting the CAST registers will be configured, each
            // with having a fix offset from the base. The offset will be SLICE_SIZE. This will assume symmetrical
            // allocation and contigious slices. For this logic we will always configure the address of smallest slice
            // index
            static constexpr uint32_t base0Offset = offsetof(nn_public::VpuDPUInvariantRegisters, base_adr[0]);
            static constexpr uint32_t base1Offset = offsetof(nn_public::VpuDPUInvariantRegisters, base_adr[1]);

            builder.create<ELFNPU37XX::RelocOp>(output.getLoc(), invariant, regsOffset + base0Offset,
                                                ELFNPU37XX::RelocationType::R_VPU_32_MULTICAST_BASE, sourceSym, addend,
                                                "Base (base_adr[0]) offset in DPU reloc");
            builder.create<ELFNPU37XX::RelocOp>(output.getLoc(), invariant, regsOffset + base1Offset,
                                                ELFNPU37XX::RelocationType::R_VPU_32_MULTICAST_BASE, sourceSym, addend,
                                                "Base (base_adr[1]) offset in DPU reloc");

            if (auto outputSparsityMap = invariant.getOutputSparsityMapBuff()) {
                declarator = mlir::cast<VPURT::DeclareBufferOp>(outputSparsityMap.getDefiningOp());

                symTabOfInput = declarator.getSection() == VPURT::BufferSection::CMX_NN
                                        ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(
                                                  CMXMappingSymtabValue.getDefiningOp())
                                        : relocationManager.getSymTab(input);

                relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

                builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                addend = declarator.getByteOffset();

                bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(output.getType()).getMemSpace();
                bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
                VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

                if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                    sourceSym = elfCMXMappingSyms[static_cast<int>(
                            vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                    auto secIdx = bufferMemSpace.getIndex().value_or(0);
                    addend += secIdx * NNCMX_SLICE_SIZE;
                } else {
                    sourceSym = relocationManager.getSymbol(targetSection);
                }

                builder.create<ELFNPU37XX::RelocOp>(
                        output.getLoc(), invariant, regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, sp_base),
                        ELFNPU37XX::RelocationType::R_VPU_32_MULTICAST_BASE, sourceSym, addend,
                        "Output sparsity map (sp_base) in DPU invariant registers reloc");
            }
        }

        // weights table
        if (auto weightTable = invariant.getWeightTable()) {
            declarator = mlir::cast<VPURT::DeclareBufferOp>(weightTable.getDefiningOp());

            symTabOfInput =
                    declarator.getSection() == VPURT::BufferSection::CMX_NN
                            ? mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp())
                            : relocationManager.getSymTab(input);
            relocSection = relocationManager.getRelocSection(targetSection, symTabOfInput);

            builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            addend = 0;

            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(weightTable.getType()).getMemSpace();
            bufferSection = VPURT::symbolizeBufferSection(bufferMemSpace.getLeafName());
            VPUX_THROW_UNLESS(bufferSection.has_value(), "Buffer with no section associated");

            if (bufferSection.value() == VPURT::BufferSection::CMX_NN) {
                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                auto secIdx = bufferMemSpace.getIndex().value_or(0);
                addend += secIdx * NNCMX_SLICE_SIZE;
            } else {
                sourceSym = relocationManager.getSymbol(targetSection);
            }

            // wt_offset needs to be set even if there is no weights operand in MAXPOOL or AVEPOOL
            if (!invariant.getWeights() &&
                (opType == VPUIP::NCETaskType::MAXPOOL || opType == VPUIP::NCETaskType::AVEPOOL)) {
                builder.create<ELFNPU37XX::RelocOp>(
                        weightTable.getLoc(), invariant,
                        regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, wt_offset),
                        ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                        "Weights table (wt_offset, for max pool and ave pool) in DPU reloc");
            }

            addend += declarator.getByteOffset();
            weightTableStartSym = sourceSym;
            weightTableStartAddend = addend;
            builder.create<ELFNPU37XX::RelocOp>(
                    weightTable.getLoc(), invariant,
                    regsOffset + offsetof(nn_public::VpuDPUInvariantRegisters, weight_start),
                    ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, addend,
                    "Weights table (weight_start) in DPU invariant registers reloc");
        }

        // variant to invariant relocation
        auto children = invariant.getResult().getUsers();
        for (auto child : children) {
            auto variant = mlir::dyn_cast<VPUMI37XX::DPUVariantOp>(child);
            if (variant == nullptr) {
                continue;
            }

            auto invariantVal = variant.getInvariant();

            auto targetSection = relocationManager.getSection(variant.getResult());

            auto relocSection = relocationManager.getRelocSection(
                    targetSection,
                    mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp()));

            auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

            sourceSym = elfCMXMappingSyms[static_cast<int>(ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_RTM_IVAR)];

            builder.create<ELFNPU37XX::RelocOp>(
                    invariant.getLoc(), variant, variant.getOffsetOfWithinOperation(invariantVal),
                    ELFNPU37XX::RelocationType::R_VPU_32_RTM, sourceSym, sizeof(nn_public::VpuDPUInvariant),
                    "Variant to invariant (invariantVal) in DPU reloc");

            if (invariant.getWeightTable()) {
                builder.create<ELFNPU37XX::RelocOp>(
                        variant.getLoc(), variant, offsetof(nn_public::VpuDPUVariant, weight_table_offset_),
                        ELFNPU37XX::RelocationType::R_VPU_32_SUM, weightTableStartSym, weightTableStartAddend,
                        "Invariant weight table (weight_table_offset_) in DPU reloc");
            }

            // in case of invariant, the input drives the specific cluster dispatching. We control this based on the
            // variant's cluster_ field .
            bufferMemSpace = mlir::cast<vpux::NDTypeInterface>(invariant.getInput().getType()).getMemSpace();
            int64_t clusterIdx = bufferMemSpace.getIndex().value_or(0);

            sourceSym = elfCMXMappingSyms[static_cast<int>(ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_FIFO_BASE)];

            // zero variant cluster_ index also should be relocated to TILE1 in case of multitile
            builder.create<ELFNPU37XX::RelocOp>(
                    variant.getLoc(), variant, offsetof(nn_public::VpuDPUVariant, cluster_),
                    ELFNPU37XX::RelocationType::R_VPU_32, sourceSym, clusterIdx,
                    "Variant cluster (at idx " + std::to_string(clusterIdx) + ") in DPU reloc");
        }
    }

    return;
}

void ConvertVPUMI37XX2ELFPass::createDMARelocs(mlir::func::FuncOp& funcOp, mlir::MLIRContext* ctx,
                                               mlir::SmallVector<int64_t>& dmaCount,
                                               mlir::Operation::operand_range dmaTasks,
                                               mlir::SmallVector<ELFNPU37XX::CreateSectionOp>& dmaSections) {
    ELFNPU37XX::ElfSectionInterface targetSection;
    ELFNPU37XX::CreateSymbolTableSectionOp symTab;
    ELFNPU37XX::CreateRelocationSectionOp relocSection;
    ELFNPU37XX::SymbolOp sourceSym;

    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));

    for (auto listHead : dmaTasks) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getPort();
        auto listElemCount = dmaCount[listIdx];

        ELFNPU37XX::CreateRelocationSectionOp createInputRelocationSectionOp = builderFunc.create<
                ELFNPU37XX::CreateRelocationSectionOp>(
                mlir::UnknownLoc::get(ctx),
                vpux::ELFNPU37XX::SectionType::get(ctx),        // mlir::Type
                ".rlt.DMA_NetInput" + std::to_string(listIdx),  // llvm::StringRef secName,
                networkInputSymTabValue,                        // sourceSymbolTableSection,
                dmaSections[listIdx].getResult(),               // targetSection,
                vpux::ELFNPU37XX::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_JIT |
                        vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_USERINPUT  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                               // secFlags,
        );

        mlir::Region& regInputRelocSec = createInputRelocationSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkInputRelocSec = new mlir::Block();

        regInputRelocSec.push_back(blkInputRelocSec);

        mlir::OpBuilder builderInputRelocSec(blkInputRelocSec, blkInputRelocSec->begin());

        ELFNPU37XX::CreateRelocationSectionOp createOutputRelocationSectionOp = builderFunc.create<
                ELFNPU37XX::CreateRelocationSectionOp>(
                mlir::UnknownLoc::get(ctx),
                vpux::ELFNPU37XX::SectionType::get(ctx),         // mlir::Type
                ".rlt.DMA_NetOutput" + std::to_string(listIdx),  // llvm::StringRef secName,
                networkOutputSymTabValue,                        // sourceSymbolTableSection,
                dmaSections[listIdx].getResult(),                // targetSection,
                vpux::ELFNPU37XX::SectionFlagsAttr::SHF_INFO_LINK | vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_JIT |
                        vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_USEROUTPUT  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                                // secFlags,
        );
        mlir::Region& regOutputRelocSec = createOutputRelocationSectionOp.getOperation()->getRegion(0);
        mlir::Block* blkOutputRelocSec = new mlir::Block();
        regOutputRelocSec.push_back(blkOutputRelocSec);

        mlir::OpBuilder builderOutputRelocSec(blkOutputRelocSec, blkOutputRelocSec->begin());

        mlir::OpBuilder builderProfOutputRelocSec(ctx);
        if (profOutputSymTabValue) {
            ELFNPU37XX::CreateRelocationSectionOp createProfOutputRelocationSectionOp =
                    builderFunc.create<ELFNPU37XX::CreateRelocationSectionOp>(
                            mlir::UnknownLoc::get(ctx),
                            vpux::ELFNPU37XX::SectionType::get(ctx),          // mlir::Type
                            ".rlt.DMA_ProfOutput" + std::to_string(listIdx),  // llvm::StringRef secName,
                            profOutputSymTabValue,                            // sourceSymbolTableSection,
                            dmaSections[listIdx].getResult(),                 // targetSection,
                            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_INFO_LINK |
                                    vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_JIT |
                                    vpux::ELFNPU37XX::SectionFlagsAttr::
                                            VPU_SHF_PROFOUTPUT  // vpux::ELFNPU37XX::SectionFlagsAttr
                                                                // secFlags,
                    );
            mlir::Region& regProfOutputRelocSec = createProfOutputRelocationSectionOp.getOperation()->getRegion(0);
            mlir::Block* blkProfOutputRelocSec = new mlir::Block();
            regProfOutputRelocSec.push_back(blkProfOutputRelocSec);

            builderProfOutputRelocSec = mlir::OpBuilder(blkProfOutputRelocSec, blkProfOutputRelocSec->begin());
        }

        targetSection = dmaSections[listIdx];
        auto cmxRelocationSection = relocationManager.getRelocSection(targetSection, relocationManager.getCMXSymTab());
        auto cmxBuilder = mlir::OpBuilder::atBlockEnd(cmxRelocationSection.getBlock());
        auto ddrRelocationSection = relocationManager.getRelocSection(targetSection, ddrSymbolTable);
        auto ddrBuilder = mlir::OpBuilder::atBlockEnd(ddrRelocationSection.getBlock());

        // Need to go through individual DMA task lists
        for (auto dmaIdx = 0; dmaIdx < listElemCount; ++dmaIdx) {
            auto dmaOp = mlir::dyn_cast_or_null<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp());
            // input addr
            if (auto dmaInputArg = mlir::dyn_cast<mlir::BlockArgument>(dmaOp.getInput())) {
                auto offset = offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src);
                createBlockArgReloc(dmaOp, builderInputRelocSec, builderOutputRelocSec, offset,
                                    vpux::ELFNPU37XX::RelocationType::R_VPU_64, dmaInputArg);
            } else {
                auto dmaInputArg_ = dmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>();
                if (dmaInputArg_ && (dmaInputArg_.getMemorySpace() == VPURT::BufferSection::NetworkInput)) {
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaInputArg_.getSectionIndex().value());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with input {} which has multiple section indexes {}",
                                      dmaOp, dmaInputArg_, funcArgIndex);
                    auto inputOffset = dmaInputArg_.getByteOffset();
                    auto funcArg = funcOp.getArgument(funcArgIndex[0]);
                    if (mlir::Value netInputSymValue = lookupELFSymbol(networkInputSymTabValue, funcArg)) {
                        builderInputRelocSec.create<ELFNPU37XX::RelocOp>(
                                builderInputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64,  // relocationType
                                netInputSymValue,                            // ::mlir::Value sourceSymbol
                                inputOffset,                                 // int64_t addend
                                dmaOp.getOperationName().str() + " input from network input dma reloc");
                    }
                } else if (dmaInputArg_ && (dmaInputArg_.getMemorySpace() == VPURT::BufferSection::NetworkOutput)) {
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaInputArg_.getSectionIndex().value());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with input {} which has multiple section indexes {}",
                                      dmaOp, dmaInputArg_, funcArgIndex);
                    auto inputOffset = dmaInputArg_.getByteOffset();
                    auto funcArg =
                            funcOp.getArgument(funcArgIndex[0] + funcOp.getNumArguments() - funcOp.getNumResults());
                    if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, funcArg)) {
                        builderOutputRelocSec.create<ELFNPU37XX::RelocOp>(
                                builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, src),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64,  // relocationType
                                netOutputSymValue,                           // ::mlir::Value sourceSymbol
                                inputOffset,                                 // int64_t addend
                                dmaOp.getOperationName().str() + " input from network output dma reloc");
                    }
                } else {
                    auto dmaInput = dmaOp.getInput();
                    auto inputMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaInput.getType()).getMemSpace();
                    std::optional<VPURT::BufferSection> bufferSection;
                    if (inputMemSpace) {
                        bufferSection = VPURT::symbolizeBufferSection(inputMemSpace.getLeafName());
                        VPUX_THROW_WHEN(!bufferSection.has_value(),
                                        "Unrecognizable memory space {0} of input {1} DMA {2}", inputMemSpace, dmaInput,
                                        dmaOp);
                    }

                    if (bufferSection == VPURT::BufferSection::CMX_NN) {
                        auto sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                        auto addend = ELFNPU37XX::getOffsetOfOpInSection(dmaInput);
                        cmxBuilder.create<ELFNPU37XX::RelocOp>(
                                dmaInput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaInput),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64, sourceSym, addend,
                                dmaOp.getOperationName().str() + " input from cmx dma reloc");
                    } else if (bufferSection == VPURT::BufferSection::Constant ||
                               mlir::isa<Const::DeclareOp>(dmaInput.getDefiningOp())) {
                        auto sourceSym = symbolMap["sym_constSection"];
                        auto addend = ELFNPU37XX::getOffsetOfOpInSection(dmaInput, constSectionOp, offsetCache);
                        ddrBuilder.create<ELFNPU37XX::RelocOp>(
                                dmaInput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaInput),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64, sourceSym, addend,
                                dmaOp.getOperationName().str() + " input from constant dma reloc");
                    } else if (bufferSection == VPURT::BufferSection::DDR) {
                        auto sourceSym = symbolMap["sym_bufferSection"];
                        auto addend = ELFNPU37XX::getOffsetOfOpInSection(dmaInput, scratchBufferSectionOp, offsetCache);
                        ddrBuilder.create<ELFNPU37XX::RelocOp>(
                                dmaInput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaInput),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64, sourceSym, addend,
                                dmaOp.getOperationName().str() + " input from DDR dma reloc");
                    } else {
                        auto symTab = relocationManager.getSymTab(dmaInput);
                        auto relocSection = relocationManager.getRelocSection(targetSection, symTab);
                        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());
                        ELFNPU37XX::SymbolOp sourceSym;
                        size_t addend = 0;
                        if (bufferSection == VPURT::BufferSection::Register) {
                            sourceSym = elfCMXMappingSyms[static_cast<int>(
                                    vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_HW_REGISTER)];
                            addend = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(dmaInput.getDefiningOp())
                                             .getByteOffset();
                        } else {
                            auto dmaInputSection = relocationManager.getSection(dmaInput);
                            sourceSym = ELFNPU37XX::RelocationManager::getSymbol(dmaInputSection);
                            mlir::Value dmaInputSectionValue = dmaInputSection.getOperation()->getResult(0);
                            addend = ELFNPU37XX::getOffsetOfOpInSection(dmaInput, dmaInputSectionValue, offsetCache);
                        }
                        builder.create<ELFNPU37XX::RelocOp>(
                                dmaInput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaInput),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64, sourceSym, addend,
                                dmaOp.getOperationName().str() + " input dma reloc");
                    }
                }
            }

            // output addr
            auto outputBuffs = dmaOp.getOutputBuffs();

            if (auto dmaOutputArg = mlir::dyn_cast<mlir::BlockArgument>(outputBuffs[0])) {
                VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as blockArgument with multiple outputs");
                auto offset = offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst);
                createBlockArgReloc(dmaOp, builderInputRelocSec, builderOutputRelocSec, offset,
                                    vpux::ELFNPU37XX::RelocationType::R_VPU_64, dmaOutputArg);
            } else {
                auto dmaOutputArg_ = outputBuffs[0].getDefiningOp<VPURT::DeclareBufferOp>();
                VPUX_THROW_UNLESS(dmaOutputArg_,
                                  "Encountered DMA op {} with output {} which is neither mlir::BlockArgument, nor "
                                  "VPURT::DeclareBufferOp",
                                  dmaOp, dmaOutputArg_);

                if (dmaOutputArg_.getMemorySpace() == VPURT::BufferSection::NetworkOutput) {
                    VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as NetworkOut with multiple outputs");
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaOutputArg_.getSectionIndex().value());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with output {} which has multiple secion indexes {}",
                                      dmaOp, dmaOutputArg_, funcArgIndex);
                    auto outputOffset = dmaOutputArg_.getByteOffset();
                    auto funcArg =
                            funcOp.getArgument(funcArgIndex[0] + funcOp.getNumArguments() - funcOp.getNumResults());
                    if (mlir::Value netOutputSymValue = lookupELFSymbol(networkOutputSymTabValue, funcArg)) {
                        builderOutputRelocSec.create<ELFNPU37XX::RelocOp>(
                                builderOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64,  // relocationType
                                netOutputSymValue,                           // ::mlir::Value sourceSymbol
                                outputOffset,                                // int64_t addend
                                dmaOp.getOperationName().str() + " output from network output dma reloc");
                    }
                } else if (dmaOutputArg_.getMemorySpace() == VPURT::BufferSection::ProfilingOutput) {
                    VPUX_THROW_WHEN(outputBuffs.size() != 1, "have first arg as NetworkOut with multiple outputs");
                    auto funcArgIndex = parseIntArrayAttr<int64_t>(dmaOutputArg_.getSectionIndex().value());
                    VPUX_THROW_UNLESS(funcArgIndex.size() == 1,
                                      "Encountered DMA op {} with output {} which has multiple secion indexes {}",
                                      dmaOp, dmaOutputArg_, funcArgIndex);
                    VPUX_THROW_UNLESS(funcArgIndex[0] == 0, "Only profiling output index 0 is supported, got '{0}'",
                                      funcArgIndex[0]);
                    auto outputOffset = dmaOutputArg_.getByteOffset();
                    auto funcArg = funcOp.getArgument(funcOp.getNumArguments() - 1);
                    if (mlir::Value profOutputSymValue = lookupELFSymbol(profOutputSymTabValue, funcArg)) {
                        builderProfOutputRelocSec.create<ELFNPU37XX::RelocOp>(
                                builderProfOutputRelocSec.getUnknownLoc(), dmaOp.getResult(),
                                offsetof(nn_public::VpuDMATask, transaction_) + offsetof(vpu_dma_descriptor_t, dst),
                                vpux::ELFNPU37XX::RelocationType::R_VPU_64,  // relocationType
                                profOutputSymValue,                          // ::mlir::Value sourceSymbol
                                outputOffset,                                // int64_t addend
                                dmaOp.getOperationName().str() + " output from profiling dma reloc");
                    }
                } else {
                    auto dmaOutput = outputBuffs[0];
                    auto outputMemSpace = mlir::cast<vpux::NDTypeInterface>(dmaOutput.getType()).getMemSpace();
                    std::optional<VPURT::BufferSection> bufferSection;
                    if (outputMemSpace) {
                        bufferSection = VPURT::symbolizeBufferSection(outputMemSpace.getLeafName());
                        VPUX_THROW_WHEN(!bufferSection.has_value(),
                                        "Unrecognizable memory space {0} of output {1} DMA {2}", outputMemSpace,
                                        dmaOutput, dmaOp);
                    }
                    // in case of broadcast output using OR relocation. The DST will have the default MASK value for
                    // multicast;
                    auto relocType = outputBuffs.size() > 1 ? vpux::ELFNPU37XX::RelocationType::R_VPU_64_OR
                                                            : vpux::ELFNPU37XX::RelocationType::R_VPU_64;

                    if (bufferSection == VPURT::BufferSection::CMX_NN) {
                        auto sourceSym = elfCMXMappingSyms[static_cast<int>(
                                vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR)];
                        auto addend = ELFNPU37XX::getOffsetOfOpInSection(dmaOutput);
                        cmxBuilder.create<ELFNPU37XX::RelocOp>(
                                dmaOutput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaOutput), relocType,
                                sourceSym, addend, dmaOp.getOperationName().str() + " output from CMX dma reloc");
                    } else if (bufferSection == VPURT::BufferSection::DDR) {
                        auto sourceSym = symbolMap["sym_bufferSection"];
                        auto addend =
                                ELFNPU37XX::getOffsetOfOpInSection(dmaOutput, scratchBufferSectionOp, offsetCache);
                        ddrBuilder.create<ELFNPU37XX::RelocOp>(
                                dmaOutput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaOutput), relocType,
                                sourceSym, addend, dmaOp.getOperationName().str() + " output from DDR dma reloc");
                    } else {
                        auto symTab = relocationManager.getSymTab(dmaOutput);
                        auto relocSection = relocationManager.getRelocSection(targetSection, symTab);
                        auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());
                        ELFNPU37XX::SymbolOp sourceSym;
                        size_t addend = 0;
                        if (bufferSection == VPURT::BufferSection::Register) {
                            sourceSym = elfCMXMappingSyms[static_cast<int>(
                                    vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_HW_REGISTER)];
                            addend = mlir::dyn_cast<vpux::VPURT::DeclareBufferOp>(dmaOutput.getDefiningOp())
                                             .getByteOffset();
                        } else {
                            auto dmaOutputSection = relocationManager.getSection(dmaOutput);
                            auto dmaOutputSectionValue = dmaOutputSection.getOperation()->getResult(0);
                            sourceSym = ELFNPU37XX::RelocationManager::getSymbol(dmaOutputSection);
                            addend = ELFNPU37XX::getOffsetOfOpInSection(dmaOutput, dmaOutputSectionValue, offsetCache);
                        }
                        builder.create<ELFNPU37XX::RelocOp>(
                                dmaOutput.getLoc(), dmaOp, dmaOp.getOffsetOfWithinOperation(dmaOutput), relocType,
                                sourceSym, addend, dmaOp.getOperationName().str() + " output dma reloc");
                    }
                }
            }

            // link_address
            if (static_cast<uint32_t>(listElemCount) > dmaOp.getType().getValue() + 1) {
                cmxBuilder.create<ELFNPU37XX::RelocOp>(
                        cmxRelocationSection.getLoc(), dmaOp, offsetof(nn_public::VpuDMATask, transaction_),
                        vpux::ELFNPU37XX::RelocationType::R_VPU_32_RTM,
                        elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_RTM_DMA0) +
                                          listIdx]
                                .getResult(),
                        sizeof(nn_public::VpuDMATask), dmaOp.getOperationName().str() + " link address dma reloc");
            }

            // barriers masks
            {
                symTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
                relocSection = relocationManager.getRelocSection(targetSection, symTab);
                auto builder = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

                sourceSym = elfCMXMappingSyms[static_cast<int>(
                        vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_BARRIERS_START)];
                builder.create<ELFNPU37XX::RelocOp>(relocSection.getLoc(), dmaOp,
                                                    offsetof(nn_public::VpuDMATask, transaction_.barriers.prod_mask),
                                                    vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT, sourceSym, 0);

                builder.create<ELFNPU37XX::RelocOp>(relocSection.getLoc(), dmaOp,
                                                    offsetof(nn_public::VpuDMATask, transaction_.barriers.cons_mask),
                                                    vpux::ELFNPU37XX::RelocationType::R_VPU_64_LSHIFT, sourceSym, 0);
            }

            listHead = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getNextDMAIdx();
        }
    }
}

void ConvertVPUMI37XX2ELFPass::safeRunOnModule() {
    mlir::MLIRContext* ctx = &(getContext());
    mlir::func::FuncOp funcOp;
    mlir::ModuleOp moduleOp = getOperation();

    ShaveBinaryResources::loadElfData(moduleOp);

    _log.trace("ConvertVPUMI37XX2ELFPass::safeRunOnFunc(): START\n {0}\n", moduleOp);

    net::NetworkInfoOp netInfo;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, funcOp);

    relocationManager.init(funcOp);

    // We use this constructor: OpBuilder(Operation *op, Listener *listener=nullptr)
    mlir::OpBuilder builderFunc(&(funcOp.getBody().front().back()));
    auto mappedInferenceOps = funcOp.getOps<VPUMI37XX::MappedInferenceOp>();
    VPUX_THROW_UNLESS(!mappedInferenceOps.empty(), "MappedInferenceOp could not be located.");
    auto mappedInferenceOp = *(mappedInferenceOps.begin());

    auto dmaCount = parseIntArrayAttr<int64_t>(mappedInferenceOp.getDmaCount());
    auto barrierCount = mappedInferenceOp.getBarrierCount();
    auto rangeCount = mappedInferenceOp.getActKernelRangesCount();
    auto invoCount = mappedInferenceOp.getActKernelInvocationsCount();
    auto invariantCount = mappedInferenceOp.getInvariantCount();
    auto variantCount = mappedInferenceOp.getVariantCount();

    auto dmaTasks = mappedInferenceOp.getDmaTasks();
    auto barrierTasks = mappedInferenceOp.getBarrierTasks();
    auto actKernelRanges = mappedInferenceOp.getActKernelRanges();
    auto actKernelInvocations = mappedInferenceOp.getActKernelInvocations();
    auto invariantTasks = mappedInferenceOp.getInvariantTasks();
    auto variantTasks = mappedInferenceOp.getVariantTasks();

    auto trivialIndexType = VPURegMapped::IndexType::get(ctx, 0);

    //
    // Sections Creation
    //

    auto dmaSectionOps = createDMASections(funcOp, ctx, dmaCount, dmaTasks);

    auto barrierSectionOp = createSection<vpux::VPUMI37XX::ConfigureBarrierOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.BarrierConfigs", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto kernelTextSectionOp = createSection<vpux::VPUMI37XX::DeclareKernelTextOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.KernelText", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE);

    auto kernelDataSectionOp = createSection<vpux::VPUMI37XX::DeclareKernelArgsOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.KernelData", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_WRITE);

    auto kernelParamsSectionOp = createSection<vpux::VPUMI37XX::KernelParamsOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.KernelParams", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto actKernelRangesSectionOp = createSection<vpux::VPUMI37XX::ActKernelRangeOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.ActKernelRanges", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto actKernelInvosSectionOp = createSection<vpux::VPUMI37XX::ActKernelInvocationOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.ActKernelInvocations", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    mappedInferenceSectionOp = createSection<vpux::VPUMI37XX::MappedInferenceOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.MappedInference", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto invariantsSectionOp = createSection<vpux::VPUMI37XX::DPUInvariantOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.DPUInvariants", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto variantsSectionOp = createSection<vpux::VPUMI37XX::DPUVariantOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".text.DPUVariants", vpux::ELFNPU37XX::SectionTypeAttr::SHT_PROGBITS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto metadataSectionOp = builderFunc.create<ELFNPU37XX::CreateMetadataSectionOp>(
            builderFunc.getUnknownLoc(),
            vpux::ELFNPU37XX::SectionType::get(ctx),       // mlir::Type
            ".metadata",                                   // llvm::StringRef secName,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE,  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                      // int64_t secInfo,
            vpux::VPUMI37XX::NetworkMetadataOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto performanceMetricsOp = createSection<vpux::VPUMI37XX::PerformanceMetricsOp, ELFNPU37XX::CreateSectionOp>(
            funcOp, ctx, ".perf.metrics", vpux::ELFNPU37XX::SectionTypeAttr::VPU_SHT_PERF_METRICS,
            vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE);

    auto builderPerfSec = mlir::OpBuilder::atBlockEnd(performanceMetricsOp.getBlock());
    builderPerfSec.create<VPUMI37XX::PerformanceMetricsOp>(mlir::UnknownLoc::get(ctx), trivialIndexType);

    auto builderMetadataSec = mlir::OpBuilder::atBlockEnd(metadataSectionOp.getBlock());
    builderMetadataSec.create<VPUMI37XX::NetworkMetadataOp>(mlir::UnknownLoc::get(ctx), trivialIndexType);

    //
    // The following sections are created as additional versioning & platform information that needs to be
    // serialized to the ELF
    //

    //
    // ELF ABI Version
    //
    auto elfABINoteSectionOp = builderFunc.create<ELFNPU37XX::CreateSectionOp>(
            builderFunc.getUnknownLoc(),
            ELFNPU37XX::SectionType::get(ctx),                    // mlir::Type
            ".note.LoaderABIVersion",                             // llvm::StringRef secName,
            ELFNPU37XX::SectionTypeAttr::SHT_NOTE,                // ELFNPU37XX::SectionTypeAttr secType,
            ELFNPU37XX::SectionFlagsAttr::SHF_NONE,               // ELFNPU37XX::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                             // int64_t secInfo,
            ELFNPU37XX::ABIVersionOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto builderABINoteSec = mlir::OpBuilder::atBlockEnd(elfABINoteSectionOp.getBlock());
    builderABINoteSec.create<ELFNPU37XX::ABIVersionOp>(mlir::UnknownLoc::get(ctx));

    //
    // Mapped Inference Version
    //
    auto MINoteSectionOp = builderFunc.create<ELFNPU37XX::CreateSectionOp>(
            builderFunc.getUnknownLoc(),
            ELFNPU37XX::SectionType::get(ctx),                               // mlir::Type
            ".note.MappedInferenceVersion",                                  // llvm::StringRef secName,
            ELFNPU37XX::SectionTypeAttr::SHT_NOTE,                           // ELFNPU37XX::SectionTypeAttr secType,
            ELFNPU37XX::SectionFlagsAttr::SHF_NONE,                          // ELFNPU37XX::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                                        // int64_t secInfo,
            VPUMI37XX::MappedInferenceVersionOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto builderMINoteSec = mlir::OpBuilder::atBlockEnd(MINoteSectionOp.getBlock());
    builderMINoteSec.create<VPUMI37XX::MappedInferenceVersionOp>(builderMINoteSec.getUnknownLoc());

    //
    // Platform Information
    //
    auto platformInfoSectionOp = builderFunc.create<ELFNPU37XX::CreateSectionOp>(
            builderFunc.getUnknownLoc(),
            ELFNPU37XX::SectionType::get(ctx),                     // mlir::Type
            ".meta.PlatformInfo",                                  // llvm::StringRef secName,
            ELFNPU37XX::SectionTypeAttr::VPU_SHT_PLATFORM_INFO,    // ELFNPU37XX::SectionTypeAttr secType,
            ELFNPU37XX::SectionFlagsAttr::SHF_NONE,                // ELFNPU37XX::SectionFlagsAttr secFlags,
            elf::VPU_SH_INFO_FOR_VPU,                              // int64_t secInfo,
            VPUMI37XX::PlatformInfoOp::getAlignmentRequirements()  // int64_t secAddrAlign
    );

    auto builderPlatformInfoSec = mlir::OpBuilder::atBlockEnd(platformInfoSectionOp.getBlock());
    builderPlatformInfoSec.create<VPUMI37XX::PlatformInfoOp>(builderPlatformInfoSec.getUnknownLoc());

    if (!netInfo.getProfilingOutputsInfo().empty()) {
        auto profilingSectionOp = builderFunc.create<ELFNPU37XX::CreateProfilingSectionOp>(
                builderFunc.getUnknownLoc(),
                vpux::ELFNPU37XX::SectionType::get(ctx),       // mlir::Type
                ".profiling",                                  // llvm::StringRef secName,
                vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE,  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
                elf::VPU_SH_INFO_FOR_VPU,                      // int64_t secInfo,
                vpux::VPUMI37XX::ProfilingMetadataOp::getAlignmentRequirements()  // int64_t secAddrAlign
        );

        SmallVector<mlir::Operation*> profOps(funcOp.getOps<VPUMI37XX::ProfilingMetadataOp>());
        VPUX_THROW_UNLESS(profOps.size() == 1, "Found {0} VPUMI37XX::ProfilingMetadataOps, but should be exact 1",
                          profOps.size());
        // Move ProfilingMetadataOp from funcOp to CreateProfilingSectionOp
        auto* block = profilingSectionOp.getBlock();
        profOps.front()->moveBefore(block, block->begin());
    }

    _log.trace("ConvertVPUMI37XX2ELFPass, after sections creation:\n {0} \n", moduleOp);

    //
    // Create Symbols for the relevant sections
    //

    vpux::ELFNPU37XX::SymbolTypeEnumAttr typeSym;
    mlir::IntegerAttr sizeSym;
    mlir::IntegerAttr valueSym;
    mlir::UnitAttr isBuiltin = nullptr;
    mlir::SmallVector<vpux::ELFNPU37XX::SymbolOp> dmaSectionSyms;

    // dmaCount size is always > 0, even when having 0 DMA operations in the network,
    // but in that case dmaSectionOps size will be 0. Check this to avoid out-of-bounds access.
    const bool doesNetworkHaveDMAOperations = !dmaSectionOps.empty();
    if (doesNetworkHaveDMAOperations) {
        for (size_t listIdx = 0; listIdx < dmaCount.size(); ++listIdx) {
            dmaSectionSyms.push_back(builderFunc.create<ELFNPU37XX::SymbolOp>(
                    mlir::UnknownLoc::get(ctx),
                    vpux::ELFNPU37XX::SymbolType::get(ctx),                                  // mlir::Type
                    dmaSectionOps[listIdx].getResult(),                                      // mlir::Value inputArg
                    isBuiltin,                                                               // mlir::UnitAttr
                    mlir::StringAttr::get(ctx, "sym_dmaSection" + std::to_string(listIdx)),  // mlir::StringAttr
                    typeSym,  // vpux::ELFNPU37XX::SymbolTypeEnumAttr
                    sizeSym,  // size
                    valueSym  // value
                    ));

            symbolMap["sym_dmaSection" + std::to_string(listIdx)] = dmaSectionSyms[listIdx].getResult();
        }
    }

    auto barrierSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),            // mlir::Type
            barrierSectionOp.getResult(),                      // mlir::Value inputArg
            isBuiltin,                                         // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_barrierSection"),  // mlir::StringAttr
            typeSym,                                           // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                           // size
            valueSym                                           // value
    );

    symbolMap["sym_barrierSection"] = barrierSectionSym.getResult();

    auto actKernelRangeSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),                   // mlir::Type
            actKernelRangesSectionOp.getResult(),                     // mlir::Value inputArg
            isBuiltin,                                                // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_actKernelRangeSection"),  // mlir::StringAttr
            typeSym,                                                  // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                                  // size
            valueSym                                                  // value
    );

    symbolMap["sym_actKernelRangeSection"] = actKernelRangeSectionSym.getResult();

    auto actKernelInvoSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),           // mlir::Type
            actKernelInvosSectionOp.getResult(),              // mlir::Value inputArg
            isBuiltin,                                        // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_actKernelInvo"),  // mlir::StringAttr
            typeSym,                                          // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                          // size
            valueSym                                          // value
    );

    symbolMap["sym_actKernelInvo"] = actKernelInvoSectionSym.getResult();

    auto kernelTextSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),               // mlir::Type
            kernelTextSectionOp.getResult(),                      // mlir::Value inputArg
            isBuiltin,                                            // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_kernelTextSection"),  // mlir::StringAttr
            typeSym,                                              // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                              // size
            valueSym                                              // value
    );

    symbolMap["sym_kernelTextSection"] = kernelTextSectionSym.getResult();

    auto kernelDataSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),               // mlir::Type
            kernelDataSectionOp.getResult(),                      // mlir::Value inputArg
            isBuiltin,                                            // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_kernelDataSection"),  // mlir::StringAttr
            typeSym,                                              // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                              // size
            valueSym                                              // value
    );

    symbolMap["sym_kernelDataSection"] = kernelDataSectionSym.getResult();

    auto kernelParamsSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SymbolType::get(ctx),                 // mlir::Type
            kernelParamsSectionOp.getResult(),                      // mlir::Value inputArg
            isBuiltin,                                              // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "sym_kernelParamsSection"),  // mlir::StringAttr
            typeSym,                                                // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                                // size
            valueSym                                                // value
    );

    symbolMap["sym_kernelParamsSection"] = kernelParamsSectionSym.getResult();

    auto inVariantsSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx), vpux::ELFNPU37XX::SymbolType::get(ctx), invariantsSectionOp, isBuiltin,
            mlir::StringAttr::get(ctx, "sym_inVariantsSection"), typeSym, sizeSym, valueSym);

    symbolMap["sym_inVariantsSection"] = inVariantsSectionSym.getResult();

    auto variantsSectionSym = builderFunc.create<ELFNPU37XX::SymbolOp>(
            mlir::UnknownLoc::get(ctx), vpux::ELFNPU37XX::SymbolType::get(ctx), variantsSectionOp, isBuiltin,
            mlir::StringAttr::get(ctx, "sym_variantsSection"), typeSym, sizeSym, valueSym);

    symbolMap["sym_variantsSection"] = variantsSectionSym.getResult();

    //
    // Creation of SymTabs
    //

    createNetworkIOSymtab(funcOp, ctx, netInfo);
    bufferSymTabValue = createBuffersSecAndSymtab(funcOp, ctx);
    CMXMappingSymtabValue = createCMXMappingSymtab(funcOp, ctx);

    ELFNPU37XX::CreateSymbolTableSectionOp CMXMappingSymtabOp =
            mlir::dyn_cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
    relocationManager.initCMXSymTab(CMXMappingSymtabOp);

    tasksSymbolTable = builderFunc.create<ELFNPU37XX::CreateSymbolTableSectionOp>(
            mlir::UnknownLoc::get(ctx),
            vpux::ELFNPU37XX::SectionType::get(ctx),      // mlir::Type
            mlir::StringAttr::get(ctx, ".symtab.tasks"),  // mlir::StringAttr secName,
            vpux::ELFNPU37XX::SectionFlagsAttrAttr::get(
                    ctx,
                    vpux::ELFNPU37XX::SectionFlagsAttr::SHF_NONE),  // vpux::ELFNPU37XX::SectionFlagsAttr secFlags,
            isBuiltin                                               // mlir::UnitAttr
    );

    mlir::Region& regTasksSymTabOp = tasksSymbolTable.getOperation()->getRegion(0);
    mlir::Block* blkTasksSymTabOp = new mlir::Block();
    regTasksSymTabOp.push_back(blkTasksSymTabOp);
    mlir::OpBuilder builderTasksSymTab(blkTasksSymTabOp, blkTasksSymTabOp->begin());

    for (auto listHead : dmaTasks) {
        auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getPort();
        builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                                dmaSectionSyms[listIdx].getResult());
    }
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            barrierSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            kernelTextSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            kernelDataSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            kernelParamsSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            actKernelRangeSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            actKernelInvoSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            inVariantsSectionSym.getResult());
    builderTasksSymTab.create<ELFNPU37XX::PutOpInSectionOp>(builderTasksSymTab.getUnknownLoc(),
                                                            variantsSectionSym.getResult());

    auto vpux_entry_SymbolTypeEnumAttr =
            vpux::ELFNPU37XX::SymbolTypeEnumAttr::get(ctx, vpux::ELFNPU37XX::SymbolTypeEnum::VPU_STT_ENTRY);
    auto mappedInferenceSym = builderTasksSymTab.create<ELFNPU37XX::SymbolOp>(
            builderTasksSymTab.getUnknownLoc(),
            vpux::ELFNPU37XX::SymbolType::get(ctx),               // mlir::Type
            mappedInferenceOp.getResult(),                        // mlir::Value inputArg
            isBuiltin,                                            // mlir::UnitAttr
            mlir::StringAttr::get(ctx, "MappedInference_entry"),  // mlir::StringAttr
            vpux_entry_SymbolTypeEnumAttr,                        // vpux::ELFNPU37XX::SymbolTypeEnumAttr
            sizeSym,                                              // size
            valueSym                                              // value
    );

    symbolMap["MappedInference_entry"] = mappedInferenceSym.getResult();

    _log.trace("ConvertVPUMI37XX2ELFPass, after symtabs creation:\n {0} \n", moduleOp);

    //
    // create general relocs for the tasks
    //

    createDMARelocs(funcOp, ctx, dmaCount, dmaTasks, dmaSectionOps);
    _log.trace("ConvertVPUMI37XX2ELFPass, after DMA Relocs creation:\n {0} \n", moduleOp);

    bool shaveScratchAccess = false;
    bool shaveConstAccess = false;
    createKernelParamsRelocs(funcOp, shaveScratchAccess, shaveConstAccess);
    if (shaveScratchAccess) {
        VPUX_THROW_UNLESS(scratchBufferSectionOp != nullptr, "Scratch buffer section result is null");
        auto currFlagsAttrVal = scratchBufferSectionOp.getSecFlags();
        currFlagsAttrVal = currFlagsAttrVal | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
        scratchBufferSectionOp.setSecFlagsAttr(ELFNPU37XX::SectionFlagsAttrAttr::get(ctx, currFlagsAttrVal));
    }

    if (shaveConstAccess) {
        VPUX_THROW_UNLESS(constSectionOp != nullptr, "Const buffer section result is null");
        auto currFlagsAttrVal = constSectionOp.getSecFlags();
        currFlagsAttrVal = currFlagsAttrVal | ELFNPU37XX::SectionFlagsAttr::VPU_SHF_PROC_SHAVE;
        constSectionOp.setSecFlagsAttr(ELFNPU37XX::SectionFlagsAttrAttr::get(ctx, currFlagsAttrVal));
    }

    createActKernelRelocs(funcOp, actKernelRangesSectionOp, kernelTextSectionOp, actKernelInvosSectionOp,
                          kernelDataSectionOp, kernelParamsSectionOp);
    setupActKernelRtConfigs(funcOp, moduleOp, ctx);
    _log.trace("ConvertVPUMI37XX2ELFPass, after Shave Relocs creation:\n {0} \n", moduleOp);

    createDPURelocs(funcOp);
    _log.trace("ConvertVPUMI37XX2ELFPass, after ActKernel Relocs creation:\n {0} \n", moduleOp);

    //
    // create relocs for the tasks in MappedInference
    //

    auto relocSection = relocationManager.getRelocSection(mappedInferenceSectionOp, tasksSymbolTable);
    auto builderMappedInfRelocSec = mlir::OpBuilder::atBlockEnd(relocSection.getBlock());

    // Refresh range after mapped inference was updated
    // There are 3 posssible cases
    // Case1: 2 DPU 2 DMA - both dma_lists are ocupied, nothing to add
    // Case2: 2 DPU 1 DMA - On this case, both tiles are using a single DMA engine.
    //                    - We'll create relocations that will populate the dma_tasks[0] and clear out dma_tasks[1]
    // Case3: 1 DPU 1 DMA - Each tile will use it's own DMA engine.
    //                    - For this case we'll create relocations that will enable only one of the dma_tasks lists.
    //                    - using tile0 dma_tasks[0] is initialized, dma_tasks[1] is cleared
    //                    - using tile1 dma_tasks[0] is cleared, dma_tasks[1] is initialized
    //                    - the initialization is tied to VPU_NNRD_SYM_FIFO_BASE (possible values: 0x0 for tile0, 0x1
    //                    for tile1, 0x0 for both tiles)
    dmaTasks = mappedInferenceOp.getDmaTasks();

    if (dmaTasks.size() != 1) {
        // Case1
        for (auto listHead : dmaTasks) {
            auto listIdx = mlir::cast<VPUMI37XX::NNDMAOp>(listHead.getDefiningOp()).getPort();
            builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(

                    builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                    mappedInferenceOp.getOffsetOfWithinOperation(listHead), vpux::ELFNPU37XX::RelocationType::R_VPU_64,
                    dmaSectionSyms[listIdx].getResult(), 0, "listHead in mapped inference reloc");
        }
    } else {
        // Case2/Case3
        // If we got here it means that the configuration requested a single tile blob
        // We need to update also the barrier index. For single tile use case
        // there are only 32 barriers used (0 to 32 for tile0 and 32 to 64 tile1).
        createBarriersConfigRelocs(funcOp);
        _log.trace("ConvertVPUMI37XX2ELFPass, after BarriersConfig Relocs creation:\n {0} \n", moduleOp);
        createAKISingleTileConfigRelocs(funcOp, actKernelInvosSectionOp);

        // Addind a set of relocations for the same fields. That is because the relocation infrastructure don't allow to
        // use multiple symbols for the same relocation.
        for (size_t i = 0; i != nn_public::VPU_MAX_DMA_ENGINES; i++) {
            // First stage:
            // Set the pointer to DMA tasks for each engine.
            // This is a required step as we don't have a clean possibility to select on wich engine would run.
            builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                    builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                    offsetof(nn_public::VpuMappedInference, dma_tasks) +
                            sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>) * i +
                            offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address),
                    vpux::ELFNPU37XX::RelocationType::R_VPU_64, dmaSectionSyms[0].getResult(), 0);

            auto rtSymTab = mlir::cast<ELFNPU37XX::CreateSymbolTableSectionOp>(CMXMappingSymtabValue.getDefiningOp());
            auto relocSectionSymTab = relocationManager.getRelocSection(mappedInferenceSectionOp, rtSymTab);
            auto builderRtSymTab = mlir::OpBuilder::atBlockEnd(relocSectionSymTab.getBlock());

            auto sourceSym =
                    elfCMXMappingSyms[static_cast<int>(vpux::ELFNPU37XX::CMXMappingSymbol::VPU_NNRD_SYM_FIFO_BASE)];

            // Second stage:
            // Add a relocation for the same DMA task to "select" the engine.
            // The deciding factor is VPU_NNRD_SYM_FIFO_BASE symbol that can have the values 0x0, 0x1, 0x0.
            // Basedd on this symbol we multiply the field with 0 or 1 (tile index).
            // for tile 0, dmalist active : 0, so formula is addr[0] *= 1, addr[1] *= 0
            auto operation = i == 0 ? vpux::ELFNPU37XX::RelocationType::R_VPU_64_MULT_SUB
                                    : vpux::ELFNPU37XX::RelocationType::R_VPU_64_MULT;
            auto addend = i == 0 ? 1 : 0;

            builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                    builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                    offsetof(nn_public::VpuMappedInference, dma_tasks) +
                            sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>) * i +
                            offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, address),
                    operation, sourceSym, addend);

            // same with task length
            builderRtSymTab.create<ELFNPU37XX::RelocOp>(
                    builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                    offsetof(nn_public::VpuMappedInference, dma_tasks) +
                            sizeof(nn_public::VpuTaskReference<nn_public::VpuDMATask>) * i +
                            offsetof(nn_public::VpuTaskReference<nn_public::VpuDMATask>, count),
                    operation, sourceSym, addend);
        }
    }

    if (barrierCount > 0) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                mappedInferenceOp.getOffsetOfWithinOperation(barrierTasks), vpux::ELFNPU37XX::RelocationType::R_VPU_64,
                barrierSectionSym.getResult(), 0, "barrierTasks in mapped inference reloc");
    }

    if (rangeCount > 0) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                mappedInferenceOp.getOffsetOfWithinOperation(actKernelRanges),
                vpux::ELFNPU37XX::RelocationType::R_VPU_64, actKernelRangeSectionSym.getResult(), 0,
                "actKernelRanges in mapped inference reloc");
    }

    if (invoCount > 0) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                mappedInferenceOp.getOffsetOfWithinOperation(actKernelInvocations),
                vpux::ELFNPU37XX::RelocationType::R_VPU_64, actKernelInvoSectionSym.getResult(), 0,
                "actKernelInvocations in mapped inference reloc");
    }

    if (invariantCount > 0) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                mappedInferenceOp.getOffsetOfWithinOperation(invariantTasks),
                vpux::ELFNPU37XX::RelocationType::R_VPU_64, inVariantsSectionSym.getResult(), 0,
                "invariantTasks in mapped inference reloc");
    }

    if (variantCount > 0) {
        builderMappedInfRelocSec.create<ELFNPU37XX::RelocOp>(
                builderMappedInfRelocSec.getUnknownLoc(), mappedInferenceOp.getResult(),
                mappedInferenceOp.getOffsetOfWithinOperation(variantTasks), vpux::ELFNPU37XX::RelocationType::R_VPU_64,
                variantsSectionSym.getResult(), 0, "variantTasks in mapped inference reloc");
    }

    _log.trace("Convert2VPUIPRegMappedAndELFPass::safeRunOnFunc(): FINISH\n {0}\n", moduleOp);
}
}  // namespace

//
// createConvertVPUMI37XX2ELFPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUMI37XX2ELFPass(Logger log) {
    return std::make_unique<ConvertVPUMI37XX2ELFPass>(log);
}
