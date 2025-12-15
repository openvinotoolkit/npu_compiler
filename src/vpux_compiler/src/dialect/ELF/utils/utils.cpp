//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/ELF/utils/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>

ArrayRef<uint8_t> vpux::ELF::getDataAndSizeOfElfSection(ArrayRef<uint8_t> elfBlob,
                                                        ArrayRef<StringRef> possibleSecNames) {
    auto accessor = elf::DDRAccessManager<elf::DDRAlwaysEmplace>(elfBlob.data(), elfBlob.size());
    auto elfReader = elf::Reader<elf::ELF_Bitness::Elf32>(&accessor);

    const uint8_t* secData = nullptr;
    uint32_t secSize = 0;

    bool secFound = false;

    for (size_t i = 0; i < elfReader.getSectionsNum(); ++i) {
        const auto& section = elfReader.getSection(i);
        const auto secName = section.getName();
        const auto sectionHeader = section.getHeader();

        for (auto& possibleSecName : possibleSecNames) {
            if (strcmp(secName, possibleSecName.data()) == 0) {
                secSize = sectionHeader->sh_size;
                secData = section.getData<uint8_t>();
                secFound = true;
                break;
            }
        }
    }
    VPUX_THROW_UNLESS(secFound, "Section {0} not found in ELF", possibleSecNames);

    return {secData, secSize};
}

size_t vpux::ELF::math::gcd(size_t a, size_t b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

size_t vpux::ELF::math::lcm(size_t a, size_t b) {
    return (a / vpux::ELF::math::gcd(a, b)) * b;
}

size_t vpux::ELF::getOffsetOfSymRef(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef) {
    auto referencedOp = symRefMap.lookupSymbol(symRef);
    auto binarySizeOp = mlir::dyn_cast<ELF::BinarySizeOpInterface>(referencedOp);

    VPUX_THROW_UNLESS(binarySizeOp, "The relocInfo can't be retrieved for a non-binaryOpIf type reference");

    return binarySizeOp.getMemoryOffset();
}

vpux::ELF::MainOp vpux::ELF::getElfMainOp(mlir::ModuleOp moduleOp) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp netFunc;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, netFunc);
    return getElfMainOp(netFunc);
}

vpux::ELF::MainOp vpux::ELF::getElfMainOp(mlir::func::FuncOp funcOp) {
    auto mainOps = to_small_vector(funcOp.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    return mainOps[0];
}

ArrayRef<uint8_t> vpux::ELF::getKernelELF(mlir::Operation* operation, StringRef kernelPath,
                                          ArrayRef<StringRef> sectionNames) {
    const auto& kernelInfo = ShaveBinaryResourcesCache::getCache(operation->getContext());
    const auto archKind = config::getArch(operation);
    const auto arch = ShaveBinaryResources::getSwKernelArchString(archKind);
    llvm::ArrayRef<uint8_t> elfBlob = kernelInfo.getElf(kernelPath, arch);

    return sectionNames.empty() ? elfBlob : vpux::ELF::getDataAndSizeOfElfSection(elfBlob, sectionNames);
}

mlir::SymbolRefAttr vpux::ELF::composeSectionObjectSymRef(ELF::ElfSectionInterface sectionIface, mlir::Operation* op) {
    auto sectionSymIface = mlir::cast<mlir::SymbolOpInterface>(sectionIface.getOperation());
    auto opSymIface = mlir::cast<mlir::SymbolOpInterface>(op);

    auto opRef = mlir::FlatSymbolRefAttr::get(opSymIface.getNameAttr());
    return mlir::SymbolRefAttr::get(sectionSymIface.getNameAttr(), {opRef});
}

//
// SymbolReferenceMap
//

mlir::Operation* ELF::SymbolReferenceMap::lookupSymbol(mlir::SymbolRefAttr symRef) {
    // convention for symbols naming: @section_name::@symbol_name
    // @section_name can be retrieved via SymbolRefAttr::getRootReference()
    // and @symbol_name from SymbolRefAttr::getLeafReference()
    // however there are some exceptions from this rule, having a @symbol_name directly under ElfMainOp

    VPUX_THROW_UNLESS(symRef, "Symbol reference is null");
    auto symbolRoot = symRef.getRootReference();
    auto symbolLeaf = symRef.getLeafReference();
    auto sectionSymbolContainerIt = _sectionSymbolContainers.find(symbolRoot);
    if (sectionSymbolContainerIt == _sectionSymbolContainers.end()) {
        auto symbolRootOp = _elfMainSymbolTable.lookup(symbolRoot);
        VPUX_THROW_UNLESS(symbolRootOp, "Symbol {0} not found under elfMain", symbolRoot.str());

        if (!symbolRootOp->hasTrait<mlir::OpTrait::SymbolContainer>() || symbolRoot == symbolLeaf) {
            return symbolRootOp;
        }

        auto insertRes = _sectionSymbolContainers.insert({symbolRoot, mlir::SymbolTable(symbolRootOp)});
        sectionSymbolContainerIt = insertRes.first;
    }

    auto symbolOp = sectionSymbolContainerIt->second.lookup(symbolLeaf);
    VPUX_THROW_UNLESS(symbolOp, "No op found for symbol {0}::{1}", symbolRoot, symbolLeaf);

    return symbolOp;
}

void ELF::SymbolReferenceMap::walkAllSymbols() {
    auto elfOp = _elfMainSymbolTable.getOp();

    for (mlir::Region& region : elfOp->getRegions()) {
        for (mlir::Block& block : region) {
            for (mlir::Operation& nestedOp : block) {
                if (nestedOp.hasTrait<mlir::OpTrait::SymbolContainer>()) {
                    auto symbol = mlir::cast<mlir::SymbolOpInterface>(&nestedOp);
                    auto insertRes =
                            _sectionSymbolContainers.insert({symbol.getNameAttr(), mlir::SymbolTable(&nestedOp)});

                    VPUX_THROW_UNLESS(insertRes.second, "ElfMain expected to contain uniquely named symbols {0}",
                                      elfOp);
                }
            }
        }
    }
}

//
// Platform Information
//

namespace {

elf::platform::ArchKind mapArchKindToElfArchKind(config::ArchKind archKind) {
    switch (archKind) {
    case config::ArchKind::NPU37XX:
        VPUX_THROW("NPU3720 is not supported");  // in new backend
    case config::ArchKind::NPU40XX:
        return elf::platform::ArchKind::VPUX40XX;
    case config::ArchKind::NPU50XX:
        return elf::platform::ArchKind::VPUX501X;
    case config::ArchKind::UNKNOWN:
        break;
    }
    VPUX_THROW("Unknown architecture");
}

elf::platform::ArchKind mapPlatformToElfArchKind(config::Platform platform) {
    switch (platform) {
    case config::Platform::NPU3720:
        VPUX_THROW("NPU3720 is not supported");  // in new backend
    case config::Platform::NPU4000:
        return elf::platform::ArchKind::VPUX40XX;
    case config::Platform::NPU5000:
    case config::Platform::NPU5010:
        return elf::platform::ArchKind::VPUX501X;
    }
    VPUX_THROW("Invalid platform");
}

}  // namespace

elf::platform::ArchKind vpux::ELF::getElfArchKind(mlir::Operation* op) {
    auto platform = config::getPlatform(op);
    if (platform.has_value()) {
        return mapPlatformToElfArchKind(platform.value());
    } else {
        return mapArchKindToElfArchKind(config::getArch(op));
    }
}

std::pair<uint8_t, uint8_t> vpux::ELF::reduceWaitMaskTo8bit(uint64_t waitMask) {
    uint8_t barrier_group = 0;
    uint8_t barrier_mask = 0;
    for (uint64_t mask = waitMask, group = 1; mask > 0; mask >>= 8, ++group) {
        if (mask & 0xff) {
            if (barrier_group == 0) {
                barrier_group = static_cast<unsigned char>(group);
                barrier_mask = mask & 0xff;
            } else {
                barrier_group = 0;
                barrier_mask = 0;
                break;
            }
        }
    }
    return {barrier_group, barrier_mask};
}

mlir::MemRefType vpux::ELF::getLinearMemrefType(mlir::MLIRContext* ctx, int64_t memrefSize, mlir::Type dataType,
                                                VPU::MemoryKind memKind) {
    VPUX_THROW_UNLESS(dataType.isIntOrFloat(), "Data Type of the MemRef must be an Integer or Float Type");

    const auto memrefShape = SmallVector<int64_t>{memrefSize};
    auto memKindAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(memKind));
    const auto memKindSymbolAttr = vpux::IndexedSymbolAttr::get(ctx, memKindAttr);
    unsigned int perm[1] = {0};
    auto map = mlir::AffineMap::getPermutationMap(to_small_vector(perm), ctx);

    auto memrefType = mlir::MemRefType::get(memrefShape, dataType, map, memKindSymbolAttr);
    return memrefType;
}

mlir::SymbolRefAttr vpux::ELF::moveOpToSection(mlir::Operation* op, SectionMapper& sectionMap,
                                               mlir::OpBuilder& builder) {
    auto wrapOp = mlir::dyn_cast<ELF::WrappableOpInterface>(op);
    if (!wrapOp) {
        return {};
    }

    auto maybeSignature = wrapOp.getSectionSignature();
    if (!maybeSignature.has_value()) {
        return {};
    }

    const auto& signature = maybeSignature.value();

    auto createSection = [&](const ELF::SectionSignature& signature, bool memFootprint, size_t opAling) {
        if (memFootprint) {
            auto sec = builder.create<ELF::DataSectionOp>(builder.getUnknownLoc(),
                                                          signature.getName(),  // llvm::StringRef secName
                                                          opAling,              // int64_t secAddrAlign
                                                          signature.getType(),  // ELFVPUX40XX secType
                                                          signature.getFlags()  // ELFVPUX40XX secFlags
            );

            return mlir::cast<ELF::ElfSectionInterface>(sec.getOperation());
        } else {
            VPUX_THROW_UNLESS(mlir::dyn_cast<ELF::BufferLocationInterface>(op),
                              "Received op does not have type of BufferLocation!");
            auto sec = builder.create<ELF::LogicalSectionOp>(
                    builder.getUnknownLoc(),
                    signature.getName(),                                                 // llvm::StringRef secName
                    opAling,                                                             // int64_t secAddrAlign
                    signature.getType(),                                                 // ELFVPUX40XX secType
                    signature.getFlags(),                                                // ELFVPUX40XX secFlags
                    mlir::dyn_cast<ELF::BufferLocationInterface>(op).getMemorySection()  // section location
            );

            return mlir::cast<ELF::ElfSectionInterface>(sec.getOperation());
        }
    };

    auto sectionMapKey = sectionMap.find(signature);
    if (sectionMapKey != sectionMap.end()) {
        auto secInterface = sectionMapKey->second;

        auto sectionBlock = secInterface.getBlock();
        op->moveAfter(&sectionBlock->back());
    } else {
        auto hasMemFootprint = wrapOp.hasMemoryFootprint();
        auto secInterface = createSection(signature, hasMemFootprint, VPUX_DEFAULT_ALIGNMENT);
        mlir::Block* sectionBlock = secInterface.getBlock();
        op->moveBefore(sectionBlock, sectionBlock->end());

        sectionMap[signature] = secInterface;
    }

    auto symbolOp = mlir::cast<mlir::SymbolOpInterface>(op);
    auto symbolRef = mlir::FlatSymbolRefAttr::get(symbolOp.getNameAttr());
    auto symContainer = mlir::cast<mlir::SymbolOpInterface>(sectionMap[signature].getOperation());
    return mlir::SymbolRefAttr::get(symContainer.getNameAttr(), {symbolRef});
}

mlir::SymbolRefAttr vpux::ELF::cloneSectionSymbol(mlir::SymbolRefAttr from, mlir::SymbolRefAttr to) {
    auto symbolSection = from.getRootReference();
    auto symbolOpRef = mlir::FlatSymbolRefAttr::get(to.getRootReference());
    return mlir::SymbolRefAttr::get(symbolSection, {symbolOpRef});
}

void vpux::ELF::insertELFMain(mlir::func::FuncOp netFunc) {
    // create the main ELF op alongside the netFunc
    auto mainBuilder = mlir::OpBuilder(netFunc.getOperation());
    auto elf = mainBuilder.create<ELF::MainOp>(netFunc->getLoc(), "ELFMain");

    // take the body of the netFunc and put everything inside ELF Op, so we avoid clone of all OPS
    elf.getContent().takeBody(netFunc.getBody());
    auto netFuncBlock = netFunc.addEntryBlock();

    elf.getOperation()->moveBefore(netFuncBlock, netFuncBlock->end());

    // as we've moved the whole function we also moved the terminator
    auto terminator = elf.getContent().front().getTerminator();
    terminator->moveAfter(elf.getOperation());
}

size_t vpux::ELF::getOpBinarySize(vpux::NDTypeInterface type) {
    enum tensor_4d_index_t {
        TENSOR_4D_INDEX_N,
        TENSOR_4D_INDEX_C,
        TENSOR_4D_INDEX_H,
        TENSOR_4D_INDEX_W,

        TENSOR_4D_INDEX_COUNT,
    };

    // auto contentType = content.getType();
    auto memoryKind = type.getMemoryKind();
    auto memShape = to_small_vector(type.getShape());
    auto memStrides = to_small_vector(type.getStrides());
    auto dimsOrder = type.getDimsOrder();
    auto elementTypeSize = type.getElemTypeSize().count();
    auto elementTypeSizeInByte = (elementTypeSize >= CHAR_BIT) ? (elementTypeSize / CHAR_BIT) : (elementTypeSize);
    bool isNonZero4DTensor = false;

    if (memShape.size() == 4) {
        if (memShape[TENSOR_4D_INDEX_N] != 0 && memShape[TENSOR_4D_INDEX_C] != 0 && memShape[TENSOR_4D_INDEX_H] != 0 &&
            memShape[TENSOR_4D_INDEX_W] != 0) {
            isNonZero4DTensor = true;
        }
    }

    // When the highest non-unitary dim is padded, the dma using that declare buffer won't actually write data to the
    // padded region, so the strict span of the declare buffer (where data is actually being written) needs to use the
    // unstrided size in the highest non-unitary dimension.
    if (memoryKind == vpux::VPU::MemoryKind::DDR && isNonZero4DTensor && memShape.size() == memStrides.size() &&
        elementTypeSize >= CHAR_BIT) {
        if (dimsOrder == DimsOrder::NHWC) {
            if (memShape[TENSOR_4D_INDEX_N] > 1) {
                auto totalSize = memShape[TENSOR_4D_INDEX_N] *
                                 ((memStrides[TENSOR_4D_INDEX_N].count()) / (memStrides[TENSOR_4D_INDEX_H].count())) *
                                 ((memStrides[TENSOR_4D_INDEX_H].count()) / (memStrides[TENSOR_4D_INDEX_W].count())) *
                                 ((memStrides[TENSOR_4D_INDEX_W].count()) / (memStrides[TENSOR_4D_INDEX_C].count())) *
                                 elementTypeSizeInByte;
                return totalSize;
            } else {
                if (memShape[TENSOR_4D_INDEX_H] > 1) {
                    auto totalSize =
                            memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_H] *
                            ((memStrides[TENSOR_4D_INDEX_H].count()) / (memStrides[TENSOR_4D_INDEX_W].count())) *
                            ((memStrides[TENSOR_4D_INDEX_W].count()) / (memStrides[TENSOR_4D_INDEX_C].count())) *
                            elementTypeSizeInByte;
                    return totalSize;
                } else {
                    if (memShape[TENSOR_4D_INDEX_W] > 1) {
                        auto totalSize =
                                memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_H] *
                                memShape[TENSOR_4D_INDEX_W] *
                                ((memStrides[TENSOR_4D_INDEX_W].count()) / (memStrides[TENSOR_4D_INDEX_C].count())) *
                                elementTypeSizeInByte;
                        return totalSize;
                    } else {
                        auto totalSize = memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_H] *
                                         memShape[TENSOR_4D_INDEX_W] * memShape[TENSOR_4D_INDEX_C] *
                                         elementTypeSizeInByte;
                        return totalSize;
                    }
                }
            }
        }
        if (dimsOrder == DimsOrder::NCHW) {
            if (memShape[TENSOR_4D_INDEX_N] > 1) {
                auto totalSize = memShape[TENSOR_4D_INDEX_N] *
                                 ((memStrides[TENSOR_4D_INDEX_N].count()) / (memStrides[TENSOR_4D_INDEX_C].count())) *
                                 ((memStrides[TENSOR_4D_INDEX_C].count()) / (memStrides[TENSOR_4D_INDEX_H].count())) *
                                 ((memStrides[TENSOR_4D_INDEX_H].count()) / (memStrides[TENSOR_4D_INDEX_W].count())) *
                                 elementTypeSizeInByte;
                return totalSize;
            } else {
                if (memShape[TENSOR_4D_INDEX_C] > 1) {
                    auto totalSize =
                            memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_C] *
                            ((memStrides[TENSOR_4D_INDEX_C].count()) / (memStrides[TENSOR_4D_INDEX_H].count())) *
                            ((memStrides[TENSOR_4D_INDEX_H].count()) / (memStrides[TENSOR_4D_INDEX_W].count())) *
                            elementTypeSizeInByte;
                    return totalSize;
                } else {
                    if (memShape[TENSOR_4D_INDEX_H] > 1) {
                        auto totalSize =
                                memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_C] *
                                memShape[TENSOR_4D_INDEX_H] *
                                ((memStrides[TENSOR_4D_INDEX_H].count()) / (memStrides[TENSOR_4D_INDEX_W].count())) *
                                elementTypeSizeInByte;
                        return totalSize;
                    } else {
                        auto totalSize = memShape[TENSOR_4D_INDEX_N] * memShape[TENSOR_4D_INDEX_C] *
                                         memShape[TENSOR_4D_INDEX_H] * memShape[TENSOR_4D_INDEX_W] *
                                         elementTypeSizeInByte;
                        return totalSize;
                    }
                }
            }
        }
    }

    return type.getTotalAllocSize().count();
}
