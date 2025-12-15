//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

#include <stdint.h>

#include <api/vpu_dma_hw_40xx.h>
#include <gtest/gtest.h>
#include <malloc.h>
#include <random>
#include <vector>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/writer.hpp>
#include <vpux_loader/vpux_loader.hpp>

#include "common_test_utils/test_assertions.hpp"

using namespace elf;
using namespace writer;

namespace {

class HardCodedSymtabToCluster0 {
private:
    static constexpr size_t SPECIAL_SYMTAB_SIZE = 7;
    SymbolEntry symTab_[SPECIAL_SYMTAB_SIZE];

public:
    HardCodedSymtabToCluster0(): symTab_() {
        for (size_t i = 0; i < SPECIAL_SYMTAB_SIZE; ++i) {
            symTab_[i].st_info = static_cast<unsigned char>(elf64STInfo(STB_GLOBAL, STT_OBJECT));
            symTab_[i].st_other = STV_DEFAULT;
            symTab_[i].st_shndx = 0;
            symTab_[i].st_name = 0;
        }

        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_value = 0x2e014000;
        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_size = 2097152;

        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_value = 0x2e004000;
        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_ACT].st_value = 0;
        symTab_[VPU_NNRD_SYM_RTM_ACT].st_size = 0;

        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_value = 0x2e1f8000;
        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_value = 0x2e1fc000;
        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_size = 64;

        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_value = 0x0;
        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_size = 0;

        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_value = 0;
        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_size = 0;
    }

    const std::vector<SymbolEntry> symTab() const {
        return std::vector<SymbolEntry>(symTab_, symTab_ + SPECIAL_SYMTAB_SIZE);
    }
};

class DummyBufferManager : public BufferManager {
public:
    DeviceBuffer allocate(const BufferSpecs& buffSpecs) override {
        auto addr = malloc(buffSpecs.size);
        return {reinterpret_cast<uint8_t*>(addr), reinterpret_cast<uint64_t>(addr), buffSpecs.size};
    }

    void deallocate(DeviceBuffer& devBuffer) override {
        free(reinterpret_cast<void*>(devBuffer.cpu_addr()));
    }

    void lock(DeviceBuffer& devBuffer) override {
        (void)devBuffer;
    }

    void unlock(DeviceBuffer& devBuffer) override {
        (void)devBuffer;
    }

    size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) override {
        memcpy(to.cpu_addr(), from, count);
        return count;
    }
};

class NullAllocBufferManager : public BufferManager {
    DeviceBuffer allocate(const BufferSpecs& buffSpecs) override {
        (void)buffSpecs;
        return DeviceBuffer();
    }
    void deallocate(DeviceBuffer& devAddress) override {
        (void)devAddress;
    }

    void lock(DeviceBuffer& devAddress) override {
        (void)devAddress;
    }
    void unlock(DeviceBuffer& devAddress) override {
        (void)devAddress;
    }
    size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) override {
        (void)to;
        (void)from;
        (void)count;
        return 0;
    };
};

struct DummyBinObject {
    uint32_t a = 0;
    uint64_t b = 0;
};

uint32_t generateRandom(uint32_t from, uint32_t to) {
    std::mt19937 gen((std::random_device())());
    std::uniform_int_distribution<> dist(from, to);

    return dist(gen);
}

template <typename T>
BinaryDataSection<T>* generateDataSection(Writer& writer, const std::string& name, const Elf_Word type = SHT_PROGBITS) {
    auto binDataSection = writer.addBinaryDataSection<T>(name, type);
    auto iterCount = generateRandom(1, 128);
    binDataSection->setSize(iterCount * sizeof(T));
    return binDataSection;
}

template <typename T>
void populateDataSection(BinaryDataSection<T>* section) {
    for (uint32_t i = 0; i < section->getSize() / sizeof(DummyBinObject); i++) {
        section->appendData(DummyBinObject{});
    }
}

SymbolSection* generateSymbolSection(Writer& writer, const std::string& name) {
    auto symSection = writer.addSymbolSection(name);
    auto iterCount = generateRandom(1, 128);
    for (uint32_t i = 0; i < iterCount; i++) {
        auto symbol = symSection->addSymbolEntry(std::string("_symbol_") + std::to_string(i));
        symbol->setType(STT_SECTION);
        symbol->setSize(generateRandom(16, 32));
    }

    return symSection;
}

RelocationSection* generateRelocationSection(Writer& writer, const std::string& name, const Section* patchSection,
                                             SymbolSection* symSection) {
    auto relocSection = writer.addRelocationSection(name);
    relocSection->setSectionToPatch(patchSection);
    relocSection->setSymbolTable(symSection);

    return relocSection;
}

std::vector<uint8_t> generateBadUserIOElf() {
    Writer writer;

    auto binDataSection = generateDataSection<DummyBinObject>(writer, ".binData");
    auto symSection = generateSymbolSection(writer, ".symbols");
    symSection->setFlags(VPU_SHF_USERINPUT);

    for (auto i = 0; i < 2; i++) {
        auto userIOSection = generateSymbolSection(writer, std::string(".userIO.symbols") + std::to_string(i));
        userIOSection->setFlags(VPU_SHF_USERINPUT);
    }

    auto relocSection = generateRelocationSection(writer, ".reloc", binDataSection, symSection);

    auto reloc = relocSection->addRelocationEntry();
    reloc->setSymbol(&(*(
            symSection->getSymbols()[generateRandom(0, static_cast<uint32_t>(symSection->getSymbols().size()) - 1)])));
    reloc->setOffset(sizeof(DummyBinObject::a));
    reloc->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> elf(writer.getTotalSize());
    writer.generateELF(elf.data());
    writer.setSectionsStartAddr(elf.data());
    populateDataSection(binDataSection);
    return elf;
}

std::vector<uint8_t> generateBadSectionTypeElf() {
    Writer writer;

    auto binDataSection = generateDataSection<DummyBinObject>(writer, ".binData", SHT_LOUSER - 2);
    writer.prepareWriter();
    std::vector<uint8_t> elf(writer.getTotalSize());
    writer.generateELF(elf.data());
    writer.setSectionsStartAddr(elf.data());
    populateDataSection(binDataSection);
    return elf;
}

std::vector<uint8_t> generateUnrecognizedUserSectionTypeElf() {
    Writer writer;

    auto binDataSection = generateDataSection<DummyBinObject>(writer, ".binData", SHT_HIUSER - 1);
    writer.prepareWriter();
    std::vector<uint8_t> elf(writer.getTotalSize());
    writer.generateELF(elf.data());
    writer.setSectionsStartAddr(elf.data());
    populateDataSection(binDataSection);
    return elf;
}

std::vector<uint8_t> generateValidTestElf() {
    Writer writer;

    auto binDataSection = generateDataSection<DummyBinObject>(writer, ".binData");
    auto symSection = generateSymbolSection(writer, ".symbols");
    auto relocSection = generateRelocationSection(writer, ".reloc", binDataSection, symSection);
    auto reloc = relocSection->addRelocationEntry();
    auto relocSymbol =
            symSection->getSymbols()[generateRandom(0, static_cast<uint32_t>(symSection->getSymbols().size()) - 1)]
                    .get();
    relocSymbol->setRelatedSection(binDataSection);
    reloc->setSymbol(relocSymbol);
    reloc->setOffset(sizeof(DummyBinObject::a));
    reloc->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> elf(writer.getTotalSize());
    writer.generateELF(elf.data());
    writer.setSectionsStartAddr(elf.data());
    return elf;
}

std::vector<uint8_t> generateValidElfWithDmaRelocations(DmaSymbolEntry& symbolEntry) {
    Writer writer;

    auto dmaProgBitsSection = generateDataSection<DmaDescriptor>(writer, ".dmaTask");
    auto dmaSymSection = writer.addDmaSymbolSection(".dmaSymbols");
    dmaSymSection->setFlags(VPU_SHF_USERINPUT | VPU_SHF_JIT);
    auto dmaSymbol = dmaSymSection->addDmaSymbolEntry();
    dmaSymbol->setDmaSymbol(symbolEntry);

    auto relocSection = writer.addRelocationSection(".dmaReloc");
    relocSection->setSectionToPatch(dmaProgBitsSection);
    relocSection->setSpecialSymbolTable(dmaSymSection->getIndex());
    relocSection->setFlags(VPU_SHF_USERINPUT | VPU_SHF_JIT);
    auto inputReloc = relocSection->addRelocationEntry();
    inputReloc->setSpecialSymbol(dmaSymbol->getIndex());
    inputReloc->setType(R_VPU_DMA_TASK_INPUT);
    inputReloc->setOffset(0);
    inputReloc->setAddend(0);

    auto outputReloc = relocSection->addRelocationEntry();
    outputReloc->setSpecialSymbol(dmaSymbol->getIndex());
    outputReloc->setType(R_VPU_DMA_TASK_OUTPUT);
    outputReloc->setOffset(0);
    outputReloc->setAddend(0);

    writer.prepareWriter();
    std::vector<uint8_t> elf(writer.getTotalSize());
    writer.generateELF(elf.data());
    writer.setSectionsStartAddr(elf.data());
    return elf;
}

const HardCodedSymtabToCluster0 gSymTab;

TEST(ELFLoader, ThrowWhenAccessorPointerIsNull) {
    DummyBufferManager bufMgr;

    ASSERT_THROW(VPUXLoader(nullptr, &bufMgr), ArgsError);
}

TEST(ELFLoader, ThrowWhenElfHeaderIsInvalid) {
    ELFHeader elf{};
    elf.e_ident[0] = elf::ELFMAG0;
    elf.e_ident[1] = elf::EI_MAG1;
    elf.e_ident[2] = elf::EI_MAG2;
    elf.e_ident[3] = 'X';
    DummyBufferManager bufMgr;

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(&elf), sizeof(ELFHeader));

    ASSERT_THROW(VPUXLoader(&accessor, &bufMgr), HeaderError);
}

TEST(ELFLoader, ThrowWhenBufferManagerIsNull) {
    std::vector<uint8_t> elf;

    OV_ASSERT_NO_THROW(elf = generateValidTestElf());

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader(&accessor, nullptr), ArgsError);
}

TEST(ELFLoader, ThrowWhenAllocFails) {
    NullAllocBufferManager bufMgr;
    std::vector<uint8_t> elf;

    ASSERT_THROW(AllocatedDeviceBuffer(nullptr, {0, 0, 0}), ArgsError);
}

TEST(ELFLoader, ThrowWhenBadUserIO) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    OV_ASSERT_NO_THROW(elf = generateBadUserIOElf());

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader loader(&accessor, &bufMgr), SequenceError);
}

TEST(ELFLoader, ThrowWhenBadSectionType) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    OV_ASSERT_NO_THROW(elf = generateBadSectionTypeElf());

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());

    ASSERT_THROW(VPUXLoader(&accessor, &bufMgr), ImplausibleState);
}

TEST(ELFLoader, NoThrowWhenUnrecognizedUserSectionType) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    OV_ASSERT_NO_THROW(elf = generateUnrecognizedUserSectionTypeElf());

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    VPUXLoader loader(&accessor, &bufMgr);
    OV_ASSERT_NO_THROW(loader.load(gSymTab.symTab()));
}

TEST(ELFLoader, NoThrowWhenValidElf) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    OV_ASSERT_NO_THROW(elf = generateValidTestElf());

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    OV_ASSERT_NO_THROW(VPUXLoader(&accessor, &bufMgr));
    VPUXLoader loader(&accessor, &bufMgr);
    OV_ASSERT_NO_THROW(loader.load(gSymTab.symTab(), false, {}, false));
}

TEST(ELFLoader, NoThrowForDmaRelocations) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    DmaSymbolEntry dmaSymbol{0,
                             0,
                             {6, 2, 4, 0, 0, 0},
                             {1, 6, 12, 48, 0, 0},
                             {0, 0, 0, 0, 0, 0},
                             {6, 2, 4, 0, 0, 0},
                             {1, 6, 12, 48, 0, 0},
                             4};

    OV_ASSERT_NO_THROW(elf = generateValidElfWithDmaRelocations(dmaSymbol));

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    OV_ASSERT_NO_THROW(VPUXLoader(&accessor, &bufMgr));
    VPUXLoader loader(&accessor, &bufMgr);
    OV_ASSERT_NO_THROW(loader.load(gSymTab.symTab(), false, {}));

    uint8_t data = 0;

    elf::DeviceBuffer input(&data, 0xB, 0xB);
    std::vector<elf::DeviceBuffer> inputs{input};
    OV_ASSERT_NO_THROW(loader.applyJitRelocations(inputs, inputs, inputs));
}

TEST(ELFLoader, NoThrowForDmaRelocationsWithUserStrides) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    DmaSymbolEntry dmaSymbol{0,
                             0,
                             {6, 2, 4, 0, 0, 0},
                             {1, 6, 12, 48, 0, 0},
                             {0, 0, 0, 0, 0, 0},
                             {6, 2, 4, 0, 0, 0},
                             {1, 6, 12, 48, 0, 0},
                             4};

    OV_ASSERT_NO_THROW(elf = generateValidElfWithDmaRelocations(dmaSymbol));

    DDRAccessManager<elf::DDRAlwaysEmplace> accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    OV_ASSERT_NO_THROW(VPUXLoader(&accessor, &bufMgr));
    VPUXLoader loader(&accessor, &bufMgr);
    OV_ASSERT_NO_THROW(loader.load(gSymTab.symTab(), false, {}));

    uint8_t data = 0;

    elf::DeviceBuffer input(&data, 0xB, 0xB);
    input.set_user_strides({1, 12, 24, 96, 0});
    std::vector<elf::DeviceBuffer> inputs{input};
    OV_ASSERT_NO_THROW(loader.applyJitRelocations(inputs, inputs, inputs));
}

}  // namespace
