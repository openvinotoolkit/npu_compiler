//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>
#include <malloc.h>
#include <fstream>
#include <random>
#include <vpux_elf/accessor.hpp>
#include <vpux_elf/writer.hpp>
#include <vpux_loader/vpux_loader.hpp>

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

    const details::ArrayRef<SymbolEntry> symTab() const {
        return details::ArrayRef<SymbolEntry>(symTab_, SPECIAL_SYMTAB_SIZE);
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

    void lock(DeviceBuffer& devBuffer) {
        (void)devBuffer;
    }

    void unlock(DeviceBuffer& devBuffer) {
        (void)devBuffer;
    }

    size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) override {
        memcpy(to.cpu_addr(), from, count);
        return count;
    }
};

class NullAllocBufferManager : public BufferManager {
    virtual DeviceBuffer allocate(const BufferSpecs& buffSpecs) {
        (void)buffSpecs;
        return DeviceBuffer();
    }
    virtual void deallocate(DeviceBuffer& devAddress) {
        (void)devAddress;
    }

    virtual void lock(DeviceBuffer& devAddress) {
        (void)devAddress;
    }
    virtual void unlock(DeviceBuffer& devAddress) {
        (void)devAddress;
    }
    virtual size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) {
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
    for (uint32_t i = 0; i < iterCount; i++) {
        binDataSection->appendData(DummyBinObject{});
    }

    return binDataSection;
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
    reloc->setSymbol(&(*(symSection->getSymbols()[generateRandom(0, symSection->getSymbols().size() - 1)])));
    reloc->setOffset(sizeof(DummyBinObject::a));
    reloc->setAddend(0);

    return writer.generateELF();
}

std::vector<uint8_t> generateBadSectionTypeElf() {
    Writer writer;

    (void)generateDataSection<DummyBinObject>(writer, ".binData", SHT_HIUSER - 1);

    return writer.generateELF();
}

std::vector<uint8_t> generateValidTestElf() {
    Writer writer;

    auto binDataSection = generateDataSection<DummyBinObject>(writer, ".binData");
    auto symSection = generateSymbolSection(writer, ".symbols");
    auto relocSection = generateRelocationSection(writer, ".reloc", binDataSection, symSection);

    auto reloc = relocSection->addRelocationEntry();
    reloc->setSymbol(&(*(symSection->getSymbols()[generateRandom(0, symSection->getSymbols().size() - 1)])));
    reloc->setOffset(sizeof(DummyBinObject::a));
    reloc->setAddend(0);

    return writer.generateELF();
}

const HardCodedSymtabToCluster0 gSymTab;

TEST(ELFLoader, ThrowWhenAccessorPointerIsNull) {
    DummyBufferManager bufMgr;

    ASSERT_THROW(VPUXLoader(nullptr, &bufMgr, gSymTab.symTab()), ArgsError);
}

TEST(ELFLoader, ThrowWhenElfHeaderIsInvalid) {
    std::vector<uint8_t> elfBad = {0x7f, 'E', 'L', 'X'};
    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elfBad.data()), elfBad.size());

    ASSERT_THROW(VPUXLoader(&accessor, nullptr, gSymTab.symTab()), HeaderError);
}

TEST(ELFLoader, ThrowWhenBufferManagerIsNull) {
    std::vector<uint8_t> elf;

    ASSERT_NO_THROW(elf = generateValidTestElf());

    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader(&accessor, nullptr, gSymTab.symTab()), ArgsError);
}

TEST(ELFLoader, ThrowWhenAllocFails) {
    NullAllocBufferManager bufMgr;
    std::vector<uint8_t> elf;

    ASSERT_NO_THROW(elf = generateValidTestElf());

    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader(&accessor, &bufMgr, gSymTab.symTab()), AllocError);
}

TEST(ELFLoader, ThrowWhenBadUserIO) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    ASSERT_NO_THROW(elf = generateBadUserIOElf());

    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader(&accessor, &bufMgr, gSymTab.symTab()), SequenceError);
}

TEST(ELFLoader, ThrowWhenBadSectionType) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    ASSERT_NO_THROW(elf = generateBadSectionTypeElf());

    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_THROW(VPUXLoader(&accessor, &bufMgr, gSymTab.symTab()), SectionError);
}

TEST(ELFLoader, NoThrowWhenValidElf) {
    DummyBufferManager bufMgr;
    std::vector<uint8_t> elf;

    ASSERT_NO_THROW(elf = generateValidTestElf());

    ElfDDRAccessManager accessor(reinterpret_cast<const uint8_t*>(elf.data()), elf.size());
    ASSERT_NO_THROW(VPUXLoader(&accessor, &bufMgr, gSymTab.symTab()));
}

}  // namespace
