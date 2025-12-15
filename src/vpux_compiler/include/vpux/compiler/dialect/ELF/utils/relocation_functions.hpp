//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

namespace vpux {
namespace ELF {
// E#170074 consider to remove duplicate code
// Do not change the following relocation functions without changing their correspondent from VPUXLoader!
using RelocFunc = std::function<void(void*, uint64_t, uint64_t)>;

const uint32_t B21_B26_MASK = 0x07E0'0000;
const uint32_t LO_21_BIT_MASK = 0x001F'FFFF;
const uint32_t ADDRESS_MASK = ~0x00C0'0000u;

uint32_t to_dpu_multicast(uint32_t addr, unsigned int& offset1, unsigned int& offset2, unsigned int& offset3) {
    const uint32_t bare_ptr = addr & ADDRESS_MASK;
    const uint32_t broadcast_mask = (addr & ~ADDRESS_MASK) >> 20;

    static const unsigned short multicast_masks[16] = {
            0x0000, 0x0001, 0x0002, 0x0003, 0x0012, 0x0011, 0x0010, 0x0030,
            0x0211, 0x0210, 0x0310, 0x0320, 0x3210, 0x3210, 0x3210, 0x3210,
    };

    VPUX_THROW_UNLESS(broadcast_mask < 16, "Broadcast mask out of range");
    const unsigned short multicast_mask = multicast_masks[broadcast_mask];

    VPUX_THROW_UNLESS(multicast_mask != 0xffff, "Got an invalid multicast mask");

    unsigned int base_mask = (static_cast<unsigned int>(multicast_mask) & 0xf) << 20;
    offset1 *= (multicast_mask >> 4) & 0xf;
    offset2 *= (multicast_mask >> 8) & 0xf;
    offset3 *= (multicast_mask >> 12) & 0xf;

    return bare_ptr | base_mask;
}

uint32_t to_dpu_multicast_base(uint32_t addr) {
    unsigned int offset1 = 0;
    unsigned int offset2 = 0;
    unsigned int offset3 = 0;
    return to_dpu_multicast(addr, offset1, offset2, offset3);
}

const auto VPU_64_BIT_OR_B21_B26_UNSET_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                       const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint64_t*>(targetAddr);
    auto symVal = stValue;

    uint64_t B21_B26_UNSET_MASK = ~B21_B26_MASK;
    auto patchAddr = static_cast<uint64_t>(symVal + addend) & B21_B26_UNSET_MASK;
    *addr |= patchAddr;
};

const auto VPU_32_BIT_OR_B21_B26_UNSET_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                       const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    uint32_t B21_B26_UNSET_MASK = ~B21_B26_MASK;
    auto patchAddr = static_cast<uint32_t>(symVal + addend) & B21_B26_UNSET_MASK;
    *addr |= patchAddr;
};

const auto VPU_LO_21_BIT_RSHIFT_4_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                  const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddr = (static_cast<uint32_t>(symVal + addend) & LO_21_BIT_MASK) >> 4;
    *addr &= ~LO_21_BIT_MASK;
    *addr |= patchAddr;
};

const auto VPU_LO_21_BIT_Relocation = [](void* targetAddr, const uint32_t stValue,
                                         const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddr = static_cast<uint32_t>(symVal + addend) & LO_21_BIT_MASK;
    *addr &= ~LO_21_BIT_MASK;
    *addr |= patchAddr;
};

const auto VPU_16_BIT_LSB_21_RSHIFT_5_LSHIFT_16_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                                const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    const uint32_t mask = LO_21_BIT_MASK;  // mask used to only keep last 21 bits
    const uint32_t msb_16_mask = 0xFFFF0000;

    *addr &= ~msb_16_mask;
    *addr |= ((static_cast<uint32_t>(symVal + addend) & mask) >> 5) << 16;
};

const auto VPU_16_BIT_LSB_21_RSHIFT_5_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                      const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    const uint32_t mask = LO_21_BIT_MASK;  // mask used to only keep last 21 bits
    const uint32_t lsb_16_mask = 0xFFFF;

    *addr &= ~lsb_16_mask;
    *addr |= (static_cast<uint32_t>(symVal + addend) & mask) >> 5;
};

const auto VPU_16_BIT_LSB_21_RSHIFT_5_LSHIFT_CUSTOM_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                                    const elf::Elf_Sxword addend) -> void {
    // more details in ticket #E-97614
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    const uint32_t mask = LO_21_BIT_MASK;                       // mask used to only keep last 21 bits
    const uint32_t preemtion_work_around_16_mask = 0xFFFE4000;  // 1111 1111 1111 1110 0100 0000 0000 0000

    *addr &= ~preemtion_work_around_16_mask;

    auto src_value = (static_cast<uint32_t>(symVal + addend) & mask) >> 5;
    // need to convert value from this view: 0000 0000 0000 0000 1111 1111 1111 1111
    // to                                    1111 1111 1111 1110 0100 0000 0000 0000

    // set [17:31] bits
    auto converted_value = (src_value & ~1) << 16;
    // format                                1111 1111 1111 1110 0000 0000 0000 0000

    // set [14] bit
    converted_value |= (src_value & 1) << 14;
    // format                                1111 1111 1111 1110 0100 0000 0000 0000

    *addr |= converted_value;
};

const auto VPU_LO_21_BIT_SUM_Relocation = [](void* targetAddr, const uint32_t stValue,
                                             const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddr = static_cast<uint32_t>(symVal + addend) & LO_21_BIT_MASK;
    *addr += patchAddr;
};

const auto VPU_64_BIT_Relocation = [](void* targetAddr, const uint32_t stValue, const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint64_t*>(targetAddr);
    auto symVal = stValue;

    *addr = symVal + addend;
};

const auto VPU_32_BIT_Relocation = [](void* targetAddr, const uint32_t stValue, const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    *addr = static_cast<uint32_t>(symVal + addend);
};

const auto VPU_LO_21_BIT_MULTICAST_BASE_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                        const elf::Elf_Sxword addend) -> void {
    const auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddr = static_cast<uint32_t>(symVal + addend) & LO_21_BIT_MASK;
    *addr = to_dpu_multicast_base(patchAddr);
};

const auto VPU_CMX_LOCAL_RSHIFT_5_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                  const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    uint32_t CMX_TILE_SELECT_MASK = ~B21_B26_MASK;
    auto patchAddr = (static_cast<uint32_t>(symVal + addend) & CMX_TILE_SELECT_MASK) >> 5;
    *addr = patchAddr;
};

const auto VPU_HIGH_27_BIT_OR_Relocation = [](void* targetAddr, const uint32_t stValue,
                                              const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint64_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddrUnsetTile = static_cast<uint32_t>(symVal + addend) &
                              ~0xE0'0000;  // unsetting 3 tile bits as NPU5 only uses 3 bits for tile selection
    auto patchAddr = (patchAddrUnsetTile >> 4) & (0x7FFF'FFFF >> 4);  // only [30:4]
    *addr |= (static_cast<uint64_t>(patchAddr) << 37);                // set [64:37]
};

const auto VPU_32_BIT_OR_B21_B26_UNSET_LOW_16_Relocation = [](void* targetAddr, const uint32_t stValue,
                                                              const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint16_t*>(targetAddr);
    auto symVal = stValue;

    uint64_t B21_B26_UNSET_MASK = ~B21_B26_MASK;
    auto patchAddr = static_cast<uint16_t>(symVal + addend) & B21_B26_UNSET_MASK;
    *addr |= patchAddr & 0xFFFF;
};

const auto VPU_32_BIT_OR_B21_B26_UNSET_HIGH_16_Relocation = [](void* targetAddr, uint32_t stValue,
                                                               const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint16_t*>(targetAddr);
    auto symVal = stValue;

    uint64_t B21_B26_UNSET_MASK = ~B21_B26_MASK;
    auto patchAddr = static_cast<uint32_t>(symVal + addend) & B21_B26_UNSET_MASK;
    *addr |= patchAddr >> 16;
};

const auto VPU_32_OR_LO_19_LSB_21_RSHIFT_2_Relocation = [](void* targetAddr, uint32_t stValue,
                                                           const elf::Elf_Sxword addend) -> void {
    auto addr = reinterpret_cast<uint32_t*>(targetAddr);
    auto symVal = stValue;

    auto patchAddr = (static_cast<uint32_t>(symVal + addend) & LO_21_BIT_MASK) >> 2;
    *addr &= ~(LO_21_BIT_MASK >> 2);
    *addr |= patchAddr;
};
}  // namespace ELF
}  // namespace vpux
