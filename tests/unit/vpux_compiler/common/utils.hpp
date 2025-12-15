//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/dialect.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>
#include <bitset>
#include <random>

class MLIR_UnitBase : public testing::Test {
public:
    MLIR_UnitBase() {
        registry = vpux::createDialectRegistry();
    }

protected:
    mlir::DialectRegistry registry;
};

class NPUSpecific_UnitTest : public MLIR_UnitBase {
public:
    NPUSpecific_UnitTest(vpux::config::ArchKind arch) {
        // We need to register hw-specific interfaces (e.g. NCEOpInterface) for VPU NCE ops
        auto interfacesRegistry = vpux::createInterfacesRegistry(arch);
        interfacesRegistry->registerInterfaces(registry);
        ctx.appendDialectRegistry(registry);
        ctx.loadDialect<vpux::VPU::VPUDialect>();
    }
    mlir::MLIRContext ctx;
};

namespace vpux::VPU::arch37xx {
class UnitTest : public NPUSpecific_UnitTest {
public:
    UnitTest(): NPUSpecific_UnitTest(vpux::config::ArchKind::NPU37XX) {
    }
};
}  // namespace vpux::VPU::arch37xx

namespace vpux::VPU::arch40xx {
class UnitTest : public NPUSpecific_UnitTest {
public:
    UnitTest(): NPUSpecific_UnitTest(vpux::config::ArchKind::NPU40XX) {
    }
};
}  // namespace vpux::VPU::arch40xx

using MappedRegValues = std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>;
template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedUnitBase : public testing::TestWithParam<std::pair<MappedRegValues, HW_REG_TYPE>> {
public:
    MLIR_RegMappedUnitBase() {
        ctx = std::make_unique<mlir::MLIRContext>();
    }
    void compare() {
        const auto params = this->GetParam();

        // initialize regMapped register with values
        auto defValues = REG_MAPPED_TYPE::getZeroInitilizationValues();
        vpux::VPURegMapped::updateRegMappedInitializationValues(defValues, params.first);

        auto regMappedDMADesc = REG_MAPPED_TYPE::get(*builder, defValues);

        // serialize regMapped register
        auto serializedRegMappedDMADesc = regMappedDMADesc.serialize();

        // compare
        EXPECT_EQ(sizeof(params.second), serializedRegMappedDMADesc.size());
        EXPECT_TRUE(memcmp(&params.second, serializedRegMappedDMADesc.data(), sizeof(params.second)) == 0);
    }

    std::unique_ptr<mlir::MLIRContext> ctx;
    std::unique_ptr<mlir::OpBuilder> builder;
};

template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedNPUReg37XXUnitBase : public MLIR_RegMappedUnitBase<HW_REG_TYPE, REG_MAPPED_TYPE> {
public:
    MLIR_RegMappedNPUReg37XXUnitBase() {
        this->ctx->template loadDialect<vpux::NPUReg37XX::NPUReg37XXDialect>();
        this->builder = std::make_unique<mlir::OpBuilder>(this->ctx.get());
    }
};

template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedNPUReg40XXUnitBase : public MLIR_RegMappedUnitBase<HW_REG_TYPE, REG_MAPPED_TYPE> {
public:
    MLIR_RegMappedNPUReg40XXUnitBase() {
        this->ctx->template loadDialect<vpux::NPUReg40XX::NPUReg40XXDialect>();
        this->builder = std::make_unique<mlir::OpBuilder>(this->ctx.get());
    }
};
template <typename HW_REG_TYPE, typename REG_MAPPED_TYPE>
class MLIR_RegMappedNPUReg50XXUnitBase : public MLIR_RegMappedUnitBase<HW_REG_TYPE, REG_MAPPED_TYPE> {
public:
    MLIR_RegMappedNPUReg50XXUnitBase() {
        this->ctx->template loadDialect<vpux::NPUReg50XX::NPUReg50XXDialect>();
        this->builder = std::make_unique<mlir::OpBuilder>(this->ctx.get());
    }
};

template <typename HWRegistersType, typename NPUDescriptorType>
class NPUReg_RegisterUnitBase : public testing::Test {
protected:
    NPUReg_RegisterUnitBase() = default;

    void SetUp() override {
        reference = {};
    }

    bool isContentEqual() const {
        const auto actualStorage = actual.getStorage();
        return std::memcmp(&reference, actualStorage.data(), sizeof(reference)) == 0;
    }

    template <class Field, class Register = std::tuple_element_t<0, typename Field::Registers>>
    auto testPositiveValue() {
        const auto value = generateUnsignedTestValue<Field>();
        writeValueToDescriptor<Field, Register>(value);

        const auto actualValue = actual.template read<Register, Field>();
        EXPECT_EQ(actualValue, value);
        return value;
    }

    template <class Field, class Register = std::tuple_element_t<0, typename Field::Registers>>
    auto testNegativeValue() {
        if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::UINT) {
            return 0ull;
        }
        const auto value = generateUnsignedTestValue<Field>();
        if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::SINT) {
            const auto signedValue = -1 * static_cast<int64_t>(value);

            actual.template write<Register, Field>(signedValue);
            auto actualSignedValue = actual.template read<Register, Field>();

            // there are cases where the descriptor does not write internally the entire 2's complement representation
            // of the negative value (e.g.: SINT field with size 9). Therefore before testing the read value it is
            // necessary to reconstruct the 2's complement.
            // *** use case: SINT field, 9 bits
            // *** signedValue - to be written by the descriptor: -128
            // *** signedValue representation:
            //              1111111111111111111111111111111111111111111111111111111110000000
            // *** the descriptor writes internally 9 bits: 110000000
            // *** actualSignedValue - value read by the descriptor:
            //              0000000000000000000000000000000000000000000000000000000110000000 (= 0x180 = 384)
            // *** => replace the 0s from pos 9 -> pos 63 with 1s in order to obtain -128
            actualSignedValue |= ~getBitsSet<Field::SIZE.count()>();
            EXPECT_EQ(actualSignedValue, llvm::bit_cast<uint64_t>(signedValue));

            return llvm::bit_cast<uint64_t>(signedValue);
        } else {
            // set the sign bit
            const auto negativeValue = value | (1ull << (Field::SIZE.count() - 1));
            writeValueToDescriptor<Field, Register>(negativeValue);

            const auto actualNegativeValue = actual.template read<Register, Field>();
            EXPECT_EQ(actualNegativeValue, negativeValue);

            return negativeValue;
        }
    }

    NPUDescriptorType actual;
    HWRegistersType reference;

private:
    // to test big vs little endian use random set bits distribution
    // so that for a big enough fields there would be consequent bytes
    // with different values
    template <size_t bitsCount>
    static uint64_t generateRandom() {
        std::mt19937 generator(42);
        std::uniform_int_distribution<> distribution(0, 1);
        std::stringstream bitSequence;
        for ([[maybe_unused]] auto _ : vpux::irange(bitsCount)) {
            bitSequence << distribution(generator);
        }
        return std::bitset<bitsCount>(bitSequence.str()).to_ullong();
    }

    // generates non-negative value that fits into range of the field
    template <class Field>
    static uint64_t generateUnsignedTestValue() {
        if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::UINT) {
            return generateRandom<Field::SIZE.count()>();
        } else {
            return generateRandom<Field::SIZE.count() - 1>();
        }
    }

    template <size_t size>
    static constexpr uint64_t getBitsSet() {
        static_assert(size <= 64, "No support for more than 64 bits");
        if constexpr (size == 64) {
            return llvm::bit_cast<uint64_t>(int64_t{-1});
        } else {
            return (uint64_t{1} << size) - 1;
        }
    }

    template <class Field, class Register = std::tuple_element_t<0, typename Field::Registers>>
    void writeValueToDescriptor(const uint64_t value) {
        if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::UINT) {
            actual.template write<Register, Field>(value);
        } else if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::SINT) {
            actual.template write<Register, Field>(static_cast<int64_t>(value));
        } else if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::FP) {
            if constexpr (Field::SIZE.count() == 64) {
                actual.template write<Register, Field>(llvm::bit_cast<double>(value));
            } else if constexpr (Field::SIZE.count() == 32) {
                actual.template write<Register, Field>(llvm::bit_cast<float>(static_cast<uint32_t>(value)));
            } else if constexpr (Field::SIZE.count() == 16) {
                actual.template write<Register, Field>(vpux::type::float16::from_bits(static_cast<uint16_t>(value)));
            }
        } else if constexpr (Field::TYPE == vpux::VPURegMapped::RegFieldDataType::BF) {
            actual.template write<Register, Field>(vpux::type::bfloat16::from_bits(static_cast<uint16_t>(value)));
        }
    }
};

#define HELPER_TEST_NPU_REGISTER_FIELD(TestFixtureName, TestName, FieldType, DescriptorMember, LeftShiftBitsCount) \
    TEST_F(TestFixtureName, TestName##Test) {                                                                      \
        ASSERT_EQ(sizeof(reference), actual.size());                                                               \
                                                                                                                   \
        auto value = testPositiveValue<FieldType>();                                                               \
        reference.DescriptorMember = value << LeftShiftBitsCount;                                                  \
        ASSERT_TRUE(isContentEqual());                                                                             \
                                                                                                                   \
        if constexpr (FieldType::TYPE != vpux::VPURegMapped::RegFieldDataType::UINT) {                             \
            const auto value = testNegativeValue<FieldType>();                                                     \
            reference.DescriptorMember = value;                                                                    \
            ASSERT_TRUE(isContentEqual());                                                                         \
        }                                                                                                          \
    }

#define HELPER_TEST_NPU_MULTIPLE_REGS_FIELD(TestFixtureName, TestName, ParentRegType, FieldType, DescriptorMember, \
                                            LeftShiftBitsCount)                                                    \
    TEST_F(TestFixtureName, TestName##Test) {                                                                      \
        ASSERT_EQ(sizeof(reference), actual.size());                                                               \
                                                                                                                   \
        auto value = testPositiveValue<FieldType, ParentRegType>();                                                \
        reference.DescriptorMember = value << LeftShiftBitsCount;                                                  \
        ASSERT_TRUE(isContentEqual());                                                                             \
                                                                                                                   \
        if constexpr (FieldType::TYPE != vpux::VPURegMapped::RegFieldDataType::UINT) {                             \
            const auto value = testNegativeValue<FieldType, ParentRegType>();                                      \
            reference.DescriptorMember = value;                                                                    \
            ASSERT_TRUE(isContentEqual());                                                                         \
        }                                                                                                          \
    }
