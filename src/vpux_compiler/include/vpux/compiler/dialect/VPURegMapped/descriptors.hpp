//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/bit.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include "vpux/compiler/core/developer_build_utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type/bfloat16.hpp"
#include "vpux/utils/core/type/float16.hpp"
#include "vpux_elf/utils/version.hpp"

#include <mlir/IR/OpImplementation.h>

namespace vpux::VPURegMapped::detail {

std::pair<mlir::ParseResult, std::optional<elf::Version>> parseVersion(mlir::AsmParser&);

namespace {

constexpr auto FLOAT16_BIT_SIZE = sizeof(uint16_t) * CHAR_BIT;
constexpr auto FLOAT32_BIT_SIZE = sizeof(float) * CHAR_BIT;
constexpr auto FLOAT64_BIT_SIZE = sizeof(double) * CHAR_BIT;

template <class... Registers>
struct Union {
    static constexpr auto START = vpux::Byte{};
    static constexpr auto END = vpux::Byte{std::max({
            Registers::OFFSET + Registers::SIZE...,
    })};
    static_assert(END >= START, "Invalid registers pack");
    static constexpr auto SIZE = END - START;
};

template <auto subject, auto... candidates>
constexpr auto isAnyOf() {
    return ((subject == candidates) || ...);
}

template <auto subject, auto... candidates>
constexpr auto isNoneOf() {
    return !isAnyOf(subject, candidates...);
}

template <class Field>
constexpr auto isIntegralImpl() {
    return isAnyOf<Field::TYPE, ::vpux::VPURegMapped::RegFieldDataType::UINT,
                   ::vpux::VPURegMapped::RegFieldDataType::SINT>();
}

template <class Field>
constexpr auto isFloatingPointImpl() {
    return isAnyOf<Field::TYPE, ::vpux::VPURegMapped::RegFieldDataType::FP,
                   ::vpux::VPURegMapped::RegFieldDataType::BF>();
}

template <class Field>
constexpr auto IS_INTEGRAL = isIntegralImpl<Field>();

template <class Field>
constexpr auto IS_FLOATING_POINT = isFloatingPointImpl<Field>();

template <class Target, class... Candidates>
struct Contains : std::bool_constant<(std::is_same_v<Target, Candidates> || ...)> {};
template <class Target, class... Candidates>
struct Contains<Target, std::tuple<Candidates...>> : Contains<Target, Candidates...> {};

template <class T, class = void>
struct TypeExtractor {
    using type = T;
};

template <class T>
struct TypeExtractor<T, std::enable_if_t<std::is_enum_v<T>>> {
    using type = std::underlying_type_t<T>;
};

template <class SpecificField, const char* name, class ParentRegisters, size_t offsetInBits, size_t sizeInBits,
          ::vpux::VPURegMapped::RegFieldDataType type, uint32_t major, uint32_t minor, uint32_t patch>
struct FieldTemplate {
    static constexpr auto NAME = std::string_view{name};
    using Registers = ParentRegisters;
    static constexpr auto OFFSET = vpux::Bit{int64_t{offsetInBits}};
    static constexpr auto SIZE = vpux::Bit{int64_t{sizeInBits}};
    static constexpr auto TYPE = type;
    static constexpr auto DEFAULT_VERSION = elf::Version{major, minor, patch};

    static_assert(SIZE.count() != 0, "Field of zero size is unsupported");
    static_assert(SIZE.count() <= sizeof(uint64_t) * CHAR_BIT, "Field of size more than 8 bytes is unsupported");
    static_assert((TYPE != ::vpux::VPURegMapped::RegFieldDataType::SINT) || SIZE.count() > 1,
                  "Signed field must have more than one bit");
    static_assert((TYPE != ::vpux::VPURegMapped::RegFieldDataType::FP) || SIZE.count() == FLOAT16_BIT_SIZE ||
                          SIZE.count() == FLOAT32_BIT_SIZE || SIZE.count() == FLOAT64_BIT_SIZE,
                  "Floating-point field must have size of 16, 32 or 64 bits");
    static_assert((TYPE != ::vpux::VPURegMapped::RegFieldDataType::BF) || SIZE.count() == FLOAT16_BIT_SIZE,
                  "bfloat16 field must have size of 16 bits");

    template <class Register, class Descriptor>
    static void print(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        static_assert(Contains<SpecificField, typename Register::Fields>::value,
                      "Given register doesn't contain this field");
        static_assert(Contains<Register, typename Descriptor::Registers>::value,
                      "Given descriptor doesn't contain this register");

        printer.printNewline();
        printer << vpux::VPURegMapped::stringifyEnum(TYPE) << " " << NAME << " = "
                << getFormattedValue(descriptor.template read<Register, SpecificField>());

        // print comma at the end of field unconditionally (even for the last field in a register)
        // ex.:
        // R0 {
        //     F0,
        //     F1,
        // },
        // the reason is mlir::AsmPrinter ignores whitespaces including newline characters
        // if we print no commas and there're optional components in the Field format
        // (e.g. required version), when you will check if version is there you will see
        // name of the next field on the next line and fail with something like: got
        // unexpected keyword "<F1-name>" instead of "requires"
        printer << ',';
    }

    template <class Register, class Descriptor>
    static std::optional<elf::Version> getVersion(const Descriptor& descriptor) {
        std::optional<elf::Version> maybeFieldVersion;
        if (descriptor.template read<Register, SpecificField>()) {
            maybeFieldVersion = DEFAULT_VERSION;
        }
        return maybeFieldVersion;
    }

    template <class Register, class Descriptor>
    static std::pair<mlir::ParseResult, std::ostringstream> parse(mlir::AsmParser& parser, Descriptor& descriptor) {
        using namespace std::string_literals;
        using ResultType = std::pair<mlir::ParseResult, std::ostringstream>;

        if (std::string parsedType; parser.parseKeywordOrString(&parsedType).failed()) {
            return ResultType{mlir::failure(), "failed to parse field name"s};
        } else if (const auto maybeType = symbolizeRegFieldDataType(parsedType);
                   !maybeType.has_value() || maybeType.value() != TYPE) {
            return ResultType{mlir::failure(), std::ostringstream{"invalid field type \""}
                                                       << parsedType << "\", expected " << stringifyEnum(TYPE).str()};
        }

        if (std::string parsedName; parser.parseKeywordOrString(&parsedName).failed()) {
            return ResultType{mlir::failure(), "failed to parse field name"s};
        } else if (parsedName != NAME) {
            return ResultType{mlir::failure(),
                              std::ostringstream{"invalid field name \""} << parsedName << "\", expected " << NAME};
        }

        if (parser.parseEqual().failed()) {
            return ResultType{mlir::failure(), "failed to parse '='"s};
        }

        auto value = uint64_t{};
        if (parser.parseInteger(value).failed()) {
            return ResultType{mlir::failure(), "failed to parse field value"s};
        }

        if (parser.parseComma().failed()) {
            return ResultType{mlir::failure(), "failed to parse ','"s};
        }

        descriptor.template write<Register, SpecificField>(value);
        return ResultType{mlir::success(), ""s};
    }
};

// The register level provides the opportunity to reuse field definitions
// across multiple generations, making it worthwhile to preserve.
template <class SpecificRegister, const char* name, size_t offsetInBytes, size_t sizeInBytes, class... FieldsPack>
struct RegisterTemplate {
    static constexpr auto NAME = std::string_view{name};
    static constexpr auto OFFSET = vpux::Byte{offsetInBytes};
    static constexpr auto SIZE = vpux::Byte{sizeInBytes};
    // until C++26 there's no indexing of parameter pack
    // use std::tuple and std::tuple_element_t to index pack
    using Fields = std::tuple<FieldsPack...>;

    template <class Descriptor>
    static void print(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        static_assert(Contains<SpecificRegister, typename Descriptor::Registers>::value);

        printer.printNewline();

        constexpr auto bitSize = SIZE.template to<vpux::Bit>();

        printer << NAME;

        using FirstFieldType = std::tuple_element_t<0, Fields>;
        if constexpr (NAME == FirstFieldType::NAME && bitSize == FirstFieldType::SIZE) {
            printer << " = " << vpux::VPURegMapped::stringifyEnum(FirstFieldType::TYPE) << ' '
                    << getFormattedValue(descriptor.template read<SpecificRegister, FirstFieldType>());
            // see Field printer for why comma is needed here
            printer << ',';
        } else {
            printer << " {";
            printer.increaseIndent();
            (FieldsPack::template print<SpecificRegister>(printer, descriptor), ...);
            printer.decreaseIndent();
            printer.printNewline();
            printer << "}";
        }
    }

    template <class Descriptor>
    static std::optional<elf::Version> getVersion(const Descriptor& descriptor) {
        std::optional<elf::Version> maybeRegisterVersion;

        (
                [&] {
                    const auto maybeFieldVersion = FieldsPack::template getVersion<SpecificRegister>(descriptor);
                    if (maybeFieldVersion.has_value()) {
                        maybeRegisterVersion = std::max(maybeRegisterVersion.value_or(maybeFieldVersion.value()),
                                                        maybeFieldVersion.value());
                    }
                }(),
                ...);

        return maybeRegisterVersion;
    }

    template <class Descriptor>
    static std::pair<mlir::ParseResult, std::ostringstream> parse(mlir::AsmParser& parser, Descriptor& descriptor) {
        using namespace std::string_literals;
        using ResultType = std::pair<mlir::ParseResult, std::ostringstream>;

        if (std::string parsedName; parser.parseKeywordOrString(&parsedName).failed()) {
            return ResultType{mlir::failure(), "failed to parse register name"s};
        } else if (parsedName != NAME) {
            return ResultType{mlir::failure(),
                              std::ostringstream{"invalid register name \""} << name << "\", expected " << NAME};
        }

        StringRef keyword;
        if (parser.parseOptionalKeyword(&keyword).succeeded()) {
            if (keyword != "allowOverlap") {
                return ResultType{mlir::failure(), std::ostringstream{"unknown keyword \""}
                                                           << keyword.str() << R"(", expected "allowOverlap")"};
            }
            // ignoring the keyword for now
        }

        if (parser.parseOptionalEqual().succeeded()) {
            assert(sizeof...(FieldsPack) == 1);
            using SingleFieldType = std::tuple_element_t<0, Fields>;

            if (std::string type; parser.parseKeywordOrString(&type).failed()) {
                return ResultType{mlir::failure(), "failed to parse field type"s};
            } else if (const auto maybeType = symbolizeEnum<RegFieldDataType>(type);
                       !maybeType.has_value() || maybeType.value() != SingleFieldType::TYPE) {
                return ResultType{mlir::failure(), std::ostringstream{"invalid field data type \""}
                                                           << type << "\", expected "
                                                           << stringifyEnum(SingleFieldType::TYPE).str()};
            }

            auto value = uint64_t{};
            if (parser.parseInteger(value).failed()) {
                return ResultType{mlir::failure(), "failed to parse field value"s};
            }

            if (parser.parseComma().failed()) {
                return ResultType{mlir::failure(), "failed to parse ','"s};
            }

            descriptor.template write<SpecificRegister, SingleFieldType>(value);
        } else {
            if (parser.parseLBrace().failed()) {
                return ResultType{mlir::failure(), "failed to parse '{'"s};
            }

            auto status = mlir::ParseResult::success();
            std::ostringstream errorMessage;
            ([&] {
                std::tie(status, errorMessage) = FieldsPack::template parse<SpecificRegister>(parser, descriptor);
                return status.succeeded();
            }() &&
             ...);

            if (status.failed()) {
                return ResultType{status, std::move(errorMessage)};
            }

            if (parser.parseRBrace().failed()) {
                return ResultType{mlir::failure(), "failed to parse '}'"s};
            }
        }

        return ResultType{mlir::success(), ""s};
    }
};

template <class Descriptor, const char* name, class... RegistersPack>
class DescriptorTemplate {
public:
    static constexpr auto NAME = std::string_view{name};
    using Registers = std::tuple<RegistersPack...>;

    DescriptorTemplate() {
        _storage.resize(Union<RegistersPack...>::SIZE.count());
    }

    size_t size() const {
        return _storage.size();
    }

    template <class Register, class Field, class U>
    void write(U userValue) {
        using ::vpux::VPURegMapped::RegFieldDataType;
        using namespace ::vpux::type;

        static_assert(isAnyOf<Field::TYPE, RegFieldDataType::SINT, RegFieldDataType::UINT, RegFieldDataType::FP,
                              RegFieldDataType::BF>());

        // decay in case U is const or reference type, otherwise is_same would fail
        using X = std::decay_t<U>;

        static_assert(Contains<Register, Registers>::value, "Given register isn't a part of the descriptor");
        static_assert(Contains<Field, typename Register::Fields>::value, "Given field isn't a part of the register");

        constexpr auto size = Field::SIZE.count();

        if constexpr (std::is_same_v<uint64_t, X>) {
            // uint64_t as argument is a special case since it's used by printer/parser
            // temporarily trust the input without checks and just forward bits to storage
            // E#137584
            write<Register::OFFSET.count(), Field::OFFSET.count(), size>(userValue);
        } else if constexpr (std::is_floating_point_v<X> || std::is_same_v<float16, X> || std::is_same_v<bfloat16, X>) {
            static_assert(IS_FLOATING_POINT<Field>, "floating-point value can be set to floating-point field only");
            if constexpr (std::is_floating_point_v<X>) {
                static_assert(Field::TYPE == RegFieldDataType::FP,
                              "floating-point to bfloat conversion is unsupported");
                static_assert(size == FLOAT64_BIT_SIZE || (size == FLOAT32_BIT_SIZE && std::is_same_v<float, X>),
                              "value can't fit into field");
            } else {
                static_assert((std::is_same_v<float16, X> && Field::TYPE == RegFieldDataType::FP) ||
                                      (std::is_same_v<bfloat16, X> && Field::TYPE == RegFieldDataType::BF),
                              "floating-point/bfloat value and type mismatch");
                static_assert(size == FLOAT16_BIT_SIZE, "floating-point upcast for fp16 and bf16 is unsupported");
            }

            if constexpr (size == FLOAT64_BIT_SIZE) {
                // upcast to double to handle float argument
                write<Register::OFFSET.count(), Field::OFFSET.count(), size>(
                        llvm::bit_cast<uint64_t>(static_cast<double>(userValue)));
            } else if constexpr (size == FLOAT32_BIT_SIZE) {
                write<Register::OFFSET.count(), Field::OFFSET.count(), size>(llvm::bit_cast<uint32_t>(userValue));
            } else if constexpr (size == FLOAT16_BIT_SIZE) {
                write<Register::OFFSET.count(), Field::OFFSET.count(), size>(userValue.to_bits());
            }
        } else if constexpr (std::is_enum_v<X> || std::is_integral_v<X>) {
            static_assert((!std::is_same_v<bool, X> && !std::is_enum_v<X>) || IS_INTEGRAL<Field>,
                          "bool or enum to floating-point field is unsupported");

            if constexpr (std::is_same_v<bool, X>) {
                // no boundary check in case of bool value
                // boundary check for bool emits warning bool is always less than 1
                write<Register::OFFSET.count(), Field::OFFSET.count(), size>(userValue);
            } else {
                [[maybe_unused]] constexpr auto max = getIntegralFieldMaxValue<Field>();
                // to cover both enum and not enum cases
                using BackboneT = typename TypeExtractor<X>::type;

                if constexpr (std::is_unsigned_v<BackboneT> || Field::TYPE == RegFieldDataType::UINT) {
                    assert(static_cast<uint64_t>(userValue) <= max);
                } else {
                    // avoid cast to uint64_t if value maybe negative
                    // cast to int64_t for max is safe as size <= 64 and max positive value <= int64_t::max
                    assert(static_cast<int64_t>(userValue) <= int64_t{max});
                }

                if constexpr (std::is_signed_v<BackboneT>) {
                    // check for min only in case of signed type as unsigned is always >= 0
                    assert(static_cast<int64_t>(userValue) >= getIntegralFieldMinValue<Field>());
                }

                if constexpr (Field::TYPE == RegFieldDataType::SINT) {
                    const auto bitCasted = llvm::bit_cast<uint64_t>(static_cast<int64_t>(userValue));
                    // mask before writing in case if value was negative due to 2's complement format
                    const auto masked = bitCasted & getBitsSet<Field::SIZE.count()>();
                    write<Register::OFFSET.count(), Field::OFFSET.count(), size>(masked);
                } else {
                    write<Register::OFFSET.count(), Field::OFFSET.count(), size>(static_cast<uint64_t>(userValue));
                }
            }
        } else {
            assert(false && "unsupported value type");
        }
    }

    template <class Field, class U>
    void write(U&& userValue) {
        static_assert(std::tuple_size_v<typename Field::Registers> == 1,
                      "Ambiguous call to write, field has more than one parent register");
        write<std::tuple_element_t<0, typename Field::Registers>, Field>(std::forward<U>(userValue));
    }

    template <class Register, class Field>
    uint64_t read() const {
        // see write implementation about part0, part1 and part2 patterns

        static_assert(Contains<Register, Registers>::value, "Given register isn't a part of the descriptor");
        static_assert(Contains<Field, typename Register::Fields>::value, "Given field isn't a part of the register");

        // don't convert Field::OFFSET from Bit to Byte via to<vpux::Byte> as it'll throw
        // if Field::OFFSET isn't divisible by CHAR_BIT
        const auto address = _storage.data() + Register::OFFSET.count() + Field::OFFSET.count() / CHAR_BIT;
        constexpr auto inByteFieldOffset = Field::OFFSET.count() % CHAR_BIT;
        constexpr auto part0Size = std::min(Field::SIZE.count(), CHAR_BIT - inByteFieldOffset);
        constexpr auto part1n2Size = Field::SIZE.count() - part0Size;

        auto value = uint64_t{};
        if constexpr (part0Size != 0) {
            constexpr auto part0Mask = getBitsSet<part0Size>() << inByteFieldOffset;
            const auto part0Value = static_cast<uint64_t>(((*address) & part0Mask) >> inByteFieldOffset);
            value |= part0Value;
        }

        constexpr auto part2Size = part1n2Size % CHAR_BIT;
        constexpr auto part1Size = part1n2Size - part2Size;
        static_assert(part1Size % CHAR_BIT == 0);
        // seems like some compilers (e.g. clang) may complain variable isn't used
        // if both "if constexpr" below evaluate to false
        [[maybe_unused]] constexpr auto part0ByteOffset = size_t{1};
        [[maybe_unused]] constexpr auto part1ByteCount = part1Size / CHAR_BIT;

        if constexpr (part1Size != 0) {
            for (size_t i = 0; i < part1ByteCount; ++i) {
                const auto part1Value = static_cast<uint64_t>(address[part0ByteOffset + i]);
                value |= part1Value << (part0Size + i * CHAR_BIT);
            }
        }

        if constexpr (part2Size != 0) {
            constexpr auto part2Mask = getBitsSet<part2Size>();
            const auto part2Value = static_cast<uint64_t>(address[part0ByteOffset + part1ByteCount] & part2Mask);
            value |= part2Value << (part0Size + part1Size);
        }

        return value;
    }

    template <class Field>
    auto read() const {
        static_assert(std::tuple_size_v<typename Field::Registers> == 1,
                      "Ambiguous call to read, field has more than one parent register");
        return read<std::tuple_element_t<0, typename Field::Registers>, Field>();
    }

    bool operator==(const Descriptor& rhs) const {
        // no need to take into account custom versions
        // if values are the same, version will be as well
        return _storage == rhs._storage;
    }

    llvm::ArrayRef<uint8_t> getStorage() const {
        return _storage;
    }

    MutableArrayRef<uint8_t> getStorage() {
        return _storage;
    }

    // E#166553
    template <class OldDescriptor>
    void copyFrom(OldDescriptor& obj) {
        // in some cases we need an easy way to copy from a c structure directly to the corresponding descriptor
        // to enable descriptor features
        static_assert(Union<RegistersPack...>::SIZE.count() == sizeof(OldDescriptor),
                      "Size of template does not match the storage size!");
        std::copy_n(reinterpret_cast<uint8_t*>(&obj), _storage.size(), _storage.begin());
    }

    void print(mlir::AsmPrinter& printer) const {
        if constexpr (isDeveloperBuild()) {
            printer << '<';
            printer.increaseIndent();
            printer.printNewline();
            printer << NAME << " {";
            printer.increaseIndent();

            (RegistersPack::print(printer, static_cast<const Descriptor&>(*this)), ...);

            printer.decreaseIndent();
            printer.printNewline();
            printer << "}";

            const auto maybeVersion = getDescriptorVersion();
            if (maybeVersion.has_value()) {
                printer << " requires " << maybeVersion.value().getMajor() << ":" << maybeVersion.value().getMinor()
                        << ":" << maybeVersion.value().getPatch();
            }

            printer.decreaseIndent();
            printer.printNewline();
            printer << '>';
        } else {
            // On non-developer / non-debug builds, the printer and parser implementation is simplified so that the
            // descriptor is dumped in the form of a hexadecimal string. This is done in order to reduce the binary size
            // of the project, as these print / parse methods are created for each class that ends up generated during
            // build (each descriptor, register and field class)
            std::string_view data(reinterpret_cast<const char*>(getStorage().data()), getStorage().size());
            printer << '"' << llvm::toHex(data) << '"';
        }
    }

    static std::optional<Descriptor> parse(mlir::AsmParser& parser) {
        if constexpr (isDeveloperBuild()) {
            if (parser.parseLess().failed()) {
                return {};
            }

            if (std::string parsedName; parser.parseKeywordOrString(&parsedName).failed()) {
                return {};
            } else if (parsedName != Descriptor::NAME) {
                parser.emitError(parser.getCurrentLocation())
                        << "invalid descriptor name \"" << parsedName << "\", expected " << Descriptor::NAME;
                return {};
            }

            auto result = Descriptor{};

            if (parser.parseLBrace().failed()) {
                return {};
            }

            auto status = mlir::ParseResult::success();
            std::ostringstream errorMessage;
            ([&] {
                std::tie(status, errorMessage) = RegistersPack::parse(parser, result);
                return status.succeeded();
            }() &&
             ...);

            if (status.failed()) {
                parser.emitError(parser.getCurrentLocation()) << errorMessage.str();
                return {};
            }

            if (parser.parseRBrace().failed()) {
                return {};
            }

            const auto [versionParsing, maybeVersion] = parseVersion(parser);
            if (versionParsing.failed()) {
                return {};
            }

            if (parser.parseGreater().failed()) {
                return {};
            }

            return result;
        } else {
            // On non-developer / non-debug builds, the printer and parser implementation is simplified so that the
            // descriptor is dumped in the form of a hexadecimal string. This is done in order to reduce the binary size
            // of the project, as these print / parse methods are created for each class that ends up generated during
            // build (each descriptor, register and field class)
            std::string data;
            if (parser.parseString(&data).failed()) {
                return {};
            }
            auto result = Descriptor{};
            auto buffer = result.getStorage();
            auto binaryString = llvm::fromHex(data);
            std::copy_n(binaryString.data(), buffer.size(), buffer.begin());
            return result;
        }
    }

    std::optional<elf::Version> getDescriptorVersion() const {
        std::optional<elf::Version> maybeDescriptorVersion;

        (
                [&] {
                    const auto maybeRegisterVersion = RegistersPack::getVersion(static_cast<const Descriptor&>(*this));
                    if (maybeRegisterVersion.has_value()) {
                        maybeDescriptorVersion = std::max(maybeDescriptorVersion.value_or(maybeRegisterVersion.value()),
                                                          maybeRegisterVersion.value());
                    }
                }(),
                ...);

        return maybeDescriptorVersion;
    }

    llvm::hash_code hashValue() const {
        return llvm::hash_value(getStorage());
    }

private:
    template <size_t registerOffsetInBytes, size_t fieldOffsetInBits, size_t fieldSizeInBits>
    void write(uint64_t value) {
        // note: schemas below follow convention of having Least Significant Bits (LSB)
        //       on the right side and Most Significant Bits (MSB) on the left
        //       Intel uses little-endian format where LSB read first
        //
        //   Byte 0   Byte 1   Byte 2   Byte 3   Byte 4   Byte 5   Byte 6   Byte 7
        // |xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|
        //  76543210       98 <- bits ordering
        //
        // value represents sequence of bits to be written to the descriptor
        // specifically, to the descriptor in position occupied by a given field
        // since field location isn't necessary byte-aligned, we split value into 3 parts
        //
        // example:
        //
        // descriptor slice containing bits occupied by the field
        // S - start of the descriptor
        // RO - Register offset (always byte-aligned) = 2 bytes
        // FO - Field offset (not byte-aligned) = 11 bits = 1 byte + 3 bit
        // P0 - Part 0 of the field of size 5 bit (it can be up to 8 bits)
        // (FO % 8) + sizeof(P0) = 8 bit = 1 byte
        // P1 - Part 1 of the field of size 2 bytes (always byte-aligned)
        // P2 - Part 2 of the field of size 4 bits (it can be up to 7 bits)
        // sizeof(field) = sizeof(P0) + sizeof(P1) + sizeof(P2)
        //
        // S                 RO           P0   FO          P1               P2
        // |xxxxxxxx|xxxxxxxx|xxxxxxxx|[xxxxx]xxx|[xxxxxxxx|xxxxxxxx]|xxxx[xxxx]|xxxxxxxx|
        //
        // depending on field configuration (size, offset), some of the parts maybe omitted
        // e.g. if sizeof(field) = 5, FO = 0, then sizeof(P0) = 5, sizeof(P1) = sizeof(P2) = 0
        //
        // value generic case (64 bit): [unused][P2][P1][P0]
        //                  unused                         P2            P1           P0
        // |[xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxxx|xxxxxxx][x|xxx][xxxxx|xxxxxxxx|xxx][xxxxx]|
        //
        // unused bits maybe present if sizeof(field) < 64
        //
        // function basically matches P0, P1 and P2 in the value with corresponding positions
        // in the descriptor

        // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic): pointer arithmetic is required by algorithm

        auto address = _storage.data() + registerOffsetInBytes + fieldOffsetInBits / CHAR_BIT;
        constexpr auto inByteFieldOffset = fieldOffsetInBits % CHAR_BIT;
        constexpr auto part0Size = std::min(fieldSizeInBits, CHAR_BIT - inByteFieldOffset);
        constexpr auto part1n2Size = fieldSizeInBits - part0Size;

        if constexpr (part0Size != 0) {
            constexpr auto part0Mask = getBitsSet<part0Size>();
            const auto part0Value = value & part0Mask;

            // clear out whatever were in P0 position originally
            // so bitwise OR would result in bits from value only
            address[0] &= ~(part0Mask << inByteFieldOffset);
            address[0] |= static_cast<uint8_t>(part0Value << inByteFieldOffset);
        }

        constexpr auto part2Size = part1n2Size % 8;
        constexpr auto part1Size = part1n2Size - part2Size;
        static_assert(part1Size % CHAR_BIT == 0);
        [[maybe_unused]] constexpr auto part0ByteOffset = size_t{1};
        [[maybe_unused]] constexpr auto part1ByteCount = part1Size / 8;

        if constexpr (part2Size != 0) {
            constexpr auto part2Mask = getBitsSet<part2Size>();
            const auto part2Value = value >> (part0Size + part1Size);
            // clear out whatever were in P1 position originally
            // so bitwise OR would result in bits from value only
            address[part0ByteOffset + part1ByteCount] &= ~part2Mask;
            address[part0ByteOffset + part1ByteCount] |= part2Value;
        }

        if constexpr (part1Size != 0) {
            value >>= part0Size;

            for (size_t i = 0; i < part1ByteCount; ++i) {
                address[part0ByteOffset + i] = static_cast<uint8_t>(value);
                value >>= CHAR_BIT;
            }
        }

        // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    template <size_t size>
    static constexpr uint64_t getBitsSet() {
        constexpr auto maxBitSize = sizeof(uint64_t) * CHAR_BIT;
        static_assert(size <= maxBitSize, "No support for more than 64 bits");
        if constexpr (size == maxBitSize) {
            return llvm::bit_cast<uint64_t>(int64_t{-1});
        } else {
            return (uint64_t{1} << size) - 1;
        }
    }

    // use uint64_t as return type since it can hold max value for both
    // uint64_t and int64_t
    template <class Field>
    static constexpr uint64_t getIntegralFieldMaxValue() {
        constexpr auto maxBitSize = sizeof(uint64_t) * CHAR_BIT;
        static_assert(Field::SIZE.count() <= maxBitSize, "No support for more than 64 bits");
        static_assert(IS_INTEGRAL<Field>, "No support for non-integral types");

        using ::vpux::VPURegMapped::RegFieldDataType;

        if constexpr (Field::SIZE.count() == maxBitSize) {
            if constexpr (Field::TYPE == RegFieldDataType::UINT) {
                return std::numeric_limits<uint64_t>::max();
            } else {
                return uint64_t{std::numeric_limits<int64_t>::max()};
            }
        } else {
            if constexpr (Field::TYPE == RegFieldDataType::UINT) {
                return getBitsSet<Field::SIZE.count()>();
            } else {
                return getBitsSet<Field::SIZE.count() - 1>();
            }
        }
    }

    // use int64_t as return value since it can hold min value for both
    // uint64_t and int64_t
    template <class Field>
    static constexpr int64_t getIntegralFieldMinValue() {
        constexpr auto maxBitSize = sizeof(uint64_t) * CHAR_BIT;
        static_assert(Field::SIZE.count() <= maxBitSize, "No support for more than 64 bits");
        static_assert(IS_INTEGRAL<Field>, "No support for non-integral types");

        if constexpr (Field::TYPE == ::vpux::VPURegMapped::RegFieldDataType::UINT) {
            return 0;
        } else {
            // conversion to int64_t here is safe since Field is signed
            return -1 * int64_t{getIntegralFieldMaxValue<Field>()} - 1;
        }
    }

    mlir::SmallVector<std::uint8_t> _storage;
};

template <class Descriptor>
// NOLINTNEXTLINE(readability-identifier-naming): hash_value is required by MLIR
llvm::hash_code hash_value(const Descriptor& descriptor) {
    return descriptor.hashValue();
}

}  // namespace
}  // namespace vpux::VPURegMapped::detail
