//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include "vpux/utils/core/type/bfloat16.hpp"
#include "vpux/utils/core/type/float16.hpp"
#include "vpux_elf/utils/version.hpp"

namespace vpux::VPURegMapped::detail {

template <class... Registers>
struct Union {
    static constexpr auto start = vpux::Byte{std::min({
            vpux::Byte{},
            Registers::_offset...,
    })};
    static constexpr auto end = vpux::Byte{std::max({
            Registers::_offset + Registers::_size...,
    })};
    static_assert(end >= start, "Invalid registers pack");
    static constexpr auto size = end - start;
};

template <auto subject, auto... candidates>
constexpr auto is_any_of() {
    return ((subject == candidates) || ...);
}

template <auto subject, auto... candidates>
constexpr auto is_none_of() {
    return !is_any_of(subject, candidates...);
}

template <class Field>
constexpr auto is_integral_impl() {
    return is_any_of<Field::_type, ::vpux::VPURegMapped::RegFieldDataType::UINT,
                     ::vpux::VPURegMapped::RegFieldDataType::SINT>();
}

template <class Field>
constexpr auto is_floating_point_impl() {
    return is_any_of<Field::_type, ::vpux::VPURegMapped::RegFieldDataType::FP,
                     ::vpux::VPURegMapped::RegFieldDataType::BF>();
}

template <class Field>
static constexpr auto is_integral = is_integral_impl<Field>();

template <class Field>
static constexpr auto is_floating_point = is_floating_point_impl<Field>();

template <class Target, class Candidates>
struct contains;
template <class Target, class... Candidates>
struct contains<Target, std::tuple<Candidates...>> : std::disjunction<std::is_same<Target, Candidates>...> {};

template <class Tuple, template <class, auto, auto> class Functor, class... CallArgs>
struct Mapper;

template <class... Ts, template <class, auto, auto> class Functor, class... CallArgs>
struct Mapper<std::tuple<Ts...>, Functor, CallArgs...> {
    template <class... Args>
    static mlir::LogicalResult map(Args&&... args) {
        return impl(std::make_index_sequence<sizeof...(Ts)>(), std::forward<Args>(args)...);
    }

private:
    template <size_t... indexes, class... Args>
    static mlir::LogicalResult impl(std::index_sequence<indexes...> sequence, Args&&... args) {
        return mlir::LogicalResult::success(
                (Functor<Ts, sequence.size(), indexes>::template call<CallArgs...>(std::forward<Args>(args)...)
                         .succeeded() &&
                 ...));
    }
};

std::pair<mlir::ParseResult, std::optional<elf::Version>> parseVersion(mlir::AsmParser&);
template <const char* name, class ParentRegisters, size_t offsetInBits, size_t sizeInBits,
          ::vpux::VPURegMapped::RegFieldDataType type, uint32_t major, uint32_t minor, uint32_t patch>
struct FieldTemplate {
    static constexpr auto _name = std::string_view{name};
    using Registers = ParentRegisters;
    static constexpr auto _offset = vpux::Bit{int64_t{offsetInBits}};
    static constexpr auto _size = vpux::Bit{int64_t{sizeInBits}};
    static constexpr auto _type = type;
    static constexpr auto _defaultVersion = elf::Version{major, minor, patch};

    static_assert(_size.count() != 0, "Field of zero size is unsupported");
    static_assert(_size.count() <= sizeof(uint64_t) * CHAR_BIT, "Field of size more than 8 bytes is unsupported");
    static_assert((_type != ::vpux::VPURegMapped::RegFieldDataType::SINT) || _size.count() > 1,
                  "Signed field must have more than one bit");
    static_assert((_type != ::vpux::VPURegMapped::RegFieldDataType::FP) || _size.count() == 16 || _size.count() == 32 ||
                          _size.count() == 64,
                  "Floating-point field must have size of 16, 32 or 64 bits");
    static_assert((_type != ::vpux::VPURegMapped::RegFieldDataType::BF) || _size.count() == 16,
                  "bfloat16 field must have size of 16 bits");
};

// The register level provides the opportunity to reuse field definitions
// across multiple generations, making it worthwhile to preserve.
template <const char* name, class ParentDescriptors, size_t offsetInBytes, size_t sizeInBytes, class... FieldsPack>
struct RegisterTemplate {
    static constexpr auto _name = std::string_view{name};
    using Descriptors = ParentDescriptors;
    static constexpr auto _offset = vpux::Byte{offsetInBytes};
    static constexpr auto _size = vpux::Byte{sizeInBytes};
    // until C++26 there's no indexing of parameter pack
    // use std::tuple and std::tuple_element_t to index pack
    using Fields = std::tuple<FieldsPack...>;
};

template <class Field, size_t size = 1, size_t index = 0>
struct FieldPrinter {
    template <class Register, class Descriptor>
    static auto call(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        static_assert(contains<Register, typename Field::Registers>::value,
                      "Given register doesn't contain this field");
        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given descriptor doesn't contain this register");

        printer.printNewline();
        printer << vpux::VPURegMapped::stringifyEnum(Field::_type) << " " << Field::_name << " = "
                << getFormattedValue(descriptor.template read<Register, Field>());

        if constexpr (index < size - 1) {
            printer << ',';
        }

        return mlir::success();
    }
};

template <class Field, size_t size = 1, size_t index = 0>
struct FieldVersion {
    template <class Register, class Descriptor>
    static auto call(std::optional<elf::Version>& maxVersion, const Descriptor& descriptor) {
        static_assert(contains<Register, typename Field::Registers>::value,
                      "Given register doesn't contain this field");
        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given descriptor doesn't contain this register");

        if (descriptor.template read<Register, Field>()) {
            maxVersion = std::max(maxVersion.value_or(Field::_defaultVersion), Field::_defaultVersion);
        }

        return mlir::success();
    }
};

template <class Register, size_t size = 1, size_t index = 0>
struct RegisterVersion {
    template <class Descriptor>
    static auto call(std::optional<elf::Version>& maxVersion, const Descriptor& descriptor) {
        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given descriptor doesn't contain this register");

        return Mapper<typename Register::Fields, FieldVersion, Register>::map(maxVersion, descriptor);
    }
};

template <class T, class = void>
struct TypeExtractor {
    using type = T;
};

template <class T>
struct TypeExtractor<T, std::enable_if_t<std::is_enum_v<T>>> {
    using type = std::underlying_type_t<T>;
};

template <class Field, size_t fieldsCount = 1, size_t index = 0>
struct FieldParser {
    template <class Register, class Descriptor>
    static auto call(Descriptor& result, mlir::AsmParser& parser) {
        if (std::string type; parser.parseKeywordOrString(&type).failed()) {
            return mlir::failure();
        } else if (const auto maybeType = vpux::VPURegMapped::symbolizeRegFieldDataType(type);
                   !maybeType.has_value() || maybeType.value() != Field::_type) {
            parser.emitError(parser.getCurrentLocation())
                    << "unknown field data type \"" << type << "\", expected "
                    << vpux::VPURegMapped::stringifyRegFieldDataType(Field::_type);
            return mlir::failure();
        }

        if (std::string name; parser.parseKeywordOrString(&name).failed()) {
            return mlir::failure();
        } else if (name != Field::_name) {
            parser.emitError(parser.getCurrentLocation())
                    << "unknown field name \"" << name << "\", expected " << Field::_name;
            return mlir::failure();
        }

        if (parser.parseEqual().failed()) {
            return mlir::failure();
        }

        auto value = uint64_t{};
        if (parser.parseInteger(value).failed()) {
            return mlir::failure();
        }

        const auto [status, maybeVersion] = parseVersion(parser);
        if (status.failed()) {
            return mlir::failure();
        }

        if constexpr (index < fieldsCount - 1) {
            if (parser.parseComma().failed()) {
                return mlir::failure();
            }
        }

        result.template write<Register, Field>(value);
        return mlir::success();
    }
};

template <class Register, size_t size = 1, size_t index = 0>
struct RegisterPrinter {
    template <class Descriptor>
    static auto call(mlir::AsmPrinter& printer, const Descriptor& descriptor) {
        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given descriptor doesn't contain this register");

        printer.printNewline();

        constexpr auto bitSize = Register::_size.template to<vpux::Bit>();

        printer << Register::_name;

        using FirstFieldType = std::tuple_element_t<0, typename Register::Fields>;
        if constexpr (Register::_name == FirstFieldType::_name && bitSize == FirstFieldType::_size) {
            printer << " = " << vpux::VPURegMapped::stringifyEnum(FirstFieldType::_type) << ' '
                    << getFormattedValue(descriptor.template read<Register, FirstFieldType>());
        } else {
            printer << " {";
            printer.increaseIndent();
            if (Mapper<typename Register::Fields, FieldPrinter, Register>::map(printer, descriptor).failed()) {
                assert(false && "printer unexpectedly failed");
            }
            printer.decreaseIndent();
            printer.printNewline();
            printer << "}";
        }

        if constexpr (index < size - 1) {
            printer << ',';
        }

        return mlir::success();
    }
};

template <class Register, size_t registersCount = 1, size_t index = 0>
struct RegisterParser {
    template <class Descriptor>
    static auto call(Descriptor& result, mlir::AsmParser& parser) {
        if (std::string name; parser.parseKeywordOrString(&name).failed()) {
            return mlir::failure();
        } else if (name != Register::_name) {
            parser.emitError(parser.getCurrentLocation())
                    << "invalid register name \"" << name << "\", expected " << Register::_name;
        }

        StringRef keyword;
        if (parser.parseOptionalKeyword(&keyword).succeeded()) {
            if (keyword != "allowOverlap") {
                parser.emitError(parser.getCurrentLocation())
                        << "unknown keyword \"" << keyword << "\", expected \"allowOverlap\"";
                return mlir::failure();
            }
            // ignoring the keyword for now
        }

        if (parser.parseOptionalEqual().succeeded()) {
            assert(std::tuple_size_v<typename Register::Fields> == 1);
            using SingleFieldType = std::tuple_element_t<0, typename Register::Fields>;

            if (std::string type; parser.parseKeywordOrString(&type).failed()) {
                return mlir::failure();
            } else if (const auto maybeType =
                               vpux::VPURegMapped::symbolizeEnum<vpux::VPURegMapped::RegFieldDataType>(type);
                       !maybeType.has_value() || maybeType.value() != SingleFieldType::_type) {
                parser.emitError(parser.getCurrentLocation())
                        << "invalid field data type \"" << type << "\", expected "
                        << vpux::VPURegMapped::stringifyRegFieldDataType(SingleFieldType::_type);
                return mlir::failure();
            }

            auto value = uint64_t{};
            if (parser.parseInteger(value).failed()) {
                return mlir::failure();
            }

            const auto [parsingStatus, maybeVersion] = parseVersion(parser);
            if (parsingStatus.failed()) {
                return mlir::failure();
            }

            result.template write<Register, SingleFieldType>(value);
        } else {
            if (parser.parseLBrace().failed()) {
                return mlir::failure();
            }

            if (Mapper<typename Register::Fields, FieldParser, Register>::map(result, parser).failed()) {
                return mlir::failure();
            }

            if (parser.parseRBrace().failed()) {
                return mlir::failure();
            }
        }

        if constexpr (index < registersCount - 1) {
            if (parser.parseComma().failed()) {
                return mlir::failure();
            }
        }

        return mlir::success();
    }
};

template <class Descriptor, const char* name, class... RegistersPack>
class DescriptorTemplate {
public:
    static constexpr auto _name = std::string_view{name};
    using Registers = std::tuple<RegistersPack...>;

    DescriptorTemplate() {
        storage.resize(Union<RegistersPack...>::size.count());
    }

    size_t size() const {
        return storage.size();
    }

    template <class Register, class Field, class U>
    void write(U userValue) {
        using ::vpux::VPURegMapped::RegFieldDataType;
        using namespace ::vpux::type;

        static_assert(is_any_of<Field::_type, RegFieldDataType::SINT, RegFieldDataType::UINT, RegFieldDataType::FP,
                                RegFieldDataType::BF>());

        // decay in case U is const or reference type, otherwise is_same would fail
        using X = std::decay_t<U>;

        static_assert(contains<Register, Registers>::value, "Given register isn't a part of the descriptor");
        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given register isn't a part of the descriptor");
        static_assert(contains<Register, typename Field::Registers>::value, "Given field isn't a part of the register");

        constexpr auto size = Field::_size.count();

        if constexpr (std::is_same_v<uint64_t, X>) {
            // uint64_t as argument is a special case since it's used by printer/parser
            // temporarily trust the input without checks and just forward bits to storage
            // E#137584
            write<Register::_offset.count(), Field::_offset.count(), size>(userValue);
        } else if constexpr (std::is_floating_point_v<X> || std::is_same_v<float16, X> || std::is_same_v<bfloat16, X>) {
            static_assert(is_floating_point<Field>, "floating-point value can be set to floating-point field only");
            if constexpr (std::is_floating_point_v<X>) {
                static_assert(Field::_type == RegFieldDataType::FP,
                              "floating-point to bfloat conversion is unsupported");
                static_assert(size == 64 || (size == 32 && std::is_same_v<float, X>), "value can't fit into field");
            } else {
                static_assert((std::is_same_v<float16, X> && Field::_type == RegFieldDataType::FP) ||
                                      (std::is_same_v<bfloat16, X> && Field::_type == RegFieldDataType::BF),
                              "floating-point/bfloat value and type mismatch");
                static_assert(size == 16, "floating-point upcast for fp16 and bf16 is unsupported");
            }

            if constexpr (size == 64) {
                // upcast to double to handle float argument
                write<Register::_offset.count(), Field::_offset.count(), size>(
                        llvm::bit_cast<uint64_t>(static_cast<double>(userValue)));
            } else if constexpr (size == 32) {
                write<Register::_offset.count(), Field::_offset.count(), size>(llvm::bit_cast<uint32_t>(userValue));
            } else if constexpr (size == 16) {
                write<Register::_offset.count(), Field::_offset.count(), size>(userValue.to_bits());
            }
        } else if constexpr (std::is_enum_v<X> || std::is_integral_v<X>) {
            static_assert((!std::is_same_v<bool, X> && !std::is_enum_v<X>) || is_integral<Field>,
                          "bool or enum to floating-point field is unsupported");

            if constexpr (std::is_same_v<bool, X>) {
                // no boundary check in case of bool value
                // boundary check for bool emits warning bool is always less than 1
                write<Register::_offset.count(), Field::_offset.count(), size>(userValue);
            } else {
                [[maybe_unused]] constexpr auto max = getIntegralFieldMaxValue<Field>();
                // to cover both enum and not enum cases
                using BackboneT = typename TypeExtractor<X>::type;

                if constexpr (std::is_unsigned_v<BackboneT> || Field::_type == RegFieldDataType::UINT) {
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

                if constexpr (Field::_type == RegFieldDataType::SINT) {
                    const auto bitCasted = llvm::bit_cast<uint64_t>(static_cast<int64_t>(userValue));
                    // mask before writing in case if value was negative due to 2's complement format
                    const auto masked = bitCasted & getBitsSet<Field::_size.count()>();
                    write<Register::_offset.count(), Field::_offset.count(), size>(masked);
                } else {
                    write<Register::_offset.count(), Field::_offset.count(), size>(static_cast<uint64_t>(userValue));
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

        static_assert(contains<Descriptor, typename Register::Descriptors>::value,
                      "Given register isn't a part of the descriptor");
        static_assert(contains<Register, typename Field::Registers>::value, "Given field isn't a part of the register");

        // don't convert Field::_offset from Bit to Byte via to<vpux::Byte> as it'll throw
        // if Field::_offset isn't divisible by CHAR_BIT
        const auto address = storage.data() + Register::_offset.count() + Field::_offset.count() / CHAR_BIT;
        constexpr auto inByteFieldOffset = Field::_offset.count() % CHAR_BIT;
        constexpr auto part0Size = std::min(Field::_size.count(), CHAR_BIT - inByteFieldOffset);
        constexpr auto part1n2Size = Field::_size.count() - part0Size;

        auto value = uint64_t{};
        if constexpr (part0Size != 0) {
            constexpr auto part0Mask = getBitsSet<part0Size>() << inByteFieldOffset;
            const auto part0Value = static_cast<uint64_t>(((*address) & part0Mask) >> inByteFieldOffset);
            value |= part0Value;
        }

        constexpr auto part2Size = part1n2Size % 8;
        constexpr auto part1Size = part1n2Size - part2Size;
        static_assert(part1Size % 8 == 0);
        // seems like some compilers (e.g. clang) may complain variable isn't used
        // if both "if constexpr" below evaluate to false
        [[maybe_unused]] constexpr auto part0ByteOffset = size_t{1};
        [[maybe_unused]] constexpr auto part1ByteCount = part1Size / 8;

        if constexpr (part1Size != 0) {
            for (size_t i = 0; i < part1ByteCount; ++i) {
                const auto part1Value = static_cast<uint64_t>(address[part0ByteOffset + i]);
                value |= part1Value << (part0Size + i * 8);
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
        return storage == rhs.storage;
    }

    llvm::ArrayRef<uint8_t> getStorage() const {
        return storage;
    }

    void print(mlir::AsmPrinter& printer) const {
        printer << '<';
        printer.increaseIndent();
        printer.printNewline();
        printer << _name << " {";
        printer.increaseIndent();

        if (Mapper<Registers, RegisterPrinter>::map(printer, static_cast<const Descriptor&>(*this)).failed()) {
            assert(false && "printer unexpectedly failed");
        }

        printer.decreaseIndent();
        printer.printNewline();
        printer << "}";

        const auto maybeVersion = getDescriptorVersion();
        if (maybeVersion.has_value()) {
            printer << " requires " << maybeVersion.value().getMajor() << ":" << maybeVersion.value().getMinor() << ":"
                    << maybeVersion.value().getPatch();
        }

        printer.decreaseIndent();
        printer.printNewline();
        printer << '>';
    }

    static std::optional<Descriptor> parse(mlir::AsmParser& parser) {
        if (parser.parseLess().failed()) {
            return {};
        }

        if (std::string parsedName; parser.parseKeywordOrString(&parsedName).failed()) {
            return {};
        } else if (parsedName != Descriptor::_name) {
            parser.emitError(parser.getCurrentLocation())
                    << "invalid descriptor name \"" << parsedName << "\", expected " << Descriptor::_name;
            return {};
        }

        auto result = Descriptor{};

        if (parser.parseLBrace().failed()) {
            return {};
        }

        if (Mapper<typename Descriptor::Registers, RegisterParser>::map(result, parser).failed()) {
            return {};
        }

        if (parser.parseRBrace().failed()) {
            return {};
        }

        const auto [status, maybeVersion] = parseVersion(parser);
        if (status.failed()) {
            return {};
        }

        if (parser.parseGreater().failed()) {
            return {};
        }

        return result;
    }

    std::optional<elf::Version> getDescriptorVersion() const {
        std::optional<elf::Version> maxVersion;

        if (Mapper<typename Descriptor::Registers, RegisterVersion>::map(maxVersion,
                                                                         static_cast<const Descriptor&>(*this))
                    .failed()) {
            return std::nullopt;
        }

        return maxVersion;
    }

    llvm::hash_code hash_value() const {
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

        auto address = storage.data() + registerOffsetInBytes + fieldOffsetInBits / CHAR_BIT;
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
        static_assert(part1Size % 8 == 0);
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
                value >>= 8;
            }
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

    // use uint64_t as return type since it can hold max value for both
    // uint64_t and int64_t
    template <class Field>
    static constexpr uint64_t getIntegralFieldMaxValue() {
        static_assert(Field::_size.count() <= 64, "No support for more than 64 bits");
        static_assert(is_integral<Field>, "No support for non-integral types");

        using ::vpux::VPURegMapped::RegFieldDataType;

        if constexpr (Field::_size.count() == 64) {
            if constexpr (Field::_type == RegFieldDataType::UINT) {
                return std::numeric_limits<uint64_t>::max();
            } else {
                return uint64_t{std::numeric_limits<int64_t>::max()};
            }
        } else {
            if constexpr (Field::_type == RegFieldDataType::UINT) {
                return getBitsSet<Field::_size.count()>();
            } else {
                return getBitsSet<Field::_size.count() - 1>();
            }
        }
    }

    // use int64_t as return value since it can hold min value for both
    // uint64_t and int64_t
    template <class Field>
    static constexpr int64_t getIntegralFieldMinValue() {
        static_assert(Field::_size.count() <= 64, "No support for more than 64 bits");
        static_assert(is_integral<Field>, "No support for non-integral types");

        if constexpr (Field::_type == ::vpux::VPURegMapped::RegFieldDataType::UINT) {
            return 0;
        } else {
            // conversion to int64_t here is safe since Field is signed
            return -1 * int64_t{getIntegralFieldMaxValue<Field>()} - 1;
        }
    }

    mlir::SmallVector<std::uint8_t> storage;
};

template <class Descriptor>
llvm::hash_code hash_value(const Descriptor& descriptor) {
    return descriptor.hash_value();
}

}  // namespace vpux::VPURegMapped::detail
