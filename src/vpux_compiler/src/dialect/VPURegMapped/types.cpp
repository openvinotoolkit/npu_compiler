//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/YAMLParser.h>

#include <llvm/Support/Debug.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPURegMapped/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::VPURegMapped::VPURegMappedDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPURegMapped/types.cpp.inc>
            >();
}

//
// Dialect hooks
//

namespace {
mlir::ParseResult parseRequiredMiVersion(mlir::AsmParser& parser, uint32_t& major, uint32_t& minor, uint32_t& patch) {
    StringRef keyword;
    if (parser.parseOptionalKeyword(&keyword).failed()) {
        return mlir::failure();
    }
    if (keyword != "requires") {
        parser.emitError(parser.getCurrentLocation()) << "version keyword \"" << keyword << "\", expected \"requires\"";
        return mlir::failure();
    }

    if (mlir::failed(parser.parseInteger(major))) {
        return mlir::failure();
    }
    if (mlir::failed(parser.parseColon())) {
        return mlir::failure();
    }
    if (mlir::failed(parser.parseInteger(minor))) {
        return mlir::failure();
    }
    if (mlir::failed(parser.parseColon())) {
        return mlir::failure();
    }
    if (mlir::failed(parser.parseInteger(patch))) {
        return mlir::failure();
    }

    return mlir::success();
}
}  // namespace

llvm::hash_code elf::hash_value(const elf::Version& version) {
    return llvm::hash_combine(llvm::hash_value(version.getMajor()), llvm::hash_value(version.getMinor()),
                              llvm::hash_value(version.getPatch()));
}

//
// IndexType
//

VPURegMapped::IndexType VPURegMapped::IndexType::get(mlir::MLIRContext* context, uint32_t value) {
    return get(context, 0, 0, value);
}

VPURegMapped::IndexType VPURegMapped::IndexType::get(mlir::MLIRContext* context, uint32_t listIdx, uint32_t value) {
    return get(context, 0, listIdx, value);
}

void VPURegMapped::IndexType::print(mlir::AsmPrinter& printer) const {
    printer << "<" << getTileIdx() << ":" << getListIdx() << ":" << getValue() << ">";
}

mlir::Type VPURegMapped::IndexType::parse(mlir::AsmParser& parser) {
    if (parser.parseLess()) {
        return mlir::Type();
    }

    uint32_t tile = 0;
    if (parser.parseInteger(tile)) {
        return {};
    }

    if (parser.parseColon()) {
        return {};
    }

    uint32_t list = 0;
    if (parser.parseInteger(list)) {
        return {};
    }

    if (parser.parseColon()) {
        return {};
    }

    uint32_t id = 0;
    if (parser.parseInteger(id)) {
        return {};
    }

    if (parser.parseGreater()) {
        return {};
    }

    return get(parser.getContext(), tile, list, id);
}

//
// RegFieldType
//

mlir::LogicalResult vpux::VPURegMapped::RegFieldType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t width, uint32_t pos, uint64_t value,
        std::string name, vpux::VPURegMapped::RegFieldDataType /*dataType*/, elf::Version /*requiredMIVersion*/) {
#ifdef NDEBUG
    VPUX_UNUSED(emitError);
    VPUX_UNUSED(width);
    VPUX_UNUSED(pos);
    VPUX_UNUSED(value);
    VPUX_UNUSED(name);
#else
    if (calcMinBitsRequirement(value) > width) {
        return printTo(emitError(),
                       "RegFieldType - provided width {0} is not enough to store provided value {1} for field {2}",
                       width, value, name);
    }
    if (width == 0 || width > Byte(sizeof(value)).to<Bit>().count()) {
        return printTo(emitError(), "RegFieldType - not supported width {0} for field {1}", width, name);
    }
    if (pos + width > Byte(sizeof(value)).to<Bit>().count()) {
        return printTo(emitError(), "RegFieldType - position of start {0} + width {1} Out of Range for field {2}.", pos,
                       width, name);
    }
    if (name.empty()) {
        return printTo(emitError(), "RegFieldType - name is empty.");
    }
#endif

    return mlir::success();
}

llvm::FormattedNumber getFormattedValue(uint64_t value) {
    return value > 9 ? llvm::format_hex(value, 2, true) : llvm::format_decimal(value, 1);
}

void VPURegMapped::RegFieldType::print(mlir::AsmPrinter& printer) const {
    auto requiredMiVersion = getRequiredMIVersion();
    printer << VPURegMapped::stringifyEnum(getDataType()) << " " << getName() << " at " << getPos() << " size "
            << getWidth() << " = " << getFormattedValue(getValue());

    if (requiredMiVersion.checkValidity()) {
        printer << " requires " << requiredMiVersion.getMajor() << ":" << requiredMiVersion.getMinor() << ":"
                << requiredMiVersion.getPatch();
    }
}

mlir::Type VPURegMapped::RegFieldType::parse(mlir::AsmParser& parser) {
    uint32_t width, pos;
    uint64_t value;
    std::string name;
    std::string dataType;
    uint32_t vMajor{}, vMinor{}, vPatch{};

    if (parser.parseKeywordOrString(&dataType)) {
        return {};
    }

    if (parser.parseKeywordOrString(&name)) {
        return {};
    }

    if (parser.parseKeyword("at")) {
        return {};
    }
    if (parser.parseInteger(pos)) {
        return {};
    }

    if (parser.parseKeyword("size")) {
        return {};
    }
    if (parser.parseInteger(width)) {
        return {};
    }

    if (parser.parseEqual()) {
        return {};
    }
    if (parser.parseInteger(value)) {
        return {};
    }
    auto requiredMiVersion = mlir::succeeded(parseRequiredMiVersion(parser, vMajor, vMinor, vPatch))
                                     ? elf::Version(vMajor, vMinor, vPatch)
                                     : elf::Version();

    return get(parser.getContext(), width, pos, value, name,
               VPURegMapped::symbolizeEnum<RegFieldDataType>(dataType).value(), requiredMiVersion);
}

//
// RegisterType
//

Byte vpux::VPURegMapped::RegisterType::getSizeInBytes() const {
    return Byte(getSize());
}

std::vector<uint8_t> vpux::VPURegMapped::RegisterType::serialize() const {
    std::vector<uint8_t> result(getSizeInBytes().count(), 0);

    uint64_t serializedReg = 0;
    auto fieldsAttrs = getRegFields().getValue();
    for (const auto& fieldAttr : fieldsAttrs) {
        auto pos = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getPos();
        auto value = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getValue();
        auto currentFieldMap = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getMap();

        auto shiftedValue = value << pos;
        serializedReg |= (shiftedValue & currentFieldMap);

        // value and currentFieldMap has max allowed size - 64 bit
        // result should contain first getSize() bytes only
    }
    auto dataPtr = result.data();
    memcpy(dataPtr, &serializedReg, getSizeInBytes().count());
    return result;
}

vpux::VPURegMapped::RegFieldType vpux::VPURegMapped::RegisterType::getField(const std::string& name) const {
    auto fieldsAttrs = getRegFields().getValue();
    auto fieldIter = std::find_if(fieldsAttrs.begin(), fieldsAttrs.end(), [&](mlir::Attribute fieldAttr) {
        return fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getName() == name;
    });
    VPUX_THROW_UNLESS(fieldIter != fieldsAttrs.end(), "Field with name {0} is not found in register {1}", name,
                      this->getName());
    return fieldIter->cast<VPURegMapped::RegisterFieldAttr>().getRegField();
}

elf::Version vpux::VPURegMapped::RegisterType::getRequiredMIVersion() const {
    auto fieldsAttrs = getRegFields().getValue();

    elf::Version maxVersion;
    llvm::for_each(fieldsAttrs, [&maxVersion](mlir::Attribute fieldAttr) {
        auto currVersion = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getRequiredMIVersion();
        maxVersion = std::max(maxVersion, currVersion);
    });

    return maxVersion;
}

mlir::LogicalResult vpux::VPURegMapped::RegisterType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, uint32_t size, std::string name, uint32_t address,
        ::mlir::ArrayAttr regFields, bool allowOverlap) {
#ifdef NDEBUG
    VPUX_UNUSED(emitError);
    VPUX_UNUSED(size);
    VPUX_UNUSED(name);
    VPUX_UNUSED(address);
    VPUX_UNUSED(regFields);
    VPUX_UNUSED(allowOverlap);
#else
    if (name.empty()) {
        return printTo(emitError(), "RegisterType - name is empty.");
    }

    uint32_t totalWidth(0);
    uint32_t currentAddress(0x0);
    std::map<std::string, uint64_t> wholeRegisterMap;
    auto fieldsAttrs = regFields.getValue();
    for (const auto& fieldAttr : fieldsAttrs) {
        auto width = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getWidth();
        auto pos = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getPos();
        auto value = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getValue();
        auto name = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getName();
        auto dataType = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getDataType();
        auto currentFieldMap = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getMap();
        auto requiredMiVersion = fieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().getRequiredMIVersion();
        totalWidth += width;

        // check overlaping
        auto overlapIter = std::find_if(wholeRegisterMap.begin(), wholeRegisterMap.end(),
                                        [currentFieldMap](const std::pair<std::string, uint64_t>& map) {
                                            return map.second & currentFieldMap;
                                        });
        if (!allowOverlap && overlapIter != wholeRegisterMap.end()) {
            return printTo(
                    emitError(),
                    "RegisterType - Overlap with {0} detected. Start position {1} and width {2} for {3} are invalid. "
                    "If you are sure it's not a mistake, please allow fields overlap for this register explicitly",
                    overlapIter->first, pos, width, name);
        }
        wholeRegisterMap[name] = currentFieldMap;

        if (mlir::failed(vpux::VPURegMapped::RegFieldType::verify(emitError, width, pos, value, name, dataType,
                                                                  requiredMiVersion))) {
            return printTo(emitError(), "RegisterType - invalid.");
        }

        if (address < currentAddress) {
            return printTo(emitError(), "RegisterType - address {0} for {1} is invalid", address, name);
        }
        currentAddress = address;
    }

    if (!allowOverlap && (totalWidth > size * CHAR_BIT)) {
        return printTo(emitError(), "RegisterType - {0} - invalid size {1}.", name, totalWidth);
    }
#endif

    return mlir::success();
}

void VPURegMapped::RegisterType::print(mlir::AsmPrinter& printer) const {
    printer << getName() << " offset " << getAddress() << " size " << getSize();
    if (getAllowOverlap()) {
        printer << " allowOverlap";
    }
    auto regFields = getRegFields().getValue();
    auto firstRegField = regFields.front().cast<VPURegMapped::RegisterFieldAttr>().getRegField();
    if (getName() == firstRegField.getName() && getSize() * CHAR_BIT == firstRegField.getWidth()) {
        printer << " = " << VPURegMapped::stringifyEnum(firstRegField.getDataType()) << " "
                << getFormattedValue(firstRegField.getValue());
        auto requiredMiVersion = firstRegField.getRequiredMIVersion();
        if (requiredMiVersion.checkValidity()) {
            printer << " requires " << requiredMiVersion.getMajor() << ":" << requiredMiVersion.getMinor() << ":"
                    << requiredMiVersion.getPatch();
        }
        return;
    }
    printer << " {";
    printer.increaseIndent();
    for (auto regFieldAttr : regFields) {
        printer.printNewline();
        regFieldAttr.cast<VPURegMapped::RegisterFieldAttr>().getRegField().print(printer);
        if (regFieldAttr != regFields.back()) {
            printer << ",";
        }
    }
    printer.decreaseIndent();
    printer.printNewline();
    printer << "}";
}

mlir::Type VPURegMapped::RegisterType::parse(mlir::AsmParser& parser) {
    std::string name;
    mlir::ArrayAttr regFields;
    mlir::SmallVector<mlir::Attribute> regFieldsVec;
    uint32_t size, address;
    bool allowOverlap = false;

    if (parser.parseKeywordOrString(&name)) {
        return {};
    }

    if (parser.parseKeyword("offset")) {
        return {};
    }
    if (parser.parseInteger(address)) {
        return {};
    }

    if (parser.parseKeyword("size")) {
        return {};
    }
    if (parser.parseInteger(size)) {
        return {};
    }

    StringRef keyword;
    if (mlir::succeeded(parser.parseOptionalKeyword(&keyword))) {
        if (keyword != "allowOverlap") {
            parser.emitError(parser.getCurrentLocation())
                    << "version keyword \"" << keyword << "\", expected \"allowOverlap\"";
            return {};
        }
        allowOverlap = true;
    }

    if (mlir::succeeded(parser.parseOptionalEqual())) {
        std::string dataType;
        uint64_t value;
        uint32_t vMajor{}, vMinor{}, vPatch{};

        if (parser.parseKeywordOrString(&dataType)) {
            return {};
        }

        if (parser.parseInteger(value)) {
            return {};
        }
        auto requiredMiVersion = mlir::succeeded(parseRequiredMiVersion(parser, vMajor, vMinor, vPatch))
                                         ? elf::Version(vMajor, vMinor, vPatch)
                                         : elf::Version();

        auto regFieldType =
                RegFieldType::get(parser.getContext(), size * CHAR_BIT, 0, value, name,
                                  VPURegMapped::symbolizeEnum<RegFieldDataType>(dataType).value(), requiredMiVersion);
        auto regFieldAttr = RegisterFieldAttr::get(parser.getContext(), regFieldType);
        regFields = parser.getBuilder().getArrayAttr(regFieldAttr);
    } else {
        auto parseRegField = [&]() -> mlir::ParseResult {
            if (auto regFieldType = RegFieldType::parse(parser).dyn_cast_or_null<RegFieldType>()) {
                auto registerFieldAttr = parser.getChecked<RegisterFieldAttr>(parser.getContext(), regFieldType);
                regFieldsVec.push_back(registerFieldAttr);
                return mlir::success();
            }
            return mlir::failure();
        };

        if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces, parseRegField)) {
            return mlir::Type();
        }
        regFields = parser.getBuilder().getArrayAttr(regFieldsVec);
    }

    return get(parser.getContext(), size, name, address, regFields, allowOverlap);
}

//
// RegMappedType
//

std::vector<uint8_t> vpux::VPURegMapped::RegMappedType::serialize() const {
    auto regAttrs = getRegs().getValue();
    std::vector<uint8_t> result(getWidth().count(), 0);
    std::for_each(regAttrs.begin(), regAttrs.end(), [&result](const mlir::Attribute& regAttr) {
        auto reg = regAttr.cast<VPURegMapped::RegisterAttr>().getReg();
        auto serializedRegister = reg.serialize();

        auto resultIter = result.begin() + Byte(reg.getAddress()).count();
        for (auto serializedRegIter = serializedRegister.begin(); serializedRegIter != serializedRegister.end();
             ++serializedRegIter, ++resultIter) {
            *resultIter |= *serializedRegIter;
        }
    });

    return result;
}

Byte vpux::VPURegMapped::RegMappedType::getWidth() const {
    auto regAttrs = getRegs().getValue();
    Byte regMappedSize(0);
    for (auto regAttr = regAttrs.begin(); regAttr != regAttrs.end(); regAttr++) {
        auto reg = regAttr->cast<VPURegMapped::RegisterAttr>().getReg();
        auto boundingRegMappedWidth = Byte(reg.getAddress()) + reg.getSizeInBytes();
        regMappedSize = boundingRegMappedWidth > regMappedSize ? boundingRegMappedWidth : regMappedSize;
    }
    return regMappedSize;
}

vpux::VPURegMapped::RegisterType vpux::VPURegMapped::RegMappedType::getRegister(const std::string& name) const {
    auto regsAttrs = getRegs().getValue();

    auto regIter = std::find_if(regsAttrs.begin(), regsAttrs.end(), [&](mlir::Attribute regAttr) {
        return regAttr.cast<VPURegMapped::RegisterAttr>().getReg().getName() == name;
    });
    VPUX_THROW_UNLESS(regIter != regsAttrs.end(), "Register with name {0} is not found in Mapped Register {1}", name,
                      this->getName());
    return regIter->cast<VPURegMapped::RegisterAttr>().getReg();
}

elf::Version vpux::VPURegMapped::RegMappedType::getRequiredMIVersion() const {
    auto regsAttrs = getRegs().getValue();

    elf::Version maxVersion;
    llvm::for_each(regsAttrs, [&maxVersion](mlir::Attribute fieldAttr) {
        auto currVersion = fieldAttr.cast<VPURegMapped::RegisterAttr>().getReg().getRequiredMIVersion();
        maxVersion = std::max(maxVersion, currVersion);
    });

    return maxVersion;
}

mlir::LogicalResult vpux::VPURegMapped::RegMappedType::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, std::string name, ::mlir::ArrayAttr regs) {
#ifdef NDEBUG
    VPUX_UNUSED(emitError);
    VPUX_UNUSED(name);
    VPUX_UNUSED(regs);
#else
    if (name.empty()) {
        return printTo(emitError(), "RegMappedType - name is empty.");
    }

    auto regAttrs = regs.getValue();
    for (const auto& regAttr : regAttrs) {
        auto reg = regAttr.cast<VPURegMapped::RegisterAttr>().getReg();
        auto regSize = reg.getSize();
        auto name = reg.getName();
        auto address = reg.getAddress();
        auto regFields = reg.getRegFields();
        auto allowOverlap = reg.getAllowOverlap();
        if (mlir::failed(vpux::VPURegMapped::RegisterType::verify(emitError, regSize, name, address, regFields,
                                                                  allowOverlap))) {
            return printTo(emitError(), "RegMappedType {0} - invalid.", name);
        }
    }
#endif

    return mlir::success();
}

void VPURegMapped::RegMappedType::print(mlir::AsmPrinter& printer) const {
    printer.increaseIndent();
    printer.printNewline();
    printer << getName() << " {";
    printer.increaseIndent();
    auto regs = getRegs().getValue();
    for (auto regAttr : regs) {
        printer.printNewline();
        regAttr.cast<VPURegMapped::RegisterAttr>().getReg().print(printer);
        if (regAttr != regs.back()) {
            printer << ",";
        }
    }
    printer.decreaseIndent();
    printer.printNewline();
    printer << "}";
    printer.decreaseIndent();
    printer.printNewline();
}

mlir::Type VPURegMapped::RegMappedType::parse(mlir::AsmParser& parser) {
    std::string name;
    mlir::ArrayAttr regs;
    mlir::SmallVector<mlir::Attribute> regsVec;

    if (parser.parseKeywordOrString(&name)) {
        return {};
    }

    auto parseReg = [&]() -> mlir::ParseResult {
        if (auto regType = RegisterType::parse(parser).dyn_cast_or_null<RegisterType>()) {
            auto registerAttr = parser.getChecked<RegisterAttr>(parser.getContext(), regType);
            regsVec.push_back(registerAttr);
            return mlir::success();
        }
        return mlir::failure();
    };

    if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Braces, parseReg)) {
        return mlir::Type();
    }
    regs = parser.getBuilder().getArrayAttr(regsVec);

    return get(parser.getContext(), std::move(name), regs);
}
