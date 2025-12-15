//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_TYPES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_TYPES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/attributes.td"

class VPURegMapped_Type<string name, list<Trait> traits = [],
                          string baseCppClass = "::mlir::Type">
        : TypeDef<VPURegMapped_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

class VPURegMapped_Version {
    int major = 0;
    int minor = 0;
    int patch = 0;
}

//
// RegField Type
//

def VPURegMapped_RegField : VPURegMapped_Type<"RegField"> {
    let summary = "VPURegMapped RegField Type";

    let description = "This object represents closely a RegField Type";

    let genAccessors = 1;

    let parameters = (ins
        "uint32_t":$width,
        "uint32_t":$pos,
        "uint64_t":$value,
        "std::string":$name,
        "vpux::VPURegMapped::RegFieldDataType":$dataType,
        "elf::Version":$requiredMIVersion
    );

    let extraClassDeclaration = [{
        uint64_t getMap() const {
            if(getWidth() == (sizeof(uint64_t) * CHAR_BIT)) {
                return std::numeric_limits<uint64_t>::max();
            }
            uint64_t currentFieldMap = ((1ULL << getWidth()) - 1) << getPos();
            return currentFieldMap;
        }
    }];
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// Register Type
//

def VPURegMapped_Register : VPURegMapped_Type<"Register"> {
    let summary = "VPURegMapped Register Type";

    let description = "This object represents closely a Register Type";

    let genAccessors = 1;

    let parameters = (ins
        "uint32_t":$size,
        "std::string":$name,
        "uint32_t":$address,
        VPURegMapped_RegisterFieldArrayAttr:$regFields,
        "bool":$allowOverlap
    );

    let extraClassDeclaration = [{
        Byte getSizeInBytes() const;
        std::vector<uint8_t> serialize() const;
        vpux::VPURegMapped::RegFieldType getField(const std::string& name) const;
        elf::Version getRequiredMIVersion() const;
    }];

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// RegMapped Type
//

def VPURegMapped_RegMapped : VPURegMapped_Type<"RegMapped"> {
    let summary = "VPURegMapped RegMapped Type";

    let description = "This object represents closely a RegMapped Type";

    let genAccessors = 1;

    let parameters = (ins
        "std::string":$name,
        VPURegMapped_RegisterArrayAttr:$regs
    );

    let extraClassDeclaration = [{
        Byte getWidth() const;
        std::vector<uint8_t> serialize() const;
        vpux::VPURegMapped::RegisterType getRegister(const std::string& name) const;
        elf::Version getRequiredMIVersion() const;
    }];

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// RegField Specialization Types
//

class VPURegMapped_RegFieldWrapper<Dialect dialect, string name = "", int width = 0, int pos = 0, string dataType = "UINT", VPURegMapped_Version version = VPURegMapped_Version<>> :
        TypeDef<dialect, "RegField_" # name, [], "vpux::VPURegMapped::RegFieldType"> {

    let mnemonic = "RegField_" # name;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
    static elf::Version getVersion() {
        return elf::Version(}] # version.major # [{, }] # version.minor # [{, }] # version.patch #[{);
    }
    static vpux::VPURegMapped::RegFieldType get(mlir::MLIRContext *context, vpux::VPURegMapped::RegFieldValue regFieldVal) {
        auto requiredVersion = regFieldVal.version.checkValidity() ? regFieldVal.version : getVersion();
        return vpux::VPURegMapped::RegFieldType::get(context, }] # width # [{, }] # pos # [{, regFieldVal.value, "}] # name # [{", vpux::VPURegMapped::RegFieldDataType::}] # dataType # [{, requiredVersion}] # [{);
    }
    static vpux::VPURegMapped::RegFieldType get(mlir::MLIRContext *context, uint64_t value) {
        return get(context, vpux::VPURegMapped::RegFieldValue(value, getVersion()));
    }
    static constexpr uint8_t getRegFieldWidth() {
        return }] # width # [{;
    }
    static constexpr uint8_t getRegFieldPos() {
        return }] # pos # [{;
    }
    static constexpr vpux::VPURegMapped::RegFieldDataType getRegFieldDataType() {
        return vpux::VPURegMapped::RegFieldDataType::}] # dataType # [{;
    }
  }];

    // E#135163: revisit tablegen code after new design adoption
    // e.g. these fields could get better names (without protective
    // "_" in the beggining), removing unnecessary code in the class
    // above and etc.
    string _name = name;
    int _offset = pos;
    int _size = width;
    string _type = dataType;
    VPURegMapped_Version _version = version;
}

//
// Register Type
//

class VPURegMapped_RegisterWrapper<Dialect dialect, string regDialectStr, string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
         TypeDef<dialect, "Register_" # name, [], "vpux::VPURegMapped::RegisterType"> {
    list<string> regFieldList = !foreach(name, regFieldTypes, regDialectStr # "RegField_" # name # "Type::get(builder.getContext(), values[\"" # name # "\"])," );
    string regFieldStr = !interleave(regFieldList, "\n");

    list<string> regFieldParams = !foreach(name, regFieldTypes, "const uint64_t " # name);
    list<string> regFieldParamsList = !foreach(name, regFieldTypes, regDialectStr # "RegField_" # name # "Type::get(builder.getContext(), " # name # ")," );

    list<string> regFieldListZeroValues = !foreach(name, regFieldTypes, "{\"" # name # "\", 0}," );
    string regFieldStrZeroValues = !interleave(regFieldListZeroValues, "\n");

    list<string> regFieldListResetValues = !foreach(name, regFieldTypes, "{\"" # name # "\", {extractFieldValueFromRegValue<RegField_" # name # "Type>(), RegField_" # name # "Type::getVersion()}}," );
    string regFieldStrResetValues = !interleave(regFieldListResetValues, "\n");

    list<string> regFieldListValues = !foreach(name, regFieldTypes, "{\"" # name # "\", extractVersionedFieldValue(regType, \"" # name # "\")}," );
    string regFieldStrValues = !interleave(regFieldListValues, "\n");


    let mnemonic = "Register_" # name;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
    static vpux::VPURegMapped::RegisterType get(mlir::OpBuilder builder, std::map<std::string, vpux::VPURegMapped::RegFieldValue> values) {
        auto regFields = getVPURegMapped_RegisterFieldArrayAttr(builder, { }] # "\n" # regFieldStr # "\n" # [{ });
        return vpux::VPURegMapped::RegisterType::get(builder.getContext(), }] # size # [{, "}] # name # [{", }] # address # [{, regFields, }] # allowOverlap # [{ );
    };
    static constexpr unsigned getRegSize() {
        return }] # size # [{;
    }
    static constexpr unsigned getFieldsInfo() {
        return }] # size # [{;
    }
    template<typename REG_FIELD>
    static constexpr uint64_t extractFieldValueFromRegValue() {
        constexpr auto shiftValue = REG_FIELD::getRegFieldPos() + REG_FIELD::getRegFieldWidth();
        static_assert(shiftValue <= 64, "extractFieldValueFromRegValue: Incorrect shiftValue");
        if (shiftValue < 64) {
            return (((long long int)}] # resetValue # [{) & (( 1ull << shiftValue) - 1)) >> REG_FIELD::getRegFieldPos();
        }
        return ((long long int)}] # resetValue # [{) >> REG_FIELD::getRegFieldPos();
    }
    static uint64_t extractFieldValue(vpux::VPURegMapped::RegisterType regType, std::string regFieldName) {
        return regType.getField(regFieldName).getValue();
    }
    static vpux::VPURegMapped::RegFieldValue extractVersionedFieldValue(vpux::VPURegMapped::RegisterType regType, std::string regFieldName) {
        auto regField = regType.getField(regFieldName);
        return vpux::VPURegMapped::RegFieldValue(regField.getValue(), regField.getRequiredMIVersion());
    }
    static std::map<std::string, vpux::VPURegMapped::RegFieldValue> getZeroInitilizationValues() {
        return { }] # regFieldStrZeroValues # [{ };
    }
    static std::map<std::string, vpux::VPURegMapped::RegFieldValue> getResetInitilizationValues() {
        return { }] # regFieldStrResetValues # [{ };
    }
    static std::map<std::string, vpux::VPURegMapped::RegFieldValue> getValues(vpux::VPURegMapped::RegMappedType regMapType) {
        auto regType = regMapType.getRegister("}] # name # [{");
        return { }] # regFieldStrValues # [{ };
    }
  }];

    string _name = name;
    int _offset = address;
    int _size = size;
    list<string> _fields = regFieldTypes;
}

//
// RegMapped Specialization Types
//

class VPURegMapped_RegMappedWrapper<Dialect dialect, string regDialectStr, string name = "", list<string> registerTypes = []> :
         TypeDef<dialect, "RegMapped_" # name, [], "vpux::VPURegMapped::RegisterType"> {
    list<string> regsList = !foreach(name, registerTypes, regDialectStr # "Register_" # name # "Type::get(builder, values[\"" # name # "\"])," );
    string regsStr = !interleave(regsList, "\n");

    list<string> regsListZeroValues = !foreach(name, registerTypes, "{ \"" # name # "\", " # regDialectStr # "Register_" # name # "Type::getZeroInitilizationValues() }," );
    string regStrZeroValues = !interleave(regsListZeroValues, "\n");

    list<string> regsListResetValues = !foreach(name, registerTypes, "{ \"" # name # "\", " # regDialectStr # "Register_" # name # "Type::getResetInitilizationValues() }," );
    string regStrResetValues = !interleave(regsListResetValues, "\n");

    list<string> regsListValues = !foreach(name, registerTypes, "{ \"" # name # "\", " # regDialectStr # "Register_" # name # "Type::getValues(regMapType) }," );
    string regStrValues = !interleave(regsListValues, "\n");


    let mnemonic = "RegMapped_" # name;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
    static vpux::VPURegMapped::RegMappedType get(mlir::OpBuilder builder, std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>> values) {
        auto regs = getVPURegMapped_RegisterArrayAttr(builder, { }] # "\n" # regsStr # "\n" # [{ });
        return vpux::VPURegMapped::RegMappedType::get(builder.getContext(), "}] # name # [{", regs);
    };
    static std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>> getZeroInitilizationValues() {
        return { }] # regStrZeroValues # [{ };
    }
    static std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>> getResetInitilizationValues() {
        return { }] # regStrResetValues # [{ };
    }
    static std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>> getValues(vpux::VPURegMapped::RegMappedType regMapType) {
        return { }] # regStrValues # [{ };
    }
  }];

    string _name = name;
    list<string> _registers = registerTypes;
}

//
// IndexType
//

def VPURegMapped_IndexType : VPURegMapped_Type<"Index"> {
    let summary = "VPUregMapped Index type";

    let description = [{
      An index type containing the value as a parameter
    }];

    let parameters = (ins
        "uint32_t":$tileIdx,
        "uint32_t":$listIdx,
        "uint32_t":$value
    );

    let builders = [
      TypeBuilder<(ins "uint32_t":$value)>,
      TypeBuilder<(ins "uint32_t":$listIdx, "uint32_t":$value)>
    ];

    let hasCustomAssemblyFormat = 1;
}

#endif
