//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_VPUREGMAPPED_ATTR_INTERFACES

include "mlir/IR/AttrTypeBase.td"

class DescriptorAttrBase<Dialect dialect, string descriptorType, string descriptorMnemonic> : AttrDef<dialect, descriptorMnemonic> {
    let mnemonic = descriptorMnemonic;
    let parameters = (ins descriptorType:$regMapped);
    let hasCustomAssemblyFormat = 1;

    let extraClassDefinition = [{
        ::mlir::Attribute $cppClass::parse(::mlir::AsmParser& odsParser, ::mlir::Type) {
            using DescriptorType = decltype($cppClass{}.getRegMapped());
            auto maybeDescriptor = DescriptorType::parse(odsParser);
            if (!maybeDescriptor.has_value()) {
                return {};
            }
            return get(odsParser.getContext(), maybeDescriptor.value());
        }

        void $cppClass::print(::mlir::AsmPrinter& odsPrinter) const {
            getRegMapped().print(odsPrinter);
        }
    }];
}

class PropertyBase<string registerName, string attr, string dialect> : Property<registerName> {
    let convertToAttribute = [{
        return }] # dialect # [{::}] # attr # [{::get($_ctxt, $_storage);
    }];
    let convertFromAttribute = [{
        auto desc = mlir::cast<}] # dialect # [{::}] # attr # [{>($_attr).getRegMapped();
        $_storage = std::move(desc);
        return ::mlir::success();
    }];

    let writeToMlirBytecode = [{
        $_writer.writeAttribute(}] # dialect # [{::}] # attr # [{::get($_ctxt, $_storage))
    }];

    let readFromMlirBytecode = [{
        auto attr = }] # dialect # [{::}] # attr # [{::get(state.getContext(), prop.descriptor);
        reader.readAttribute(attr);
        prop.descriptor = attr.getRegMapped();
    }];

    let hashProperty = "hash_value($_storage)";
    let storageType = registerName;
}


#endif
