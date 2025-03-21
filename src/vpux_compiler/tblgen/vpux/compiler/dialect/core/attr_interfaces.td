//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_CORE_ATTR_INTERFACES
#define VPUX_COMPILER_CORE_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// HwSpecificMemRefFieldInterface
//

def HwSpecificMemRefFieldInterface : AttrInterface<"HwSpecificMemRefField"> {
    let description = [{
        Interface the implementors of which are stored within the HW-specific fields section of VPU compiler's MemRef attribute.
    }];

    let cppNamespace = "vpux";

    let methods = [
        StaticInterfaceMethod<
            "Returns the HW-specific field key in the MemRef",
            "::mlir::StringLiteral", "memRefKey", (ins),
            /*defaultImplementation=*/ [{
                return ConcreteAttr::getMnemonic();
            }]
        >,
    ];

    let extraClassDeclaration = [{
        // Allow implicit conversion to a subclass of this interface
        template<typename T,
                 std::enable_if_t<std::is_base_of<vpux::HwSpecificMemRefField::Trait<T>, T>::value, bool> = true>
        operator T() const {
            assert(this != nullptr && "The assumption of the conversion is that `this` is always valid");
            return llvm::cast_if_present<T>(*this);
        }
    }];
}

#endif
