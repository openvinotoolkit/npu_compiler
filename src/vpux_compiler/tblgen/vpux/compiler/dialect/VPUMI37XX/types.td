//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPUMI37XX_TYPES
#define VPUX_COMPILER_DIALECT_VPUMI37XX_TYPES

include "vpux/compiler/dialect/core/attributes.td"
include "vpux/compiler/dialect/VPUMI37XX/dialect.td"
include "vpux/compiler/dialect/VPUMI37XX/attributes.td"

class VPUMI37XX_Type<string name, list<Trait> traits = [],
                          string baseCppClass = "::mlir::Type">
        : TypeDef<VPUMI37XX_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

#endif
