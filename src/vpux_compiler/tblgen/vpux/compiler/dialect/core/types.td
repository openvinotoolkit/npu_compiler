//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_CORE_TYPES
#define VPUX_COMPILER_CORE_TYPES

include "mlir/IR/OpBase.td"

include "vpux/compiler/dialect/core/dialect.td"

def Bool8 : I<8>;

def I4 : I<4>;
def SI4 : SI<4>;
def UI4 : UI<4>;
def I2 : I<2>;
def SI2 : SI<2>;
def UI2 : UI<2>;


def vpux_QuantileType : Type<CPred<"::llvm::isa<::vpux::type::QuantileType>($_self)">, "quantile type">;

class Core_Type<string name, list<Trait> traits = []>
    : TypeDef<Core_Dialect, name, traits> {
  let mnemonic = name;
}

def BoundedTensorType : Core_Type<"BoundedTensor"> {
    let extraClassDeclaration = [{
        static bool classof(mlir::Type type);
        static BoundedTensorType get(mlir::Type type, BoundsRef bounds);
        [[nodiscard]] BoundsRef getBounds() const;
        [[nodiscard]] BoundedShape getDynamicShape() const;
        [[nodiscard]] BoundedTensorType changeBounds(BoundsRef bounds) const;
    }];
}

def DynamicDimsMaskTensorType : Core_Type<"DynamicDimsMaskTensor"> {
    let extraClassDeclaration = [{
        static bool classof(mlir::Type type);
        static DynamicDimsMaskTensorType get(mlir::Type type, DynamicDimsMaskRef dynamicDimsMask);
        [[nodiscard]] DynamicDimsMaskRef getDynamicDimsMask() const;
        [[nodiscard]] DimsMaskedShape getDynamicShape() const;
        [[nodiscard]] DynamicDimsMaskTensorType changeDynamicDimsMask(DynamicDimsMaskRef dynamicDimsMask) const;
    }];
}

#endif
