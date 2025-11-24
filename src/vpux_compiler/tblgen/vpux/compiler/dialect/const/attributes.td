//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES
#define VPUX_COMPILER_DIALECT_CONST_ATTRIBUTES

include "vpux/compiler/dialect/const/dialect.td"
include "vpux/compiler/dialect/const/attr_interfaces.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

//
// Base classes
//

class Const_Attr<string name, list<Trait> traits = [], string baseCppClass = "mlir::Attribute"> :
        AttrDef<Const_Dialect, name, traits, baseCppClass> {
    let mnemonic = name;
}

//
// OpaqueI64ElementsAttr
//

def Const_OpaqueI64ElementsAttr : Const_Attr<"OpaqueI64Elements", [ElementsAttrInterface, TypedAttrInterface]> {
    let summary = "An opaque representation of a si64 array";
    let description = [{
      Const_OpaqueI64ElementsAttr is a constant attribute that represents an array of 64-bit integers.
      This attribute stores the data for the 64-bit integer array it represents and contains the necessary
      memory to hold these constant values, ensuring that the data is directly accessible within the MLIR context.
    }];

    let parameters = (ins AttributeSelfTypeParameter<"", "mlir::ShapedType">:$type,
                          ArrayRefParameter<"int64_t">:$value);
    let builders = [
        AttrBuilderWithInferredContext<(ins "mlir::ShapedType":$type,
                                          "mlir::ArrayRef<int64_t>":$value), [{
            return $_get(type.getContext(), type, value);
        }]>,
    ];

    let extraClassDeclaration = [{
        using ValueType = ArrayRef<int64_t>;

    }];

    let storageNamespace = "details";
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//
// TransformAttrInterfaceArrayAttr
//

def Const_TransformAttrInterfaceArrayAttr :
    ArrayOfAttr<Const_Dialect, "TransformAttrInterfaceArray", "TransformAttrInterfaceArray", "vpux::Const::TransformAttrInterface">;

//
// TransformAttrInterfaceArrayAttr
//

def Const_ContentArrayAttr :
    ArrayOfAttr<Const_Dialect, "ContentArray", "ContentArray", "vpux::Const::ContentAttr">;

//
// SymElementsAttr
//

def Const_SymElementsAttr : Const_Attr<"SymElements", [
    DeclareAttrInterfaceMethods<ElementsAttrInterface>
]> {
    let summary = "Can reference Rodata containing the elements";
    let description = [{
        This attribute implements mlir::ElementsAttr "interface". It can be used
        instead of mlir::DenseElementsAttr or mlir::DenseResourceElementsAttr to
        reference a const.Rodata op.

        Because of some quirks of MLIR, proper verification is done by Const.DeclareOp.
        Using this attribute on its own should be done with caution.
    }];

    let parameters = (ins
        "::mlir::SymbolRefAttr":$sym_name,
        // We need an additional user-informed type attribute because we cannot lookup
        // the underlying elements to which the symbol points to in all situations.
        // Verifying if this type matches the type of the underlying elements is
        // done by Const.DeclareOp using the 'SymbolOpUserInterface' and the function
        // verifySymbolUses().
        "::mlir::ShapedType":$type
    );

    let hasCustomAssemblyFormat = 1;
}

// Use custom implementation instead of SymbolRefArrayAttr because SymbolRefArrayAttr
// does not overload the return type and would require us to do extra casting.
def Const_SymbolRefAttrArrayAttr : ArrayOfAttr<Const_Dialect,
    "SymElementsAttrArray", "SymElementsAttrArray", "::mlir::SymbolRefAttr">;

//
// ContentAttr
//

def Const_ContentAttr : Const_Attr<"Content"> {
    let summary = "Lazy folded constant content";

    let description = [{
        This object holds base constant and transformations applied to it. It
        provides an API to get transformed values on the fly.
    }];

    let parameters = (ins
        "mlir::ElementsAttr":$baseContent,
        "vpux::Const::TransformAttrInterfaceArrayAttr":$transformations,
        "vpux::NDTypeInterface":$finalType,
        "mlir::UnitAttr":$isSplat
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "::mlir::ElementsAttr":$baseContent), [{
                auto transformationsAttr = vpux::Const::TransformAttrInterfaceArrayAttr::get(
                    baseContent.getContext(), {});
                const auto [finalType, isSplat] = vpux::Const::inferFinalTypeAndSplat(
                    baseContent, {});
                const auto splatAttr = isSplat ? mlir::UnitAttr::get(baseContent.getContext()) : nullptr;
                return $_get(baseContent.getContext(), baseContent, transformationsAttr, finalType, splatAttr);
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent,
                 "mlir::ArrayRef<vpux::Const::TransformAttrInterface>":$transformations), [{
                auto transformationsAttr = vpux::Const::TransformAttrInterfaceArrayAttr::get(
                    baseContent.getContext(), transformations);
                const auto [finalType, isSplat] = vpux::Const::inferFinalTypeAndSplat(
                    baseContent, transformations);
                const auto splatAttr = isSplat ? mlir::UnitAttr::get(baseContent.getContext()) : nullptr;
                return $_get(baseContent.getContext(), baseContent, transformationsAttr, finalType, splatAttr);
            }]
        >,

        // Used for cases when splatness can't be inferred from data such as parsing IR with dense resources
        AttrBuilderWithInferredContext<
            (ins "mlir::ElementsAttr":$baseContent,
                 "mlir::UnitAttr":$splatAttr,
                 "mlir::ArrayRef<vpux::Const::TransformAttrInterface>":$transformations), [{
                auto transformationsAttr = vpux::Const::TransformAttrInterfaceArrayAttr::get(
                    baseContent.getContext(), transformations);
                const auto finalType = vpux::Const::inferFinalType(
                    mlir::cast<vpux::NDTypeInterface>(baseContent.getType()), transformations);
                return $_get(baseContent.getContext(), baseContent, transformationsAttr, finalType, splatAttr);
            }]
        >,
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let genAccessors = 0;
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        using ValueType = ::vpux::Const::Content;

        ValueType fold(bool bypassCache = false) const;

        mlir::ElementsAttr getBaseContent() const;
        mlir::ArrayRef<vpux::Const::TransformAttrInterface> getTransformations() const;
        vpux::NDTypeInterface getType() const;
        bool isSplat() const;
        vpux::Const::TransformAttrInterfaceArrayAttr getTransformationsAttr() const;

        // ContentSetup interface
        struct SpecialSetupCallable {  // lambda that could be assigned
            ::mlir::ElementsAttr baseContent;
            using return_type = ContentAttr;
            return_type operator()(const detail::ContentSetupBase& setup) const {
                return ContentAttr::get(baseContent, setup);
            }
        };

        auto transform() const {
            return vpux::Const::SpecializedContentSetup(getBaseContent().getType(), getTransformations(),
                SpecialSetupCallable{getBaseContent()});
        }

        llvm::hash_code getTransformationHash() const;
        static llvm::hash_code getTransformationHash(
            mlir::ArrayRef<vpux::Const::TransformAttrInterface> transformations);

        // special version of get() that is used when we create Content
        // property from the setup object
        static ContentAttr get(mlir::ElementsAttr base, const detail::ContentSetupBase& setup);

        static mlir::LogicalResult verifyDenseResource(FuncRef<mlir::InFlightDiagnostic()> emitError, mlir::DenseResourceElementsAttr denseResource,
                                                bool isSplat);
    }];

    let storageType = "vpux::Const::ContentAttr";
    let returnType = "vpux::Const::Content";
    let convertFromStorage = "$_self.fold()";
}

//
// MultiContentSymbolAttr
//

def Const_MultiContentSymbolAttr : Const_Attr<"MultiContentSymbol"> {
    let summary = "Represents the content in const.MultiDeclare";

    let description = [{
        This attribute contains a symbol which points to a const.RodataBundle, a type and
        a list of transformations.

        To access the underlying a data, a "dereferenced version", namely an instance of
        Const::MultiContentAttr must be acquired through functionality in Const::MultiDeclare.

        ```MLIR
        @Bundle::@Symbol : tensor<4x4xf32>, [#Const.Add<3.0>]
        ```
    }];

    let parameters = (ins
        // the symbol that references a 'const.RodataBundle' op
        "::mlir::SymbolRefAttr":$bundleSymbol,
        // and its type
        "::vpux::NDTypeInterface":$bundleSymbolType,
        // a list of transformations that would be applied
        "::vpux::Const::TransformAttrInterfaceArrayAttr":$transformations,
        // The resulting type if these transformations were applied to a buffer of type 'bundleSymbolType'. This is inferred.
        "::vpux::NDTypeInterface":$finalType
    );

    let skipDefaultBuilders = 1;
    let builders = [
        AttrBuilder<(ins "::mlir::SymbolRefAttr":$bundleSymbol, "::vpux::NDTypeInterface":$bundleSymbolType, "::vpux::Const::TransformAttrInterfaceArrayAttr":$transformations), [{
                auto transformationsAttr = ::vpux::Const::TransformAttrInterfaceArrayAttr::get($_ctxt, transformations);
                auto finalType = ::vpux::Const::inferFinalType(bundleSymbolType, transformations);
                return $_get($_ctxt, bundleSymbol, bundleSymbolType, transformationsAttr, finalType);
            }]
        >
    ];

    let hasCustomAssemblyFormat = 1;
}

def Const_ElementsAttrArrayAttr : ArrayOfAttr<Const_Dialect,
        "ElementsAttrArray", "ElementsAttrArray", "::mlir::ElementsAttr"> {
    // the default parser generator would use ArrayOfAttr's '$self.getValue()' which leads to a
    // compile error
    let convertFromStorage = "$_self";
}

//
// MultiContentAttr
//

def Const_MultiContentAttr : Const_Attr<"MultiContent"> {
    let summary = "Represents the _dereferenced_ content in const.MultiDeclare";

    let description = [{
        This attribute contains an array of mlir::ElementsAttr attributes, an array of transformations
        and a final type. This attribute *should only* be acquired by calling Const::MultiDeclareOp::dereferenceMultiContentSymbol().
        It _can_ be parsed and printed but is generally not designed to.

        Assume the user parses some IR of the following form:
        ```MLIR
        const.Data @Data {
            const.Rodata @weights_0 dense<1.0> : tensor<4x4xf32>
        }

        const.BundleData @BundleStore {
            const.RodataBundle @bundle = [@Data::@weights_0, @Data::@weights_0] : tensor<4x4xf32>
        }

        // ...
        %cst = const.MultiDeclare tensor<4x4xf32> = @BundleStore::@bundle [<list of transformations>] : tensor<4x4xf32>
        // ...
        ```
        Then `@BundleStore::@bundle [<list of transformations>] : tensor<4x4xf32>` is of type `MultiContentSymbolAttr`.
        To allow passes to access the underlying tensor data, the bundle symbol must first be dereferenced. The
        result is `MultiContentAttr` where the underlying Dense(Resource)ElementsAttr are directly accessible.
    }];

    let parameters = (ins
        // array of mlir::DenseElementsAttr or mlir::DenseResourceElementsAttr
        Const_ElementsAttrArrayAttr:$baseContent,
        // array of transformations
        "vpux::Const::TransformAttrInterfaceArrayAttr":$transformations,
        // The type when applying all transformations to any of the tensors in 'baseContent'. The verifier ensures that all underlying tensors
        // in 'baseContent' have the same type.
        "vpux::NDTypeInterface":$finalType
    );

    let genVerifyDecl = 1;
    let assemblyFormat = "$baseContent `:` $finalType `,` $transformations";

    let builders = [
        AttrBuilder<
                (ins "::mlir::ArrayRef<::mlir::ElementsAttr>":$baseContent, "mlir::ArrayRef<vpux::Const::TransformAttrInterface>":$transformations), [{
                    // Assume that finalType is equal for *all* Dense(Resource)Elements in baseContent. If this is actually the case is checked in
                    // the verifier.
                    VPUX_THROW_WHEN(baseContent.empty(), "'baseContent' is expected to have at least 1 element");
                    VPUX_THROW_WHEN(::llvm::any_of(baseContent, [](auto attr) { return attr == nullptr; }), "Unexpected null elements in 'baseContent'");

                    auto baseContentAttr = vpux::Const::ElementsAttrArrayAttr::get($_ctxt, baseContent);
                    auto transformationsAttr = vpux::Const::TransformAttrInterfaceArrayAttr::get($_ctxt, transformations);
                    auto finalType = vpux::Const::inferFinalTypeAndSplat(baseContent[0], transformations).first;

                    return $_get($_ctxt, baseContentAttr, transformationsAttr, finalType);
                }]
            >
    ];
}

//
// CastElemTypeAttr
//

def Const_CastElemTypeAttr : Const_Attr<"CastElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Cast constant content element type";

    let parameters = (ins
        "mlir::Type":$elemType,
        "llvm::hash_code":$stableHashValue
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::Type":$elemType), [{
                auto hash = stableHashForCastElemType(elemType);
                return $_get(elemType.getContext(), elemType, hash);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// Const_ConvertElemTypeAttr
//

def Const_ConvertElemTypeAttr : Const_Attr<"ConvertElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Convert constant content element type";

    let description = [{
        This transformation performs actual data type conversion. The supported
        conversions are:
        * Sub-byte type unpacking (e.g. i4 -> i8)
        * Quantized to quantized type conversion
    }];

    let parameters = (ins
        "mlir::Type":$elemType,
        "llvm::hash_code":$stableHashValue
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::Type":$elemType), [{
                auto hash = stableHashForConvertElemType(elemType);
                return $_get(elemType.getContext(), elemType, hash);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// DequantizeAttr
//

def Const_DequantizeAttr : Const_Attr<"Dequantize",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Dequantize constant content";
}

//
// QuantizeAttr
//

def Const_QuantizeAttr : Const_Attr<"Quantize",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Converts const values to quantized type";

    let parameters = (ins
        "mlir::quant::QuantizedType":$targetType
    );

    let builders = [

        AttrBuilder<
            (ins "mlir::quant::QuantizedType":$targetType), [{
                return $_get($_ctxt, targetType);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

//
// RescaleAttr
//

def Const_RescaleAttr : Const_Attr<"Rescale",
    [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
  let summary = "Rescale constant content";

  let parameters = (ins
    "vpux::Const::ContentAttr":$scale
  );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$scale), [{
                    auto tensorType = mlir::RankedTensorType::get({1}, scale.getType());
                    auto denseAttr = mlir::DenseElementsAttr::get(tensorType, scale);
                    auto contentAttr = vpux::Const::ContentAttr::get(denseAttr); 
                    return vpux::Const::RescaleAttr::get(contentAttr);
                }]
        >,
        AttrBuilderWithInferredContext<
                (ins "vpux::Const::ContentAttr":$scale), [{
                    return $_get(scale.getContext(), scale);
                }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// AddAttr
//

def Const_AddAttr : Const_Attr<"Add",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Add constant content";

    let parameters = (ins
        "mlir::FloatAttr":$bias
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::FloatAttr":$bias), [{
                return $_get(bias.getContext(), bias);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReshapeAttr
//

def Const_ReshapeAttr : Const_Attr<"Reshape",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reshape constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$shape), [{
                return $_get(shape.getContext(), shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReorderAttr
//

def Const_ReorderAttr : Const_Attr<"Reorder",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reorder constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ReverseAttr
//

def Const_ReverseAttr : Const_Attr<"Reverse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Reverse constant content on specified axis";

    let description = [{
        This transformation reverses content on a specified axis.
        For example, considering an input with dimensions (d0, d1, d2, d3),
        reversing on axis d1 will result in the data of every subpart (d2, d3)
        being reversed.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$axis
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis), [{
                return $_get(axis.getContext(), axis);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}


//
// PadWithZeroAttr
//

def Const_PadWithZeroAttr : Const_Attr<"PadWithZero",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Pad constant content with zeros";

    let parameters = (ins
        "mlir::ArrayAttr":$padBefore,
        "mlir::ArrayAttr":$padAfter
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$padBefore, "mlir::ArrayAttr":$padAfter), [{
                return $_get(padBefore.getContext(), padBefore, padAfter);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SubViewAttr
//

def Const_SubViewAttr : Const_Attr<"SubView",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["supportsSubByteStorageType"]>]> {
    let summary = "Extract subview from constant content";

    let parameters = (ins
        "mlir::ArrayAttr":$offset,
        "mlir::ArrayAttr":$shape
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$offset, "mlir::ArrayAttr":$shape), [{
                return $_get(offset.getContext(), offset, shape);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// BroadcastAttr
//

def Const_BroadcastAttr : Const_Attr<"Broadcast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Broadcast axis by value of constant content";

    let parameters = (ins
        "mlir::IntegerAttr":$axis,
        "mlir::IntegerAttr":$value
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$axis, "mlir::IntegerAttr":$value), [{
                return $_get(value.getContext(), axis, value);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// TransposeAttr
//

def Const_TransposeAttr : Const_Attr<"Transpose",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Transpose constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$order
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$order), [{
                return $_get(order.getContext(), order);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// MemPermuteAttr
//

def Const_MemPermuteAttr : Const_Attr<"MemPermute",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Permute constant content";

    let parameters = (ins
        "mlir::AffineMapAttr":$dstOrder,
        "mlir::AffineMapAttr":$memPerm
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$dstOrder, "mlir::AffineMapAttr":$memPerm), [{
                return $_get(dstOrder.getContext(), dstOrder, memPerm);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// LayoutCastAttr
//

def Const_LayoutCastAttr : Const_Attr<"LayoutCast",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Change constant content layout";

    let parameters = (ins
        "mlir::AffineMapAttr":$dstOrder
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::AffineMapAttr":$dstOrder), [{
                return $_get(dstOrder.getContext(), dstOrder);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// WeightsTablePtrs
//

def Const_RelocateWeightsTableAttr : Const_Attr<"RelocateWeightsTable",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Patches offsets in the weights table";

    let description = [{
        The weights table is used by the DPU to describe the constants associated with the operation.
        It has OCx4 elements, where each element is a 32-bit integer:
          0  [weights_ptr0, sparsity_ptr0, mult_shift0, bias0]
          1  [weights_ptr1, sparsity_ptr1, mult_shift1, bias1]
          ...
         OC  [weights_ptrC, sparsity_ptrC, mult_shiftC, biasC]
        Each row corresponds to an output channel of the DPU operation, where:
          - `weights_ptr` corresponds to the offset in the cluster where the channel's weights are found
          - `sparsity_ptr` corresponds to the offset in the cluster where the channel's weights sparsity map is found
          - `mult_shift` corresponds to the multiply and shift values applied to the output
          - `bias` corresponds to the bias added to the output

        This transformation relocates the pointers from the weights table based on the parameters below:
        - weightsPtr - the base address of the weights associated with the weights table,
                       which will be added to all weights_ptr entries from the table
        - sparsityPtr - the base address of the weights sparsity map associated with the weights table,
                        which will be added to all sparsity_ptr entries from the table
        - offsets - an array of channel offsets, which describes what clusters each channel belongs to
                  - for example, if the associated weights are segmented equally over OC across two clusters,
                    the `offsets` parameter could contain [0, OC/2]; this will cause the second half of the
                    channels to have their pointers reset since this subset of values will be used by the second cluster only
        - weightsTableSize - the size of the weights table, in bytes
        - weightsElemBitSize - the elements size of the weights, in bits
        - weightsCompression - in case the weights are sparse, this parameter indicates how many elements are found
                               in the weights per output channel; it is used internally to correctly determine the new pointers
        - channelOffset - the starting offset of the channels that are being relocated; this is used in case the transformation
                          works on a slice of the channels to ensure the correct pointers are computed
                        - for example, if the transformation receives channels [16-31] for the table, the `channelOffset` would
                          be set to 16, so that the first relocated pointer corresponds to channel 16 instead of channel 0
        - originalOC - when the ODU has the possibility to write less than 16 channels, the weights table still needs to be
                        aligned to 16, so to skip the expand on weights, this attribute tells the real number of channels which
                        need a proper value, the rest up to 16 are just a copy of the first valid pointer value from the table
                        this value refers to the global view of the WT, not the current slice
                     - when this attribute is set to 0, it's functionality is disabled
    }];

    let parameters = (ins
        "mlir::ArrayAttr":$weightsPtr,
        "mlir::IntegerAttr":$sparsityPtr,
        "mlir::ArrayAttr":$offsets,
        "mlir::IntegerAttr":$weightsTableSize,
        "mlir::IntegerAttr":$weightsElemBitSize,
        "VPUIP::SparsityCompressionAttr":$weightsCompression,
        "mlir::IntegerAttr":$channelOffset,
        "mlir::IntegerAttr":$originalOC
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$weightsPtr, "mlir::IntegerAttr":$sparsityPtr, "mlir::ArrayAttr":$offsets, "mlir::IntegerAttr":$weightsTableSize,
                 "mlir::IntegerAttr":$weightsElemBitSize, "VPUIP::SparsityCompressionAttr":$weightsCompression, "mlir::IntegerAttr":$channelOffset, "mlir::IntegerAttr":$originalOC), [{
                return $_get(weightsPtr.getContext(), weightsPtr, sparsityPtr, offsets, weightsTableSize, weightsElemBitSize, weightsCompression, channelOffset, originalOC);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SwizzleConstant
//

def Const_SwizzleConstantAttr : Const_Attr<"SwizzleConstant",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement",
                                                                    "supportsSubByteStorageType"]>]> {
    let summary = "Swizzles the constant based on swizzle key";

    let description = [{
        This transformation applies swizzling transformation to a constant content.
        Configuration of transformation depends on architecture and swizzleKey.
        Constant content will be size aligned (alignSize flag) if its default size
        does not meet HW requirements for this transformation.
    }];

    let parameters = (ins
        "mlir::IntegerAttr":$swizzleKey,
        "mlir::IntegerAttr":$arch
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::IntegerAttr":$swizzleKey, "mlir::IntegerAttr": $arch), [{
                return $_get(swizzleKey.getContext(), swizzleKey, arch);
            }]
        >
    ];

    let storageNamespace = "details";
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}


//
// FuseWeightsAttr
//

def Const_FuseWeightsAttr : Const_Attr<"FuseWeights",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Specialization of Fuse transform with constants specific";

    let parameters = (ins
        "mlir::RankedTensorType":$fusedType,
        OptionalParameter<"Const::ContentAttr">:$weightsTable,
        OptionalParameter<"Const::ContentAttr">:$weights,
        OptionalParameter<"Const::ContentAttr">:$sparsity,
        OptionalParameter<"Const::ContentAttr">:$activations
    );

    let storageNamespace = "details";
    let assemblyFormat = [{
        `<`
        $fusedType
        (`,` `weightsTable` `=` `<` custom<ContentAttr>($weightsTable)^ `>`)?
        (`,` `weights` `=` `<` custom<ContentAttr>($weights)^ `>`)?
        (`,` `sparsity` `=` `<` custom<ContentAttr>($sparsity)^ `>`)?
        (`,` `activations` `=` `<` custom<ContentAttr>($activations)^ `>`)?
        `>`
    }];
}


//
// FuseAttr
//

def Const_FuseAttr : Const_Attr<"Fuse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Fuse constants";

    let parameters = (ins
        "mlir::RankedTensorType":$fusedType,
        "vpux::Const::ContentArrayAttr":$constants
    );

    let storageNamespace = "details";
    let assemblyFormat = [{
        `<`
        $fusedType `,` `constants` `=` `<` $constants `>`
        `>`
    }];
}

//
// ExpandDilatedAttr
//

def Const_ExpandDilatedAttr : Const_Attr<"ExpandDilated",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Expand constant content with zeros according to dilations";

    let parameters = (ins
        "mlir::ArrayAttr":$dilations
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$dilations), [{
                return $_get(dilations.getContext(), dilations);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// GetSparsityMapAttr
//

def Const_GetSparsityMapAttr : Const_Attr<"GetSparsityMap",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
    ]> {
    let summary = "Generate sparsity map";

    let storageNamespace = "details";
}

//
// Sparsify
//

def Const_SparsifyAttr : Const_Attr<"Sparsify", [
        DeclareAttrInterfaceMethods<Const_TransformAttrInterface, ["getPositionRequirement"]>
    ]> {
    let summary = "Remove zeros / zero-points from constant";

    let parameters = (ins
        "mlir::BoolAttr":$compressOutputType,
        "mlir::ElementsAttr":$numActualElements
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType), [{
                return $_get(compressOutputType.getContext(), compressOutputType, nullptr);
            }]
        >,

        AttrBuilderWithInferredContext<
            (ins "mlir::BoolAttr":$compressOutputType, "mlir::ElementsAttr":$numActualElements), [{
                return $_get(compressOutputType.getContext(), compressOutputType, numActualElements);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ChangeShapeAndElemTypeAttr
//

def Const_ChangeShapeAndElemTypeAttr : Const_Attr<"ChangeShapeAndElemType",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Change the shape and element type of a constant";

    let description = [{
        This transformation atomically changes the shape and the element type of a constant.
        It can only be applied to constants that have a per-axis quantized element type
        and the only difference between the original element type and the new one must be
        the quantization dimension. The transformation is needed to avoid any intermediary
        invalid state when the reshape also changes the quantization dimention.
    }];

    let parameters = (ins
        "mlir::ArrayAttr":$shape,
        "mlir::Type":$elemType,
        "llvm::hash_code":$stableHashValue
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$shape, "mlir::Type":$elemType), [{
                auto hash = stableHashForChangeShapeAndElemType(shape, elemType);
                return $_get(shape.getContext(), shape, elemType, hash);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

def Const_ScalarMultInverseAttr : Const_Attr<"ScalarMultInverse",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "Apply scalar multiplicative inverse to the constant content";
}

//
// AffineReshapeAttr
//

def Const_AffineReshapeAttr : Const_Attr<"AffineReshape",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "A special set of Reshape-equivalent operations for which the output layout can be automatically deduced";

    let description = [{
        This transformation is semantically equivalent to the AffineReshape operations in various dialects. Its main purpose is
        to be used in the respective folders. It is equivalent to Reshape except that a rank change is allowed for non-identity
        input orderings.

        'dimMapping' is an array of arrays of integers. Axis i of the input tensor is mapped to the output axes described by the
        integer array dimMapping[i]. A single input axis can be mapped to multiple output axes (e.g. dimMapping = [[0, 1]]),
        in which case the tensor shape at this axis is specified by the 'shapeValue' array.
        Multiple input axes can be mapped to the same output axis (e.g. dimMapping = [[0], [0]]). These axes are combined using
        a reshape operation.

        'shapeValue' is an array of integers that defines the output shape.

        AffineReshape is a view-like transformation. This means no elements get moved around in memory. However, for certain values of
        'dimMapping', the logical view can look like a permutation. To combat this, an output ordering is inferred such that
        it looks like a Reshape transformation under the given input ordering. If no such output ordering can be found,
        then that combination of input ordering and output shape is not permitted.

        Examples

        ```MLIR
            const.Declare tensor<6xi32> = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>, [#const.AffineReshape<[[0, 1]], [6]>]
            // logical view: [0, 1, 2, 3, 4, 5]

            const.Declare tensor<3x2xi32> = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>, [#const.AffineReshape<[[0], [1]], [3, 2]>]
            // logical view: [[0, 1],
            //                [2, 3],
            //                [4, 5]]

            const.Declare tensor<3x2xi32, {order = #CN}> = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32, {order = #NC}>, [#const.AffineReshape<[[1], [0]], [3, 2]>]
            // dimMapping makes the logical view look like a transpose. But just reshaping without adjusting the ordering results in the wrong logical view:
            //     [[0, 1],
            //      [2, 3],
            //      [4, 5]]
            // Inferring the output order to #CN gives us the correct logical view:
            //     [[0, 3],
            //      [1, 4],
            //      [2, 5]]
        ```
    }];

    let parameters = (ins
        // array of array of IntegerAttrs (int64_t)
        "mlir::ArrayAttr":$dimMapping,
        // array of IntegerAttrs (int64_t)
        "mlir::ArrayAttr":$shapeValue
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$dimMapping, "mlir::ArrayAttr":$shapeValue), [{
                return $_get(dimMapping.getContext(), dimMapping, shapeValue);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    // #const.AffineReshape<$dimMapping, $shapeValue>
    let hasCustomAssemblyFormat = 1;
}

//
// InterpolateAttr
//

def Const_InterpolateAttr : Const_Attr<"Interpolate",
        [DeclareAttrInterfaceMethods<Const_TransformAttrInterface>]> {
    let summary = "An attribute for specifying interpolation operations with configurable parameters";

    let description = [{
        This attribute is used to define interpolation operations, which are commonly used in image processing and other domains
        where resizing or resampling of data is required. The attribute allows specifying the interpolation method and other
        relevant parameters.

        'axes' is an array of integers that specifies the axes along which interpolation is applied.

        'sizes' is an array of integers that specify the target sizes for each dimension.

        'mode' is a string attribute that specifies the interpolation method (e.g., CUBIC, NEAREST).

        'coordMode' is a string attribute that specifies how to transform the coordinate (e.g. HALF_PIXEL).

        'nearestMode' is a string attribute that specifies the nearest neighbor rounding mode (e.g. ROUND_PREFER_FLOOR).

        'antialias' is a boolean attribute that indicates whether antialiasing should be applied.

        'padsBegin' and 'padsEnd' are arrays of integers that specify the padding at the beginning and end of each dimension.

        'cubeCoeff' is a floating-point attribute that specifies the coefficient for cubic interpolation.

        Examples

        ```MLIR
            const.Declare tensor<1x384x27x27xf32> = dense<...> : tensor<1x384x37x37xf32>, [#const.Interpolate<[2, 3], [27, 27], "CUBIC", "HALF_PIXEL", "FLOOR", false, [0, 0, 0, 0], [0, 0, 0, 0], -7.500000e-01 : f64>]
            // This specifies a CUBIC interpolation operation on axes 2 and 3, resizing to 27x27, with half pixel coordinate mode,
        ```
    }];

    let parameters = (ins
        // array of IntegerAttrs (int64_t) for axes
        "mlir::ArrayAttr":$axes,
        // array of IntegerAttrs (int64_t) for sizes
        "mlir::ArrayAttr":$sizes,
        // string attribute for interpolation mode
        "mlir::StringAttr":$mode,
        // string attribute for coordinate mode
        "mlir::StringAttr":$coordMode,
        // string attribute for nearest mode
        "mlir::StringAttr":$nearestMode,
        // boolean attribute for antialiasing
        "mlir::BoolAttr":$antialias,
        // array of IntegerAttrs (int64_t) for padding at the beginning
        "mlir::ArrayAttr":$padsBegin,
        // array of IntegerAttrs (int64_t) for padding at the end
        "mlir::ArrayAttr":$padsEnd,
        // float attribute for cubic coefficient
        "mlir::FloatAttr":$cubeCoeff
    );

    let builders = [
        AttrBuilderWithInferredContext<
            (ins "mlir::ArrayAttr":$axes, "mlir::ArrayAttr":$sizes, "mlir::StringAttr":$mode, 
                 "mlir::StringAttr":$coordMode, "mlir::StringAttr":$nearestMode, "mlir::BoolAttr":$antialias, "mlir::ArrayAttr":$padsBegin, "mlir::ArrayAttr":$padsEnd, "mlir::FloatAttr":$cubeCoeff), [{
                return $_get(mode.getContext(), axes, sizes, mode, coordMode, nearestMode, antialias, padsBegin, padsEnd, cubeCoeff);
            }]
        >
    ];

    let storageNamespace = "details";
    let genVerifyDecl = 1;
    let skipDefaultBuilders = 1;
    // #const.Interpolate<$axes, $sizes, $mode, $coordMode, $nearestMode, $antialias, $padsBegin, $padsEnd, $cubeCoeff>
    let hasCustomAssemblyFormat = 1;
}

#endif
