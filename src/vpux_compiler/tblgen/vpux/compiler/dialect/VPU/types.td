//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_TYPES
#define VPUX_COMPILER_DIALECT_VPU_TYPES

include "vpux/compiler/dialect/core/type_interfaces.td"
include "vpux/compiler/dialect/VPU/dialect.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/dialect/core/type_interfaces.td"

include "mlir/IR/BuiltinTypeInterfaces.td"

// Base class for VPU dialect types.
class VPU_Type<string name, list<Trait> traits = []>
    : TypeDef<VPU_Dialect, name, traits> {
  let mnemonic = name;
}

//
// DistributedTensor
//

def VPU_DistributedTensor :
        VPU_Type<
            "DistributedTensor",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<ClusterTypeInterface>,
                VPU_DistributedTypeInterface,
                TensorTypeInterface
            ]
        > {
    let summary = "VPU tensor type to describe the tensor tiling";
    let description = [{ This type of tensor is used together with the ClusterTiling operation
                            to describe a tile operation between clusters }];
    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::AffineMapAttr":$order,
        "vpux::IndexedSymbolAttr":$memSpace,
        "DistributionInfoAttr":$distribution
    );

    let genAccessors = 0;

    let extraClassDeclaration = [{
        mlir::RankedTensorType getCompactType() const;

        mlir::AffineMapAttr getOrder() const;
        DistributionInfoAttr getDistribution() const;
        DistributedTensorType cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape, mlir::Type elementType) const;
    }];

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SparseTensor
//

def VPU_SparseTensor :
        VPU_Type<
            "SparseTensor",
            [
                GroupedTypeInterface,
                TensorTypeInterface,
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<VPU_DistributedTypeInterface,
                    ["containsDistributedTypes", "getDistributedTypes", "changeShapeForExplicitDistribution",
                     "changeShapeElemTypeForExplicitDistribution", "changeTypeComponentsForExplicitDistribution",
                     "extractDenseTileForExplicitDistribution", "extractViewTileForExplicitDistribution"]>
            ]
        > {
    let summary = "VPU SparseTensor Type";

    let description = [{
        This type represents a sparse tensor as a group of data and metadata.
        The metadata is represented by the sparsity map and, in some instances,
        the storage element table.

        The data and metadata have to be of one of the following types:
        - mlir::RankedTensorType
        - VPU::DistributedTensorType

        The `isWeights` attribute is used to mark cases where a sparse tensor is
        consumed as weights by the IDU of the user operation. In such cases, the
        weights set size of the sparsity map (i.e. ICxKYxKW) needs to be aligned
        to 16 bytes for every output channel.

        The `sparsityCompression` attribute is utilized for weights sparsity to
        identify the number of elements per output channel that are present in
        the data after removing the sparse values.
    }];

    let parameters = (ins
        "mlir::Type":$data,
        "mlir::Type":$sparsityMap,
        "mlir::Type":$storageElementTable,
        "mlir::UnitAttr":$isWeights,
        "VPU::SparsityCompressionAttr":$sparsityCompression,
        VPU_SEAttr:$seAttr
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::Type":$data,
                CArg<"mlir::Type", "{}">:$sparsityMap,
                CArg<"mlir::Type", "{}">:$storageElementTable,
                CArg<"mlir::UnitAttr", "{}">:$isWeights,
                CArg<"VPU::SparsityCompressionAttr", "{}">:$sparsityCompression,
                CArg<"VPU::SEAttr", "{}">:$seAttr
            ), [{
            return $_get(data.getContext(), data, sparsityMap, storageElementTable, isWeights, sparsityCompression, seAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        SparseTensorType cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape, mlir::Type elementType) const;
    }];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

#endif
