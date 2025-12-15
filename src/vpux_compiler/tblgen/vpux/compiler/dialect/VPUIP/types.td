//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPUIP_TYPES
#define VPUX_COMPILER_DIALECT_VPUIP_TYPES

include "vpux/compiler/dialect/VPUIP/dialect.td"
include "vpux/compiler/dialect/VPUIP/attributes.td"
include "vpux/compiler/dialect/VPU/type_interfaces.td"
include "vpux/compiler/dialect/VPU/attributes.td"
include "vpux/compiler/dialect/core/type_interfaces.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.td"

// Base class for VPUIP dialect types.
class VPUIP_Type<string name, list<Trait> traits = []>
    : TypeDef<VPUIP_Dialect, name, traits> {
  let mnemonic = name;
}

//
// DistributedBuffer
//

def VPUIP_DistributedBuffer :
        VPUIP_Type<
            "DistributedBuffer",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<ClusterTypeInterface>,
                ShapedTypeInterface,
                DeclareTypeInterfaceMethods<Bufferization_BufferLikeTypeInterface>,
                VPU_DistributedTypeInterface,
            ]
        > {
    let summary = "VPUIP buffer type to describe the buffer tiling";
    let description = [{ This type of buffer is used to describe an operation tiled across clusters }];
    let genAccessors = 0;

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::MemRefLayoutAttrInterface":$layout,
        "vpux::IndexedSymbolAttr":$memSpace,
        "VPU::DistributionInfoAttr":$distribution,
        "VPUIP::SparsityCompressionAttr":$sparsityCompression
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::MLIRContext*":$context,
                "llvm::ArrayRef<int64_t>":$shape,
                "mlir::Type":$elementType,
                "mlir::MemRefLayoutAttrInterface":$layout,
                "vpux::IndexedSymbolAttr":$memSpace,
                "VPU::DistributionInfoAttr":$distribution,
                CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsityCompression
            ), [{
            return $_get(context, shape, elementType, layout, memSpace, distribution, sparsityCompression);
        }]>
    ];

    let extraClassDeclaration = [{
        mlir::MemRefType getCompactType() const;

        mlir::MemRefLayoutAttrInterface getLayout() const;
        VPU::DistributionInfoAttr getDistribution() const;
        VPUIP::SparsityCompressionAttr getSparsityCompression() const;
        ::mlir::ShapedType cloneWith(std::optional<mlir::ArrayRef<int64_t>> shape,
                                        mlir::Type elementType) const;
        Byte getAllocSizeOfCluster(size_t clusterId) const;
    }];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// SparseBuffer
//

def VPUIP_SparseBuffer :
        VPUIP_Type<
            "SparseBuffer",
            [
                GroupedTypeInterface,
                ShapedTypeInterface,
                DeclareTypeInterfaceMethods<Bufferization_BufferLikeTypeInterface>,
                DeclareTypeInterfaceMethods<NDTypeInterface>,
                DeclareTypeInterfaceMethods<VPU_DistributedTypeInterface,
                    ["containsDistributedTypes", "getDistributedTypes", "changeShapeForExplicitDistribution",
                     "changeShapeElemTypeForExplicitDistribution", "changeTypeComponentsForExplicitDistribution",
                     "extractDenseTileForExplicitDistribution", "extractViewTileForExplicitDistribution"]>
            ]
        > {
    let summary = "VPUIP Sparse Buffer Type";

    let description = [{
        This type represents a sparse buffer as a group of data and metadata.
        The metadata is represented by the sparsity map and, in some instances,
        the storage element table.

        The data and metadata have to be of one of the following types:
        - mlir::MemRefType
        - VPU::DistributedBufferType

        The `isWeights` attribute is used to mark cases where a sparse buffer is
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
        "VPUIP::SparsityCompressionAttr":$sparsityCompression,
        "VPU::SEAttr":$seAttr
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::Type":$data,
                CArg<"mlir::Type", "{}">:$sparsityMap,
                CArg<"mlir::Type", "{}">:$storageElementTable,
                CArg<"mlir::UnitAttr", "{}">:$isWeights,
                CArg<"VPUIP::SparsityCompressionAttr", "{}">:$sparsityCompression,
                CArg<"VPU::SEAttr", "{}">:$seAttr
            ), [{
            return $_get(data.getContext(), data, sparsityMap, storageElementTable, isWeights, sparsityCompression, seAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        ::mlir::ShapedType cloneWith(std::optional<mlir::ArrayRef<int64_t>> /*shape*/,
                                    mlir::Type /*elementType*/) const;
    }];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// ITIBuffer
//

def VPUIP_ITIBuffer :
        VPUIP_Type<
            "ITIBuffer",
            [
                DeclareTypeInterfaceMethods<NDTypeInterface>
            ]
        > {
    let summary = "VPUIP ITI Buffer Type";

    let description = [{
        This buffer type is used to represent buffers with halo regions.
        Halo regions are chuncks of the whole buffer that are read from a different
        cluster (for input ITI buffers) or that are written by another cluster in the
        current one (for output ITI buffers). This type is used to model buffers in a
        multiclustering scenario.

        * `iduSegmentation` - attribute used to differentiate between input or output
        ITI buffers
        * `inwardHaloRegions` - array of halo regions from other clusters
        * `outwardHaloRegions` - array of halo regions that the current buffer will
        share with buffers in a different cluster
    }];

    let genAccessors = 0;

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::MemRefLayoutAttrInterface":$layout,
        "vpux::IndexedSymbolAttr":$memSpace,
        "mlir::UnitAttr": $iduSegmentation,
        ArrayRefParameter<"HaloRegionAttr">: $inwardHaloRegions,
        ArrayRefParameter<"OutwardHaloRegionAttr">: $outwardHaloRegions
    );

    let extraClassDeclaration = [{
        mlir::MemRefLayoutAttrInterface getLayout() const;
        mlir::UnitAttr getIduSegmentation() const;
        ArrayRef<HaloRegionAttr> getInwardHaloRegions() const;
        ArrayRef<OutwardHaloRegionAttr> getOutwardHaloRegions() const;
    }];

    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

//
// BoundedBuffer
//
def VPUIP_BoundedBuffer :
    VPUIP_Type<
        "BoundedBuffer",
        [
            DeclareTypeInterfaceMethods<NDTypeInterface>,
            ShapedTypeInterface,
            DeclareTypeInterfaceMethods<Bufferization_BufferLikeTypeInterface>,
            DeclareTypeInterfaceMethods<VPU_DistributedTypeInterface,
                ["containsDistributedTypes", "getDistributedTypes", "changeShapeForExplicitDistribution",
                 "changeShapeElemTypeForExplicitDistribution", "changeTypeComponentsForExplicitDistribution",
                 "extractDenseTileForExplicitDistribution", "extractViewTileForExplicitDistribution"]>
        ]
    > {
    let summary = "VPUIP Bounded Buffer Type";
    let description = [{
        The BoundedBuffer type represents dynamic tensor constrained by predefined upper bounds.

        *  `data` - an upper-bounded memref buffer type to hold the actual data.
        It defines the maximum capacity and the type of the elements stored within the buffer.
        * `dynamicShape` - a buffer type that contains the actual size of the data buffer.
        The size of the buffer equals to the rank of the data buffer. For 4D tensor - the shape is 4.
    }];

    let parameters = (ins
        "mlir::Type":$data,
        "mlir::Type":$dynamicShape
    );

    let builders = [
        TypeBuilderWithInferredContext<
            (ins
                "mlir::Type":$data,
                "mlir::Type":$dynamicShape
            ), [{
            return $_get(data.getContext(), data, dynamicShape);
        }]>
    ];

    let extraClassDeclaration = [{
        ::mlir::ShapedType cloneWith(std::optional<mlir::ArrayRef<int64_t>> /*shape*/,
                                    mlir::Type /*elementType*/) const;
    }];

    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
    let hasCustomAssemblyFormat = 1;
}

#endif
