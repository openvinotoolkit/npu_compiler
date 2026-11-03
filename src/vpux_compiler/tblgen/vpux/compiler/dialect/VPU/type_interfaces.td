//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_TYPE_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//
// ClusterTypeInterface
//

def ClusterTypeInterface : TypeInterface<"ClusterTypeInterface"> {
    let description = [{
        Interface for generating cluster-aware information for types.
    }];

    let cppNamespace = "vpux";

    let methods = [
        InterfaceMethod<[{
            @brief Retrieve the array of compute shapes
            @warning An important thing to consider with regards to compute shapes,
                     is that modes like SEGMENTED and OVERLAPPED take precedence over
                     DUPLICATED and MULTICASTED.
                     In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
                     tensor with shape [1, 64, 4, 4], the compute shape in each cluster is
                     [1, 16, 4, 4], which is needed when tiling and generating workloads,
                     while the allocated shape is [1, 64, 4, 4] (because of duplicated)
                     information which is needed for scheduler and strategy manager,
                     in order to estimate memory
                     In an example of OVERLAPPED over H with uniform segmentation for
                     4 clusters, a tensor of shape [1, 64, 22, 16] will have the following
                     compute distribution across clusters:
                     [1 64 6 16] [1 64 6 16] [1 64 5 16] [1 64 5 16]
            }],
            "SmallVector<Shape>", "getPerClusterComputeShapes", (ins)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of compute shape offsets with regards to the full buffer
            @warning An important thing to consider with regards to compute offsets,
                     is that modes like SEGMENTED and OVERLAPPED take precedence over
                     DUPLICATED and MULTICASTED.
             }],
            "SmallVector<Shape>", "getPerClusterComputeShapeOffsets", (ins)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of memory shapes
            @warning An important thing to consider with regards to memory shapes,
                     is that modes like DUPLICATED and MULTICASTED take precedence over
                     SEGMENTED and OVERLAPPED.
                     In an example case of a "SEGMENTED | DUPLICATED" (needed for SplitOverK)
                     tensor with shape [1, 64, 4, 4], the memory shape in each cluster is
                     [1, 64, 4, 4], which is the allocated shape (because of duplicated)
                     information which is needed for scheduler and strategy manager,
                     in order to estimate memory
                     In an example of OVERLAPPED over H with k3x3s1 pad (1, 1, 1, 1) and
                     uniform segmentation across 4 clusters, a tensor of shape [1, 64, 22, 16]
                     will have the following memory distribution across clusters:
                     [1 64 7 16] [1 64 8 16] [1 64 7 16] [1 64 6 16]
            }],
            "SmallVector<Shape>", "getPerClusterMemoryShapes", (ins)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of memory shape offsets with regards to the full buffer
            @warning An important thing to consider with regards to memory shape offsets,
                     is that modes like DUPLICATED and MULTICASTED take precedence over
                     SEGMENTED and OVERLAPPED.
             }],
            "SmallVector<Shape>", "getPerClusterMemoryShapeOffsets", (ins)
        >,

        InterfaceMethod<[{
            @brief Get largest compact compute shape
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "Shape", "getLargestCompactShape", (ins)
        >,

        InterfaceMethod<[{
            @brief Get the compact compute shape for a specific cluster
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "Shape", "getCompactShape", (ins "int64_t":$tileInd)
        >,

        InterfaceMethod<[{
            @brief Retrieve the array of padding for each cluster
            @warning This function is needed for getting padding in OVERLAPPED mode
            }],
            "SmallVector<vpux::PadInfo>", "getPerClusterPadding", (ins "vpux::PadInfo":$kernel_padding)
        >,

       InterfaceMethod<[{
            @brief Retrieve the array of strided compute shapes
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "SmallVector<StridedShape>", "getPerClusterMemoryStridedShapes", (ins)
        >,

       InterfaceMethod<[{
            @brief Non-throwing counterpart of getPerClusterMemoryStridedShapes
            @warning Returns failure instead of throwing whenever the strides cannot be
                     split per cluster, so a caller that has to fall back rather than abort
                     does not need to replicate the preconditions of each implementation.
                     `logCb` receives the diagnostic the throwing variant would report.
            }],
            "mlir::FailureOr<SmallVector<StridedShape>>", "getPerClusterMemoryStridedShapesOrFailure",
            (ins "vpux::LogCb":$logCb)
        >,

        InterfaceMethod<[{
            @brief Get largest strided compute shape
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "StridedShape", "getLargestStridedShape", (ins)
        >,

        InterfaceMethod<[{
            @brief Get the strided compute shape for a specific cluster
            @warning This function should not be used for memory size calculation,
                     because it does not retrieve the true allocate shape in cases
                     of broadcasting.
            }],
            "StridedShape", "getStridedShape", (ins "int64_t":$tileInd)
        >,

        InterfaceMethod<[{
            @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
            them. This method creates DistributedType with requested shape and DistributedAttr with
            memory_shapes/memory_offsets/computes_shapes/compute_offsets adjusted for the new shape.
            }],
            "vpux::NDTypeInterface", "changeShapeForExplicitDistribution",
                (ins "vpux::ShapeRef":$shape, "vpux::VPU::DistributionInfoAttr":$distributedAttr)
        >,

        InterfaceMethod<[{
            @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
            them. This method creates DistributedType with requested shape and element type and DistributedAttr with
            memory_shapes/memory_offsets/computes_shapes/compute_offsets adjusted for the new shape.
            }],
            "vpux::NDTypeInterface", "changeShapeElemTypeForExplicitDistribution",
                (ins "vpux::ShapeRef":$shape, "mlir::Type":$elemType, "vpux::VPU::DistributionInfoAttr":$distributedAttr)
        >,

        InterfaceMethod<[{
            @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
            them. This method creates DistributedType with requested type components. If shape is one of the changed
            components, it will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offsets
            adjusted for the new shape. Otherwise, it leaves the DistributedAttr untouched.
            }],
            "vpux::NDTypeInterface", "changeTypeComponentsForExplicitDistribution",
                (ins "const TypeComponents&":$typeComponents, "vpux::VPU::DistributionInfoAttr":$distributedAttr)
        >,

        InterfaceMethod<[{
            @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
            them. This method creates DistributedType obtained by extracting a dense tile from the original DistributedType.
            It will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offsets
            adjusted for the resulting dense tile.
            }],
            "vpux::NDTypeInterface", "extractDenseTileForExplicitDistribution",
                (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape, "vpux::VPU::DistributionInfoAttr":$distributedAttr)
        >,

        InterfaceMethod<[{
            @brief When having explicit per cluster memory/compute shapes/offsets, changing the type's shapes invalidates
            them. This method creates DistributedType obtained by extracting a view tile from the original DistributedType.
            It will also update the DistributedAttr with memory_shapes/memory_offsets/computes_shapes/compute_offsets
            adjusted for the resulting view tile.
            }],
            "vpux::NDTypeInterface", "extractViewTileForExplicitDistribution",
                (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape, "vpux::ShapeRef":$tileElemStrides,
                "vpux::VPU::DistributionInfoAttr":$distributedAttr)
        >,
    ];
}


//
// DistributedTypeInterface
//

def VPU_DistributedTypeInterface : TypeInterface<"DistributedTypeInterface"> {
    let description = [{
        Interface for types that work with distributed components.
        It is compatible with types that containg multiple types internally.
    }];

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<
            "Returns true if the components are distributed types",
            "bool", "containsDistributedTypes", (ins),
            [{}],
            [{
                return true;
            }]
        >,

        InterfaceMethod<
            "Returns the distributed components",
            "SmallVector<mlir::Type>", "getDistributedTypes", (ins),
            [{}],
            [{
                return SmallVector<mlir::Type>{$_type};
            }]
        >,

        InterfaceMethod<[{
            @brief Change the type's shapes when the DistributedType has an explicit DistributionInfoAttr.
            Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
            for its components, based on the one that was passed as an arg.
            }],
            "vpux::NDTypeInterface", "changeShapeForExplicitDistribution",
                (ins "vpux::ShapeRef":$shape, "vpux::VPU::DistributionInfoAttr":$distributedAttr),
            [{}],
            [{
                return $_type.changeShapeForExplicitDistribution(shape, distributedAttr);
            }]
        >,

        InterfaceMethod<[{
            @brief Change the type's shapes and elem type when the DistributedType has an explicit DistributionInfoAttr.
            Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
            for its components, based on the one that was passed as an arg.
            }],
            "vpux::NDTypeInterface", "changeShapeElemTypeForExplicitDistribution",
                (ins "vpux::ShapeRef":$shape, "mlir::Type":$elemType, "vpux::VPU::DistributionInfoAttr":$distributedAttr),
            [{}],
            [{
                return $_type.changeShapeElemTypeForExplicitDistribution(shape, elemType, distributedAttr);
            }]
        >,

        InterfaceMethod<[{
            @brief Change the type's components when the DistributedType has an explicit DistributionInfoAttr.
            Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
            for its components, based on the one that was passed as an arg.
            }],
            "vpux::NDTypeInterface", "changeTypeComponentsForExplicitDistribution",
                (ins "const TypeComponents&":$typeComponents, "vpux::VPU::DistributionInfoAttr":$distributedAttr),
            [{}],
            [{
                return $_type.changeTypeComponentsForExplicitDistribution(typeComponents, distributedAttr);
            }]
        >,

        InterfaceMethod<[{
            @brief Extract a dense tile from a DistributedType with an explicit DistributionInfoAttr.
            Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
            for its components, based on the one that was passed as an arg.
            }],
            "vpux::NDTypeInterface", "extractDenseTileForExplicitDistribution",
                (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape, "vpux::VPU::DistributionInfoAttr":$distributedAttr),
            [{}],
            [{
                return $_type.extractDenseTileForExplicitDistribution(tileOffsets, tileShape, distributedAttr);
            }]
        >,

        InterfaceMethod<[{
            @brief Extract a view tile from a DistributedType with an explicit DistributionInfoAttr.
            Each type implementing this interface will decide the appropriate explicit DistributionInfoAttr
            for its components, based on the one that was passed as an arg.
            }],
            "vpux::NDTypeInterface", "extractViewTileForExplicitDistribution",
                (ins "vpux::ShapeRef":$tileOffsets, "vpux::ShapeRef":$tileShape, "vpux::ShapeRef":$tileElemStrides,
                "vpux::VPU::DistributionInfoAttr":$distributedAttr),
            [{}],
            [{
                return $_type.extractViewTileForExplicitDistribution(tileOffsets, tileShape, tileElemStrides, distributedAttr);
            }]
        >,
    ];
}

#endif
