//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ATTR_INTERFACES
#define VPUX_COMPILER_DIALECT_VPU_ATTR_INTERFACES

include "mlir/IR/OpBase.td"

//
// SEAttrInterface
//

def VPU_SEAttrInterface : AttrInterface<"SEAttr"> {
    let description = [{
        Interface for Storage Elements attributes.
    }];

    let cppNamespace = "vpux::VPU";

    let methods = [
        InterfaceMethod<[{
            Infer the output shape that is obtained after applying the transformation
            represented by the attribute over the input shape.

            For example, a nearest-neighbor interpolate attribute with a scale of [2, 2]
            for data which has a [1, 16, 3, 3] shape would return the [1, 16, 6, 6] shape.
            }],
            "vpux::Shape", "inferOutputShape",
            (ins "vpux::ShapeRef":$inputShape)
        >,

        InterfaceMethod<[{
            Infers the input shape necessary to produce the given output shape, when
            applying the transformation represented by the attribute.

            For example, a nearest-neighbor interpolate attribute with a scale of [2, 2]
            whose output has a [1, 16, 6, 6] shape would return the [1, 16, 3, 3] shape.
            }],
            "vpux::Shape", "backInferInputShape",
            (ins "vpux::ShapeRef":$outputShape)
        >,

        InterfaceMethod<[{
            Infers the input coordinates of the value used to produce the value found at
            the given output coordinates. The input shape may be additionally passed in
            order to help correctly infer the input coordinates (e.g. to restrict them
            in a valid range).

            This can be exemplified with a nearest-neighbor interpolate operation which
            has an ASYMMETRIC coordinate transformation and a FLOOR nearest mode. The input
            shape is [1, 16, 3, 3], the scale is [2, 2], so the output shape is [1, 16, 6, 6]:
            ```
                input           input coordinates ([H, W])
                0 1 2           [0, 0] [0, 1] [0, 2]
                3 4 5           [1, 0] [1, 1] [1, 2]
                6 7 8           [2, 0] [2, 1] [2, 2]

                output          back-inferred input coordinates ([H, W])
                0 0 1 1 2 2     [0, 0] [0, 0] [0, 1] [0, 1] [0, 2] [0, 2]
                0 0 1 1 2 2     [0, 0] [0, 0] [0, 1] [0, 1] [0, 2] [0, 2]
                3 3 4 4 5 5     [1, 0] [1, 0] [1, 1] [1, 1] [1, 2] [1, 2]
                3 3 4 4 5 5     [1, 0] [1, 0] [1, 1] [1, 1] [1, 2] [1, 2]
                6 6 7 7 8 8     [2, 0] [2, 0] [2, 1] [2, 1] [2, 2] [2, 2]
                6 6 7 7 8 8     [2, 0] [2, 0] [2, 1] [2, 1] [2, 2] [2, 2]
            ```
            }],
            "vpux::Shape", "backInferInputCoord",
            (ins "vpux::ShapeRef":$outputCoord, "vpux::ShapeRef":$inputShape)
        >,

        InterfaceMethod<[{
            Given an output tile and the shape of the input data, this method infers the
            input tile and SEAttr required to produce the output tile.

            This can be exemplified with a nearest-neighbor interpolate operation which
            has an ASYMMETRIC coordinate transformation and a FLOOR nearest mode. The input
            shape is [1, 16, 3, 3], the scale is [2, 2], so the output shape is [1, 16, 6, 6]:
            ```
                input             output
                0 1 2           0 0 1 1 2 2
                3 4 5           0 0 1 1 2 2  <-
                6 7 8           3 3 4 4 5 5  <-
                                3 3 4 4 5 5
                                6 6 7 7 8 8
                                6 6 7 7 8 8
            ```

            If the target output tile contains the second and third line:

                - outputTileOffset: [0, 0, 1, 0]
                - outputTileShape:  [1, 16, 2, 6]

            The method will return:

                - inputTileOffset: [0, 0, 0, 0]
                - inputTileShape:  [1, 16, 2, 3]
                - SEInterpolateAttr adapted with internal parameters:
                    - offsets: [0, 0, 1, 0]
                    - sizes:   [1, 16, 2, 6]
            }],
            "vpux::VPU::SEAttr", "extractTile",
            (ins "vpux::ShapeRef":$outputTileOffset, "vpux::ShapeRef":$outputTileShape, "vpux::ShapeRef":$inputShape,
                 "vpux::Shape&":$inputTileOffset, "vpux::Shape&":$inputTileShape)
        >,

        InterfaceMethod<[{
            Infers input tile shape and offset based on the output tile shape and offset and input shape}],
            "std::pair<vpux::Shape, vpux::Shape>", "inferInputTileShapeAndOffset",
            (ins "vpux::ShapeRef":$outputTileOffset, "vpux::ShapeRef":$outputTileShape, "vpux::ShapeRef":$inputShape)
        >,

        InterfaceMethod<[{
            Computes the Storage Element offsets for the given input data after applying
            the transformation represented by the current attribute.

            For example, let's take an example of a nearest-neighbor interpolate attribute
            with scale [2, 2] for data which has a 1x16x3x3xui8 type, compact strides and a
            Storage Element size of 16. The addresses of each Storage Element for the data
            are:
            ```
                0x0   0x10  0x20
                0x30  0x40  0x50
                0x60  0x70  0x80
            ```

            The resulting offsets would be:
            ```
                0x0   0x0   0x10  0x10  0x20  0x20
                0x0   0x0   0x10  0x10  0x20  0x20
                0x30  0x30  0x40  0x40  0x50  0x50
                0x30  0x30  0x40  0x40  0x50  0x50
                0x60  0x60  0x70  0x70  0x80  0x80
                0x60  0x60  0x70  0x70  0x80  0x80
            ```
            }],
            "std::vector<int32_t>", "computeSEOffsets",
            (ins "vpux::ShapeRef":$dataShape, "vpux::StridesRef":$dataStrides, "vpux::Byte":$elemSize, "int64_t":$seSize)
        >,

        InterfaceMethod<[{
            The attribute generates the input Storage Element pointers based on some internal parameters
            which represent a general input transformation. Beside these parameters, it may also contain
            some extra parameters that represent how to tile the transformed input obtained from the
            general parameters.

            This method returns the values of the extra tiling-related parameters, which are represented
            by offsets and sizes. If there are no such parameters, nothing is returned.
            }],
            "std::optional<VPU::SETileInfo>", "getTileInfo", (ins)
        >
    ];
}

def VPU_PPEAttrInterface : AttrInterface<"PPEAttr"> {
    let description = [{
        Interface PPE attributes. The interface doesn't contain any method since it is used only to
        mark other attributes for further casting.
    }];

    let cppNamespace = "vpux::VPU";
}

def VPU_MPEEngineAttrInterface : AttrInterface<"MPEEngineAttr"> {
    let description = [{
        Interface MPEEngine attributes. The interface doesn't contain any method since it is used only to
        mark other attributes for further casting.
    }];

    let cppNamespace = "vpux::VPU";
}

#endif
