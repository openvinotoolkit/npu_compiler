//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: module @CastRegular
module @CastRegular {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf16>
    }

    func.func @main() -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.CastElemType<f16>]
        return %cst : tensor<2x2xf16>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf16>
    // CHECK:       [[TYPE_CORRECTION:%.+]] = IE.Convert([[NGRAPH_CST]]) {dstElemType = f16}
    // CHECK:       return [[TYPE_CORRECTION]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

!qElemType = !quant.uniform<i8:f32:0, {0.05, 0.1}>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<i8:f32:0, {5.000000e-02,1.000000e-01}>

// CHECK: module @CastToQuantizedType
module @CastToQuantizedType {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xi8>
    }

    func.func @main() -> tensor<2x2xi8> {
        %cst = const.Declare tensor<2x2x!qElemType> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.CastElemType<!qElemType>]

        // do quant-cast to satisfy compiler's requirement - output cannot be
        // quantized type.
        %workaround = VPU.QuantizeCast(%cst) { dstElemType = i8 }
            : tensor<2x2x!qElemType> -> tensor<2x2xi8>
        return %workaround : tensor<2x2xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x2xf32>) -> tensor<2x2xsi8>
    // CHECK:       [[TYPE_CORRECTION:%.+]] = IE.Convert([[NGRAPH_CST]]) {dstElemType = i8}
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[TYPE_CORRECTION]]) {dstElemType = [[QTYPE]]}

    // CHECK:       [[COMPATIBILITY_CAST:%.+]] = IE.QuantizeCast([[CAST]])
    // CHECK:       return [[COMPATIBILITY_CAST]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400112233"
        }
    }
#-}

!qElemType = !quant.uniform<i8:f32, 0.05>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<i8:f32, 5.000000e-02>

// CHECK: module @CastFromQuantizedType
module @CastFromQuantizedType {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> tensor<2x2xf32> {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<2x2xi8>,
            [#const.CastElemType<!qElemType>, #const.CastElemType<f32>]
        return %cst : tensor<2x2xf32>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x2xi8>) -> tensor<2x2xf32>
    // CHECK:       [[FIRST_CAST:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE]]}

    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[FIRST_CAST]]) {dstElemType = i8}
    // CHECK:       [[TYPE_CORRECTION:%.+]] = IE.Convert([[CAST]]) {dstElemType = f32}

    // CHECK:       return [[TYPE_CORRECTION]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16:0, {0.0067938112745098041,0.0056410845588235293,0.0042681525735294113,0.0017099417892156863,0.0041800704656862744,0.0015409581801470588,0.006744025735294118,0.0017927581188725489,0.0015265969669117647,0.0019196155024509803}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0067938112745098041:128,0.0056410845588235293:128,0.0042681525735294113:128,0.0017099417892156863:128,0.0041800704656862744:128,0.0015409581801470588:128,0.006744025735294118:128,0.0017927581188725489:128,0.0015265969669117647:128,0.0019196155024509803:128}>

// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16:0, {0.0067938112745098041,0.0056410845588235293,0.0042681525735294113,0.0017099417892156863,0.0041800704656862744,0.0015409581801470588,0.006744025735294118,0.0017927581188725489,0.0015265969669117647,0.0019196155024509803}>
// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16:0, {0.0067938112745098041:128,0.0056410845588235293:128,0.0042681525735294113:128,0.0017099417892156863:128,0.0041800704656862744:128,0.0015409581801470588:128,0.006744025735294118:128,0.0017927581188725489:128,0.0015265969669117647:128,0.0019196155024509803:128}>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: module @PositiveZeroPointDelta
module @PositiveZeroPointDelta {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<10x1x1x1xi8>
    }

    func.func @main() -> tensor<10x1x1x1xi8> {
        // Note: #const.CastElemType here is only used to satisfy the constraints of allowed @main function IO types.
        %cst = const.Declare tensor<10x1x1x1xi8> = dense_resource<vpux_ow_1> : tensor<10x1x1x1xi8>,
            [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<i8>]
        return %cst : tensor<10x1x1x1xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<10x1x1x1xi8>) -> tensor<10x1x1x1xi8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_I8]]}
    // CHECK:       [[PER_AXIS_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = si8}
    // CHECK:       [[PER_AXIS_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_1]]) {dstElemType = [[QTYPE_1]]}
    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PER_AXIS_CAST_2]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<10x1x1x1x[[QTYPE_1]]> -> tensor<10x1x1x1x[[QTYPE_2]]>
    // CHECK:       [[PER_AXIS_CAST_3:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = ui8}
    // CHECK:       [[PER_AXIS_CAST_4:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_3]]) {dstElemType = [[QTYPE_U8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_4]]) {dstElemType = i8}
    // CHECK:       return [[IO_CAST_2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<u8:f16:0, {0.0067938112745098041:128,0.0056410845588235293:128,0.0042681525735294113:128,0.0017099417892156863:128,0.0041800704656862744:128,0.0015409581801470588:128,0.006744025735294118:128,0.0017927581188725489:128,0.0015265969669117647:128,0.0019196155024509803:128}>
!qElemType2 = !quant.uniform<i8:f16:0, {0.0067938112745098041,0.0056410845588235293,0.0042681525735294113,0.0017099417892156863,0.0041800704656862744,0.0015409581801470588,0.006744025735294118,0.0017927581188725489,0.0015265969669117647,0.0019196155024509803}>

// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16:0, {0.0067938112745098041:128,0.0056410845588235293:128,0.0042681525735294113:128,0.0017099417892156863:128,0.0041800704656862744:128,0.0015409581801470588:128,0.006744025735294118:128,0.0017927581188725489:128,0.0015265969669117647:128,0.0019196155024509803:128}>
// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16:0, {0.0067938112745098041,0.0056410845588235293,0.0042681525735294113,0.0017099417892156863,0.0041800704656862744,0.0015409581801470588,0.006744025735294118,0.0017927581188725489,0.0015265969669117647,0.0019196155024509803}>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: module @NegativeZeroPointDelta
module @NegativeZeroPointDelta {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<10x1x1x1xi8>
    }

    func.func @main() -> tensor<10x1x1x1xi8> {
        // Note: #const.CastElemType here is only used to satisfy the constraints of allowed @main function IO types.
        %cst = const.Declare tensor<10x1x1x1xi8> = dense_resource<vpux_ow_1> : tensor<10x1x1x1xi8>,
            [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<i8>]
        return %cst : tensor<10x1x1x1xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<10x1x1x1xi8>) -> tensor<10x1x1x1xi8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_U8]]}
    // CHECK:       [[PER_AXIS_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = ui8}
    // CHECK:       [[PER_AXIS_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_1]]) {dstElemType = [[QTYPE_1]]}
    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PER_AXIS_CAST_2]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<10x1x1x1x[[QTYPE_1]]> -> tensor<10x1x1x1x[[QTYPE_2]]>
    // CHECK:       [[PER_AXIS_CAST_3:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = si8}
    // CHECK:       [[PER_AXIS_CAST_4:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_3]]) {dstElemType = [[QTYPE_I8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_4]]) {dstElemType = i8}
    // CHECK:       return [[IO_CAST_2]]

}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30"
    }
  }
#-}

!qElemType1 = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>

// CHECK: module @QuantizedPadValuePerAxis
module @QuantizedPadValuePerAxis {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<4x1x2x1xsi8>
    }

    func.func @main() -> (tensor<4x1x2x1xsi8>) {
        // Note: #const.CastElemType<si8> here is only used to satisfy the constraints of allowed @main function IO types.
        %cst = const.Declare tensor<4x1x2x1xsi8> = dense_resource<vpux_ow_0> : tensor<4x1x1x1xsi8>,
                    [#const.CastElemType<!qElemType1>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 1, 0]>, #const.CastElemType<si8>]
        return %cst : tensor<4x1x2x1xsi8>
    }
    // CHECK: func.func @init([[ARG0:%.+]]: tensor<4x1x1x1xsi8>) -> tensor<4x1x2x1xsi8>
    // CHECK:   [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK:   [[PAD0:%.+]] = IE.Pad([[CAST0]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.280000e+02 : f64,
    // CHECK-SAME: pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 1, 0]} : tensor<4x1x1x1x[[QTYPE1]]> -> tensor<4x1x2x1x[[QTYPE1]]>
    // CHECK:   [[CAST1:%.+]] = IE.QuantizeCast([[PAD0]]) {dstElemType = si8}
    // CHECK:   return [[CAST1]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30"
    }
  }
#-}

!qElemType1 = !quant.uniform<u8:f16, 0.5:120>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:120>

// CHECK: module @QuantizedPadValue
module @QuantizedPadValue {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x5x1x1xsi8>
    }

    func.func @main() -> (tensor<1x5x1x1xsi8>) {
        // Note: #const.CastElemType<si8> here is only used to satisfy the constraints of allowed @main function IO types.
        %cst = const.Declare tensor<1x5x1x1xsi8> = dense_resource<vpux_ow_0> : tensor<1x4x1x1xsi8>,
            [#const.CastElemType<!qElemType1>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.CastElemType<si8>]
        return %cst : tensor<1x5x1x1xsi8>
    }

    // CHECK: func.func @init([[ARG0:%.+]]: tensor<1x4x1x1xsi8>)
    // CHECK:   [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK:   [[PAD:%.+]] = IE.Pad([[CAST0]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.200000e+02 : f64,
    // CHECK-SAME: pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 1, 0, 0]} : tensor<1x4x1x1x[[QTYPE1]]> -> tensor<1x5x1x1x[[QTYPE1]]>
    // CHECK:   [[CAST1:%.+]] = IE.QuantizeCast([[PAD]]) {dstElemType = si8}
    // CHECK:   return [[CAST1]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:127>
!qElemType2 = !quant.uniform<i8:f16, 0.478921568627451:127>

// CHECK: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00:127>
// CHECK: [[QTYPE2:!.+]] = !quant.uniform<i8:f16, 0.478921568627451:127>

module @QuantizedToQuantizedCast {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<10x1x1x1xi8>
    }

    func.func @main() -> tensor<10x1x1x1xi8> {
        // Note: #const.CastElemType<i8> here is only used to satisfy the constraints of allowed @main function IO types.
        %cst = const.Declare tensor<10x1x1x1xi8> = dense_resource<vpux_ow_1> : tensor<10x1x1x1xi8>,
            [#const.CastElemType<!qElemType1>, #const.CastElemType<!qElemType2>, #const.CastElemType<i8>]
        return %cst : tensor<10x1x1x1xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<10x1x1x1xi8>) -> tensor<10x1x1x1xi8>
    // CHECK:       [[IO_CAST_0:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE1]]}
    // CHECK:       [[QCAST_0:%.+]] = IE.QuantizeCast([[IO_CAST_0]]) {dstElemType = i8}
    // CHECK:       [[QCAST_1:%.+]] = IE.QuantizeCast([[QCAST_0]]) {dstElemType = [[QTYPE2]]}
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[QCAST_1]]) {dstElemType = i8}
    // CHECK:       return [[IO_CAST_1]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16, 1.0>
!qElemType2 = !quant.uniform<u8:f16, 1.0:128>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: module @QuantizedToQuantizedConversion_1D
module @QuantizedToQuantizedConversion_1D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<10xui8>
    }

    func.func @main() -> tensor<10xui8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<10xui8> = dense_resource<vpux_ow_1> : tensor<10xsi8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<ui8>]
        return %cst : tensor<10xui8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<10xsi8>) -> tensor<10xui8>
    // CHECK:       [[CAST_QTYPE1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[RESHAPE_TO_4D:%.+]] = IE.Reshape([[CAST_QTYPE1]]) {shape_value = [1, 1, 1, 10]}
    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPE_TO_4D]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:      tensor<1x1x1x10x[[QTYPE_1]]> -> tensor<1x1x1x10x[[QTYPE_2]]>
    // CHECK:       [[RESHAPE_FROM_4D:%.+]] = IE.Reshape([[AVG_POOL]]) {shape_value = [10]}

    // CHECK:       [[CAST_U8:%.+]] = IE.QuantizeCast([[RESHAPE_FROM_4D]]) {dstElemType = ui8}
    // CHECK:       return [[CAST_U8]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16, 1.0>
!qElemType2 = !quant.uniform<u8:f16, 1.0:128>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: module @QuantizedToQuantizedConversion_5D
module @QuantizedToQuantizedConversion_5D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x1x1x1x2xui8>
    }

    func.func @main() -> tensor<5x1x1x1x2xui8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<5x1x1x1x2xui8> = dense_resource<vpux_ow_1> : tensor<5x1x1x1x2xsi8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<ui8>]
        return %cst : tensor<5x1x1x1x2xui8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<5x1x1x1x2xsi8>) -> tensor<5x1x1x1x2xui8>
    // CHECK:       [[CAST_QTYPE1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[CAST_QTYPE1]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1, 1]}
    // CHECK-SAME:      tensor<5x1x1x1x2x[[QTYPE_1]]> -> tensor<5x1x1x1x2x[[QTYPE_2]]>

    // CHECK:       [[CAST_U8:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = ui8}
    // CHECK:       return [[CAST_U8]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16:1, {0.1, 0.2, 0.3, 0.4, 0.5}>
!qElemType2 = !quant.uniform<u8:f16:1, {0.1:128, 0.2:128, 0.3:128, 0.4:128, 0.5:128}>
// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16:1, {1.000000e-01,2.000000e-01,3.000000e-01,4.000000e-01,5.000000e-01}>
// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128,5.000000e-01:128}>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: module @QuantizedToQuantizedConversionPerAxis_2D
module @QuantizedToQuantizedConversionPerAxis_2D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x5xui8>
    }

    func.func @main() -> tensor<2x5xui8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<2x5xui8> = dense_resource<vpux_ow_1> : tensor<2x5xsi8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<ui8>]
        return %cst : tensor<2x5xui8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x5xsi8>) -> tensor<2x5xui8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_I8]]}

    // CHECK:       [[PER_AXIS_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = si8}
    // CHECK:       [[PER_AXIS_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_1]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[RESHAPE_TO_4D:%.+]] = IE.Reshape([[PER_AXIS_CAST_2]]) {shape_value = [1, 1, 2, 5]}
    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[RESHAPE_TO_4D]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:      tensor<1x1x2x5x[[QTYPE_1]]> -> tensor<1x1x2x5x[[QTYPE_2]]>
    // CHECK:       [[RESHAPE_FROM_4D:%.+]] = IE.Reshape([[AVG_POOL]]) {shape_value = [2, 5]}

    // CHECK:       [[PER_AXIS_CAST_3:%.+]] = IE.QuantizeCast([[RESHAPE_FROM_4D]]) {dstElemType = ui8}
    // CHECK:       [[PER_AXIS_CAST_4:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_3]]) {dstElemType = [[QTYPE_U8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_AXIS_CAST_4]]) {dstElemType = ui8}
    // CHECK:       return [[IO_CAST_2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16:3, {8.9134667433944398E-4:-24, 3.9134667433944398E-4:-24}>
!qElemType2 = !quant.uniform<u8:f16:3, {8.9134667433944398E-4:104, 3.9134667433944398E-4:104}>
// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16:3, {8.9134667433944398E-4:-24,3.9134667433944397E-4:-24}>
// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16:3, {8.9134667433944398E-4:104,3.9134667433944397E-4:104}>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>


// CHECK: module @QuantizedToQuantizedConversionPerAxisNegativeZp
module @QuantizedToQuantizedConversionPerAxisNegativeZp {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x1x1x2xui8>
    }

    func.func @main() -> tensor<5x1x1x2xui8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<5x1x1x2xui8> = dense_resource<vpux_ow_1> : tensor<5x1x1x2xsi8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<ui8>]
        return %cst : tensor<5x1x1x2xui8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<5x1x1x2xsi8>) -> tensor<5x1x1x2xui8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_I8]]}

    // CHECK:       [[PER_TENSOR_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = si8}
    // CHECK:       [[PER_TENSOR_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_1]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PER_TENSOR_CAST_2]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:      tensor<5x1x1x2x[[QTYPE_1]]> -> tensor<5x1x1x2x[[QTYPE_2]]>

    // CHECK:       [[PER_TENSOR_CAST_3:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = ui8}
    // CHECK:       [[PER_TENSOR_CAST_4:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_3]]) {dstElemType = [[QTYPE_U8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_4]]) {dstElemType = ui8}
    // CHECK:       return [[IO_CAST_2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16, 8.9134667433944398E-4:-24>
!qElemType2 = !quant.uniform<u8:f16, 8.9134667433944398E-4:104>
// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16, 8.9134667433944398E-4:-24>
// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16, 8.9134667433944398E-4:104>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK: module @QuantizedToQuantizedConversionPerTensorNegativeZp
module @QuantizedToQuantizedConversionPerTensorNegativeZp {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x1x1x2xui8>
    }

    func.func @main() -> tensor<5x1x1x2xui8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<5x1x1x2xui8> = dense_resource<vpux_ow_1> : tensor<5x1x1x2xsi8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<ui8>]
        return %cst : tensor<5x1x1x2xui8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<5x1x1x2xsi8>) -> tensor<5x1x1x2xui8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_I8]]}

    // CHECK:       [[PER_TENSOR_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = si8}
    // CHECK:       [[PER_TENSOR_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_1]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PER_TENSOR_CAST_2]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:      tensor<5x1x1x2x[[QTYPE_1]]> -> tensor<5x1x1x2x[[QTYPE_2]]>

    // CHECK:       [[PER_TENSOR_CAST_3:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = ui8}
    // CHECK:       [[PER_TENSOR_CAST_4:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_3]]) {dstElemType = [[QTYPE_U8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_4]]) {dstElemType = ui8}
    // CHECK:       return [[IO_CAST_2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000411223344551122334455"
        }
    }
#-}

!qElemType1 = !quant.uniform<u8:f16, 8.9134667433944398E-4:104>
!qElemType2 = !quant.uniform<i8:f16, 8.9134667433944398E-4:-24>
// CHECK-DAG: [[QTYPE_U8:!.+]] = !quant.uniform<u8:f16, 8.9134667433944398E-4:104>
// CHECK-DAG: [[QTYPE_I8:!.+]] = !quant.uniform<i8:f16, 8.9134667433944398E-4:-24>
// CHECK-DAG: [[QTYPE_1:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>
// CHECK-DAG: [[QTYPE_2:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: module @QuantizedToQuantizedConversionPerTensorNegativeOutZp
module @QuantizedToQuantizedConversionPerTensorNegativeOutZp {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x1x1x2xsi8>
    }

    func.func @main() -> tensor<5x1x1x2xsi8> {
        // Note: surrounding casts is to abide I/O requirements
        %cst = const.Declare tensor<5x1x1x2xsi8> = dense_resource<vpux_ow_1> : tensor<5x1x1x2xui8>,
        [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>, #const.CastElemType<si8>]
        return %cst : tensor<5x1x1x2xsi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<5x1x1x2xui8>) -> tensor<5x1x1x2xsi8>
    // CHECK:       [[IO_CAST_1:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE_U8]]}

    // CHECK:       [[PER_TENSOR_CAST_1:%.+]] = IE.QuantizeCast([[IO_CAST_1]]) {dstElemType = ui8}
    // CHECK:       [[PER_TENSOR_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_1]]) {dstElemType = [[QTYPE_1]]}

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[PER_TENSOR_CAST_2]])
    // CHECK-SAME:      {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME:      tensor<5x1x1x2x[[QTYPE_1]]> -> tensor<5x1x1x2x[[QTYPE_2]]>

    // CHECK:       [[PER_TENSOR_CAST_3:%.+]] = IE.QuantizeCast([[AVG_POOL]]) {dstElemType = si8}
    // CHECK:       [[PER_TENSOR_CAST_4:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_3]]) {dstElemType = [[QTYPE_I8]]}
    // CHECK:       [[IO_CAST_2:%.+]] = IE.QuantizeCast([[PER_TENSOR_CAST_4]]) {dstElemType = si8}
    // CHECK:       return [[IO_CAST_2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: module @Reverse
module @Reverse {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> tensor<2x2xf32> {
        %NOT_supported = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.Reverse<0 : i64>]

        // Note: the "supported" constant is here to show that the exact same
        // constant, given a different transformation, ends up in init schedule
        %supported = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.CastElemType<f16>]

        return %NOT_supported : tensor<2x2xf32>
    }

    // CHECK:   func.func @init([[CST:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf16>
    // CHECK-NEXT:  [[CVT:%.+]] = IE.Convert([[CST]])
    // CHECK-NEXT:  return [[CVT]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: module @ExpandDilated
module @ExpandDilated {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<1x1x3x3xf32>
    }

    func.func @main() -> tensor<1x1x3x3xf32> {
        %NOT_supported = const.Declare tensor<1x1x3x3xf32> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf32>,
            [#const.ExpandDilated<[2, 2]>]

        // Note: the "supported" constant is here to show that the exact same
        // constant, given a different transformation, ends up in init schedule
        %supported = const.Declare tensor<1x1x2x2xf16> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf32>,
            [#const.CastElemType<f16>]

        return %NOT_supported : tensor<1x1x3x3xf32>
    }

    // CHECK:   func.func @init([[CST:%.+]]: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf16>
    // CHECK-NEXT:  [[CVT:%.+]] = IE.Convert([[CST]])
    // CHECK-NEXT:  return [[CVT]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x00000004aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccd6"
        }
    }
#-}

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK: module @AffineReshape
module @AffineReshape {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<1x1x3x3xf32>
    }

    func.func @main() -> tensor<1x1x3x3xf16, {order = #NCWH}> {
        // Note: CastElemType is only here to avoid this constant from being ignored as #const.AffineReshape is view-like.
        %cst = const.Declare tensor<1x1x3x3xf16, {order = #NCWH}> = dense_resource<vpux_ow_1> : tensor<1x1x3x3xf32>,
            [#const.AffineReshape<[[0], [1], [3], [2]], [1, 1, 3, 3]>, #const.CastElemType<f16>]
        return %cst : tensor<1x1x3x3xf16, {order = #NCWH}>
    }

    // CHECK:           func.func @init([[CST:%.+]]: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf16, {order = #NCWH}>
    // CHECK-NEXT:          [[AFFINE:%.+]] = IE.AffineReshape([[CST]])
    // CHECK{LITERAL}:           {dim_mapping = [[0], [1], [3], [2]], shape_value = [1, 1, 3, 3]}
    // CHECK-NEXT:          [[CVT:%.+]] = IE.Convert([[AFFINE]]) {dstElemType = f16}
    // CHECK-NEXT:          return [[CVT]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x01000000000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f7071727374757677"
        }
    }
#-}

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK: [[NCWH:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK: [[NHWC:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @LayoutCastAttr
module @LayoutCastAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x3x4x5xui8>
    }

    func.func @main() -> tensor<2x3x4x5xui8, {order = #NCWH}> {
        // Note: Reorder is only here to avoid this constant from being ignored as #const.LayoutCast is view-like.
        %cst = const.Declare tensor<2x3x4x5xui8, {order = #NCWH}> = dense_resource<vpux_ow_1> : tensor<2x3x4x5xui8>, [#const.LayoutCast<#NHWC>, #const.Reorder<#NCWH>]
        return %cst : tensor<2x3x4x5xui8, {order = #NCWH}>
    }

    // CHECK:           func.func @init([[CST:%.+]]: tensor<2x3x4x5xui8>) -> tensor<2x3x4x5xui8, {order = [[NCWH]]}>
    // CHECK-NEXT:          [[L_CST:%.+]] = IE.LayoutCast([[CST]]) {dst_order = [[NHWC]]}
    // CHECK-NEXT:          [[REO:%.+]] = IE.Reorder([[L_CST]]) {dstOrder = [[NCWH]]}
    // CHECK-NEXT:          return [[REO]]

}


// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}


// CHECK: module @BroadcastAttr
module @BroadcastAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<16x8xf32>
    }

    func.func @main() -> tensor<16x8xf32> {
        %cst = const.Declare tensor<16x8xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Broadcast<0 : i64, 16 : i64>, #const.Broadcast<1 : i64, 8 : i64>]
        return %cst : tensor<16x8xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<4x1xf32>) -> tensor<16x8xf32>
    // CHECK-NEXT:       [[SH_0:%.+]] = const.Declare tensor<2xi64> = dense<[16, 1]> : tensor<2xi64>
    // CHECK-NEXT:       [[BR_0:%.+]] = IE.Broadcast([[IN]], [[SH_0]]) : tensor<4x1xf32>, tensor<2xi64> -> tensor<16x1xf32>
    // CHECK-NEXT:       [[SH_1:%.+]] = const.Declare tensor<2xi64> = dense<[16, 8]> : tensor<2xi64>
    // CHECK-NEXT:       [[BR_1:%.+]] = IE.Broadcast([[BR_0]], [[SH_1]]) : tensor<16x1xf32>, tensor<2xi64> -> tensor<16x8xf32>
    // CHECK-NEXT:   return [[BR_1]]
}


// -----


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}


// CHECK: module @AddAttr
module @AddAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x3x2xf32>
    }

    func.func @main() -> tensor<2x3x2xf32> {
        %cst = const.Declare tensor<2x3x2xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.Add<1.27e-03>]
        return %cst : tensor<2x3x2xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<2x3x2xf32> {
    // CHECK-NEXT:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.270000e-03> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK-NEXT:       [[RET:%.+]] = IE.Add([[IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-NEXT:       return [[RET]]
}


// -----


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}


// CHECK: module @ZeroPadAttr
module @ZeroPadAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x10x13xf32>
    }

    func.func @main() -> tensor<5x10x13xf32> {
        %cst = const.Declare tensor<5x10x13xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.PadWithZero<[1, 3, 5], [2, 4, 6]>]
        return %cst : tensor<5x10x13xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<5x10x13xf32>
    // CHECK-NEXT:       [[RET:%.+]] = IE.Pad([[IN]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [1, 3, 5], pads_end_attr = [2, 4, 6]}
    // CHECK-NEXT:       return [[RET]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}

// per axis quantization with zero point
!qElemType = !quant.uniform<i8:f32:0, {0.05:21, 0.1:21}>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<i8:f32:0, {5.000000e-02:21,1.000000e-01:21}>
// aligned with padded shape
!qElemTypeP = !quant.uniform<i8:f32:0, {0.0:21, 0.05:21, 0.1:21, 0.0:21, 0.0:21}>
// CHECK: [[QTYPEP:!.+]] = !quant.uniform<i8:f32:0, {5.000000e-02:21,5.000000e-02:21,1.000000e-01:21,1.000000e-01:21,1.000000e-01:21}>

// CHECK: module @ZeroPadQuantized
module @ZeroPadQuantized {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<5x10x13xi8>
    }

    func.func @main() -> tensor<5x10x13xi8> {
        %cst = const.Declare tensor<5x10x13x!qElemTypeP> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.CastElemType<!qElemType>, #const.PadWithZero<[1, 3, 5], [2, 4, 6]>]

        // do quant-cast to satisfy compiler's requirement - output cannot be quantized type.
        %workaround = VPU.QuantizeCast(%cst) { dstElemType = i8 }
            : tensor<5x10x13x!qElemTypeP> -> tensor<5x10x13xi8>
        return %workaround : tensor<5x10x13xi8>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<5x10x13xsi8>
    // CHECK-NEXT:       [[DATA_I8:%.+]] = IE.Convert([[IN]]) {dstElemType = i8}
    // CHECK-NEXT:       [[DATA_Q:%.+]] = IE.QuantizeCast([[DATA_I8]]) {dstElemType = [[QTYPE]]} : tensor<2x3x2xi8> -> tensor<2x3x2x[[QTYPE]]>
    // CHECK-NEXT:       [[PADED:%.+]] = IE.Pad([[DATA_Q]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 2.100000e+01 : f64, pads_begin_attr = [1, 3, 5], pads_end_attr = [2, 4, 6]} : tensor<2x3x2x[[QTYPE]]> -> tensor<5x10x13x[[QTYPEP]]>
    // CHECK-NEXT:       [[RET:%.+]] = IE.QuantizeCast([[PADED]]) {dstElemType = si8}
    // CHECK-NEXT:       return [[RET]]
}


// -----


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}

// CHECK: module @RescaleAttr
module @RescaleAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x3x2xf32>
    }

    func.func @main() -> tensor<2x3x2xf32> {
        %cst = const.Declare tensor<2x3x2xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.Rescale<0.66666666666666663e-05 : f32>]
        return %cst : tensor<2x3x2xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
    // CHECK-NEXT:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<6.66666664E-6> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK-NEXT:       [[RET:%.+]] = IE.Multiply([[IN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-NEXT:       return [[RET]]
}


// -----


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}

// Inverts all values in the tensor, i.e. 1 / x for each x in the tensor.
// CHECK: module @ScalarMultInverseAttr
module @ScalarMultInverseAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x3x2xf32>
    }

    func.func @main() -> tensor<2x3x2xf32> {
        %cst = const.Declare tensor<2x3x2xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.ScalarMultInverse]
        return %cst : tensor<2x3x2xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
    // CHECK-NEXT:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-NEXT:       [[RET:%.+]] = IE.Divide([[CST]], [[IN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-NEXT:       return [[RET]]
}

// -----


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x04000000000000000000803f0000004000004040000080400000a0400000c0400000e04000000041000010410000204100003041"
        }
    }
#-}

#transpose_map = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK: [[HCW:#.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK: module @TransposeAttr
module @TransposeAttr {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<3x2x2xf32>
    }

    func.func @main() -> tensor<3x2x2xf32> {
        %cst = const.Declare tensor<3x2x2xf32> = dense_resource<vpux_ow_1> : tensor<2x3x2xf32>, [#const.Transpose<#transpose_map>]
        return %cst : tensor<3x2x2xf32>
    }

    // CHECK:        func.func @init([[IN:%.+]]: tensor<2x3x2xf32>) -> tensor<3x2x2xf32>
    // CHECK-NEXT:       [[RET:%.+]] = IE.Transpose([[IN]]) {order_value = [[HCW]]} : tensor<2x3x2xf32> -> tensor<3x2x2xf32>
    // CHECK-NEXT:       return [[RET]]
}


// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30AEB00E30AEB00E30AEB00E30AEB00E30AEB00E30"
    }
  }
#-}

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: module @MemPermuteConversion
module @MemPermuteConversion {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x4x2x3xsi8>
    }

    func.func @main() -> (tensor<1x4x2x3xsi8>) {
        %cst = const.Declare tensor<1x4x2x3xsi8> = dense_resource<vpux_ow_0> : tensor<1x2x3x4xsi8>,
                    [#const.MemPermute<#NCHW, #NWCH>]
        return %cst : tensor<1x4x2x3xsi8>
    }
    // CHECK: func.func @init([[ARG0:%.+]]: tensor<1x2x3x4xsi8>) -> tensor<1x4x2x3xsi8>
    // CHECK:   [[SHCAST:%.+]] = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs([[ARG0:%.+]] : tensor<1x2x3x4xsi8>) -> tensor<1x2x3x4xsi8>
    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[SHCAST:%.+]]) {dst_order = #NCHW} : tensor<1x2x3x4xsi8> -> tensor<1x2x3x4xsi8>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[LAYOUTCAST:%.+]]) {order_value = #NWCH} : tensor<1x2x3x4xsi8> -> tensor<1x4x2x3xsi8>
    // CHECK:   [[SHCAST2:%.+]] = IE.ShapeCast {shape = [1, 4, 2, 3]} inputs([[TRANSPOSE:%.+]] : tensor<1x4x2x3xsi8>) -> tensor<1x4x2x3xsi8>
    // CHECK:   [[LAYOUTCAST2:%.+]] = IE.LayoutCast([[SHCAST2:%.+]]) {dst_order = #NCHW} : tensor<1x4x2x3xsi8> -> tensor<1x4x2x3xsi8>
    // CHECK:   return [[LAYOUTCAST2]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30AEB00E30AEB00E30AEB00E30AEB00E30AEB00E30"
    }
  }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @MemPermuteConversionNoTranspose
module @MemPermuteConversionNoTranspose {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x2x3x4xsi8, {order = #NHWC}>
    }

    func.func @main() -> (tensor<1x2x3x4xsi8, {order = #NHWC}>) {
        %cst = const.Declare tensor<1x2x3x4xsi8, {order = #NHWC}> = dense_resource<vpux_ow_0> : tensor<1x2x3x4xsi8>,
                            [#const.MemPermute<#NHWC, #NHWC>]
        return %cst : tensor<1x2x3x4xsi8, {order = #NHWC}>
    }
    // CHECK: func.func @init([[ARG0:%.+]]: tensor<1x2x3x4xsi8>) -> tensor<1x2x3x4xsi8, {order = #NHWC}>
    // CHECK:   [[SHCAST:%.+]] = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs([[ARG0:%.+]] : tensor<1x2x3x4xsi8>) -> tensor<1x2x3x4xsi8>
    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[SHCAST:%.+]]) {dst_order = #NCHW} : tensor<1x2x3x4xsi8> -> tensor<1x2x3x4xsi8>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[LAYOUTCAST:%.+]]) {order_value = #NHWC} : tensor<1x2x3x4xsi8> -> tensor<1x3x4x2xsi8>
    // CHECK:   [[SHCAST2:%.+]] = IE.ShapeCast {shape = [1, 2, 3, 4]} inputs([[TRANSPOSE:%.+]] : tensor<1x3x4x2xsi8>) -> tensor<1x2x3x4xsi8>
    // CHECK:   [[LAYOUTCAST2:%.+]] = IE.LayoutCast([[SHCAST2:%.+]]) {dst_order = #NHWC} : tensor<1x2x3x4xsi8> -> tensor<1x2x3x4xsi8, {order = #NHWC}>
    // CHECK:   return [[LAYOUTCAST2]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30AEB0"
    }
  }
#-}

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: module @MemPermuteConversion3D
module @MemPermuteConversion3D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x3x2xsi8>
    }

    func.func @main() -> (tensor<1x3x2xsi8>) {
        %cst = const.Declare tensor<1x3x2xsi8> = dense_resource<vpux_ow_0> : tensor<1x2x3xsi8>,
                    [#const.MemPermute<#CHW, #map>]
        return %cst : tensor<1x3x2xsi8>
    }
    // CHECK: func.func @init([[ARG0:%.+]]: tensor<1x2x3xsi8>) -> tensor<1x3x2xsi8>
    // CHECK:   [[SHCAST:%.+]] = IE.ShapeCast {shape = [1, 2, 3]} inputs([[ARG0:%.+]] : tensor<1x2x3xsi8>) -> tensor<1x2x3xsi8>
    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[SHCAST:%.+]]) {dst_order = #CHW} : tensor<1x2x3xsi8> -> tensor<1x2x3xsi8>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[LAYOUTCAST:%.+]]) {order_value = #map} : tensor<1x2x3xsi8> -> tensor<1x3x2xsi8>
    // CHECK:   [[SHCAST2:%.+]] = IE.ShapeCast {shape = [1, 3, 2]} inputs([[TRANSPOSE:%.+]] : tensor<1x3x2xsi8>) -> tensor<1x3x2xsi8>
    // CHECK:   [[LAYOUTCAST2:%.+]] = IE.LayoutCast([[SHCAST2:%.+]]) {dst_order = #CHW} : tensor<1x3x2xsi8> -> tensor<1x3x2xsi8>
    // CHECK:   return [[LAYOUTCAST2]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        vpux_ow_0: "0x10000000AEB00E30AEB0"
    }
  }
#-}

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: module @MemPermuteConversionNoTranspose3D
module @MemPermuteConversionNoTranspose3D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x2x3xsi8, {order = #map}>
    }

    func.func @main() -> (tensor<1x2x3xsi8, {order = #map}>) {
        %cst = const.Declare tensor<1x2x3xsi8, {order = #map}> = dense_resource<vpux_ow_0> : tensor<1x2x3xsi8>,
                            [#const.MemPermute<#map, #map>]
        return %cst : tensor<1x2x3xsi8, {order = #map}>
    }
    // CHECK: func.func @init([[ARG0:%.+]]: tensor<1x2x3xsi8>) -> tensor<1x2x3xsi8, {order = #map}>
    // CHECK:   [[SHCAST:%.+]] = IE.ShapeCast {shape = [1, 2, 3]} inputs([[ARG0:%.+]] : tensor<1x2x3xsi8>) -> tensor<1x2x3xsi8>
    // CHECK:   [[LAYOUTCAST:%.+]] = IE.LayoutCast([[SHCAST:%.+]]) {dst_order = #CHW} : tensor<1x2x3xsi8> -> tensor<1x2x3xsi8>
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[LAYOUTCAST:%.+]]) {order_value = #map} : tensor<1x2x3xsi8> -> tensor<1x3x2xsi8>
    // CHECK:   [[SHCAST2:%.+]] = IE.ShapeCast {shape = [1, 2, 3]} inputs([[TRANSPOSE:%.+]] : tensor<1x3x2xsi8>) -> tensor<1x2x3xsi8>
    // CHECK:   [[LAYOUTCAST2:%.+]] = IE.LayoutCast([[SHCAST2:%.+]]) {dst_order = #map} : tensor<1x2x3xsi8> -> tensor<1x2x3xsi8, {order = #map}>
    // CHECK:   return [[LAYOUTCAST2]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400112233"
        }
    }
#-}

!qElemType = !quant.uniform<i8:f32:0, {0.05, 0.1}>
!qElemType1 = !quant.uniform<i8:f32:1, {0.05, 0.1}>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<i8:f32:1, {5.000000e-02,1.000000e-01}>

// CHECK: module @DequantizePerAxis4D
module @DequantizePerAxis4D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2x1x1xf32>
        DataInfo "output2" : tensor<2x2x1x1xi8>
    }

    func.func @main() -> (tensor<2x2x1x1xf32>, tensor<2x2x1x1xi8>) {
        %NOT_supported = const.Declare tensor<2x2x1x1xf32> = dense_resource<vpux_ow_1> : tensor<2x2x1x1xi8>,
            [#const.CastElemType<!qElemType>, #const.Dequantize]

        // Added the last cast to differentiate between the constants
        %processed_cst = const.Declare tensor<2x2x1x1xi8> = dense_resource<vpux_ow_1> : tensor<2x2x1x1xi8>,
            [#const.CastElemType<!qElemType1>, #const.Dequantize, #const.CastElemType<i8>]

        return %NOT_supported, %processed_cst : tensor<2x2x1x1xf32>, tensor<2x2x1x1xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x2x1x1xi8>) -> tensor<2x2x1x1xi8>
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE]]}
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CAST]])
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[DEQUANT]])
    // CHECK:       return [[CONVERT]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400112233"
        }
    }
#-}

!qElemType = !quant.uniform<i8:f32:0, {0.05, 0.1}>
!qElemType1 = !quant.uniform<i8:f32:1, {0.05, 0.1}>
// CHECK: [[QTYPE:!.+]] = !quant.uniform<i8:f32:1, {5.000000e-02,1.000000e-01}>

// CHECK: module @DequantizePerAxis5D
module @DequantizePerAxis5D {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2x1x1x1xf32>
        DataInfo "output2" : tensor<2x2x1x1x1xi8>
    }

    func.func @main() -> (tensor<2x2x1x1x1xf32>, tensor<2x2x1x1x1xi8>) {
        %NOT_supported = const.Declare tensor<2x2x1x1x1xf32> = dense_resource<vpux_ow_1> : tensor<2x2x1x1x1xi8>,
            [#const.CastElemType<!qElemType>, #const.Dequantize]

        // Added the last cast to differentiate between the constants
        %processed_cst = const.Declare tensor<2x2x1x1x1xi8> = dense_resource<vpux_ow_1> : tensor<2x2x1x1x1xi8>,
            [#const.CastElemType<!qElemType1>, #const.Dequantize, #const.CastElemType<i8>]

        return %NOT_supported, %processed_cst : tensor<2x2x1x1x1xf32>, tensor<2x2x1x1x1xi8>
    }

    // CHECK:   func.func @init([[NGRAPH_CST:%.+]]: tensor<2x2x1x1x1xi8>) -> tensor<2x2x1x1x1xi8>
    // CHECK:       [[CAST:%.+]] = IE.QuantizeCast([[NGRAPH_CST]]) {dstElemType = [[QTYPE]]}
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[CAST]])
    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[DEQUANT]])
    // CHECK:       return [[CONVERT]]
}
