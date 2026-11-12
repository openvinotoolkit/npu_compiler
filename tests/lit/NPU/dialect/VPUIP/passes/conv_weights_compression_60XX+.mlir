//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --conv-weights-compression %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvWeightsMoreThan4IC
module @CompressConvWeightsMoreThan4IC {

// CHECK-LABEL: @main
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x16x112x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @main(%arg0: memref<1x16x112x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]> {
    %scale_table_const = const.Declare memref<64x1x1x1xf32> = dense<1.0> : tensor<64x1x1x1xf32>
    %bias_table_const = const.Declare memref<64x1x1x1xf32> = dense<2.0> : tensor<64x1x1x1xf32>
    %weights_const = const.Declare memref<64x16x7x7x!qElemType, {order = #NHWC}> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %weights_CMX = memref.alloc() : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%weights_const : memref<64x16x7x7x!qElemType, {order = #NHWC}>)
        outputs(%weights_CMX : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
         -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %scale_table_CMX = memref.alloc() : memref<64x1x1x1xf32, [@CMX_NN, 0]>
    %scale_table = VPUIP.Copy
        inputs(%scale_table_const : memref<64x1x1x1xf32>)
        outputs(%scale_table_CMX : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
         -> memref<64x1x1x1xf32, [@CMX_NN, 0]>
    %bias_table_CMX = memref.alloc() : memref<64x1x1x1xf32, [@CMX_NN, 0]>
    %bias_table = VPUIP.Copy
        inputs(%bias_table_const : memref<64x1x1x1xf32>)
        outputs(%bias_table_CMX : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
         -> memref<64x1x1x1xf32, [@CMX_NN, 0]>
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          task_type = #VPUIP.nce_task_type<CONV>
      }>
      input(%arg0 : memref<1x16x112x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      weights(%weights : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table_scale(%scale_table : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
      weight_table_bias(%bias_table : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x112x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>

            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK-DAG:        [[WEIGHTS_CONST:%.+]] = const.Declare memref<64x16x7x7x!qElemType2, {order = #NHWC}>
    //CHECK-DAG:        [[SCALE_TABLE_CONST:%.+]] = const.Declare memref<64x1x1x1xf32>
    //CHECK-DAG:        [[BIAS_TABLE_CONST:%.+]] = const.Declare memref<64x1x1x1xf32>
    //CHECK:        [[WEIGHTS_CMX:%.+]] = memref.alloc() : memref<64x16x7x7x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS:%.+]] = VPUIP.Copy inputs([[WEIGHTS_CONST]] : memref<64x16x7x7x!qElemType2, {order = #NHWC}>) outputs([[WEIGHTS_CMX]] :
    //CHECK-SAME:           memref<64x16x7x7x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x16x7x7x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[SCALE_TABLE_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:        [[SCALE_TABLE:%.+]] = VPUIP.Copy inputs([[SCALE_TABLE_CONST]] : memref<64x1x1x1xf32>) outputs([[SCALE_TABLE_CMX]] : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:        [[BIAS_TABLE_CMX:%.+]] = memref.alloc() : memref<64x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:        [[BIAS_TABLE:%.+]] = VPUIP.Copy inputs([[BIAS_TABLE_CONST]] : memref<64x1x1x1xf32>) outputs([[BIAS_TABLE_CMX]] : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:        [[OUTPUTS:%.+]] = memref.alloc() : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[NCE_OP:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64} <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input([[ARG0]] : memref<1x16x112x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS]] : memref<64x16x7x7x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_scale([[SCALE_TABLE]] : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_bias([[BIAS_TABLE]] : memref<64x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[ARG0]] : memref<1x16x112x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUTPUTS]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[OUTPUTS]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>}
    //CHECK:       return [[NCE_OP]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
}

}
