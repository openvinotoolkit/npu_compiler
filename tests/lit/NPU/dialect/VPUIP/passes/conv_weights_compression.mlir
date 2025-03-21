//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --conv-weights-compression %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvWeightsMoreThan4IC
func.func @CompressConvWeightsMoreThan4IC(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %2 = memref.alloc() : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst_0 : memref<64x16x7x7x!qElemType, #NHWC>)
        outputs(%2 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst : memref<64x1x1x4xsi32>)
        outputs(%3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>

            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[cst:%.*]] = const.Declare memref<64x13x7x7x!qElemType2, #NHWC>
    //CHECK-DAG:        [[cst_0:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK:        [[VAR0:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR1:%.*]] = VPUIP.Copy inputs([[cst_0]] : memref<64x1x1x4xsi32>) outputs([[VAR0]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[VAR2:%.*]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR3:%.*]] = memref.alloc() : memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>
    //CHECK:        [[VAR4:%.*]] = VPUIP.Copy inputs([[cst]] : memref<64x13x7x7x!qElemType2, #NHWC>) outputs([[VAR3]] :
    //CHECK-SAME:           memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>
    //CHECK:        [[VAR5:%.*]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs([[VAR4]] : memref<64x13x7x7x!qElemType2, {order = #NHWC, strides = [640, 1, 91, 13]}, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR6:%.*]] = VPUIP.NCEClusterTask {cm_sp_pattern = 8191 : i64,
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2], minimumHardwareExecutionCost = 375613 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[VAR5]] : memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[VAR1]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input(%arg0 : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[VAR2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[VAR2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>}
    //CHECK:       return [[VAR6]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @DoNotCompressSparseConvWeights
func.func @DoNotCompressSparseConvWeights(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst_weights = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> : tensor<64x3x7x7xf16>,
        [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>, #const.Sparsify<false>]
    %cst_weights_sm = const.Declare memref<64x1x1x896xi1> = dense<1.0> : tensor<64x3x7x7xf16>,
        [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>, #const.GetSparsityMap]
    %weights_sparse_ddr = VPUIP.GroupSparseBuffer(%cst_weights, %cst_weights_sm) {is_weights}
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType, #NHWC>, sparsity_map=memref<64x1x1x896xi1>, is_weights>
    %weights_cmx = memref.alloc() : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights_sm_cmx = memref.alloc() : memref<64x1x1x896xi1, [@CMX_NN, 0]>
    %weights_sparse_cmx = VPUIP.GroupSparseBuffer(%weights_cmx, %weights_sm_cmx) {is_weights}
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>
    %weights_sparse = VPUIP.Copy
        inputs(%weights_sparse_ddr : !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType, #NHWC>, sparsity_map=memref<64x1x1x896xi1>, is_weights>)
        outputs(%weights_sparse_cmx : !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>)
        -> !VPUIP.SparseBuffer<data=memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>, sparsity_map=memref<64x1x1x896xi1, [@CMX_NN, 0]>, is_weights>

    %weights_data, %weights_sm = VPUIP.UngroupSparseBuffer(%weights_sparse) {resultSegmentSizes = array<i32: 1, 1, 0>}
        -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>, memref<64x1x1x896xi1, [@CMX_NN, 0]>

    %cst_weights_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %weights_table_cmx = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst_weights_table : memref<64x1x1x4xsi32>)
        outputs(%weights_table_cmx : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
        -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

    %output_cmx = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %output = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights_data : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weights_sparsity_map(%weights_sm : memref<64x1x1x896xi1, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output_cmx : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output_cmx : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
      variants: {
          DPUTask { outEnd = [111, 111, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, outStart = [0, 0, 0] }
      }
      PPE : {
      }
    return %output : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare memref<64x16x7x7x!qElemType2, #NHWC>
    //CHECK-DAG:   [[CST_WEIGHTS_SM:%.+]] = const.Declare memref<64x1x1x896xi1>
    //CHECK:       [[WEIGHTS_SPARSE_DDR:%.+]] = VPUIP.GroupSparseBuffer([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]]) {is_weights}

    //CHECK:       [[WEIGHTS_DATA_CMX:%.+]] = memref.alloc() : memref<64x16x7x7x!qElemType2, #NHWC, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_SM_CMX:%.+]] = memref.alloc() : memref<64x1x1x896xi1, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_SPARSE_CMX:%.+]] = VPUIP.GroupSparseBuffer([[WEIGHTS_DATA_CMX]], [[WEIGHTS_SM_CMX]]) {is_weights}

    //CHECK:       [[WEIGHTS_SPARSE:%.+]] = VPUIP.Copy inputs([[WEIGHTS_SPARSE_DDR]]
    //CHECK-SAME:                                      outputs([[WEIGHTS_SPARSE_CMX]]
    //CHECK:       [[WEIGHTS_DATA:%.+]], [[WEIGHTS_SM:%.+]] = VPUIP.UngroupSparseBuffer([[WEIGHTS_SPARSE]])

    //CHECK:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK:       [[WEIGHTS_TABLE_CMX:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:       [[WEIGHTS_TABLE:%.+]] = VPUIP.Copy inputs([[CST_WEIGHTS_TABLE]]
    //CHECK-SAME:                                     outputs([[WEIGHTS_TABLE_CMX]]

    //CHECK:       [[OUTPUT_CMX:%.+]] = memref.alloc() : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:       [[OUTPUT:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME:          input(%arg0
    //CHECK-SAME:          weights([[WEIGHTS_DATA]]
    //CHECK-SAME:          weights_sparsity_map([[WEIGHTS_SM]]
    //CHECK-SAME:          weight_table([[WEIGHTS_TABLE]]
    //CHECK-SAME:          parent_input(%arg0
    //CHECK-SAME:          parent_output([[OUTPUT_CMX]]
    //CHECK-SAME:          outputs([[OUTPUT_CMX]]
    //CHECK-SAME:          -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
    //CHECK:       return [[OUTPUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<64x1x1x4xsi32,
    affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 2 : i64}>

!WeightsDistributed = !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 2 : i64}>

!OutputDistributed = !VPUIP.DistributedBuffer<1x64x112x112x!qElemType3, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// CHECK-LABEL: @TiledCompressConvWeights
func.func @TiledCompressConvWeights(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
    -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType3, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    %output = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, @CMX_NN,
        {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %weights = VPUIP.Copy inputs(%cst_0 : memref<64x16x7x7x!qElemType, #NHWC>) outputs(%2 : !WeightsDistributed) -> !WeightsDistributed

    %3 = VPURT.AllocDistributed -> !WeightsTableDistributed
    %weights_table = VPUIP.Copy inputs(%cst : memref<64x1x1x4xsi32>) outputs(%3 : !WeightsTableDistributed) -> !WeightsTableDistributed

    %NCEOp = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                kernel_size = [7, 7],
                kernel_strides = [2, 2],
                minimumHardwareExecutionCost = 269263 : i64,
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
            weights(%weights : !WeightsDistributed)
            weight_table(%weights_table : !WeightsTableDistributed)
            parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
            parent_output(%output : !OutputDistributed)
            outputs(%output : !OutputDistributed) -> !OutputDistributed variants :
            {
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [63, 55, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    pad = #VPU.Padding<left = 3 : i64, right = 0 : i64, top = 3 : i64, bottom = 0 : i64>,
                    outStart = [0, 0, 0]
                }
                DPUTask {
                    cluster_id = 0 : i64, outEnd = [111, 55, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>,
                    outStart = [64, 0, 0]
                }
                DPUTask {
                    cluster_id = 1 : i64, outEnd = [63, 111, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    pad = #VPU.Padding<left = 3 : i64, right = 0 : i64, top = 0 : i64, bottom = 2 : i64>,
                    outStart = [0, 56, 0]
                }
                DPUTask {
                    cluster_id = 1 : i64, outEnd = [111, 111, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>,
                    outStart = [64, 56, 0]
                }
            } PPE : {
                PPETask {
                    ppe = #VPU.PPEStub<>
                }
            }

    return %NCEOp : !OutputDistributed

    //CHECK-DAG:        [[CST_WEIGHTS_TABLE:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK-DAG:        [[CST_WEIGHTS:%.*]] = const.Declare memref<64x3x7x7x!qElemType2, #NHWC>
    //CHECK-DAG:        [[OUT_BUFFER:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[WEIGHTS_TABLE_ALLOC:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32,
    //CHECK-SAME:           #NCHW, @CMX_NN,
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[CLUSTER_WEIGHTS_TABLE:%.*]] = VPUIP.Copy
    //CHECK-SAME:           inputs([[CST_WEIGHTS_TABLE]] : memref<64x1x1x4xsi32>)
    //CHECK-SAME:           outputs([[WEIGHTS_TABLE_ALLOC]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)

    //CHECK:        [[WEIGHTS_ALLOC:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2,
    //CHECK-SAME:           {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[CLUSTER_WEIGHTS:%.*]] = VPUIP.Copy
    //CHECK-SAME:           inputs([[CST_WEIGHTS]] : memref<64x3x7x7x!qElemType2, #NHWC>)
    //CHECK-SAME:           outputs([[WEIGHTS_ALLOC]] : !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2, {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2, {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN
    //CHECK:        [[WEIGHTS_SHAPE_CAST:%.*]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]}
    //CHECK-SAME:       inputs([[CLUSTER_WEIGHTS]] : !VPUIP.DistributedBuffer<64x3x7x7x!qElemType2,
    //CHECK-SAME:           {order = #NHWC, strides = [160, 1, 21, 3]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType2, #NHWC, @CMX_NN
    //CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64}
    //CHECK:       [[CONV_OUT:%.*]] = VPUIP.NCEClusterTask {cm_sp_pattern = 7 : i64,
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    //CHECK-SAME:           kernel_size = [7, 7], kernel_strides = [2, 2], minimumHardwareExecutionCost = 269263 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input(%arg0
    //CHECK-SAME:           weights([[WEIGHTS_SHAPE_CAST]]
    //CHECK-SAME:           weight_table([[CLUSTER_WEIGHTS_TABLE]]
    //CHECK-SAME:           parent_input(%arg0
    //CHECK-SAME:           parent_output([[OUT_BUFFER]]
    //CHECK-SAME:           outputs([[OUT_BUFFER]]
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<1x64x112x112x!qElemType1, #NHWC, @CMX_NN,
    //CHECK-SAME:           {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        return [[CONV_OUT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 6.8494566078261131E-4:127>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @CompressConvWeightsSharedTable
func.func @CompressConvWeightsSharedTable(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst_wt = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_w1 = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %cst_w2 = const.Declare memref<64x64x1x1x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x64x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 0]>]

    %weights1_cmx = memref.alloc() : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPUIP.Copy
        inputs(%cst_w1 : memref<64x16x7x7x!qElemType, #NHWC>)
        outputs(%weights1_cmx : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %wt1_cmx = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table1 = VPUIP.Copy
        inputs(%cst_wt : memref<64x1x1x4xsi32>)
        outputs(%wt1_cmx : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output1 = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %conv1 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      weights(%weights1 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table1 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>

            }
      }
      PPE :  {
      }

    %weights2_cmx = memref.alloc() : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights2 = VPUIP.Copy
        inputs(%cst_w2 : memref<64x64x1x1x!qElemType, #NHWC>)
        outputs(%weights2_cmx : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %wt2_cmx = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table2 = VPUIP.Copy
        inputs(%cst_wt : memref<64x1x1x4xsi32>)
        outputs(%wt2_cmx : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output2 = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %conv2 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1],
          kernel_strides = [1, 1],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%conv1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      weights(%weights2 : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%conv1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

            }
      }
      PPE :  {
      }
    return %conv2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[cst_wt1:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK-DAG:        [[cst_wt2:%.*]] = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    //CHECK:        [[WT1_CMX:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WT1:%.*]] = VPUIP.Copy inputs([[cst_wt1]] : memref<64x1x1x4xsi32>) outputs([[WT1_CMX]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[CONV1:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WT1]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[WT2_CMX:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WT2:%.*]] = VPUIP.Copy inputs([[cst_wt2]] : memref<64x1x1x4xsi32>) outputs([[WT2_CMX]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[CONV2:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:           input([[CONV1]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WT2]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>

    //CHECK:       return [[CONV2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// CHECK-LABEL: @CompressTiledConvWeightsSharedTable
func.func @CompressTiledConvWeightsSharedTable(%arg0: memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> {
    %cst_wt = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_w1 = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %cst_w2 = const.Declare memref<64x64x1x1x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x64x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 0]>]

    %weights1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0],
        {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %weights1 = VPUIP.Copy
                inputs(%cst_w1 : memref<64x16x7x7x!qElemType, #NHWC>)
                outputs(%weights1_cmx : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0],
                    {mode = "DUPLICATED", num_clusters = 2 : i64}>)
                -> !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0],
                    {mode = "DUPLICATED", num_clusters = 2 : i64}>

    %wt1_cmx = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, [@CMX_NN, 0],
        {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %weights_table1 = VPUIP.Copy
            inputs(%cst_wt : memref<64x1x1x4xsi32>)
            outputs(%wt1_cmx : !VPUIP.DistributedBuffer<64x1x1x4xsi32,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, [@CMX_NN, 0],
                {mode = "DUPLICATED", num_clusters = 2 : i64}>)
            -> !VPUIP.DistributedBuffer<64x1x1x4xsi32,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, [@CMX_NN, 0],
                {mode = "DUPLICATED", num_clusters = 2 : i64}>

    %output1 = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            kernel_size = [7, 7],
            kernel_strides = [2, 2],
            minimumHardwareExecutionCost = 269263 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
        weights(%weights1 : !VPUIP.DistributedBuffer<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0],
                {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        weight_table(%weights_table1 : !VPUIP.DistributedBuffer<64x1x1x4xsi32,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, [@CMX_NN, 0],
            {mode = "DUPLICATED", num_clusters = 2 : i64}>)
        parent_input(%arg0 : memref<1x16x224x224x!qElemType2, #NHWC, [@CMX_NN, 0]>)
        parent_output(%output1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
        outputs(%output1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
        {
            DPUTask {
                cluster_id = 0 : i64, outEnd = [63, 55, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 3 : i64, right = 0 : i64, top = 3 : i64, bottom = 0 : i64>,
                outStart = [0, 0, 0]
            }
            DPUTask {
                cluster_id = 0 : i64, outEnd = [111, 55, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 0 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>,
                outStart = [64, 0, 0]
            }
            DPUTask {
                cluster_id = 1 : i64, outEnd = [63, 111, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 3 : i64, right = 0 : i64, top = 0 : i64, bottom = 2 : i64>,
                outStart = [0, 56, 0]
            }
            DPUTask {
                cluster_id = 1 : i64, outEnd = [111, 111, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                pad = #VPU.Padding<left = 0 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>,
                outStart = [64, 56, 0]
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }

    %weights2_cmx = memref.alloc() : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights2 = VPUIP.Copy
        inputs(%cst_w2 : memref<64x64x1x1x!qElemType, #NHWC>)
        outputs(%weights2_cmx : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %wt2_cmx = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table2 = VPUIP.Copy
        inputs(%cst_wt : memref<64x1x1x4xsi32>)
        outputs(%wt2_cmx : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output2 = memref.alloc() : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>
    %conv2 = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          kernel_size = [1, 1],
          kernel_strides = [1, 1],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%conv1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      weights(%weights2 : memref<64x64x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table2 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%conv1 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>)
      outputs(%output2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

            }
      }
      PPE :  {
      }
    return %conv2 : memref<1x64x112x112x!qElemType3, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[cst_wt1:%.*]] = const.Declare memref<64x1x1x4xsi32>
    //CHECK-DAG:        [[cst_wt2:%.*]] = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    //CHECK:        [[WT1_CMX:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW
    //CHECK-SAME:           [@CMX_NN, 0], {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WT1:%.*]] = VPUIP.Copy inputs([[cst_wt1]] : memref<64x1x1x4xsi32>)
    //CHECK-SAME:           outputs([[WT1_CMX]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0], {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0], {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[CONV1:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WT1]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, [@CMX_NN, 0], {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[WT2_CMX:%.*]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WT2:%.*]] = VPUIP.Copy inputs([[cst_wt2]] : memref<64x1x1x4xsi32>) outputs([[WT2_CMX]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[CONV2:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:           input([[CONV1]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WT2]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>

    //CHECK:       return [[CONV2]] : memref<1x64x112x112x!qElemType1, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DoNotCompressConvWeightsOCPadding
func.func @DoNotCompressConvWeightsOCPadding(%arg0: memref<1x32x2x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]> {
    %cst = const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.0> :
        tensor<1x2x2x32xf16>, [#const.Reshape<[2, 2, 32]>, #const.SubView<[0, 0, 0], [1, 2, 32]>,
        #const.Reshape<[2, 32, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [14, 0, 0, 0]>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %2 = memref.alloc() : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst : memref<16x32x1x1xf16, #NHWC>)
        outputs(%2 : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst_1 : memref<16x1x1x4xsi32>)
        outputs(%3 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    %output = memref.alloc() : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>

    %NCEOp = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 180 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%arg0 : memref<1x32x2x4xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weights_table : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%arg0 : memref<1x32x2x4xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%output : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%output : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]> variants :
        {
            DPUTask
                {
                    inEnd = [3, 1, 31], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 15], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        } PPE : {
        PPETask {
            ppe = #VPU.PPEStub<>
        }
    }

    return %NCEOp : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK-DAG:        [[WEIGHTS:%.*]] = const.Declare memref<16x32x1x1xf16, #NHWC> = dense<1.000000e+00> : tensor<1x2x2x32xf16>
    //CHECK-DAG:        [[WEIGHTS_TABLE:%.*]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:        [[ALLOC_W:%.*]] = memref.alloc() : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_CMX:%.*]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<16x32x1x1xf16, #NHWC>)
    //CHECK-SAME:           outputs([[ALLOC_W]] : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[ALLOC_WT:%.*]] = memref.alloc() : memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE_CMX:%.*]] = VPUIP.Copy inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:           outputs([[ALLOC_WT]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>

    //CHECK:        [[OUT:%.*]] = memref.alloc() : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[CLUSTER_TASK:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:               kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:               kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 180 : i64,
    //CHECK-SAME:               task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input(%arg0 : memref<1x32x2x4xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS_CMX]] : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input(%arg0 : memref<1x32x2x4xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT]] : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[OUT]] : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    //CHECK:          DPUTask {inEnd = [3, 1, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 1, 15],
    //CHECK-SAME:           outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:        }

    //CHECK:        return [[CLUSTER_TASK]] : memref<1x16x2x4xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>

// CHECK-LABEL: @DoNotCompressConvSubByteWeights
func.func @DoNotCompressConvSubByteWeights(%arg0: memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]> {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x16x7x7x!qElemType, #NHWC> = dense<1.0> :
        tensor<64x13x7x7xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]
    %2 = memref.alloc() : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst_0 : memref<64x16x7x7x!qElemType, #NHWC>)
        outputs(%2 : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
         -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table = VPUIP.Copy
        inputs(%cst : memref<64x1x1x4xsi32>)
        outputs(%3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %output = memref.alloc() : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
    %NCEOp = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          minimumHardwareExecutionCost = 375613 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
      }
      input(%arg0 : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
      weights(%weights : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
      weight_table(%weights_table : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output(%output : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>)
      outputs(%output : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>

            }
      }
      PPE :  {
      }
    return %NCEOp : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:        [[cst:%.*]] = const.Declare memref<64x16x7x7x!qElemType, #NHWC>
    //CHECK:        [[VAR3:%.*]] = memref.alloc() : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR4:%.*]] = VPUIP.Copy inputs([[cst]] : memref<64x16x7x7x!qElemType, #NHWC>) outputs([[VAR3]] :
    //CHECK-SAME:           memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[VAR6:%.*]] = VPUIP.NCEClusterTask {
    //CHECK-NOT: cm_sp_pattern
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7],
    //CHECK-SAME:           kernel_strides = [2, 2], minimumHardwareExecutionCost = 375613 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input(%arg0 : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[VAR4]] : memref<64x16x7x7x!qElemType, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WT_TABLE:%.*]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input(%arg0 : memref<1x16x224x224xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           -> memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-SAME:           variants :  {
    //CHECK:       DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>}
    //CHECK:       return [[VAR6]] : memref<1x64x112x112xf16, #NHWC, [@CMX_NN, 0]>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 1, 1], [1, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 1, 1], [1, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    240x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[128, 16, 1, 1], [112, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 16, 1, 1], [112, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    240x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[128, 1, 1, 4], [112, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [112, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x240x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 128, 1, 1], [1, 112, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 112, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

// CHECK-LABEL: @TiledCompressConvWeightsWithExplicitShapeAndOffset
// CHECK-SAME:    [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
func.func @TiledCompressConvWeightsWithExplicitShapeAndOffset(%arg0: !InputDistributed) -> !OutputDistributed {
    %cst = const.Declare memref<240x16x1x1xf16, #NHWC> = dense<1.0> :
         tensor<240x10x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>,
         #const.PadWithZero<[0, 0, 0, 0], [0, 6, 0, 0]>]
    %alloc = VPURT.AllocDistributed -> !WeightsDistributed
    %0 = VPUIP.Copy
            inputs(%cst : memref<240x16x1x1xf16, #NHWC>)
            outputs(%alloc : !WeightsDistributed)
            -> !WeightsDistributed


    %cst_0 = const.Declare memref<240x1x1x4xsi32> = dense<1> : tensor<240x1x1x4xsi32>
    %alloc_1 = VPURT.AllocDistributed -> !WeightsTableDistributed
    %1 = VPUIP.Copy
        inputs(%cst_0 : memref<240x1x1x4xsi32>)
        outputs(%alloc_1 : !WeightsTableDistributed)
        -> !WeightsTableDistributed

    %alloc_2 = VPURT.AllocDistributed -> !OutputDistributed
    %2 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 210 : i64,
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%arg0 : !InputDistributed)
            weights(%0 : !WeightsDistributed)
            weight_table(%1 : !WeightsTableDistributed)
            parent_input(%arg0 : !InputDistributed)
            parent_output(%alloc_2 : !OutputDistributed)
            outputs(%alloc_2 : !OutputDistributed)
            -> !OutputDistributed variants :
            {
                DPUTask {
                    cluster_id = 0 : i64, inEnd = [0, 0, 15], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                DPUTask {
                    cluster_id = 1 : i64, inEnd = [0, 0, 15], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 111], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            } PPE : {
                PPETask {
                    ppe = #VPU.PPEStub<>
                }
            }

    return %2 : !OutputDistributed

    //CHECK-DAG:              [[CST_WEIGHTS:%.+]] = const.Declare memref<240x10x1x1xf16, #NHWC>
    //CHECK-DAG:              [[CST_WEIGHTS_TABLE:%.+]] = const.Declare memref<240x1x1x4xsi32>

    //CHECK:                  [[WEIGHTS_TABLE_ALLOC:%.+]] = VPURT.AllocDistributed
    //CHECK:                  [[CLUSTER_WEIGHTS_TABLE:%.+]] = VPUIP.Copy
    //CHECK-SAME:                 inputs([[CST_WEIGHTS_TABLE]] : memref<240x1x1x4xsi32>)
    //CHECK-SAME:                 outputs([[WEIGHTS_TABLE_ALLOC]] : !VPUIP.DistributedBuffer<240x1x1x4xsi32

    //CHECK:                  [[OUT_BUFFER:%.+]] = VPURT.AllocDistributed

    //CHECK:                  [[WEIGHTS_ALLOC:%.+]] = VPURT.AllocDistributed
    //CHECK:                  [[CLUSTER_WEIGHTS:%.+]] = VPUIP.Copy
    //CHECK-SAME:                 inputs([[CST_WEIGHTS]] : memref<240x10x1x1xf16, #NHWC>)
    //CHECK-SAME:                 outputs([[WEIGHTS_ALLOC]] : !VPUIP.DistributedBuffer<240x10x1x1xf16
    //CHECK-SAME:                 -> !VPUIP.DistributedBuffer<240x10x1x1xf16, {order = #NHWC, strides = [16, 1, 10, 10]}, @CMX_NN,
    //CHECK-SAME:                     {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:             compute_shapes = [[128, 10, 1, 1], [112, 10, 1, 1]], compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:             memory_shapes = [[128, 10, 1, 1], [112, 10, 1, 1]], memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]}>

    //CHECK:                  [[WEIGHTS_SHAPE_CAST:%.+]] = VPUIP.ShapeCast {shape = [240, 16, 1, 1]}
    //CHECK-SAME:                 inputs([[CLUSTER_WEIGHTS]]

    //CHECK:                  [[CONV_OUT:%.+]] = VPUIP.NCEClusterTask
    //CHECK-SAME:                 input([[INPUT]]
    //CHECK-SAME:                 weights([[WEIGHTS_SHAPE_CAST]]
    //CHECK-SAME:                 weight_table([[CLUSTER_WEIGHTS_TABLE]]
    //CHECK-SAME:                 outputs([[OUT_BUFFER]]


    //CHECK:        return [[CONV_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotCompressConvWithoutPadZero
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x96x96x96xf16, #NHWC, @DDR>
func.func @DoNotCompressConvWithoutPadZero(%arg0: memref<1x96x96x96xf16, #NHWC, @DDR>) -> memref<1x32x96x96xf16, #NHWC, @DDR> {
    %cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare memref<32x32x3x3xf16, #NHWC> = dense<1.0> :
        tensor<1x1x3x9216xf16>, [#const.Reshape<[96, 32, 3, 3]>, #const.SubView<[64, 0, 0, 0], [32, 32, 3, 3]>, #const.Reorder<#NHWC>]
    %0 = VPUIP.SubView %arg0 [0, 64, 0, 0] [1, 32, 96, 96] : memref<1x96x96x96xf16, #NHWC, @DDR>
        to memref<1x32x96x96xf16, {order = #NHWC, strides = [884736, 1, 9216, 96]}, @DDR>
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %2 = VPUIP.NCEClusterTiling
        inputs(%0 as %arg2: memref<1x32x96x96xf16, #NHWC>) outputs(%1 as %arg3: memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %10 = VPUIP.Copy inputs(%arg2 : memref<1x32x96x96xf16, #NHWC>) outputs(%arg3 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN>
    }
    %3 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %4 = VPUIP.NCEClusterTiling
        inputs(%cst_0 as %arg2: memref<32x32x3x3xf16, #NHWC>) outputs(%3 as %arg3: memref<32x32x3x3xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %10 = VPUIP.Copy inputs(%arg2 : memref<32x32x3x3xf16, #NHWC>) outputs(%arg3 : memref<32x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<32x32x3x3xf16, #NHWC, @CMX_NN>
    }
    %5 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %6 = VPUIP.NCEClusterTiling
        inputs(%cst as %arg2: memref<32x1x1x4xsi32>) outputs(%5 as %arg3: memref<32x1x1x4xsi32, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
        %10 = VPUIP.Copy inputs(%arg2 : memref<32x1x1x4xsi32>) outputs(%arg3 : memref<32x1x1x4xsi32, @CMX_NN>) -> memref<32x1x1x4xsi32, @CMX_NN>
    }
    %7 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPUIP.NCEClusterTiling
        inputs(%2 as %arg2: memref<1x32x96x96xf16, #NHWC, @CMX_NN>, %4 as %arg3: memref<32x32x3x3xf16, #NHWC, @CMX_NN>, %6 as %arg4: memref<32x1x1x4xsi32, @CMX_NN>)
        outputs(%7 as %arg5: memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
        -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
        %10 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 47534 : i64, task_type = #VPUIP.nce_task_type<CONV>} input(%arg2 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) weights(%arg3 : memref<32x32x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg4 : memref<32x1x1x4xsi32, @CMX_NN>)
            parent_input(%arg2 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) parent_output(%arg5 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%arg5 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 47, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 95, 31], outStart = [0, 48, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }
    }
    %alloc = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @DDR>
    %9 = VPUIP.NCEClusterTiling
        inputs(%8 as %arg2: memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%alloc as %arg3: memref<1x32x96x96xf16, #NHWC>)
        -> memref<1x32x96x96xf16, #NHWC, @DDR> {
        %10 = VPUIP.Copy inputs(%arg2 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%arg3 : memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC>
    }

    return %9 : memref<1x32x96x96xf16, #NHWC, @DDR>

    //CHECK-DAG:    [[CST:%.*]] = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    //CHECK-DAG:    [[CST_0:%.*]] = const.Declare memref<32x32x3x3xf16, #NHWC> = dense<1.000000e+00> :
    //CHECK-SAME:       tensor<1x1x3x9216xf16>, [#const.Reshape<[96, 32, 3, 3]>, #const.SubView<[64, 0, 0, 0], [32, 32, 3, 3]>, #const.Reorder<#NHWC>]
    //CHECK:        [[VAR0:%.*]] = VPUIP.SubView [[INPUT]] [0, 64, 0, 0] [1, 32, 96, 96] : memref<1x96x96x96xf16, #NHWC, @DDR> to memref<1x32x96x96xf16, {order = #NHWC, strides = [884736, 1, 9216, 96]}, @DDR>
    //CHECK:        [[VAR1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[VAR2:%.*]] = VPUIP.NCEClusterTiling inputs([[VAR0]] as %arg1: memref<1x32x96x96xf16, #NHWC>) outputs([[VAR1]] as %arg2: memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    //CHECK:        [[VAR3:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[VAR4:%.*]] = VPUIP.NCEClusterTiling inputs([[CST_0]] as %arg1: memref<32x32x3x3xf16, #NHWC>) outputs([[VAR3]] as %arg2: memref<32x32x3x3xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       -> !VPUIP.DistributedBuffer<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}> {
    //CHECK:                VPUIP.Copy inputs(%arg1 : memref<32x32x3x3xf16, #NHWC>) outputs(%arg2 : memref<32x32x3x3xf16, #NHWC, @CMX_NN>) -> memref<32x32x3x3xf16, #NHWC, @CMX_NN>
    //CHECK:            }
    //CHECK:        [[VAR5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[VAR6:%.*]] = VPUIP.NCEClusterTiling inputs([[CST]] as %arg1: memref<32x1x1x4xsi32>) outputs([[VAR5]] as %arg2: memref<32x1x1x4xsi32, @CMX_NN>)
    //CHECK:        [[VAR7:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x96x96xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:        [[VAR8:%.*]] = VPUIP.NCEClusterTiling inputs([[VAR2]] as %arg1: memref<1x32x96x96xf16, #NHWC, @CMX_NN>, [[VAR4]] as %arg2: memref<32x32x3x3xf16, #NHWC, @CMX_NN>, [[VAR6]] as %arg3: memref<32x1x1x4xsi32, @CMX_NN>)
    //CHECK:            VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 47534 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           input(%arg1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) weights(%arg2 : memref<32x32x3x3xf16, #NHWC, @CMX_NN>) weight_table(%arg3 : memref<32x1x1x4xsi32, @CMX_NN>) parent_input(%arg1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) parent_output(%arg4 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:           outputs(%arg4 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) -> memref<1x32x96x96xf16, #NHWC, @CMX_NN> variants : {
    //CHECK:                    DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 47, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
    //CHECK:                    DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [95, 95, 31], outStart = [0, 48, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    //CHECK:                } PPE : {
    //CHECK:                    PPETask {ppe = #VPU.PPEStub<>}
    //CHECK:                }
    //CHECK:            }
    //CHECK:        [[ALLOC:%.*]] = memref.alloc() : memref<1x32x96x96xf16, #NHWC, @DDR>
    //CHECK:        [[VAR9:%.*]] = VPUIP.NCEClusterTiling inputs([[VAR8]] as %arg1: memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs([[ALLOC]] as %arg2: memref<1x32x96x96xf16, #NHWC>)
    //CHECK-SAME:       -> memref<1x32x96x96xf16, #NHWC, @DDR> {
    //CHECK:                VPUIP.Copy inputs(%arg1 : memref<1x32x96x96xf16, #NHWC, @CMX_NN>) outputs(%arg2 : memref<1x32x96x96xf16, #NHWC>) -> memref<1x32x96x96xf16, #NHWC>
    //CHECK:            }
    //CHECK:       return [[VAR9]] : memref<1x32x96x96xf16, #NHWC, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x129x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x2x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!WeightTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x64xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @DoNotCompressConvWeightsWithNonICPadAttr
// CHECK-SAME:    [[INPUT:%.+]]: !VPUIP.DistributedBuffer<1x16x129x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @DoNotCompressConvWeightsWithNonICPadAttr(%arg0: !InputDistributed) -> !OutputDistributed {
    %cst = const.Declare memref<16x16x2x1xf16, #NHWC> = dense<1.0> :
        tensor<1x1x2x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %cst_1 = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPURT.AllocDistributed -> !WeightsDistributed
    %1 = VPUIP.Copy
        inputs(%cst : memref<16x16x2x1xf16, #NHWC>)
        outputs(%0 : !WeightsDistributed)
        -> !WeightsDistributed
    %2 = VPURT.AllocDistributed -> !WeightTableDistributed
    %3 = VPUIP.Copy
        inputs(%cst_1 : memref<16x1x1x4xsi32>)
        outputs(%2 : !WeightTableDistributed)
        -> !WeightTableDistributed
    %4 = VPURT.AllocDistributed -> !OutputDistributed
    %5 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [2, 1],
            kernel_strides = [1, 1],
            minimumHardwareExecutionCost = 5571 : i64,
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%arg0 : !InputDistributed)
        weights(%1 : !WeightsDistributed)
        weight_table(%3 : !WeightTableDistributed)
        parent_input(%arg0 : !InputDistributed)
        parent_output(%4 : !OutputDistributed)
        outputs(%4 : !OutputDistributed) -> !OutputDistributed variants :
        {
            DPUTask
                {
                    cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    outEnd = [63, 63, 15], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            DPUTask
                {
                    cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    outEnd = [63, 127, 15], outStart = [0, 64, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    return %5 : !OutputDistributed

    //CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare memref<16x16x2x1xf16, #NHWC> = dense<1.000000e+00> : tensor<1x1x2x1xf32>
    //CHECK-DAG:    [[WEIGHTS_TABLE:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:        [[ALLOC_W:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x16x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS]] : memref<16x16x2x1xf16, #NHWC>)
    //CHECK-SAME:           outputs([[ALLOC_W]] : !VPUIP.DistributedBuffer<16x16x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<16x16x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[ALLOC_WT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:        [[WEIGHTS_TABLE_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS_TABLE]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:           outputs([[ALLOC_WT]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:        [[OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x16x128x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:        [[CLUSTER_TASK:%.+]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:               kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:               kernel_size = [2, 1], kernel_strides = [1, 1],
    //CHECK-SAME:               task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:           weights([[WEIGHTS_CMX]] : !VPUIP.DistributedBuffer<16x16x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE_CMX]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[OUT]] : !VPUIP.DistributedBuffer<1x16x128x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT]] : !VPUIP.DistributedBuffer<1x16x128x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<1x16x128x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> variants : {
    //CHECK:          DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 63, 15],
    //CHECK-SAME:           outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [63, 127, 15],
    //CHECK-SAME:           outStart = [0, 64, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:        }

    //CHECK:        return [[CLUSTER_TASK]] : !VPUIP.DistributedBuffer<1x16x128x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvWeightsWithReshape
// CHECK-SAME:    [[INPUT:%.+]]: memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>
func.func @CompressConvWeightsWithReshape(
    %INPUT: memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>
) -> memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]> {
    %CST_WEIGHTS = const.Declare memref<128x16x1x1xf16, #NHWC> = dense<1.0> : tensor<1x1x128x9xf32>, [
        #const.Reshape<[128, 9]>,
        #const.Reshape<[128, 9, 1, 1]>,
        #const.CastElemType<f16>,
        #const.Reorder<#NHWC>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>
    ]
    // CHECK:   [[CST_WEIGHTS:%.+]] = const.Declare memref<128x16x1x1xf16, #NHWC>

    %WEIGHTS_TABLE = const.Declare memref<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare memref<128x1x1x4xsi32>

    %ALLOC_WEIGHTS = memref.alloc() : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC_WEIGHTS:%.+]] = memref.alloc() : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>

    %WEIGHTS_COPY = VPUIP.Copy
        inputs(%CST_WEIGHTS : memref<128x16x1x1xf16, #NHWC>)
        outputs(%ALLOC_WEIGHTS : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
         -> memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:   [[WEIGHTS_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[CST_WEIGHTS]] : memref<128x16x1x1xf16, #NHWC>)
    // CHECK-SAME:  outputs([[ALLOC_WEIGHTS]] : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)

    %ALLOC_WEIGHTS_TABLE = memref.alloc() : memref<128x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC_WEIGHTS_TABLE:%.+]] = memref.alloc() : memref<128x1x1x4xsi32, [@CMX_NN, 0]>

    %WEIGHTS_TABLE_COPY = VPUIP.Copy
        inputs(%WEIGHTS_TABLE : memref<128x1x1x4xsi32>)
        outputs(%ALLOC_WEIGHTS_TABLE : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<128x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:   [[WEIGHTS_TABLE_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[WEIGHTS_TABLE]] : memref<128x1x1x4xsi32>)
    // CHECK-SAME:  outputs([[ALLOC_WEIGHTS_TABLE]] : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)

    %ALLOC_OUTPUT = memref.alloc() : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[ALLOC_OUTPUT:%.+]] = memref.alloc() : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>

    %NCE = VPUIP.NCEClusterTask {
          kernel_padding = #VPU.Padding<
            left = 0 : i64,
            right = 0 : i64,
            top = 0 : i64,
            bottom = 0 : i64
          >,
          kernel_size = [1, 1],
          kernel_strides = [1, 1],
          minimumHardwareExecutionCost = 3005 : i64,
          task_type = #VPUIP.nce_task_type<CONV>
    }
    input(%INPUT : memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    weights(%WEIGHTS_COPY : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    weight_table(%WEIGHTS_TABLE_COPY : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)
    parent_input(%INPUT : memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    parent_output(%ALLOC_OUTPUT : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    outputs(%ALLOC_OUTPUT : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>
    variants : {
        DPUTask {
             mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
             outEnd = [111, 111, 63],
             outStart = [0, 0, 0],
             pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
        }
    } PPE : {
    }
    // CHECK:   [[NCE:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  input([[INPUT]] : memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[WEIGHTS_COPY]] : memref<128x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHTS_TABLE_COPY]] : memref<128x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x16x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_output([[ALLOC_OUTPUT]] : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[ALLOC_OUTPUT]] : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>)

    return %NCE : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   return [[NCE]] : memref<1x128x256x4xf16, #NHWC, [@CMX_NN, 0]>
}
