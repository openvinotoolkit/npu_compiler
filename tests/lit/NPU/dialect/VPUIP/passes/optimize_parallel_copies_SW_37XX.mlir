//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeParallelNonConstCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x16x112x112xf32, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
func.func @OptimizeParallelNonConstCopies(
        %input: memref<1x16x112x112xf32, {order = #NHWC}>,
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
         -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Convert inputs(%input as %arg3: memref<1x16x112x112xf32, {order = #NHWC}>) outputs(%0 as %arg4: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>) on tile 0 -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x112x112xf32, {order = #NHWC}>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>
        }
    %2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %3 = VPUIP.Copy
            inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %4 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %5 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%3 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%3 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %6 = VPUIP.Copy
            inputs(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %7 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %8 = VPUIP.Copy
            inputs(%1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            outputs(%7 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %9 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %10 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%9 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%9 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %11 = VPUIP.Copy
            inputs(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    return %6, %11 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

}

// CHECK:       [[VAR0:%.+]] =  VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[ARG_0]] as [[ARG3:%.+]]: memref<1x16x112x112xf32, {order = #NHWC}>)
// CHECK:       [[VAR1:%.+]] =  VPUIP.Copy inputs([[VAR0]] : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
// CHECK:       [[VAR2:%.+]] =  VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR3:%.+]] =  VPUIP.Copy inputs([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// CHECK-NOT:   VPUIP.Copy
// CHECK:       [[VAR4:%.+]] =  VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR5:%.+]] =  VPUIP.Copy inputs([[VAR4]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_Convert(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @OptimizeParallelSubViewPatternCopies
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x16x112x113xf32, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
func.func @OptimizeParallelSubViewPatternCopies(
        %input: memref<1x16x112x113xf32, {order = #NHWC}>,
        %output1: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>,
        %output2: memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
         -> (memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>){
    %wt = const.Declare memref<16x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<16x1x1x4xsi32>
    %0 = memref.alloc() : memref<1x16x112x113xf16, {order = #NHWC}, @DDR>

    %1 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
        @VPU.SW::@builtin_Convert inputs(%input as %arg3: memref<1x16x112x113xf32, {order = #NHWC}>) outputs(%0 as %arg4: memref<1x16x112x113xf16, {order = #NHWC}, @DDR>) on tile 0 -> memref<1x16x112x113xf16, {order = #NHWC}, @DDR>  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg3, %arg4) : memref<1x16x112x113xf32, {order = #NHWC}>, memref<1x16x112x113xf16, {order = #NHWC}, @DDR>
        }
    %2 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %3 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>
    %4 = VPUIP.Copy
            inputs(%3 : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
            outputs(%2 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %5 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%4 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%5 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %7 = VPUIP.Copy
            inputs(%6 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    %8 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %9 = VPUIP.SubView %1 [0, 0, 0, 0] [1, 16, 112, 112] :
                memref<1x16x112x113xf16, {order = #NHWC}, @DDR> to memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>
    %10 = VPUIP.Copy
            inputs(%9 : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
            outputs(%8 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
             -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %11 = memref.alloc() : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
    %12 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>
        }>
        input(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        weight_table(%wt : memref<16x1x1x4xsi32, @CMX_NN>)
        parent_input(%10 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        parent_output(%11 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
        outputs(%11 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>
        variants :
        {
            DPUTask { outEnd = [16, 112, 112], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0] }
        }
        PPE : {
        }
    %13 = VPUIP.Copy
            inputs(%12 : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%output1 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>)
            -> memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

    return %7, %13 : memref<1x16x112x112xf16, {order = #NHWC}, @DDR>, memref<1x16x112x112xf16, {order = #NHWC}, @DDR>

}

// CHECK:       [[VAR0:%.+]] =  VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs([[ARG_0]] as [[ARG3:%.+]]: memref<1x16x112x113xf32, {order = #NHWC}>)
// CHECK:       [[VAR1:%.+]] =  VPUIP.SubView [[VAR0]] [0, 0, 0, 0] [1, 16, 112, 112]
// CHECK:       [[VAR2:%.+]] =  VPUIP.Copy
// CHECK-SAME:      inputs([[VAR1]] : memref<1x16x112x112xf16, {order = #NHWC, strides = [202496, 1, 1808, 16]}, @DDR>)
// CHECK:       [[VAR3:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR4:%.+]] =  VPUIP.Copy inputs([[VAR3]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// CHECK-NOT:   VPUIP.SubView
// CHECK-NOT:   VPUIP.Copy
// CHECK:       [[VAR5:%.+]] = VPUIP.NCEClusterTask
// CHECK-SAME:       input([[VAR2]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)
// CHECK:       [[VAR6:%.+]] =  VPUIP.Copy inputs([[VAR5]] : memref<1x16x112x112xf16, {order = #NHWC}, @CMX_NN>)

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8<0:254>:f16:0, {6.8494566078261131E-4:127,5.4140881759913884E-4:127,1.633063868040175E-4:127,2.6382913622330492E-4:127,0.0015323627886809701:127,3.0075550431341637E-4:127,0.0013602712726968481:127,0.0012382038934962956:127,0.0018411807891890758:127,2.6264191260488016E-4:127,0.0010926755159858643:127,2.6557371606976968E-4:127,8.7139796553634282E-4:127,4.8059178149606299E-7:127,0.0024098467639112097:127,0.0016193400452456136:127,4.7592821670329477E-4:127,0.001568063741593849:127,0.0026288621538267361:127,3.1080894817517497E-4:127,0.0024666349718889852:127,0.0015988477806406698:127,0.0023083168221270946:127,4.4035656363006654E-4:127,7.7296887326428268E-4:127,2.1079874883486529E-4:127,0.0013202947425091361:127,0.0012987030772712287:127,4.2421238746230056E-4:127,2.4158283188117772E-4:127,5.570924070876414E-4:127,1.3461924620031371E-4:127,2.8047071197840175E-4:127,0.0039018812611347109:127,1.3892022584836313E-4:127,3.0758384409851916E-4:127,2.7585416797577865E-4:127,3.095509733740739E-4:127,0.0011052948048734289:127,0.0012020447592097005:127,2.2011245857542894E-4:127,0.0015056552145424791:127,2.6557371606976968E-4:127,3.7953172495046002E-4:127,1.7592617435248817E-4:127,8.625751874578281E-4:127,0.0016026958001880195:127,4.1750900623366586E-4:127,8.2286318221430144E-4:127,0.001763350264293941:127,0.0014430034583009135:127,6.7431778889002765E-4:127,4.2953403798613959E-4:127,0.0012631090137902208:127,0.0011619765927472453:127,5.892951070793032E-4:127,5.9115041897991514E-4:127,1.6237138293859527E-4:127,4.5863459781398926E-4:127,3.1761346956876317E-4:127,6.6845418196024859E-4:127,9.7691332261393387E-4:127,2.707826692288316E-4:127,0.0025570021839592403:127}>
!qElemType2 = !quant.uniform<u8:f16, 0.0173492431640625:114>
!qElemType3 = !quant.uniform<u8:f16, 0.012699142156862745>

// CHECK-LABEL: @NotOptimizeConstCopyForCompressedConv
// CHECK-SAME: [[ARG0:%[^:]+]]: memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>,
// CHECK-SAME: [[ARG1:%[^:]+]]: memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
func.func @NotOptimizeConstCopyForCompressedConv(%arg0: memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>, %arg1: memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) -> (memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) {
    %cst = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare memref<64x1x1x208x!qElemType, {order = #NHWC}> = dense<1.0> :
        tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>,
        #const.Reshape<[64, 1, 1, 196]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>, #const.Reorder<#NHWC>]
    %2 = memref.alloc() : memref<64x1x1x208x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %weights = VPUIP.Copy
        inputs(%cst_1 : memref<64x1x1x208x!qElemType, {order = #NHWC}>)
        outputs(%2 : memref<64x1x1x208x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
         -> memref<64x1x1x208x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %3 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table_0 = VPUIP.Copy
        inputs(%cst : memref<64x1x1x4xsi32>)
        outputs(%3 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %4 = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    %weights_table_1 = VPUIP.Copy
        inputs(%cst_0 : memref<64x1x1x4xsi32>)
        outputs(%4 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
         -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>

    %output_0 = memref.alloc() : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>
    %NCEOp_0 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          task_type = #VPUIP.nce_task_type<CONV>
      }>
      input(%arg0 : memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      weights(%weights : memref<64x1x1x208x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table(%weights_table_0 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg0 : memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_output(%output_0 : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>)
      outputs(%output_0 : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
            }
      }
      PPE :  {
      }

    %output_1 = memref.alloc() : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>
    %NCEOp_1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
          kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
          kernel_size = [7, 7],
          kernel_strides = [2, 2],
          task_type = #VPUIP.nce_task_type<CONV>
      }>
      input(%arg1 : memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      weights(%weights : memref<64x1x1x208x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
      weight_table(%weights_table_1 : memref<64x1x1x4xsi32, [@CMX_NN, 0]>)
      parent_input(%arg1 : memref<1x16x224x224x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
      parent_output(%output_1 : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>)
      outputs(%output_1 : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]> variants :
      {
          DPUTask
            {
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 111, 63], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
            }
      }
      PPE :  {
      }
    return %NCEOp_0, %NCEOp_1 : memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x64x112x112x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK-DAG:    [[CST:%.+]] = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    //CHECK-DAG:    [[CST_0:%.+]] = const.Declare memref<64x1x1x208x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 1, 1, 196]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>, #const.Reorder<#NHWC>]
    //CHECK:        [[BUF0:%.+]] = memref.alloc() : memref<64x1x1x208x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS:%.+]] = VPUIP.Copy inputs([[CST_0]] : memref<64x1x1x208x!qElemType2, {order = #NHWC}>) outputs([[BUF0]] : memref<64x1x1x208x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<64x1x1x208x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[BUF1:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE0:%.+]] = VPUIP.Copy inputs([[CST]] : memref<64x1x1x4xsi32>) outputs([[BUF1]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[BUF2:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[WEIGHTS_TABLE1:%.+]] = VPUIP.Copy inputs([[CST]] : memref<64x1x1x4xsi32>) outputs([[BUF2]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<64x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:        [[BUF3:%.+]] = memref.alloc() : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[NCE0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64} <{kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2],
    //CHECK-SAME:                task_type = #VPUIP.nce_task_type<CONV>}> input([[ARG0]] : memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights([[WEIGHTS]] : memref<64x1x1x208x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weight_table([[WEIGHTS_TABLE0]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input([[ARG0]] : memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output([[BUF3]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF3]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> variants : {
    //CHECK:        [[BUF4:%.+]] = memref.alloc() : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[NCE1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 375613 : i64} <{kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2],
    //CHECK-SAME:                    task_type = #VPUIP.nce_task_type<CONV>}> input([[ARG1]] : memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) weights([[WEIGHTS]] : memref<64x1x1x208x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>) weight_table([[WEIGHTS_TABLE1]] : memref<64x1x1x4xsi32, [@CMX_NN, 0]>) parent_input([[ARG1]] : memref<1x16x224x224x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_output([[BUF4]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[BUF4]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]> variants : {
    //CHECK:        return [[NCE0]], [[NCE1]] : memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x64x112x112x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW  {
    func.func nested @builtin_Multiply(memref<*xf32>, memref<*xf16>) attributes {VPU.kernel_code = "eltwise_mul.cpp", VPU.kernel_entry = "eltwise_mul"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!FirstInputDistributed = !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!SecondInputDistributed = !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

func.func @OptimizeMultiplyParallelSameInputsCopies()
         -> (!FirstInputDistributed, !SecondInputDistributed) {
    %0 = memref.alloc() : memref<1x1x106x4xf16, @DDR>
    %1 = VPUIP.SubView %0 [0, 0, 54, 0] [1, 1, 52, 4] :
                memref<1x1x106x4xf16, @DDR> to memref<1x1x52x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %2 = VPURT.AllocDistributed -> !SecondInputDistributed
    %3 = VPUIP.Copy inputs(%1 : memref<1x1x52x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
                        outputs(%2 : !SecondInputDistributed) -> !SecondInputDistributed

    %4 = VPUIP.SubView %0 [0, 0, 54, 0] [1, 1, 52, 4] :
                memref<1x1x106x4xf16, @DDR> to memref<1x1x52x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %5 = VPURT.AllocDistributed -> !SecondInputDistributed
    %6 = VPUIP.Copy inputs(%4 : memref<1x1x52x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
                        outputs(%5 : !SecondInputDistributed) -> !SecondInputDistributed

    %7 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 54, 4] :
                memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %8 = VPURT.AllocDistributed -> !FirstInputDistributed
    %9 = VPUIP.Copy inputs(%7 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
                        outputs(%8 : !FirstInputDistributed) -> !FirstInputDistributed

    %10 = VPUIP.SubView %0 [0, 0, 0, 0] [1, 1, 54, 4] :
                memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    %11 = VPURT.AllocDistributed -> !FirstInputDistributed
    %12 = VPUIP.Copy inputs(%10 : memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>)
                        outputs(%11 : !FirstInputDistributed) -> !FirstInputDistributed

    %13 = VPURT.AllocDistributed -> !SecondInputDistributed
    %14 = VPURT.AllocDistributed -> !FirstInputDistributed
    %15:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
        inputs(%12 as %arg8: !FirstInputDistributed, %9 as %arg9: !FirstInputDistributed,
            %6 as %arg10: !SecondInputDistributed, %3 as %arg11: !SecondInputDistributed)
        outputs(%14 as %arg12: !FirstInputDistributed, %13 as %arg13: !SecondInputDistributed) on tile 0
        -> (!FirstInputDistributed, !SecondInputDistributed) {
        VPUIP.SW.Kernel.run {attrs = []}(%arg8, %arg9, %arg12) : !FirstInputDistributed, !FirstInputDistributed, !FirstInputDistributed
        VPUIP.SW.Kernel.run {attrs = []}(%arg10, %arg11, %arg13) : !SecondInputDistributed, !SecondInputDistributed, !SecondInputDistributed
    }

    return %15#0, %15#1 : !FirstInputDistributed, !SecondInputDistributed

    // CHECK:   [[ALLOC:%.+]] = memref.alloc() : memref<1x1x106x4xf16, @DDR>
    // CHECK:   [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 54, 0] [1, 1, 52, 4] : memref<1x1x106x4xf16, @DDR> to memref<1x1x52x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    // CHECK:   [[INPUT_BUFFER0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY0:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SUBVIEW0]]
    // CHECK-SAME:  outputs([[INPUT_BUFFER0]]

    // CHECK:   [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]] [0, 0, 0, 0] [1, 1, 54, 4] : memref<1x1x106x4xf16, @DDR> to memref<1x1x54x4xf16, {order = #NCHW, strides = [424, 424, 4, 1]}, @DDR>
    // CHECK:   [[INPUT_BUFFER1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[COPY1:%.+]] = VPUIP.Copy
    // CHECK-SAME:  inputs([[SUBVIEW1]]
    // CHECK-SAME:  outputs([[INPUT_BUFFER1]]

    // CHECK:   [[OUTPUT_BUFFER0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x52x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[OUTPUT_BUFFER1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1x54x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:   [[MULTIPLY:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_Multiply
    // CHECK-SAME:       inputs([[COPY1]] as [[ARG10:%[^:]+]]
    // CHECK-SAME:       [[COPY1]] as [[ARG11:%[^:]+]]
    // CHECK-SAME:       [[COPY0]] as [[ARG12:%[^:]+]]
    // CHECK-SAME:       [[COPY0]] as [[ARG13:%[^:]+]]
    // CHECK-SAME:       outputs([[OUTPUT_BUFFER1]] as [[ARG14:%[^:]+]]
    // CHECK-SAME:       [[OUTPUT_BUFFER0]] as [[ARG15:%[^:]+]]
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG10]], [[ARG11]], [[ARG14]])
    // CHECK:        VPUIP.SW.Kernel.run {attrs = []}([[ARG12]], [[ARG13]], [[ARG15]])

    // CHECK:   return [[MULTIPLY]]#0, [[MULTIPLY]]#1

}
