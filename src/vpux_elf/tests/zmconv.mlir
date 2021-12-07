#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceHW"}  {
  IERT.RunTimeResources availableMemory :  {
    MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
    MemoryResource 1982464 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
  } executors :  {
    ExecutorResource 2 of "DMA_NN"
    ExecutorResource {VPUIP.processorFrequency = 7.000000e+02 : f64} 2 of "NCE_Cluster"  {
      ExecutorResource 1 of "NCE_PerClusterDPU"
    }
  }
  func private @zmajor_conv_f16_f16_f16(%arg0: memref<1x16x16x16xf16, #NHWC, "ProgrammableInput">, %arg1: memref<1x64x16x16xf16, #NHWC, "ProgrammableOutput">) -> memref<1x64x16x16xf16, #NHWC, "ProgrammableOutput"> {
    %cst = const.Declare memref<64x16x1x1xf16, #NHWC, "GraphFile"> = #const.Content<dense<0.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]>
    %0 = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <41984> -> memref<64x16x1x1xf16, #NHWC, "CMX_NN">
    %1 = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <32768> -> memref<1x16x16x16xf16, #NHWC, "CMX_NN">
    %2 = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <0> -> memref<1x64x16x16xf16, #NHWC, "CMX_NN">
    %cst_0 = const.Declare memref<64x1x1x4xsi32, #NHWC, "GraphFile"> = #const.Content<dense<"0x00A40000FFFFFF000000803F0000000020A400000F0000010000803F0000000040A400001F0000010000803F0000000060A400002F0000010000803F0000000080A400003F0000010000803F00000000A0A400004F0000010000803F00000000C0A400005F0000010000803F00000000E0A400006F0000010000803F0000000000A500007F0000010000803F0000000020A500008F0000010000803F0000000040A500009F0000010000803F0000000060A50000AF0000010000803F0000000080A50000BF0000010000803F00000000A0A50000CF0000010000803F00000000C0A50000DF0000010000803F00000000E0A50000EF0000010000803F0000000000A60000FF0000010000803F0000000020A600000F0100010000803F0000000040A600001F0100010000803F0000000060A600002F0100010000803F0000000080A600003F0100010000803F00000000A0A600004F0100010000803F00000000C0A600005F0100010000803F00000000E0A600006F0100010000803F0000000000A700007F0100010000803F0000000020A700008F0100010000803F0000000040A700009F0100010000803F0000000060A70000AF0100010000803F0000000080A70000BF0100010000803F00000000A0A70000CF0100010000803F00000000C0A70000DF0100010000803F00000000E0A70000EF0100010000803F0000000000A80000FF0100010000803F0000000020A800000F0200010000803F0000000040A800001F0200010000803F0000000060A800002F0200010000803F0000000080A800003F0200010000803F00000000A0A800004F0200010000803F00000000C0A800005F0200010000803F00000000E0A800006F0200010000803F0000000000A900007F0200010000803F0000000020A900008F0200010000803F0000000040A900009F0200010000803F0000000060A90000AF0200010000803F0000000080A90000BF0200010000803F00000000A0A90000CF0200010000803F00000000C0A90000DF0200010000803F00000000E0A90000EF0200010000803F0000000000AA0000FF0200010000803F0000000020AA00000F0300010000803F0000000040AA00001F0300010000803F0000000060AA00002F0300010000803F0000000080AA00003F0300010000803F00000000A0AA00004F0300010000803F00000000C0AA00005F0300010000803F00000000E0AA00006F0300010000803F0000000000AB00007F0300010000803F0000000020AB00008F0300010000803F0000000040AB00009F0300010000803F0000000060AB0000AF0300010000803F0000000080AB0000BF0300010000803F00000000A0AB0000CF0300010000803F00000000C0AB0000DF0300010000803F00000000E0AB0000EF0300010000803F00000000"> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]>
    %3 = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <40960> -> memref<64x1x1x4xsi32, #NHWC, "CMX_NN">
    %4 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %5 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    VPURT.Task {isTrailingSWLayer = false} updates(%4 : !VPURT.Barrier) op :  {
        %6 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%arg0 : memref<1x16x16x16xf16, #NHWC, "ProgrammableInput">) outputs(%1 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) -> memref<1x16x16x16xf16, #NHWC, "CMX_NN">
    }
    VPURT.Task {isTrailingSWLayer = false} updates(%4 : !VPURT.Barrier) op :  {
        %6 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%cst : memref<64x16x1x1xf16, #NHWC, "GraphFile">) outputs(%0 : memref<64x16x1x1xf16, #NHWC, "CMX_NN">) -> memref<64x16x1x1xf16, #NHWC, "CMX_NN">
    }
    VPURT.Task {isTrailingSWLayer = false} updates(%4 : !VPURT.Barrier) op :  {
        %6 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%cst_0 : memref<64x1x1x4xsi32, #NHWC, "GraphFile">) outputs(%3 : memref<64x1x1x4xsi32, #NHWC, "CMX_NN">) -> memref<64x1x1x4xsi32, #NHWC, "CMX_NN">
    }
    VPURT.Task {isTrailingSWLayer = false} waits(%5 : !VPURT.Barrier) op :  {
        %6 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%2 : memref<1x64x16x16xf16, #NHWC, "CMX_NN">) outputs(%arg1 : memref<1x64x16x16xf16, #NHWC, "ProgrammableOutput">) -> memref<1x64x16x16xf16, #NHWC, "ProgrammableOutput">
    }
    VPURT.Task {isTrailingSWLayer = false} waits(%4 : !VPURT.Barrier) updates(%5 : !VPURT.Barrier) op :  {
        %6 = VPUIP.NCEClusterTask {kernel_padding = [0, 0, 0, 0], kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"} input(%1 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) weights(%0 : memref<64x16x1x1xf16, #NHWC, "CMX_NN">) weight_table(%3 : memref<64x1x1x4xsi32, #NHWC, "CMX_NN">) parent_input(%1 : memref<1x16x16x16xf16, #NHWC, "CMX_NN">) parent_output(%2 : memref<1x64x16x16xf16, #NHWC, "CMX_NN">) outputs(%2 : memref<1x64x16x16xf16, #NHWC, "CMX_NN">) -> memref<1x64x16x16xf16, #NHWC, "CMX_NN"> variants :  {
          DPUTask {end = [15, 15, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]}
        } PPE :  {
        }
    }
    return %arg1 : memref<1x64x16x16xf16, #NHWC, "ProgrammableOutput">
  }
  IE.CNNNetwork entryPoint : @zmajor_conv_f16_f16_f16 inputsInfo :  {
    DataInfo "input_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
  }
}
