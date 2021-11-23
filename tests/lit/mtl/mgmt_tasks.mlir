// This file generates a blob with management tasks, used to demonstrate that
// the runtime can process them.  It's also a lit test to help check for
// regressions in the VPUIP dialect.
//
// To generate a blob, use:
//
//    vpux-translate --export-VPUIP < mgmt_tasks.mlir > mgmt_tasks.blob
//
// RUN: vpux-opt %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceSW"}  {
  VPUIP.Graph options : "NONE" version : {contextStr = "VPUX Compiler", hash = "", majorV = 3, minorV = 11, patchV = 0}
  IERT.RunTimeResources availableMemory :  {
    MemoryResource 1073741824 bytes
    MemoryResource 31457280 bytes of "DDR" {VPUIP.bandwidth = 8, VPUIP.derateFactor = 6.000000e-01}
    MemoryResource 2097152 bytes of "CMX_NN" {VPUIP.bandwidth = 32, VPUIP.derateFactor = 1.000000e+00}
  } usedMemory :  {
  } executors :  {
    ExecutorResource 1 of "Leon_RT"
    ExecutorResource 1 of "Leon_NN"
    ExecutorResource 1 of "DMA_UPA"
    ExecutorResource 1 of "SHAVE_NN"
    ExecutorResource 1 of "NCE_Cluster"  {
      ExecutorResource 1 of "NCE_PerClusterDPU"
    }
    ExecutorResource 2 of "DMA_NN"
  }
  func private @"mgmt_task_test!quant.uniform<u8:f32, 1.000000e+00>_!quant.uniform<u8:f32, 1.000000e+00>_f16"(%arg0: memref<1x16x16x16x!quant.uniform<u8:f32, 1.000000e+00>, #NHWC, "ProgrammableInput">, %arg1: memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput">) -> memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput"> {
    %1 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    %2 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    VPURT.Task updates(%1 : !VPURT.Barrier) op :  {
      VPUIP.Empty
    }
    VPURT.Task waits(%1 : !VPURT.Barrier) updates(%2 : !VPURT.Barrier) op :  {
      VPUIP.Empty
    }
    VPURT.Task waits(%2 : !VPURT.Barrier) op :  {
      VPUIP.Empty
    }
    return %arg1 : memref<1x16x16x16xf16, #NHWC, "ProgrammableOutput">
  }
  IE.CNNNetwork entryPoint : @"mgmt_task_test!quant.uniform<u8:f32, 1.000000e+00>_!quant.uniform<u8:f32, 1.000000e+00>_f16" inputsInfo :  {
    DataInfo "input_0" : tensor<1x16x16x16xui8, {order = #NHWC}>
  } outputsInfo :  {
    DataInfo "output_0" : tensor<1x16x16x16xf16, {order = #NHWC}>
  }
}

// CHECK-LABEL: module @mainModule attributes {VPUIP.arch = "MTL", VPUIP.compilationMode = "ReferenceSW"}
