// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=MTL" --compress-weights %s | FileCheck %s

!qElemType = type !quant.uniform<u8:f16, 1.0000000000000000E-1>

// CHECK-LABEL: func @CompressQuantConstant
func @CompressQuantConstant() -> memref<256x512x3x3x!qElemType, "CMX_NN"> {
  %cst_0 = const.Declare memref<256x512x3x3x!qElemType> = #const.Content<dense<1> : tensor<256x512x3x3xui8>, [#const.QuantCast<!qElemType>]>
  %0 = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <0> -> memref<256x512x3x3x!qElemType, "CMX_NN">
  %1 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true}
    inputs(%cst_0 : memref<256x512x3x3x!qElemType>)
    outputs(%0 : memref<256x512x3x3x!qElemType, "CMX_NN">)
    -> memref<256x512x3x3x!qElemType, "CMX_NN">
  return %1 : memref<256x512x3x3x!qElemType, "CMX_NN">

  // CHECK-NOT:   VPUIP.NNDMA
  // CHECK:       %[[COMPRESSED_CST:.*]] = const.Declare memref<357152x1x1x1xui8> = #const.Content<dense<
  // CHECK-SAME:    : tensor<357152x1x1x1xui8>>
  // CHECK:       %[[ORIG_TENSOR:.*]] = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <0> -> memref<256x512x3x3x!qElemType, "CMX_NN">
  // CHECK:       %[[FLAT_TENSOR:.*]] = VPURT.DeclareBuffer "VPU_CMX_NN" [0] <0> -> memref<1179648x1x1x1xui8, "CMX_NN">
  // CHECK:       %[[COMPRESSED_DMA:.*]] = VPUIP.CompressedDMAOp {port = 0 : i64}
  // CHECK-SAME:    inputs(%[[COMPRESSED_CST]] : memref<357152x1x1x1xui8>)
  // CHECK-SAME:    outputs(%[[FLAT_TENSOR]] : memref<1179648x1x1x1xui8, "CMX_NN">)
  // CHECK-SAME:    -> memref<1179648x1x1x1xui8, "CMX_NN">
  // CHECK:       %[[ORIG_TENSOR]] : memref<256x512x3x3x!qElemType, "CMX_NN">
}
