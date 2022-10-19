// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX40XX" --adjust-memory-space %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @insertCmxCopies
func @insertCmxCopies(%arg0: tensor<1x3x256x256xui8>) -> tensor<1x3x224x224xui8> {
  %0 = VPU.M2I.Task(%arg0) {axes = [2, 3], do_csc = false, do_norm = false, inFmt = "PL_RGB24", outFmt = "PL_RGB24", sizes = [224, 224]} -> tensor<1x3x224x224xui8>
  return %0 : tensor<1x3x224x224xui8>

  //CHECK: [[VAL0:%.*]] = VPU.Copy(%arg0) {out_mem_space = [@CMX_NN, 0]} : tensor<1x3x256x256xui8> -> tensor<1x3x256x256xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
  //CHECK: [[VAL1:%.*]] = VPU.M2I.Task([[VAL0]]) {axes = [2, 3], do_csc = false, do_norm = false, inFmt = "PL_RGB24", outFmt = "PL_RGB24", sizes = [224, 224]} -> tensor<1x3x224x224xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
  //CHECK: [[VAL2:%.*]] = VPU.Copy([[VAL1]]) : tensor<1x3x224x224xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x3x224x224xui8>
  //CHECK: return [[VAL2]]
}
