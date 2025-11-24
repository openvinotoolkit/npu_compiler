//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --convert-VPUMI40XX-to-VPUASM %s | FileCheck %s
// REQUIRES: arch-NPU40XX


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @"resnet-320-pytorch" {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  config.Resources {activity_factor = 0.042571347270822207 : f64} 2 of @NCE at 1.850000e+03 MHz {
    builtin.module @ReservedMemory {
      module @DmaProfilingReservedMemory {
        config.MemoryResource 1024 bytes of @CMX_NN offset 1473536
      }
    }
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo {inferenceTiming = 18466 : i64} entryPoint : @main inputsInfo : {
    DataInfo "result.1" tensorNames = ["result.1"] : tensor<1x16x16x16xf16>
  } outputsInfo : {
    DataInfo "prob" friendlyName = "495/sink_port_0" tensorNames = ["prob"] : tensor<1x16x14x14xf16>
  }
  func.func @main() {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x16x16xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> {swizzlingKey = 0 : i64} -> memref<1x16x14x14xf16, @DDR>
    %31 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:29>
    %32 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:30>
    %33 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<0:0:31>
    %127 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:61>
    %128 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:62>
    %129 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<0:0:63>
    %224 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:14>
    %225 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:0:15>
    %767 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:14>
    %768 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<0:1:15>
    %288 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:14>
    %289 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:0:15>
    %769 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:14>
    %770 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<0:1:15>
    %351 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:29>
    %352 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:30>
    %353 = VPUMI40XX.DeclareTaskBuffer <DPUInvariant> -> !VPURegMapped.Index<1:0:31>
    %447 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:61>
    %448 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:62>
    %449 = VPUMI40XX.DeclareTaskBuffer <DPUVariant> -> !VPURegMapped.Index<1:0:63>
    %544 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:14>
    %545 = VPUMI40XX.DeclareTaskBuffer <ActKernelRange> -> !VPURegMapped.Index<1:0:15>
    %608 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:14>
    %609 = VPUMI40XX.DeclareTaskBuffer <ActKernelInvocation> -> !VPURegMapped.Index<1:0:15>
    %cst = const.Declare memref<1x1x1x2432xf16> = dense<1.0> : tensor<1x1x1x2432xf16>
    %642 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>
    %643 = VPURT.DeclareBuffer <NetworkInput> [0] <256> -> memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>
    %644 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>
    %645 = VPURT.DeclareBuffer <NetworkOutput> [0] <196> -> memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>
    %646 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %647 = VPURT.DeclareBuffer <DDR> <0> -> memref<0x0x0x0xi32, @DDR>
    %648 = VPURT.DeclareBuffer <CMX_NN> [0] <1473536> -> memref<16xui32, [@CMX_NN, 0]>
    %649 = VPURT.DeclareBuffer <CMX_NN> [0] <9472> -> memref<1x16x8x16xf16, [@CMX_NN, 0]>
    %650 = VPURT.DeclareBuffer <CMX_NN> [1] <9472> -> memref<1x16x8x16xf16, [@CMX_NN, 1]>
    %651 = VPURT.DeclareBuffer <CMX_NN> [0] <9472> -> memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 0]>
    %652 = VPURT.DeclareBuffer <CMX_NN> [1] <9472> -> memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 1]>
    %653 = VPURT.DeclareBuffer <CMX_NN> [0] <3136> -> memref<1x16x7x14xf16, [@CMX_NN, 0]>
    %654 = VPURT.DeclareBuffer <CMX_NN> [1] <3136> -> memref<1x16x7x14xf16, [@CMX_NN, 1]>
    %657 = VPURT.DeclareBuffer <CMX_NN> [0] <9472> -> memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 0]>
    %658 = VPURT.DeclareBuffer <CMX_NN> [1] <9472> -> memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 1]>
    %661 = VPURT.DeclareBuffer <CMX_NN> [0] <4864> -> memref<1x16x9x16xf16, #NHWC, [@CMX_NN, 0]>
    %662 = VPURT.DeclareBuffer <CMX_NN> [1] <4864> -> memref<1x16x9x16xf16, #NHWC, [@CMX_NN, 1]>
    %663 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    %664 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    %665 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %666 = VPURT.DeclareBuffer <CMX_NN> [1] <256> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    %667 = VPURT.DeclareBuffer <CMX_NN> [0] <11040> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>
    %668 = VPURT.DeclareBuffer <CMX_NN> [1] <11040> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>
    %669 = VPURT.DeclareBuffer <CMX_NN> [0] <9472> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>
    %670 = VPURT.DeclareBuffer <CMX_NN> [1] <9472> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>
    %671 = VPURT.DeclareBuffer <CMX_NN> [0] <1568> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>
    %672 = VPURT.DeclareBuffer <CMX_NN> [1] <1568> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>
    %673 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>
    %674 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>
    %675 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 0]>
    %676 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 1]>
    %677 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x2432xf16, [@CMX_NN, 0]>
    %678 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x1x2432xf16, [@CMX_NN, 1]>
    %679 = VPURT.DeclareBuffer <CMX_NN> [0] <4864> -> memref<1x16x16x9xf16, {order = #NWCH}, [@CMX_NN, 0]>
    %680 = VPURT.DeclareBuffer <CMX_NN> [1] <4864> -> memref<1x16x16x9xf16, {order = #NWCH}, [@CMX_NN, 1]>
    %681 = VPUMI40XX.DeclareKernelText kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %682 = VPUMI40XX.DeclareKernelEntry kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %683 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:0>
    %684 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:0>
    %685 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<0:0:1>
    %686 = VPUMI40XX.DeclareKernelArgs kernel_path("softmax") -> !VPURegMapped.Index<1:0:1>
    %687 = VPUMI40XX.KernelParams inputs(%669 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>) outputs(%673 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> !VPURegMapped.Index<0:0:0>
    %688 = VPUMI40XX.KernelParams inputs(%670 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>) outputs(%674 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([0, 0, 64, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> !VPURegMapped.Index<1:0:0>
    %689 = VPUMI40XX.KernelParams inputs(%667 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>) outputs(%671 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 0]>) kernel_type("softmax") kernel_params([0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 32, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> !VPURegMapped.Index<0:0:1>
    %690 = VPUMI40XX.KernelParams inputs(%668 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>) outputs(%672 : memref<1x16x49x1xf16, {order = #NHWC, strides = [1568, 1, 16, 16]}, [@CMX_NN, 1]>) kernel_type("softmax") kernel_params([0, 0, 64, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 64, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 49, 36, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) -> !VPURegMapped.Index<1:0:1>
    %691 = VPUMI40XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8} <0, -1> -> !VPURegMapped.Index<0:0:0>
    %692 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}(%691 : !VPURegMapped.Index<0:0:0>) <1, -1> -> !VPURegMapped.Index<0:0:1>
    %693 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 3 : ui8}(%692 : !VPURegMapped.Index<0:0:1>) <2, -1> -> !VPURegMapped.Index<0:0:2>
    %694 = VPUMI40XX.ConfigureBarrier {consumer_count = 4 : ui8, producer_count = 2 : ui8}(%693 : !VPURegMapped.Index<0:0:2>) <3, -1> -> !VPURegMapped.Index<0:0:3>
    %695 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 4 : ui8}(%694 : !VPURegMapped.Index<0:0:3>) <4, -1> -> !VPURegMapped.Index<0:0:4>
    %696 = VPUMI40XX.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}(%695 : !VPURegMapped.Index<0:0:4>) <5, -1> -> !VPURegMapped.Index<0:0:5>
    %697 = VPUMI40XX.ConfigureBarrier {consumer_count = 0 : ui8, isFinalBarrier, producer_count = 2 : ui8}(%696 : !VPURegMapped.Index<0:0:5>) <6, -1> -> !VPURegMapped.Index<0:0:6>
    %698 = VPUMI40XX.ActKernelRange taskLocation(%224 : !VPURegMapped.Index<0:0:14>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%683 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:0>
    %699 = VPUMI40XX.ActKernelRange taskLocation(%225 : !VPURegMapped.Index<0:0:15>) previousTask(%698 : !VPURegMapped.Index<0:0:0>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%685 : !VPURegMapped.Index<0:0:1>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:0:1>
    %771 = VPUMI40XX.ActKernelRange taskLocation(%767 : !VPURegMapped.Index<0:1:14>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%683 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:0>
    %772 = VPUMI40XX.ActKernelRange taskLocation(%768 : !VPURegMapped.Index<0:1:15>) previousTask(%771 : !VPURegMapped.Index<0:1:0>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%683 : !VPURegMapped.Index<0:0:0>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<0:1:1>
    %700 = VPUMI40XX.ActKernelRange taskLocation(%544 : !VPURegMapped.Index<1:0:14>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%684 : !VPURegMapped.Index<1:0:0>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:0>
    %701 = VPUMI40XX.ActKernelRange taskLocation(%545 : !VPURegMapped.Index<1:0:15>) previousTask(%700 : !VPURegMapped.Index<1:0:0>) kernel_text_index(%681 : !VPURegMapped.Index<0:0:0>) kernel_args_index(%686 : !VPURegMapped.Index<1:0:1>) kernel_entry_index(%682 : !VPURegMapped.Index<0:0:0>) kernelTaskType(@COMPUTE) -> !VPURegMapped.Index<1:0:1>
    %702 = VPUMI40XX.ActKernelInvocation taskLocation(%288 : !VPURegMapped.Index<0:0:14>) range_index(%698 : <0:0:0>) kernel_params(%687 : <0:0:0>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %703 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%289 : !VPURegMapped.Index<0:0:15>) previousTask(%702 : !VPURegMapped.Index<0:0:0>) range_index(%699 : <0:0:1>) kernel_params(%689 : <0:0:1>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %773 = VPUMI40XX.ActKernelInvocation taskLocation(%769 : !VPURegMapped.Index<0:1:14>) range_index(%771 : <0:1:0>) kernel_params(%687 : <0:0:0>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:0>
    %774 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%770 : !VPURegMapped.Index<0:1:15>) previousTask(%773 : !VPURegMapped.Index<0:1:0>) range_index(%772 : <0:1:1>) kernel_params(%687 : <0:0:0>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:1:1>
    %704 = VPUMI40XX.ActKernelInvocation taskLocation(%608 : !VPURegMapped.Index<1:0:14>) range_index(%700 : <1:0:0>) kernel_params(%688 : <1:0:0>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:0>
    %705 = VPUMI40XX.ActKernelInvocation {lastSecondaryTaskInExecutionGroup} taskLocation(%609 : !VPURegMapped.Index<1:0:15>) previousTask(%704 : !VPURegMapped.Index<1:0:0>) range_index(%701 : <1:0:1>) kernel_params(%690 : <1:0:1>) waits(%694 : !VPURegMapped.Index<0:0:3>) updates(%695 : !VPURegMapped.Index<0:0:4>) tile(1) start_after(0) clean_after(0) -> !VPURegMapped.Index<1:0:1>
    %706 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} taskLocation(%31 : !VPURegMapped.Index<0:0:29>) input(%657 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 0]>) weights(%657 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%679 : memref<1x16x16x9xf16, {order = #NWCH}, [@CMX_NN, 0]>) waits(%692 : !VPURegMapped.Index<0:0:1>) updates(%693 : !VPURegMapped.Index<0:0:2>) -> <0:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %707 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} taskLocation(%32 : !VPURegMapped.Index<0:0:30>) previousTask(%706 : !VPURegMapped.Index<0:0:0>) input(%661 : memref<1x16x9x16xf16, #NHWC, [@CMX_NN, 0]>) weights(%665 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%663 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%651 : memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 0]>) waits(%693 : !VPURegMapped.Index<0:0:2>) updates(%694 : !VPURegMapped.Index<0:0:3>) -> <0:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %708 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64} taskLocation(%33 : !VPURegMapped.Index<0:0:31>) previousTask(%707 : !VPURegMapped.Index<0:0:1>) input(%675 : memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 0]>) outputs(%653 : memref<1x16x7x14xf16, [@CMX_NN, 0]>) waits(%695 : !VPURegMapped.Index<0:0:4>) updates(%696 : !VPURegMapped.Index<0:0:5>) -> <0:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %709 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, start_after = 0 : ui64} taskLocation(%351 : !VPURegMapped.Index<1:0:29>) input(%658 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 1]>) weights(%658 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 1]>) outputs(%680 : memref<1x16x16x9xf16, {order = #NWCH}, [@CMX_NN, 1]>) waits(%692 : !VPURegMapped.Index<0:0:1>) updates(%693 : !VPURegMapped.Index<0:0:2>) -> <1:0:0> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %710 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, start_after = 0 : ui64} taskLocation(%352 : !VPURegMapped.Index<1:0:30>) previousTask(%709 : !VPURegMapped.Index<1:0:0>) input(%662 : memref<1x16x9x16xf16, #NHWC, [@CMX_NN, 1]>) weights(%666 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%664 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) outputs(%652 : memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 1]>) waits(%693 : !VPURegMapped.Index<0:0:2>) updates(%694 : !VPURegMapped.Index<0:0:3>) -> <1:0:1> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %711 = VPUMI40XX.DPUInvariant {clean_after = 0 : ui64, is_superdense, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, start_after = 0 : ui64} taskLocation(%353 : !VPURegMapped.Index<1:0:31>) previousTask(%710 : !VPURegMapped.Index<1:0:1>) input(%676 : memref<1x16x7x14xf16, #NHWC, [@CMX_NN, 1]>) outputs(%654 : memref<1x16x7x14xf16, [@CMX_NN, 1]>) waits(%695 : !VPURegMapped.Index<0:0:4>) updates(%696 : !VPURegMapped.Index<0:0:5>) -> <1:0:2> PPE : {
      VPUMI40XX.PPETask {ppe = #VPU.PPEStub<>}
    }
    %712 = VPUMI40XX.DPUVariant taskLocation(%127 : !VPURegMapped.Index<0:0:61>) calls(%706 : <0:0:0>) weights(%657 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 0]>) {end = [7, 15, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 7 : i64, yEnd = 7 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -3584 : i64, targetClusters = [1], targetWidth = 16 : i64>], inEnd = [7, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:0>
    %713 = VPUMI40XX.DPUVariant taskLocation(%128 : !VPURegMapped.Index<0:0:62>) previousTask(%712 : !VPURegMapped.Index<0:0:0>) calls(%707 : <0:0:1>) weights(%665 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%663 : memref<16x1x1x4xsi32, [@CMX_NN, 0]>) {HardLinkedAttrName, end = [13, 6, 15], inEnd = [15, 8, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:1>
    %714 = VPUMI40XX.DPUVariant taskLocation(%129 : !VPURegMapped.Index<0:0:63>) previousTask(%713 : !VPURegMapped.Index<0:0:1>) calls(%708 : <0:0:2>) {HardLinkedAttrName, end = [13, 6, 15], inEnd = [13, 6, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <0:0:2>
    %715 = VPUMI40XX.DPUVariant taskLocation(%447 : !VPURegMapped.Index<1:0:61>) calls(%709 : <1:0:0>) weights(%658 : memref<1x16x16x8xf16, #NHWC, [@CMX_NN, 1]>) {cluster_id = 1 : ui64, end = [8, 15, 15], haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 1 : i64, yEnd = 1 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 3584 : i64, targetClusters = [0], targetWidth = 16 : i64>], inEnd = [7, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<ELTWISE>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [1, 0, 0]} -> <1:0:0>
    %716 = VPUMI40XX.DPUVariant taskLocation(%448 : !VPURegMapped.Index<1:0:62>) previousTask(%715 : !VPURegMapped.Index<1:0:0>) calls(%710 : <1:0:1>) weights(%666 : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%664 : memref<16x1x1x4xsi32, [@CMX_NN, 1]>) {HardLinkedAttrName, cluster_id = 1 : ui64, end = [13, 6, 15], inEnd = [15, 8, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<CONV>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <1:0:1>
    %717 = VPUMI40XX.DPUVariant taskLocation(%449 : !VPURegMapped.Index<1:0:63>) previousTask(%716 : !VPURegMapped.Index<1:0:1>) calls(%711 : <1:0:2>) {HardLinkedAttrName, cluster_id = 1 : ui64, end = [13, 6, 15], inEnd = [13, 6, 15], inStart = [0, 0, 0], lastSecondaryTaskInExecutionGroup, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, nce_task_type = #VPUIP.nce_task_type<MAXPOOL>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, start = [0, 0, 0]} -> <1:0:2>
    %718 = VPURegMapped.ViewTaskRange(%700 -> %701 : <1:0:0> -> <1:0:1>) -> memref<2x40xui8>
    %719 = VPURegMapped.ViewTaskRange(%544 -> %545 : <1:0:14> -> <1:0:15>) -> memref<2x40xui8, [@CMX_NN, 1]>
    %720 = VPURegMapped.ViewTaskRange(%704 -> %705 : <1:0:0> -> <1:0:1>) -> memref<2x96xui8>
    %721 = VPURegMapped.ViewTaskRange(%608 -> %609 : <1:0:14> -> <1:0:15>) -> memref<2x96xui8, [@CMX_NN, 1]>
    %722 = VPURegMapped.ViewTaskRange(%698 -> %699 : <0:0:0> -> <0:0:1>) -> memref<2x40xui8>
    %723 = VPURegMapped.ViewTaskRange(%224 -> %225 : <0:0:14> -> <0:0:15>) -> memref<2x40xui8, [@CMX_NN, 0]>
    %724 = VPURegMapped.ViewTaskRange(%702 -> %703 : <0:0:0> -> <0:0:1>) -> memref<2x96xui8>
    %725 = VPURegMapped.ViewTaskRange(%288 -> %289 : <0:0:14> -> <0:0:15>) -> memref<2x96xui8, [@CMX_NN, 0]>
    %726 = VPURegMapped.ViewTaskRange(%709 -> %711 : <1:0:0> -> <1:0:2>) -> memref<3x352xui8>
    %727 = VPURegMapped.ViewTaskRange(%351 -> %353 : <1:0:29> -> <1:0:31>) -> memref<3x352xui8, [@CMX_NN, 1]>
    %728 = VPURegMapped.ViewTaskRange(%715 -> %717 : <1:0:0> -> <1:0:2>) -> memref<3x224xui8>
    %729 = VPURegMapped.ViewTaskRange(%447 -> %449 : <1:0:61> -> <1:0:63>) -> memref<3x224xui8, [@CMX_NN, 1]>
    %730 = VPURegMapped.ViewTaskRange(%706 -> %708 : <0:0:0> -> <0:0:2>) -> memref<3x352xui8>
    %731 = VPURegMapped.ViewTaskRange(%31 -> %33 : <0:0:29> -> <0:0:31>) -> memref<3x352xui8, [@CMX_NN, 0]>
    %732 = VPURegMapped.ViewTaskRange(%712 -> %714 : <0:0:0> -> <0:0:2>) -> memref<3x224xui8>
    %733 = VPURegMapped.ViewTaskRange(%127 -> %129 : <0:0:61> -> <0:0:63>) -> memref<3x224xui8, [@CMX_NN, 0]>
    %734 = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%718 : memref<2x40xui8>) outputs(%719 : memref<2x40xui8, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:0>
    %735 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%720 : memref<2x96xui8>) outputs(%721 : memref<2x96xui8, [@CMX_NN, 1]>) previousDMA(%734 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:1>
    %736 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%722 : memref<2x40xui8>) outputs(%723 : memref<2x40xui8, [@CMX_NN, 0]>) previousDMA(%735 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:2>
    %737 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%724 : memref<2x96xui8>) outputs(%725 : memref<2x96xui8, [@CMX_NN, 0]>) previousDMA(%736 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:3>
    %738 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%726 : memref<3x352xui8>) outputs(%727 : memref<3x352xui8, [@CMX_NN, 1]>) previousDMA(%737 : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:4>
    %739 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%728 : memref<3x224xui8>) outputs(%729 : memref<3x224xui8, [@CMX_NN, 1]>) previousDMA(%738 : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:5>
    %740 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%730 : memref<3x352xui8>) outputs(%731 : memref<3x352xui8, [@CMX_NN, 0]>) previousDMA(%739 : !VPURegMapped.Index<0:0:5>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:6>
    %741 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_critical, is_out_of_order, port = 0 : i64} inputs(%732 : memref<3x224xui8>) outputs(%733 : memref<3x224xui8, [@CMX_NN, 0]>) previousDMA(%740 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:7>
    %742 = VPUMI40XX.NNDMA {HardLinkedAttrName, dma_descriptor = #VPUIP.DMADescriptorAttr<numPlanes = 0 : i64, len = 0 : i64, srcWidth = 0 : i64, srcStride = 0 : i64, srcPlaneStride = 0 : i64, dstWidth = 0 : i64, dstStride = 0 : i64, dstPlaneStride = 0 : i64>, port = 0 : i64} inputs(%646 : memref<0x0x0x0xi32, @DDR>) outputs(%647 : memref<0x0x0x0xi32, @DDR>) previousDMA(%741 : !VPURegMapped.Index<0:0:7>) updates(%691 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) -> !VPURegMapped.Index<0:0:8>
    %743 = VPUMI40XX.NNDMA {HardLinkedAttrName, port = 0 : i64} inputs(%642 : memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>) outputs(%649 : memref<1x16x8x16xf16, [@CMX_NN, 0]>) previousDMA(%742 : !VPURegMapped.Index<0:0:8>) waits(%691 : !VPURegMapped.Index<0:0:0>) updates(%692 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>, outputType = memref<1x16x8x16xf16, [@CMX_NN, 0]>>) -> !VPURegMapped.Index<0:0:9>
    %744 = VPUMI40XX.NNDMA {HardLinkedAttrName, is_out_of_order, port = 0 : i64} inputs(%cst : memref<1x1x1x2432xf16>) outputs(%677, %678 : memref<1x1x1x2432xf16, [@CMX_NN, 0]>, memref<1x1x1x2432xf16, [@CMX_NN, 1]>) previousDMA(%743 : !VPURegMapped.Index<0:0:9>) updates(%693 : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x1x1x2432xf16>, outputType = !VPUIP.DistributedBuffer<1x1x1x2432xf16, {order = #NCHW, strides = [2432, 2432, 2432, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2432], [1, 1, 1, 2432]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2432], [1, 1, 1, 2432]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>>) -> !VPURegMapped.Index<0:0:10>
    %745 = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%653 : memref<1x16x7x14xf16, [@CMX_NN, 0]>) outputs(%644 : memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>) waits(%696 : !VPURegMapped.Index<0:0:5>) updates(%697 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x16x7x14xf16, [@CMX_NN, 0]>, outputType = memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>>) -> !VPURegMapped.Index<0:1:0>
    %746 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%643 : memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>) outputs(%650 : memref<1x16x8x16xf16, [@CMX_NN, 1]>) updates(%692 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x16x8x16xf16, {order = #NCHW, strides = [4096, 256, 16, 1]}, @DDR>, outputType = memref<1x16x8x16xf16, [@CMX_NN, 1]>>) -> !VPURegMapped.Index<1:0:0>
    %747 = VPUMI40XX.NNDMA {port = 1 : i64} inputs(%654 : memref<1x16x7x14xf16, [@CMX_NN, 1]>) outputs(%645 : memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>) waits(%696 : !VPURegMapped.Index<0:0:5>) updates(%697 : !VPURegMapped.Index<0:0:6>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>) dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x16x7x14xf16, [@CMX_NN, 1]>, outputType = memref<1x16x7x14xf16, {order = #NCHW, strides = [3136, 196, 14, 1]}, @DDR>>) -> !VPURegMapped.Index<1:1:0>
    %748 = VPUMI40XX.ActShaveRt kernel("nnActEntry") -> !VPURegMapped.Index<0:0:0>
    %749 = VPURegMapped.Enqueue (%734 -> %734 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:0> {taskType = #VPURegMapped.task_type<DMA>}
    %750 = VPURegMapped.Enqueue previousTaskIdx(%749 : !VPURegMapped.Index<0:0:0>) (%745 -> %745 : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:1> {taskType = #VPURegMapped.task_type<DMA>}
    %751 = VPURegMapped.Enqueue previousTaskIdx(%750 : !VPURegMapped.Index<0:0:1>) (%746 -> %746 : <1:0:0> -> <1:0:0>) -> !VPURegMapped.Index<0:0:2> {taskType = #VPURegMapped.task_type<DMA>}
    %752 = VPURegMapped.Enqueue previousTaskIdx(%751 : !VPURegMapped.Index<0:0:2>) (%747 -> %747 : <1:1:0> -> <1:1:0>) -> !VPURegMapped.Index<0:0:3> {taskType = #VPURegMapped.task_type<DMA>}
    %753 = VPURegMapped.Enqueue previousTaskIdx(%752 : !VPURegMapped.Index<0:0:3>) at(%691 : !VPURegMapped.Index<0:0:0>) (%712 -> %712 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:4> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %754 = VPURegMapped.Enqueue previousTaskIdx(%753 : !VPURegMapped.Index<0:0:4>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%753 : !VPURegMapped.Index<0:0:4>) (%715 -> %715 : <1:0:0> -> <1:0:0>) -> !VPURegMapped.Index<0:0:5> {taskType = #VPURegMapped.task_type<DPUVariant>}
    %755 = VPURegMapped.Enqueue previousTaskIdx(%754 : !VPURegMapped.Index<0:0:5>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%754 : !VPURegMapped.Index<0:0:5>) (%702 -> %702 : <0:0:0> -> <0:0:0>) -> !VPURegMapped.Index<0:0:6> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %756 = VPURegMapped.Enqueue previousTaskIdx(%755 : !VPURegMapped.Index<0:0:6>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%755 : !VPURegMapped.Index<0:0:6>) (%703 -> %703 : <0:0:1> -> <0:0:1>) -> !VPURegMapped.Index<0:0:7> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %775 = VPURegMapped.Enqueue previousTaskIdx(%756 : !VPURegMapped.Index<0:0:7>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%756 : !VPURegMapped.Index<0:0:7>) (%773 -> %773 : <0:1:0> -> <0:1:0>) -> !VPURegMapped.Index<0:0:8> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %776 = VPURegMapped.Enqueue previousTaskIdx(%775 : !VPURegMapped.Index<0:0:8>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%775 : !VPURegMapped.Index<0:0:8>) (%774 -> %774 : <0:1:1> -> <0:1:1>) -> !VPURegMapped.Index<0:0:9> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %757 = VPURegMapped.Enqueue previousTaskIdx(%776 : !VPURegMapped.Index<0:0:9>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%776 : !VPURegMapped.Index<0:0:9>) (%704 -> %704 : <1:0:0> -> <1:0:0>) -> !VPURegMapped.Index<0:0:10> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %758 = VPURegMapped.Enqueue previousTaskIdx(%757 : !VPURegMapped.Index<0:0:10>) at(%691 : !VPURegMapped.Index<0:0:0>) previousTaskIdxOnSameBarrier(%757 : !VPURegMapped.Index<0:0:10>) (%705 -> %705 : <1:0:1> -> <1:0:1>) -> !VPURegMapped.Index<0:0:11> {taskType = #VPURegMapped.task_type<ActKernelInvocation>}
    %759 = VPUMI40XX.Bootstrap inputs(%691 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %miV = VPUMI40XX.MappedInferenceVersion(11 _ 4 _ 10) -> !VPURegMapped.Index<0:0:0>
    %766 = VPUMI40XX.MappedInference dmas((%734, %745), (%746, %747) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>, !VPURegMapped.Index<1:1:0>)) invariants(%706, %709 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) variants(%712, %715 : !VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<1:0:0>) actKernelRanges((%698, %771), (%700) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>)) actKernelInvocations((%702, %773), (%704) : (!VPURegMapped.Index<0:0:0>, !VPURegMapped.Index<0:1:0>), (!VPURegMapped.Index<1:0:0>)) barriers(%691 : !VPURegMapped.Index<0:0:0>) workItemTasks(%749 : !VPURegMapped.Index<0:0:0>) bootstrapBarriers(%759 : !VPURegMapped.Index<0:0:0>) actShaveRt(%748 : !VPURegMapped.Index<0:0:0>) dmaHwpBase(%648 : memref<16xui32, [@CMX_NN, 0]>) dmaCount([[11, 1], [1, 1]]) invariantCount([3, 3]) variantCount([3, 3]) actKernelRangesCount([[2, 2], [2, 0]]) actKernelInvocationsCount([[2, 1], [2, 0]]) mediaCount(0) barrierCount(7) workItemCount(10) bootstrapBarriersCount(7) bootsrapWorkItemsCount(4) finalBarrierId(6) mappedInferenceVersion(%miV : !VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    ELF.ABIVersion(1 _ 0 _ 0) {sym_name = "LoaderABIVersion"}
    VPUMI40XX.OpRanges
  }
}

//CHECK: VPUASM.WorkItem @[[Enqueue0:.+]] idx(!VPURegMapped.Index<0:0:0>) real_task_index(!VPURegMapped.Index<0:0:0>) task_type(<DMA>) first_task(@task.dma.0.0::@NNDMA_0_0_0) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue1:.+]] idx(!VPURegMapped.Index<0:0:1>) real_task_index(!VPURegMapped.Index<0:1:0>) task_type(<DMA>) first_task(@task.dma.0.1::@NNDMA_0_1_0) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue2:.+]] idx(!VPURegMapped.Index<0:0:2>) real_task_index(!VPURegMapped.Index<1:0:0>) task_type(<DMA>) first_task(@task.dma.1.0::@NNDMA_1_0_0) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue3:.+]] idx(!VPURegMapped.Index<0:0:3>) real_task_index(!VPURegMapped.Index<1:1:0>) task_type(<DMA>) first_task(@task.dma.1.1::@NNDMA_1_1_0) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue4:.+]] idx(!VPURegMapped.Index<0:0:4>) real_task_index(!VPURegMapped.Index<0:0:0>) next_workitem_idx(!VPURegMapped.Index<0:0:5>) task_type(<DPUVariant>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_0_0_61) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue5:.+]] idx(!VPURegMapped.Index<0:0:5>) real_task_index(!VPURegMapped.Index<1:0:0>) next_workitem_idx(!VPURegMapped.Index<0:0:6>) task_type(<DPUVariant>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_DPUVariant_1_0_61) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue6:.+]] idx(!VPURegMapped.Index<0:0:6>) real_task_index(!VPURegMapped.Index<0:0:0>) next_workitem_idx(!VPURegMapped.Index<0:0:7>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_0_14) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue7:.+]] idx(!VPURegMapped.Index<0:0:7>) real_task_index(!VPURegMapped.Index<0:0:1>) next_workitem_idx(!VPURegMapped.Index<0:0:8>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_0_15) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue8:.+]] idx(!VPURegMapped.Index<0:0:8>) real_task_index(!VPURegMapped.Index<0:1:0>) next_workitem_idx(!VPURegMapped.Index<0:0:9>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_1_14) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue9:.+]] idx(!VPURegMapped.Index<0:0:9>) real_task_index(!VPURegMapped.Index<0:1:1>) next_workitem_idx(!VPURegMapped.Index<0:0:10>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_0_1_15) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue10:.+]] idx(!VPURegMapped.Index<0:0:10>) real_task_index(!VPURegMapped.Index<1:0:0>) next_workitem_idx(!VPURegMapped.Index<0:0:11>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_1_0_14) task_count(1)
//CHECK: VPUASM.WorkItem @[[Enqueue11:.+]] idx(!VPURegMapped.Index<0:0:11>) real_task_index(!VPURegMapped.Index<1:0:1>) task_type(<ActKernelInvocation>) first_task(@program.metadata.cmx::@DeclareTaskBuffer_ActKernelInvocation_1_0_15) task_count(1)

//CHECK: VPUASM.ManagedMappedInference
//CHECK-SAME: actshv_used = 7
//CHECK-SAME: dma_from_cmx_used = 3
//CHECK-SAME: dma_from_ddr_used = 3
//CHECK-SAME: dpu_used = 3
//CHECK-SAME: media_used = 0
