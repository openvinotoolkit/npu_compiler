//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-copy-between-swkernels-and-network-io %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForInOut
net.NetworkInfo entryPoint : @AddCopyForInOut inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @AddCopyForInOut
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                outputs(%arg1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
    }

    return %sw_kernel: memref<1x1x1x1000xf16, @DDR>

    // CHECK:    [[NEW_OUTPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[NEW_INPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_DDR_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:    [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:  inputs([[INPUT_COPY]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_OUTPUT_DDR_BUFF]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForInOutTwoKernels
net.NetworkInfo entryPoint : @AddCopyForInOutTwoKernels inputsInfo : {
    DataInfo "input_0" : tensor<1x1x1x1000xf16>
    DataInfo "input_1" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output_0" : tensor<1x1x1x1000xf16>
    DataInfo "output_1" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @AddCopyForInOutTwoKernels
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %input_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%input_1 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg0 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                outputs(%output_1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
                on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
    }

    return %sw_kernel#0, %sw_kernel#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DDR_INPUT_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DDR_OUTPUT_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_DDR_OUTPUT_BUFF_0]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK:   inputs([[DDR_INPUT_1]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[INPUT_COPY]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[DDR_OUTPUT_1]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[NEW_DDR_OUTPUT_BUFF_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[SW_KERNEL]]#1 : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[SW_KERNEL]]#0, [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForPrivateFuncIn
net.NetworkInfo entryPoint : @AddCopyForPrivateFuncIn inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @SwKernelPrivateFuncSwKernelIn
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncSwKernelIn(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %output_buff_alloc =  memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
        inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
        outputs(%output_buff_alloc as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
        on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
        }

    %output_copy = VPUIP.Copy inputs(%sw_kernel  : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    return %output_copy: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[ARG_0]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>) outputs([[ARG_1]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncIn
// CHECK-SAME:     ([[INPUT_ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT_ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForPrivateFuncIn(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %output_buff = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %call_op = call @SwKernelPrivateFuncSwKernelIn(%arg0, %output_buff) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    %output_copy = VPUIP.Copy inputs(%call_op : memref<1x1x1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    return %output_copy : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT_ARG]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_ALLOC]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[PRIVATE_FUNC_CALL:%.+]] = call @SwKernelPrivateFuncSwKernelIn([[INPUT_COPY]], [[OUTPUT_ALLOC]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT_ARG_COPY:%.+]] = VPUIP.Copy inputs([[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT_ARG]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[OUTPUT_ARG_COPY]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForPrivateFuncInOut
net.NetworkInfo entryPoint : @AddCopyForPrivateFuncInOut inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @SwKernelPrivateFuncInOut
// CHECK-SAME:     ([[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_2:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
        inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
        outputs(%arg1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
        on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
        }

    return %sw_kernel: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[ARG_1]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[ARG_2]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncInOut
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForPrivateFuncInOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %call_op = func.call @SwKernelPrivateFuncInOut(%arg0, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    return %call_op: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_ALLOC]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[PRIVATE_FUNC_CALL:%.+]] = call @SwKernelPrivateFuncInOut([[INPUT_COPY]], [[NEW_OUTPUT_ALLOC]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForPrivateFuncOut
net.NetworkInfo entryPoint : @AddCopyForPrivateFuncOut inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @SwKernelPrivateFuncOut
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %input_buff_alloc =  memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %input_copy = VPUIP.Copy inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%input_buff_alloc : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%input_copy as %in_buff_0: memref<1x1x1x1000xf16, @DDR>)
                outputs(%arg1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>)
                on tile 0 -> memref<1x1x1x1000xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
        }

    return %sw_kernel: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x1x1x1000xf16, @DDR>) outputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu inputs([[INPUT_COPY]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[ARG_1]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:       VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT]], [[SW_KERNEL_OUTPUT]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[SW_KERNEL]] : memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncOut
// CHECK-SAME:     ([[INPUT_ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT_ARG:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForPrivateFuncOut(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %input_alloc = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %input_arg_copy = VPUIP.Copy inputs(%arg0 :  memref<1x1x1x1000xf16, @DDR>) outputs(%input_alloc : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    %call_op = func.call @SwKernelPrivateFuncOut(%input_arg_copy, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    return %call_op: memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[INPUT_ARG_COPY:%.+]] = VPUIP.Copy inputs([[INPUT_ARG]] : memref<1x1x1x1000xf16, @DDR>) outputs([[INPUT_ALLOC]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[PRIVATE_FUNC_CALL:%.+]] = call @SwKernelPrivateFuncOut([[INPUT_ARG_COPY]], [[NEW_OUTPUT_ALLOC]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[PRIVATE_FUNC_CALL]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT_ARG]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

net.NetworkInfo entryPoint : @AddCopyForPrivateFuncInOutTwoKernels inputsInfo : {
    DataInfo "input_0" : tensor<1x1x1x1000xf16>
    DataInfo "input_1" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output_0" : tensor<1x1x1x1000xf16>
    DataInfo "output_1" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @SwKernelPrivateFuncInOutTwoKernels
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func private @SwKernelPrivateFuncInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %input_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%input_1 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg0 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>)
                outputs(%output_1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>)
                on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
    }

    return %sw_kernel#0, %sw_kernel#1: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>


    // CHECK:   [[INPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[OUTPUT:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:    [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK:    inputs([[INPUT]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_0]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[ARG_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncInOutTwoKernels
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForPrivateFuncInOutTwoKernels(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %call_op:2 = func.call @SwKernelPrivateFuncInOutTwoKernels(%arg0, %arg1) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)

    return %call_op#0, %call_op#1: memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_INPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[PRIVATE_FUNC:%.+]]:2 = call @SwKernelPrivateFuncInOutTwoKernels([[INPUT_COPY]], [[NEW_OUTPUT_BUFF]]) : (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>)
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[PRIVATE_FUNC]]#1 : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[PRIVATE_FUNC]]#0, [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// -----

net.NetworkInfo entryPoint : @AddCopyForPrivateFuncInOutWithViewOpInside inputsInfo : {
    DataInfo "input" : tensor<1x1x2x64xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x2x64xf16>
}

module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @SwKernelPrivateFuncInOutWithViewOpInside
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x2x64xf16, @DDR>, [[ARG_1:%.+]]: memref<1x1x2x64xf16, @DDR>)
func.func private @SwKernelPrivateFuncInOutWithViewOpInside(%arg0: memref<1x1x2x64xf16, @DDR>, %arg1: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %1 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %2 = VPUIP.SubView %arg0 [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %3 = VPUIP.SubView %arg1 [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%0 as %arg2: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, %2 as %arg3: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>) outputs(%1 as %arg4: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, %3 as %arg5: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>) on tile 0 -> (memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>){
    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg4) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>
    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg5) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>
    }

    %4 = VPUIP.ConcatView inputs(%results#0, %results#1 : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) outputs(%arg1 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    return %4 : memref<1x1x2x64xf16, @DDR>

    // CHECK:       [[KERNEL_IN_SLICE_0:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_OUT_SLICE_0:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_IN_SLICE_1:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_OUT_SLICE_1:%.+]] = VPUIP.SubView [[ARG_1]] [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:  inputs([[KERNEL_IN_SLICE_0]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, [[KERNEL_IN_SLICE_1]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>) outputs([[KERNEL_OUT_SLICE_0]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, [[KERNEL_OUT_SLICE_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>
    // CHECK:   [[CONCAT_RESULT:%.+]] = VPUIP.ConcatView inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>) outputs([[ARG_1]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:   return [[CONCAT_RESULT]] : memref<1x1x2x64xf16, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncInOutWithViewOpInside
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x2x64xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x2x64xf16, @DDR>)
func.func @AddCopyForPrivateFuncInOutWithViewOpInside(%arg0: memref<1x1x2x64xf16, @DDR>, %arg1: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR> {
    %call_op = func.call @SwKernelPrivateFuncInOutWithViewOpInside(%arg0, %arg1) : (memref<1x1x2x64xf16, @DDR>, memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>

    return %call_op: memref<1x1x2x64xf16, @DDR>

    // CHECK:   [[NEW_OUTPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x2x64xf16, @DDR>
    // CHECK:   [[NEW_INPUT_ALLOC:%.+]] = memref.alloc() : memref<1x1x2x64xf16, @DDR>

    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x2x64xf16, @DDR>) outputs([[NEW_INPUT_ALLOC]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:   [[PRIVATE_FUNC_CALL:%.+]] = call @SwKernelPrivateFuncInOutWithViewOpInside([[INPUT_COPY]], [[NEW_OUTPUT_ALLOC]]) : (memref<1x1x2x64xf16, @DDR>, memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[PRIVATE_FUNC_CALL]] : memref<1x1x2x64xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>

    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x2x64xf16, @DDR>
}

// -----

// CHECK-LABEL: @AddCopyForInOutWithViewOp
net.NetworkInfo entryPoint : @AddCopyForInOutWithViewOp inputsInfo : {
    DataInfo "input" : tensor<1x1x2x64xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x2x64xf16>
}
module @VPU.SW {
    func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}


// CHECK-LABEL: @AddCopyForInOutWithViewOp
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x2x64xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x2x64xf16, @DDR>)
func.func @AddCopyForInOutWithViewOp(%arg0: memref<1x1x2x64xf16, @DDR>, %arg1: memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %1 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %2 = VPUIP.SubView %arg0 [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %3 = VPUIP.SubView %arg1 [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>
    %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax inputs(%0 as %arg2: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, %2 as %arg3: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>) outputs(%1 as %arg4: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, %3 as %arg5: memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>) on tile 0 -> (memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>){
    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg2, %arg4) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>
    VPUIP.SW.Kernel.run {attrs = [0, 0]}(%arg3, %arg5) : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}>
    }
    %4 = VPUIP.ConcatView inputs(%results#0, %results#1 : memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [128, 128, 64, 1]}, @DDR>) outputs(%arg1 : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    return %4 : memref<1x1x2x64xf16, @DDR>
    // CHECK:    [[NEW_OUTPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x2x64xf16, @DDR>
    // CHECK:    [[NEW_INPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x2x64xf16, @DDR>
    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x2x64xf16, @DDR>) outputs([[NEW_INPUT_DDR_BUFF]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:       [[KERNEL_IN_SLICE_0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_OUT_SLICE_0:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUFF]] [0, 0, 0, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_IN_SLICE_1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[KERNEL_OUT_SLICE_1:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUFF]] [0, 0, 1, 0] [1, 1, 1, 64] : memref<1x1x2x64xf16, @DDR> to memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>
    // CHECK:       [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK-SAME:  inputs([[KERNEL_IN_SLICE_0]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, [[KERNEL_IN_SLICE_1]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>) outputs([[KERNEL_OUT_SLICE_0]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, [[KERNEL_OUT_SLICE_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>)
    // CHECK-SAME:  on tile 0 -> (memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>){
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>
    // CHECK:           VPUIP.SW.Kernel.run {attrs = [0, 0]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}>
    // CHECK:   [[CONCAT_RESULT:%.+]] = VPUIP.ConcatView
    // CHECK-SAME:               inputs([[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>, memref<1x1x1x64xf16, {order = #NCHW, strides = [128, 128, 64, 1]}, @DDR>)
    // CHECK-SAME:               outputs([[NEW_OUTPUT_DDR_BUFF]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT_RESULT]] : memref<1x1x2x64xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x2x64xf16, @DDR>) -> memref<1x1x2x64xf16, @DDR>
    // CHECK:   return [[OUTPUT_COPY]] : memref<1x1x2x64xf16, @DDR>
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]
module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}
// CHECK-LABEL: @AddCopyForInOutWithoutTilePattern
net.NetworkInfo entryPoint : @AddCopyForInOutWithoutTilePattern inputsInfo : {
    DataInfo "input" : tensor<1x1x1x500xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x500xf16>
}
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @AddCopyForInOutWithoutTilePattern
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x500xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x500xf16, @DDR>)
func.func @AddCopyForInOutWithoutTilePattern(%arg0: memref<1x1x1x500xf16, @DDR>, %arg1: memref<1x1x1x500xf16, @DDR>) -> memref<1x1x1x500xf16, @DDR> {
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %in_buff_0: memref<1x1x1x500xf16, @DDR>)
                outputs(%arg1 as %out_buff_0: memref<1x1x1x500xf16, @DDR>)
                on tile 0 -> memref<1x1x1x500xf16, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x500xf16, @DDR>
                , memref<1x1x1x500xf16, @DDR>
    }
    return %sw_kernel: memref<1x1x1x500xf16, @DDR>

    // CHECK:    [[NEW_OUTPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x500xf16, @DDR>
    // CHECK:    [[NEW_INPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x500xf16, @DDR>
    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x500xf16, @DDR>)
    // CHECK:                                     outputs([[NEW_INPUT_DDR_BUFF]] : memref<1x1x1x500xf16, @DDR>)
    // CHECK:    [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:   inputs([[INPUT_COPY]] as [[SW_KERNEL_INPUT:%.+]]: memref<1x1x1x500xf16, @DDR>) outputs([[NEW_OUTPUT_DDR_BUFF]] as [[SW_KERNEL_OUTPUT:%.+]]: memref<1x1x1x500xf16, @DDR>)
    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK:        inputs([[SW_KERNEL]] : memref<1x1x1x500xf16, @DDR>)
    // CHECK-SAME:   outputs([[OUTPUT]] : memref<1x1x1x500xf16, @DDR>)
    // CHECK:   return [[OUTPUT_COPY]]
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]
module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}
// CHECK-LABEL: @AddCopyForPrivateFuncInWithViewOpInMain
net.NetworkInfo entryPoint : @AddCopyForPrivateFuncInWithViewOpInMain inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


// CHECK-LABEL: @SwKernelPrivateFuncSwKernelIn
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_2:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_3:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
func.func private @SwKernelPrivateFuncSwKernelIn(%arg0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>,
                                                 %arg1: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>,
                                                 %arg2: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>,
                                                 %arg3: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) {
    %sw_kernel:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %in_buff_0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, %arg1 as %in_buff_1: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
                outputs(%arg2 as %out_buff_0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, %arg3 as %out_buff_1: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
                on tile 0 -> (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
                , memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
                , memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    }
    return %sw_kernel#0, %sw_kernel#1 : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>

    // CHECK:  [[SW_KERNEL:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:  inputs([[ARG_0]] as [[IN_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_1]] as [[IN_1:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) outputs([[ARG_2]] as [[OUT_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_3]] as [[OUT_1:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_0]], [[OUT_0]])
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_1]], [[OUT_1]])
    // CHECK:  }
    // CHECK:   return [[SW_KERNEL]]#0, [[SW_KERNEL]]#1 : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
}

// CHECK-LABEL: @AddCopyForPrivateFuncInWithViewOpInMain
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForPrivateFuncInWithViewOpInMain(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %1 = VPUIP.SubView %arg0 [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %2 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %3 = VPUIP.SubView %arg1 [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %call_op:2 = call @SwKernelPrivateFuncSwKernelIn(%0, %1, %2, %3) : (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    %4 = VPUIP.ConcatView inputs(%call_op#0, %call_op#1 : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    return %4 : memref<1x1x1x1000xf16, @DDR>

    // CHECK:    [[NEW_OUTPUT_DDR_BUF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[NEW_INPUT_DDR_BUF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_INPUT_DDR_BUF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[INPUT_SUBVIEW_0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[INPUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[OUTPUT_SUBVIEW_0:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUF]] [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[OUTPUT_SUBVIEW_1:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUF]] [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[CALL_OP:%.+]]:2 = call @SwKernelPrivateFuncSwKernelIn([[INPUT_SUBVIEW_0]], [[INPUT_SUBVIEW_1]], [[OUTPUT_SUBVIEW_0]], [[OUTPUT_SUBVIEW_1]])
    // CHECK-SAME:         (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK-SAME:       -> (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CALL_OP]]#0, [[CALL_OP]]#1 : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[NEW_OUTPUT_DDR_BUF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[CONCAT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:   return [[OUTPUT_COPY]]
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]
module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }
func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}
// CHECK-LABEL: @AddCopyForInOutWithViewOpInMainNoTilePattern
net.NetworkInfo entryPoint : @AddCopyForInOutWithViewOpInMainNoTilePattern inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1000xf16>
}
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SwKernelPrivateFuncSwKernelIn
// CHECK-SAME:     ([[ARG_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, [[ARG_1:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
func.func private @SwKernelPrivateFuncSwKernelIn(%arg0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, %arg1: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR> {
    %sw_kernel = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
        inputs(%arg0 as %in_buff_0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
        outputs(%arg1 as %out_buff_0: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
        on tile 0 -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR> {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
                , memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
        }
    return %sw_kernel: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>

    // CHECK:  [[SW_KERNEL:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK-SAME:   inputs([[ARG_0]] as [[IN_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) outputs([[ARG_1]] as [[OUT_0:%.+]]: memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK:                    VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[IN_0]], [[OUT_0]])
    // CHECK:  }
    // CHECK:   return [[SW_KERNEL]] : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
}

// CHECK-LABEL: @AddCopyForInOutWithViewOpInMainNoTilePattern
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForInOutWithViewOpInMainNoTilePattern(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR> {
    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %1 = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %call_op_0 = call @SwKernelPrivateFuncSwKernelIn(%0, %1) : (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>

    %2 = VPUIP.SubView %arg0 [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %3 = VPUIP.SubView %arg1 [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    %call_op_1 = call @SwKernelPrivateFuncSwKernelIn(%2, %3) : (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>

    %4 = VPUIP.ConcatView inputs(%call_op_0, %call_op_1 : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    return %4 : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[NEW_OUTPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[NEW_INPUT_DDR_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>)
    // CHECK:                                    outputs([[NEW_INPUT_DDR_BUFF]] : memref<1x1x1x1000xf16, @DDR>)
    // CHECK:    [[INPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[OUTPUT_SUBVIEW0:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUFF]] [0, 0, 0, 0] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[CALL_OP0:%.+]] = call @SwKernelPrivateFuncSwKernelIn([[INPUT_SUBVIEW0]], [[OUTPUT_SUBVIEW0]])
    // CHECK-SAME:    (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[INPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[INPUT_COPY]] [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[OUTPUT_SUBVIEW1:%.+]] = VPUIP.SubView [[NEW_OUTPUT_DDR_BUFF]] [0, 0, 0, 500] [1, 1, 1, 500] : memref<1x1x1x1000xf16, @DDR> to memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[CALL_OP1:%.+]] = call @SwKernelPrivateFuncSwKernelIn([[INPUT_SUBVIEW1]], [[OUTPUT_SUBVIEW1]])
    // CHECK-SAME:    (memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>) -> memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>
    // CHECK:    [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[CALL_OP0]], [[CALL_OP1]] : memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>, memref<1x1x1x500xf16, {order = #NCHW, strides = [1000, 1000, 1000, 1]}, @DDR>)
    // CHECK-SAME:         outputs([[NEW_OUTPUT_DDR_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>
    // CHECK:    [[OUTPUT_COPY:%.+]] = VPUIP.Copy
    // CHECK:        inputs([[CONCAT]] : memref<1x1x1x1000xf16, @DDR>)
    // CHECK-SAME:   outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>)
    // CHECK:   return [[OUTPUT_COPY]]
}

// -----

VPURT.SW.Runtime
    entryPoint: @VPU.SW::@runtime
    stack_configuration: [4096, 4096, 4096, 4096]

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>)
    attributes {
        VPU.kernel_code = "activation_relu.cpp",
        VPU.kernel_entry = "activation_relu",
        VPU.task_type = @COMPUTE
    }

func.func private @runtime()
    attributes {
        VPU.kernel_code = "nnActEntry"
    }
}

// CHECK-LABEL: @AddCopyForInOutThreeKernelsArgUsedTwice
net.NetworkInfo entryPoint : @AddCopyForInOutThreeKernelsArgUsedTwice inputsInfo : {
    DataInfo "input_0" : tensor<1x1x1x1000xf16>
    DataInfo "input_1" : tensor<1x1x1x1000xf16>
} outputsInfo : {
    DataInfo "output_0" : tensor<1x1x1x1000xf16>
    DataInfo "output_1" : tensor<1x1x1x1000xf16>
}

// CHECK-LABEL: @AddCopyForInOutThreeKernelsArgUsedTwice
// CHECK-SAME:     ([[INPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[OUTPUT:%.+]]: memref<1x1x1x1000xf16, @DDR>)
func.func @AddCopyForInOutThreeKernelsArgUsedTwice(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>) -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
    %input_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_1 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    %output_2 = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    %sw_kernel:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_relu
                inputs(%arg0 as %in_buff_0: memref<1x1x1x1000xf16, @DDR>, %input_1 as %in_buff_1: memref<1x1x1x1000xf16, @DDR>, %arg0 as %in_buff_2: memref<1x1x1x1000xf16, @DDR>)
                outputs(%output_1 as %out_buff_0: memref<1x1x1x1000xf16, @DDR>, %arg1 as %out_buff_1: memref<1x1x1x1000xf16, @DDR>, %output_2 as %out_buff_2: memref<1x1x1x1000xf16, @DDR>)
                on tile 0 -> (memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>) {
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_0, %out_buff_0)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_1, %out_buff_1)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
            VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%in_buff_2, %out_buff_2)
                : memref<1x1x1x1000xf16, @DDR>
                , memref<1x1x1x1000xf16, @DDR>
    }

    return %sw_kernel#0, %sw_kernel#1 : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[INPUT_COPY_OUTPUT_BUFF:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[DDR_INPUT_1:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_0:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>
    // CHECK:   [[NEW_DDR_OUTPUT_BUFF_2:%.+]] = memref.alloc() : memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[INPUT_COPY:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x1x1x1000xf16, @DDR>) outputs([[INPUT_COPY_OUTPUT_BUFF]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[SW_KERNEL:%.+]]:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_relu
    // CHECK:   inputs([[INPUT_COPY]] as [[SW_KERNEL_INPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[DDR_INPUT_1]] as [[SW_KERNEL_INPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[INPUT_COPY]] as [[SW_KERNEL_INPUT_2:%.+]]: memref<1x1x1x1000xf16, @DDR>) outputs([[NEW_DDR_OUTPUT_BUFF_0]] as [[SW_KERNEL_OUTPUT_0:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[NEW_DDR_OUTPUT_BUFF_1]] as [[SW_KERNEL_OUTPUT_1:%.+]]: memref<1x1x1x1000xf16, @DDR>, [[NEW_DDR_OUTPUT_BUFF_2]]  as [[SW_KERNEL_OUTPUT_2:%.+]]: memref<1x1x1x1000xf16, @DDR>)
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_0]], [[SW_KERNEL_OUTPUT_0]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_1]], [[SW_KERNEL_OUTPUT_1]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
    // CHECK:     VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]}([[SW_KERNEL_INPUT_2]], [[SW_KERNEL_OUTPUT_2]]) : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>

    // CHECK:   [[OUTPUT_COPY:%.+]] = VPUIP.Copy inputs([[SW_KERNEL]]#1 : memref<1x1x1x1000xf16, @DDR>) outputs([[OUTPUT]] : memref<1x1x1x1000xf16, @DDR>) -> memref<1x1x1x1000xf16, @DDR>

    // CHECK:   return [[SW_KERNEL]]#0, [[OUTPUT_COPY]] : memref<1x1x1x1000xf16, @DDR>, memref<1x1x1x1000xf16, @DDR>
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
    // CHECK-LABEL: @SWKernelDynamicInputs
  net.NetworkInfo entryPoint : @SWKernelDynamicInputs inputsInfo : {
    DataInfo "input_0" : tensor<1x8x384x384xf16>
    DataInfo "vpux_ie_shape_input_0" : tensor<4xsi32>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x8x384x384xf16>
    DataInfo "vpux_ie_shape_output_0" : tensor<4xsi32>
  }
    // CHECK-LABEL: @SWKernelDynamicInputs
  func.func @SWKernelDynamicInputs(%arg0 : memref<1x8x384x384xf16>, %arg1 : memref<4xsi32>, %arg2 : memref<1x8x384x384xf16>, %arg3 : memref<4xsi32>) -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    // CHECK-SAME:    ([[ARG_0:%.+]]: memref<1x8x384x384xf16>, [[ARG_1:%.+]]: memref<4xsi32>, [[ARG_2:%.+]]: memref<1x8x384x384xf16>, [[ARG_3:%.+]]: memref<4xsi32>

    //CHECK:        [[ALOC:%.+]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[ALOC_0:%.+]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_1:%.+]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[ALOC_2:%.+]] = memref.alloc() : memref<1x8x384x384xf16>

    //CHECK:        [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x8x384x384xf16>) outputs([[ALOC_2]] : memref<1x8x384x384xf16>)
    //CHECK:        [[COPY_1:%.+]] = VPUIP.Copy inputs([[ARG_1]] : memref<4xsi32>) outputs([[ALOC_1]] : memref<4xsi32>)

    %results:2 = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0>,
            dynamicOutputShapesMap = array<i32: 0>,
            resultSegmentSizes = array<i32: 2, 0, 0>
        }
        @VPU.SW::@builtin_dummy
            inputs(%arg0 as %arg4: memref<1x8x384x384xf16>)
            dynamicInputShapes(%arg1 : memref<4xsi32>)
            outputs(%arg2 as %arg5: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%arg3 : memref<4xsi32>)
                -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
            VPUIP.SW.Kernel.run(%arg4, %arg5) :
                memref<1x8x384x384xf16>,
                memref<1x8x384x384xf16>
    }

    //CHECK:        [[RESULT:%.*]]:2 = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 2, 0, 0>}
    //CHECK-SAME:   inputs([[COPY_0]] as {{[^:]+}}: memref<1x8x384x384xf16>) dynamicInputShapes([[COPY_1]] : memref<4xsi32>)
    //CHECK-SAME:   outputs([[ALOC_0]] as {{[^:]+}}: memref<1x8x384x384xf16>) dynamicOutputShapes([[ALOC]] : memref<4xsi32>)

    //CHECK:        [[COPY_2:%.+]] = VPUIP.Copy inputs([[RESULT]]#1 : memref<4xsi32>) outputs([[ARG_3]] : memref<4xsi32>)
    //CHECK:        [[COPY_3:%.+]] = VPUIP.Copy inputs([[RESULT]]#0 : memref<1x8x384x384xf16>) outputs([[ARG_2]] : memref<1x8x384x384xf16>)

    return %results#0, %results#1 : memref<1x8x384x384xf16>, memref<4xsi32>
    //CHECK:        return [[COPY_3]], [[COPY_2]] : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}

// -----

module {
  module @VPU.SW {
    func.func private @builtin_dummy(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "dummy.cpp", VPU.kernel_entry = "dummy"}
  }
    // CHECK-LABEL: @SWKernelDynamicInputs_1
  net.NetworkInfo entryPoint : @SWKernelDynamicInputs_1 inputsInfo : {
    DataInfo "input_0" : tensor<1x8x384x384xf16>
    DataInfo "vpux_ie_shape_input_0" : tensor<4xsi32>
    DataInfo "input_1" : tensor<1x16x384x384xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x8x384x384xf16>
    DataInfo "vpux_ie_shape_output_0" : tensor<4xsi32>
  }
    // CHECK-LABEL: @SWKernelDynamicInputs_1
  func.func @SWKernelDynamicInputs_1(%arg0 : memref<1x8x384x384xf16>, %arg1 : memref<4xsi32>, %arg2 : memref<1x16x384x384xf16>, %arg3 : memref<1x8x384x384xf16>, %arg4 : memref<4xsi32>) -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
    // CHECK-SAME:    ([[ARG_0:%.+]]: memref<1x8x384x384xf16>, [[ARG_1:%.+]]: memref<4xsi32>, [[ARG_2:%.+]]: memref<1x16x384x384xf16>, [[ARG_3:%.+]]: memref<1x8x384x384xf16>, [[ARG_4:%.+]]: memref<4xsi32>

    //CHECK:        [[ALOC:%.+]] = memref.alloc() : memref<1x8x384x384xf16>
    //CHECK:        [[ALOC_0:%.+]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[ALOC_1:%.+]] = memref.alloc() : memref<1x16x384x384xf16>
    //CHECK:        [[ALOC_2:%.+]] = memref.alloc() : memref<4xsi32>
    //CHECK:        [[ALOC_3:%.+]] = memref.alloc() : memref<1x8x384x384xf16>

    //CHECK:        [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG_0]] : memref<1x8x384x384xf16>) outputs([[ALOC_3]] : memref<1x8x384x384xf16>)
    //CHECK:        [[COPY_1:%.+]] = VPUIP.Copy inputs([[ARG_1]] : memref<4xsi32>) outputs([[ALOC_2]] : memref<4xsi32>)
    //CHECK:        [[COPY_2:%.+]] = VPUIP.Copy inputs([[ARG_2]] : memref<1x16x384x384xf16>) outputs([[ALOC_1]] : memref<1x16x384x384xf16>)

    %results:2 = VPUIP.SW.Kernel {
            dynamicInputShapesMap = array<i32: 0, -1>,
            dynamicOutputShapesMap = array<i32: 0>,
            resultSegmentSizes = array<i32: 1, 1, 0>
        }
        @VPU.SW::@builtin_DynamicReshape
            inputs(%arg0 as %arg5: memref<1x8x384x384xf16>, %arg2 as %arg6: memref<1x16x384x384xf16>)
            dynamicInputShapes(%arg1 : memref<4xsi32>)
            outputs(%arg3 as %arg7: memref<1x8x384x384xf16>)
            dynamicOutputShapes(%arg4 : memref<4xsi32>)
                -> (memref<1x8x384x384xf16>, memref<4xsi32>) {
            VPUIP.SW.Kernel.run(%arg5, %arg6, %arg7) :
                memref<1x8x384x384xf16>,
                memref<1x16x384x384xf16>,
                memref<1x8x384x384xf16>
    }
    //CHECK:        [[RESULT_0:%.*]], [[DYNAMIC_OUTPUT_SHAPES_0:%.+]] = VPUIP.SW.Kernel {dynamicInputShapesMap = array<i32: 0, -1>, dynamicOutputShapesMap = array<i32: 0>, resultSegmentSizes = array<i32: 1, 1, 0>}
    //CHECK-SAME:   inputs([[COPY_0]] as {{[^:]+}}: memref<1x8x384x384xf16>, [[COPY_2]] as {{[^:]+}}: memref<1x16x384x384xf16>) dynamicInputShapes([[COPY_1]] : memref<4xsi32>)
    //CHECK-SAME:   outputs([[ALOC]] as {{[^:]+}}: memref<1x8x384x384xf16>) dynamicOutputShapes([[ALOC_0]] : memref<4xsi32>)

    //CHECK:        [[COPY_3:%.+]] = VPUIP.Copy inputs([[RESULT_0]] : memref<1x8x384x384xf16>) outputs([[ARG_3]] : memref<1x8x384x384xf16>)
    //CHECK:        [[COPY_4:%.+]] = VPUIP.Copy inputs([[DYNAMIC_OUTPUT_SHAPES_0]] : memref<4xsi32>) outputs([[ARG_4]] : memref<4xsi32>)

    return %results#0, %results#1 : memref<1x8x384x384xf16>, memref<4xsi32>
    //CHECK:        return [[COPY_3]], [[COPY_4]] : memref<1x8x384x384xf16>, memref<4xsi32>
  }
}
