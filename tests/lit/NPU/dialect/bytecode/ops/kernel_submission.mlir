//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

bytecode.kernel_section @kernel_section {
    bytecode.kernel @my_kernel "\00\01\02\03"
}

bytecode.func_section @function_section {
    bytecode.ext.func @cmd_list_create () -> () {
        %cmd_list = bytecode.virtual_general_register
        %kernel = bytecode.virtual_general_register
        %signal = bytecode.virtual_general_register
        %wait0 = bytecode.virtual_general_register
        %wait1 = bytecode.virtual_general_register
        bytecode.cmd_list.create %cmd_list
        bytecode.cmd_list.add_kernel %cmd_list, %kernel,
                                     (%signal : !bytecode.Register),
                                     (%wait0, %wait1 : !bytecode.Register, !bytecode.Register)
        bytecode.cmd_list.close %cmd_list
        bytecode.cmd_list.exec %cmd_list, 1
        bytecode.ret
    }
    bytecode.ext.func @kernel_create (i64, i64) -> () {
        %dst  = bytecode.virtual_general_register
        %in0  = bytecode.virtual_parameter_register 0
        %out0 = bytecode.virtual_parameter_register 1
        bytecode.kernel.create %dst, @kernel_section::@my_kernel, inputs(%in0), outputs(%out0)
        bytecode.ret
    }
}

// CHECK-LABEL: bytecode.ext.func @cmd_list_create
// CHECK:         [[CMD_LIST:%.+]] = bytecode.virtual_general_register
// CHECK:         [[KERNEL:%.+]] = bytecode.virtual_general_register
// CHECK:         [[SIGNAL:%.+]] = bytecode.virtual_general_register
// CHECK:         [[WAIT0:%.+]] = bytecode.virtual_general_register
// CHECK:         [[WAIT1:%.+]] = bytecode.virtual_general_register
// CHECK:         bytecode.cmd_list.create [[CMD_LIST]]
// CHECK:         bytecode.cmd_list.add_kernel [[CMD_LIST]], [[KERNEL]], ([[SIGNAL]] : !bytecode.Register), ([[WAIT0]], [[WAIT1]] : !bytecode.Register, !bytecode.Register)
// CHECK:         bytecode.cmd_list.close [[CMD_LIST]]
// CHECK:         bytecode.cmd_list.exec [[CMD_LIST]], 1
// CHECK:         bytecode.ret

// CHECK-LABEL: bytecode.ext.func @kernel_create
// CHECK:         [[DST:%.+]]  = bytecode.virtual_general_register
// CHECK:         [[IN0:%.+]]  = bytecode.virtual_parameter_register 0
// CHECK:         [[OUT0:%.+]] = bytecode.virtual_parameter_register 1
// CHECK:         bytecode.kernel.create [[DST]], @kernel_section::@my_kernel, inputs([[IN0]]), outputs([[OUT0]])
// CHECK:         bytecode.ret
