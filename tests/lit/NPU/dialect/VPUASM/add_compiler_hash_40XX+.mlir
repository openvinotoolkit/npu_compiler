//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --add-compiler-hash %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @AddCompilerHash {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x3x4xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x3x4xf16>
    }
    func.func @main() {
        ELF.Main @ELFMain {
        }
        return
    }
}

// CHECK:    ELF.CreateSection
// CHECK-SAME: @info.compiler.hash
// CHECK-SAME: aligned(64)
// CHECK-SAME: secType(VPU_SHT_COMPILER_HASH)
// CHECK-SAME: secFlags("SHF_NONE")
// CHECK-SAME: secLocation(<DDR>)

// CHECK:    VPUASM.CompilerHash
// CHECK-SAME: @CompilerHash
// CHECK-SAME: compiler_hash("{{[0-9a-f]+}}")
