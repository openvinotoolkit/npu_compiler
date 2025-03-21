//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=false" -verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @mode attributes {VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {
}

// -----

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @arch attributes {VPU.arch = #VPU.arch_kind<NPU37XX>} {
}

// -----

// expected-error@+1 {{Available executor kind 'DMA_NN' was already added}}
module @executors {
    IE.ExecutorResource 2 of @DMA_NN
}
