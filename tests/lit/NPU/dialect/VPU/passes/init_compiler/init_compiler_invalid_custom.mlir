//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=false" -verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @mode attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>} {
}

// -----

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @arch attributes {config.arch = #config.arch_kind<NPU37XX>} {
}

// -----

// expected-error@+1 {{Available global resources was already added}}
module @executors {
    config.Resources 1 of @global {
        config.ExecutorResource 1 of @DMA_NN
    }
}
