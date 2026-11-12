//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=false" -verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @mode attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>} {
}

// -----

// expected-error@+1 {{Target platform is already set, probably you run '--init-compiler' twice}}
module @arch attributes {config.platform = #config.platform<NPU3720>} {
}

// -----

// expected-error@+1 {{Available global resources was already added}}
module @executors {
    config.Resources 1 of @global {
        config.ExecutorResource 1 of @DMA_NN
    }
}

// -----

// expected-error@+1 {{Target platform is already set, probably you run '--init-compiler' twice}}
module @arch attributes {config.platform = #config.platform<NPU4000>} {
}
