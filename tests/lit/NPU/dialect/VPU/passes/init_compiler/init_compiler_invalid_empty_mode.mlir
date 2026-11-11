//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" -verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// expected-error@+1 {{CompilationMode is already defined, probably you run '--init-compiler' twice}}
module @arch attributes {config.compilationMode = #config.compilation_mode<ReferenceSW>} {
}
