//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% revision-id=3" -verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @test attributes {config.arch = #config.arch_kind<NPU37XX>} {
}

// -----

// expected-error@+1 {{Available global resources was already added}}
module @error {
    config.Resources 1 of @global {
        config.ExecutorResource 1 of @DMA_NN
    }
}

// -----

// expected-error@+1 {{RevisionID is already defined, probably you run '--init-compiler' twice}}
module @revtest attributes {config.revisionID = #config.revision_id<REVISION_B>} {
}
