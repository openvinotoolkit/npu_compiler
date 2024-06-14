//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% revision-id=3" -verify-diagnostics %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX || arch-VPUX40XX

// expected-error@+1 {{Architecture is already defined, probably you run '--init-compiler' twice}}
module @test attributes {VPU.arch = #VPU.arch_kind<NPU30XX>} {
}

// -----

// expected-error@+1 {{Available executor kind 'DMA_NN' was already added}}
module @error {
    IE.ExecutorResource 1 of @DMA_NN
}

// -----

// expected-error@+1 {{RevisionID is already defined, probably you run '--init-compiler' twice}}
module @revtest attributes {VPU.revisionID = #VPU.revision_id<REVISION_B>} {
}
