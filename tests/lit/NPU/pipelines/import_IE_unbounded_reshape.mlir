//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-translate --platform=%platform% --import-IE ./IR/unbounded_reshape.xml 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Input: an unbounded network input reshaped into a dimension whose bound must be derived from the input bounds
// Case : the missing input bounds are reported instead of dereferencing a null bounded type

// CHECK: Reshape node 'reshape': cannot compute output bounds, the input has no upper bounds
