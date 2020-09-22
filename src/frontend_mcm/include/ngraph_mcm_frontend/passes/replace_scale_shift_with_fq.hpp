//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

// clang-format off

#include <ngraph/pass/graph_rewrite.hpp>

//
// For [Input U8] -> [Convert FP32] -> [ScaleShift] replace [ScaleShift] with [FakeQuantize].
//

class ReplaceScaleShiftWithFQ final : public ngraph::pass::GraphRewrite {
public:
    ReplaceScaleShiftWithFQ();
};

// clang-format on
