//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg60XX/mapped_inference_rewriter.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/utils.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

using namespace vpux;
using namespace NPUReg60XX;
using namespace NPUReg60XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg60xx {

mlir::LogicalResult MappedInferenceRewriter::matchAndRewrite(VPUASM::MappedInferenceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    // TODO: E#186116 Once VPUMI and VPUASM MappedInferenceOp is removed we can remove the rewriter
    rewriter.eraseOp(origOp);
    return mlir::success();
}
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
