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

#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t, 4>> broadcastEltwiseShape(llvm::ArrayRef<int64_t> shape1,
                                                               llvm::ArrayRef<int64_t> shape2,
                                                               vpux::IE::AutoBroadcastType broadcastType,
                                                               mlir::Location loc);
}  // namespace IE
}  // namespace vpux
