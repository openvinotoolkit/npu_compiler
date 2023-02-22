//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <kernels/inc/common_types.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPUIPRegMapped {

class KernelParamsSerializer {
public:
    KernelParamsSerializer() = delete;

    static SmallVector<uint8_t> createKernelParams(VPUIP::SwKernelOp swKernelOp);

    static sw_params::Location getSwParamsLocationFromMemKind(VPU::MemoryKind memKind);
    static sw_params::DataType getDataTypeFromMlirType(mlir::Type type);

private:
    static void addTensorArgToVector(SmallVector<uint8_t>& vec, mlir::Value value);
    static void addAttrsToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr);
    static void addBasicAttrToVector(SmallVector<uint8_t>& vec, mlir::Attribute attr);

    template <class T>
    static void appendValueToVector(SmallVector<uint8_t>& vec, const T& anyValue) {
        ArrayRef<uint8_t> valueAsArray(reinterpret_cast<const uint8_t*>(&anyValue), sizeof(anyValue));
        vec.insert(vec.end(), valueAsArray.begin(), valueAsArray.end());
    }
};

}  // namespace VPUIPRegMapped
}  // namespace vpux
