//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "nce2p7.h"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/logger.hpp"

#include "vpux/compiler/dialect/VPUIP/schema.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>

namespace vpux {

/**
 * Helper builder for creation activation-
 * shaves invocation in memory arguments
 */
class InvocationBuilder {
public:
    using PatchCallbackType = std::function<void(MutableArrayRef<uint8_t>, size_t)>;

public:
    InvocationBuilder(size_t dataOffset, Logger log) : _win_e_offset(dataOffset), _log(log) {}

    /**
     * register serialisation for given invocation argument, might be MemRefType or any other supported types
     * @param operand
     */
    void addArg(mlir::Value operand);
    void addTensorArg(mlir::Value value, const MVCNN::TensorReference* tenorRef);

    /**
     * actual serialising routine
     */
    SmallVector<uint8_t> store() const;

private:
    template <class T>
    void appendValue(SmallVector<char>& storage, const T& anyValue) {
        ArrayRef<char> valueAsArray(reinterpret_cast<const char*>(&anyValue), sizeof(anyValue));
        storage.insert(storage.end(), valueAsArray.begin(), valueAsArray.end());
    }

    /**
     * create a patch entry, that can be further updated
     * @tparam U structure type
     * @tparam T field type
     */
    template <class U, class T>
    PatchCallbackType createPatchPoint(const T& patcher) {
        return [offset = _scalarStorage.size(), arrayOffset = _arrayStorage.size(), this, patcher] (MutableArrayRef<uint8_t> serialStorage, size_t updateTo) {
            auto& origObject = reinterpret_cast<U&>(*(serialStorage.begin() + offset));
            patcher(origObject, checked_cast<uint32_t>(updateTo + arrayOffset));
        };
    }

private:
    size_t _win_e_offset;  //  offset of the beginning of invocation args within expected WIN_E
    Logger _log;

    SmallVector<char> _scalarStorage;   // keeps scalar elements and memref metadata
    SmallVector<char> _arrayStorage;    // keeps arrays elements

    SmallVector<PatchCallbackType> _deferredPointers;
};

}  // namespace vpux
