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

#include <ie_blob.h>

#include <cstddef>

// FIMXE: remove namespace from a header to avoid possible conflicts
namespace ie = InferenceEngine;

namespace utils {

ie::Blob::Ptr convertPrecision(const ie::Blob::Ptr& sourceData, const ie::Precision& targetPrecision);
bool isBlobAllocatedByAllocator(const ie::Blob::Ptr& blob,
                                const std::shared_ptr<InferenceEngine::IAllocator>& allocator);

ie::Blob::Ptr reallocateBlob(const ie::Blob::Ptr& blob, const std::shared_ptr<ie::IAllocator>& allocator);

std::size_t getByteSize(const ie::TensorDesc& desc);
int extractIdFromDeviceName(const std::string& name);

}  // namespace utils

ie::Blob::Ptr makeSingleValueBlob(const ie::TensorDesc& desc, float val);
ie::Blob::Ptr makeSingleValueBlob(const ie::TensorDesc& desc, int64_t val);

ie::Blob::Ptr makeScalarBlob(float val, const ie::Precision& precision = ie::Precision::FP32, size_t numDims = 1);
ie::Blob::Ptr makeScalarBlob(int64_t val, const ie::Precision& precision = ie::Precision::I64, size_t numDims = 1);

ie::Blob::Ptr toPrecision(const ie::Blob::Ptr& in, const ie::Precision& precision,
                          const std::shared_ptr<InferenceEngine::IAllocator>& alloc = nullptr);
ie::Blob::Ptr toPrecision(const ie::Blob::Ptr& in, ie::Blob::Ptr& out);
ie::Blob::Ptr toDefPrecision(const ie::Blob::Ptr& in);

inline ie::Blob::Ptr toFP32(const ie::Blob::Ptr& in) {
    return toPrecision(in, ie::Precision::FP32);
}
inline ie::Blob::Ptr toFP16(const ie::Blob::Ptr& in) {
    return toPrecision(in, ie::Precision::FP16);
}

ie::Blob::Ptr toLayout(const ie::Blob::Ptr& in, ie::Layout layout);
ie::Blob::Ptr toDefLayout(const ie::Blob::Ptr& in);
