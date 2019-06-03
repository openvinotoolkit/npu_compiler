//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vpu/blob_reader.hpp>

#include <sstream>
#include <memory>
#include <vector>
#include <string>

#include <ie_input_info.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data.hpp>

namespace vpu {

namespace {

template <typename T>
T readFromBlob(const std::vector<char>& blob, uint32_t& offset) {
    IE_ASSERT(offset + sizeof(T) <= blob.size());

    auto srcPtr = blob.data() + offset;
    offset += sizeof(T);

    return *reinterpret_cast<const T*>(srcPtr);
}

ie::Precision vpuDataTypeToIE(DataType dataType) {
    auto iePrecision = ie::Precision::UNSPECIFIED;

    switch (dataType) {
    case DataType::U8:
        iePrecision = ie::Precision::U8;
        break;
    case DataType::FP16:
        iePrecision = ie::Precision::FP16;
        break;
    case DataType::FP32:
        iePrecision = ie::Precision::FP32;
        break;
    default:
        VPU_THROW_EXCEPTION << "BlobReader error: unsupported dataType " << dataType;
    }

    return iePrecision;
}

ie::Layout vpuDimsOrderToIE(DimsOrder dimsOrder) {
    auto ieLayout = ie::Layout::ANY;

    if (DimsOrder::C == dimsOrder) {
        ieLayout = ie::Layout::C;
    } else if (DimsOrder::NC == dimsOrder) {
        ieLayout = ie::Layout::NC;
    } else if (DimsOrder::CHW == dimsOrder) {
        ieLayout = ie::Layout::CHW;
    } else if (DimsOrder::NCHW == dimsOrder) {
        ieLayout = ie::Layout::NCHW;
    } else if (DimsOrder::NHWC == dimsOrder) {
        ieLayout = ie::Layout::NHWC;
    } else {
        VPU_THROW_EXCEPTION << "BlobReader error: unsupported dimsOrder " << toString(dimsOrder);
    }

    return ieLayout;
}

ie::SizeVector vpuDimsToIE(const DimValues& dimValues) {
    auto order = DimsOrder::fromNumDims(dimValues.size());
    auto perm = order.toPermutation();

    ie::SizeVector ieDims(perm.size());
    for (int i = 0; i < perm.size(); ++i) {
        ieDims[ieDims.size() - 1 - i] = dimValues[perm[i]];
    }

    return ieDims;
}

}  // namespace

void BlobReader::parse(const std::vector<char>& blob) {
    if (blob.empty() || blob.size() < sizeof(ElfN_Ehdr) + sizeof(mv_blob_header)) {
        VPU_THROW_EXCEPTION << "BlobReader error: Blob is empty";
    }

    _pBlob = blob.data();

    _blobHeader = *reinterpret_cast<const mv_blob_header*>(blob.data() + sizeof(ElfN_Ehdr));
    if (_blobHeader.magic_number != BLOB_MAGIC_NUMBER) {
        VPU_THROW_EXCEPTION << "BlobReader error: The magic number imported blob doesn't match graph_transformer";
    }
    if (_blobHeader.blob_ver_major != BLOB_VERSION_MAJOR || _blobHeader.blob_ver_minor != BLOB_VERSION_MINOR) {
        VPU_THROW_EXCEPTION << "BlobReader error: The version of imported blob doesn't match graph_transformer";
    }

    _inputInfo.totalSize = _blobHeader.inputs_size;
    _outputInfo.totalSize = _blobHeader.outputs_size;

    auto inputInfoSecOffset = _blobHeader.input_info_section_offset;
    for (uint32_t i = 0; i < _blobHeader.inputs_count; i++) {
        auto ioIdx = readFromBlob<uint32_t>(blob, inputInfoSecOffset);
        IE_ASSERT(ioIdx == i);

        auto ioBufferOffset = readFromBlob<int32_t>(blob, inputInfoSecOffset);

        auto nameLength = readFromBlob<uint32_t>(blob, inputInfoSecOffset);
        std::string inputName(nameLength, 0);
        for (auto& c : inputName) {
            c = readFromBlob<char>(blob, inputInfoSecOffset);
        }

        // Truncate zeros
        inputName = inputName.c_str();

        auto dataType = static_cast<DataType>(readFromBlob<uint32_t>(blob, inputInfoSecOffset));
        auto orderCode = readFromBlob<uint32_t>(blob, inputInfoSecOffset);

        auto numDims = readFromBlob<uint32_t>(blob, inputInfoSecOffset);

        auto dimsOrder = DimsOrder::fromCode(orderCode);
        auto perm = dimsOrder.toPermutation();
        IE_ASSERT(perm.size() == numDims);

        DimValues vpuDims;
        for (int i = 0; i < perm.size(); ++i) {
            vpuDims.set(perm[i], readFromBlob<uint32_t>(blob, inputInfoSecOffset));
        }

        // Skip strides
        inputInfoSecOffset += perm.size() * sizeof(uint32_t);

        auto iePrecision = vpuDataTypeToIE(dataType);
        auto ieLayout    = vpuDimsOrderToIE(dimsOrder);
        auto ieDims = vpuDimsToIE(vpuDims);

        ie::TensorDesc ieDesc(iePrecision, ieDims, ieLayout);
        ie::Data inputData(inputName, ieDesc);

        ie::InputInfo input;
        input.setInputData(std::make_shared<ie::Data>(inputData));

        _networkInputs[input.name()]    = std::make_shared<ie::InputInfo>(input);
        _inputInfo.offset[input.name()] = ioBufferOffset;
    }

    auto outputInfoSecOffset = _blobHeader.output_info_section_offset;
    for (size_t i = 0; i < _blobHeader.outputs_count; i++) {
        auto ioIdx = readFromBlob<uint32_t>(blob, outputInfoSecOffset);
        IE_ASSERT(ioIdx == i);

        auto ioBufferOffset = readFromBlob<int32_t>(blob, outputInfoSecOffset);

        auto nameLength = readFromBlob<uint32_t>(blob, outputInfoSecOffset);
        std::string outputName(nameLength, 0);
        for (auto& c : outputName) {
            c = readFromBlob<char>(blob, outputInfoSecOffset);
        }

        // Truncate zeros
        outputName = outputName.c_str();

        auto dataType = static_cast<DataType>(readFromBlob<uint32_t>(blob, outputInfoSecOffset));
        auto orderCode = readFromBlob<uint32_t>(blob, outputInfoSecOffset);

        auto numDims = readFromBlob<uint32_t>(blob, outputInfoSecOffset);

        auto dimsOrder = DimsOrder::fromCode(orderCode);
        auto perm = dimsOrder.toPermutation();
        IE_ASSERT(perm.size() == numDims);

        DimValues vpuDims;
        for (int i = 0; i < perm.size(); ++i) {
            vpuDims.set(perm[i], readFromBlob<uint32_t>(blob, outputInfoSecOffset));
        }

        // Skip strides
        outputInfoSecOffset += perm.size() * sizeof(uint32_t);

        auto iePrecision = vpuDataTypeToIE(dataType);
        auto ieLayout    = vpuDimsOrderToIE(dimsOrder);
        auto ieDims = vpuDimsToIE(vpuDims);

        ie::TensorDesc ieDesc(iePrecision, ieDims, ieLayout);
        ie::Data outputData(outputName, ieDesc);

        _networkOutputs[outputData.getName()]    = std::make_shared<ie::Data>(outputData);
        _outputInfo.offset[outputData.getName()] = ioBufferOffset;
    }
}

}  // namespace vpu
