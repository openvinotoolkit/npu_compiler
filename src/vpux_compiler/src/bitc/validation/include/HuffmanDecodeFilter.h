//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "HuffmanCodebook.h"
#include "HuffmanEncodeFilter.h"

namespace vpux::bitc::huffman {

class HuffmanDecodeFilter {
public:
    bool ProcessData(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData);
    bool GetDecodeError() const;

private:
    uint32_t ExtractCodebook(const std::vector<uint8_t>& inData, std::vector<HuffmanCode>& huffmanCodes);
    HuffmanBlockInfo ExtractBlockInfo(const std::vector<uint8_t>& inData, uint32_t block, uint32_t inBitIndex);
    void ProcessBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                      const HuffmanBlockInfo& blockInfo, uint32_t block, const std::vector<HuffmanCode>& huffmanCodes);
    void CopyBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData, const HuffmanBlockInfo& blockInfo,
                   uint32_t block);
    void DecodeBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                     const HuffmanBlockInfo& blockInfo, uint32_t block, const std::vector<HuffmanCode>& huffmanCodes);
    uint32_t InsertKey(uint8_t key, std::vector<uint8_t>& outData, uint32_t outBitIndex);

    bool mDecodeError{false};
};

}  // namespace vpux::bitc::huffman
