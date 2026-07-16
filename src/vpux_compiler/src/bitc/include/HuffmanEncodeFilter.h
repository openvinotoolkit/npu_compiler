//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "HuffmanCodebook.h"

namespace vpux::bitc::huffman {

// Huffman constants
constexpr size_t HF_HEADER_BIT_SIZE = 9u;            // Huffman encode/decode header bit size
constexpr size_t HF_BLOCK_BYTE_SIZE = 64u;           // Uncompressed Huffman block size in bytes
constexpr size_t HF_BLOCK_BIT_SIZE = 512u;           // Uncompressed Huffman block size in bits
constexpr size_t HF_HEADER_BIT_MASK = 0x1FFull;      // Huffman encode/decode 9-bit header mask
constexpr size_t HF_MIN_COMPRESSED_BIT_SIZE = 128u;  // Minimum compressed Huffman block bit size

struct HuffmanBlockInfo {
    uint32_t inBitStart;
    uint32_t inBitSize;
    uint32_t outBitStart;
    uint32_t outBitSize;
    bool copyBlock;
    bool lastBlock;
};

class HuffmanEncodeFilter {
public:
    bool ProcessData(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData);

private:
    uint32_t InsertCodebook(std::vector<uint8_t>& outData, const std::vector<HuffmanCode>& huffmanCodes);
    HuffmanBlockInfo ComputeBlockInfo(const std::vector<uint8_t>& inData, uint32_t block, uint32_t outBitStart,
                                      const std::vector<HuffmanCode>& huffmanCodes) const;
    void ProcessBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                      const HuffmanBlockInfo& blockInfo, uint32_t block, const std::vector<HuffmanCode>& huffmanCodes);
    void InsertHeader(std::vector<uint8_t>& outData, const HuffmanBlockInfo& blockInfo);
    void CopyBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData, const HuffmanBlockInfo& blockInfo,
                   uint32_t block);
    void EncodeBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                     const HuffmanBlockInfo& blockInfo, uint32_t block, const std::vector<HuffmanCode>& huffmanCodes);
    void InsertCode(std::vector<uint8_t>& outData, uint32_t outBitIndex, uint8_t key,
                    const std::vector<HuffmanCode>& huffmanCodes);

    std::vector<HuffmanCode> GenerateCodebook(const std::vector<uint8_t>& inData);
};

}  // namespace vpux::bitc::huffman
