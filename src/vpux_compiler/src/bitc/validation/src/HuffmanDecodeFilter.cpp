//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "HuffmanDecodeFilter.h"
#include "commons.hpp"

#include <algorithm>

using namespace vpux::bitc;
using namespace vpux::bitc::huffman;

// Process the input data and decompress it
bool HuffmanDecodeFilter::ProcessData(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData) {
    bool success = true;
    mDecodeError = false;
    outData.clear();
    outData.reserve(inData.size());

    uint32_t inBitStart = 0;
    uint32_t numBlocks = 0;
    std::vector<HuffmanBlockInfo> blockInfo{};
    std::vector<HuffmanCode> huffmanCodes{};

    // Extract the Huffman codes from the input data
    inBitStart = ExtractCodebook(inData, huffmanCodes);

    if (inBitStart > 0) {
        // Extract the block information from the input data
        while ((BYTE_ALIGN(inBitStart) >> BYTE_SIZE_SHIFT) < inData.size()) {
            blockInfo.push_back(ExtractBlockInfo(inData, numBlocks, inBitStart));
            // Skip to next block
            inBitStart += blockInfo[numBlocks].inBitSize;
            numBlocks++;
        }
        if (numBlocks > 0) {
            blockInfo[numBlocks - 1].lastBlock = true;
        }

        // Process each block
        for (uint32_t block = 0; block < numBlocks && !mDecodeError; block++) {
            // Process each block
            ProcessBlock(inData, outData, blockInfo[block], block, huffmanCodes);
        }
    } else {
        // Error extracting codebook
        // - Decode error handling is a HW mechanism (not a failure)
        // - Any other errors should be reported as a failure.
        success = mDecodeError;
    }

    return success;
}

// Extract 64bit Huffman code lengths followed by actual codes from the start of
// the input data
uint32_t HuffmanDecodeFilter::ExtractCodebook(const std::vector<uint8_t>& inData,
                                              std::vector<HuffmanCode>& huffmanCodes) {
    huffmanCodes.resize(HF_CODEBOOK_KEYS);
    uint32_t byteIndex, bitOffset, codebookBitSize = 0;

    // Extract the Huffman code lengths from the first 64 bits of the input data
    for (uint32_t key = 0; key < HF_CODEBOOK_KEYS; key += 2) {
        byteIndex = key >> 1;  // 2 keys per byte
        if (byteIndex >= inData.size()) {
            return 0;
        }
        huffmanCodes[key].length = inData[byteIndex] & NIBBLE_MASK;
        huffmanCodes[key + 1].length = (inData[byteIndex] >> NIBBLE_BITS) & NIBBLE_MASK;

        codebookBitSize += huffmanCodes[key].length + huffmanCodes[key + 1].length;
    }
    // Check if the codebook bit size exceeds the maximum size of 135 bits (decode
    // error case)
    if (codebookBitSize > HF_MAX_CODEBOOK_BIT_SIZE) {
        mDecodeError = true;
        return 0;
    }
    uint32_t inBitIndex = HF_CODEBOOK_KEYS * NIBBLE_BITS;

    // Extract the Huffman codes from the input data
    for (uint32_t key = 0; key < HF_CODEBOOK_KEYS; key++) {
        // Check if the Huffman code length is valid (decode error case)
        if (huffmanCodes[key].length == 0) {
            mDecodeError = true;
            return 0;
        }
        // Store key
        huffmanCodes[key].key = key;

        // Extract code. It can span up to 3 bytes
        uint16_t mask = (1 << huffmanCodes[key].length) - 1;
        byteIndex = inBitIndex >> BYTE_SIZE_SHIFT;
        bitOffset = inBitIndex % BYTE_BITS;
        if (byteIndex + 2 >= inData.size()) {
            return 0;
        }
        huffmanCodes[key].code = (((inData[byteIndex + 2] << (2 * BYTE_BITS)) | (inData[byteIndex + 1] << BYTE_BITS) |
                                   inData[byteIndex]) >>
                                  bitOffset) &
                                 mask;

        // Move to next code
        inBitIndex += huffmanCodes[key].length;
    }

    // Sort the Huffman codes by length in ascending order.
    // This improves the performance of the Huffman decoding process.
    std::sort(huffmanCodes.begin(), huffmanCodes.end(), [](const HuffmanCode& a, const HuffmanCode& b) {
        return a.length < b.length;
    });

    return inBitIndex;
}

// Extract the header from the input data at the specified bit index and return
// the block size
HuffmanBlockInfo HuffmanDecodeFilter::ExtractBlockInfo(const std::vector<uint8_t>& inData, uint32_t block,
                                                       uint32_t inBitStart) {
    HuffmanBlockInfo blockInfo{0, 0, 0, 0, false, false};

    blockInfo.inBitStart = inBitStart;
    blockInfo.outBitStart = block * HF_BLOCK_BIT_SIZE;

    // Extract 9 bit header starting from inBitStart
    uint32_t byteIndex = inBitStart >> BYTE_SIZE_SHIFT;
    uint32_t bitOffset = inBitStart % BYTE_BITS;
    uint16_t header = (((inData[byteIndex + 1] << BYTE_BITS) | inData[byteIndex]) >> bitOffset) & HF_HEADER_BIT_MASK;

    blockInfo.copyBlock = (header == 0x0);
    if (!blockInfo.copyBlock) {
        // Encoded block. Block size after decide is 512 bits
        blockInfo.outBitSize = HF_BLOCK_BIT_SIZE;
        // Extract the encoded block size from the header
        blockInfo.inBitSize = header + HF_HEADER_BIT_SIZE;
    } else {
        // Copied block
        // Calculate remaining bits in the input data and check if it is less than
        // the block size
        uint32_t remainingBits = ((inData.size() << BYTE_SIZE_SHIFT) - HF_HEADER_BIT_SIZE - inBitStart);
        remainingBits &= ~(BYTE_BITS - 1);  // Actual remaining bits are byte aligned, discard any
                                            // remaining padding bits
        // Decoded block size is 512 bits or remaining bits
        blockInfo.outBitSize = HF_BLOCK_BIT_SIZE < remainingBits ? HF_BLOCK_BIT_SIZE : remainingBits;
        // Encoded block size is the same as the decoded block size + header size
        blockInfo.inBitSize = blockInfo.outBitSize + HF_HEADER_BIT_SIZE;
    }

    return blockInfo;
}

// Process the input data block and decode it
void HuffmanDecodeFilter::ProcessBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                       const HuffmanBlockInfo& blockInfo, uint32_t block,
                                       const std::vector<HuffmanCode>& huffmanCodes) {
    if (!blockInfo.copyBlock) {
        // Check if the bit size indicated by in the header is valid
        // A compressed non-final block can't be less than 128 bits (decode error
        // case)
        uint32_t inBlockSize = blockInfo.inBitSize - HF_HEADER_BIT_SIZE;
        if (!blockInfo.lastBlock && (inBlockSize < HF_MIN_COMPRESSED_BIT_SIZE)) {
            mDecodeError = true;
        } else {
            DecodeBlock(inData, outData, blockInfo, block, huffmanCodes);
        }
    } else {
        CopyBlock(inData, outData, blockInfo, block);
    }
}

// Copy the input data block bits as is
void HuffmanDecodeFilter::CopyBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                    const HuffmanBlockInfo& blockInfo, uint32_t block) {
    // Skip the header bits
    uint32_t inBitStart = blockInfo.inBitStart + HF_HEADER_BIT_SIZE;
    uint32_t inBitSize = blockInfo.inBitSize - HF_HEADER_BIT_SIZE;
    uint32_t inBitOffset = inBitStart % BYTE_BITS;
    uint32_t inByteIndex = inBitStart >> BYTE_SIZE_SHIFT;
    uint32_t outBlockStart = block * HF_BLOCK_BYTE_SIZE;
    uint32_t outBlockEnd = outBlockStart + (BYTE_ALIGN(inBitSize) >> BYTE_SIZE_SHIFT);
    outData.resize(outBlockEnd);

    if (inBitOffset % BYTE_BITS == 0) {
        // Input data is byte aligned.
        uint32_t inBlockEnd = inByteIndex + (outBlockEnd - outBlockStart);
        std::copy(inData.begin() + inByteIndex, inData.begin() + inBlockEnd, outData.begin() + outBlockStart);
    } else {
        // Input data is not byte aligned. Each byte has to be shifted with
        // inBitOffset
        for (uint32_t byteIndex = outBlockStart; (byteIndex < outBlockEnd) && (byteIndex < outData.size());
             byteIndex++) {
            outData[byteIndex] =
                    (((inData[inByteIndex + 1] << BYTE_BITS) | inData[inByteIndex]) >> inBitOffset) & BYTE_MASK;
            inByteIndex++;
        }
    }
}

// Decode the input data block using the Huffman codes
void HuffmanDecodeFilter::DecodeBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                      const HuffmanBlockInfo& blockInfo, uint32_t block,
                                      const std::vector<HuffmanCode>& huffmanCodes) {
    // Skip the header bits
    uint32_t inBitStart = blockInfo.inBitStart + HF_HEADER_BIT_SIZE;
    uint32_t inBitSize = blockInfo.inBitSize - HF_HEADER_BIT_SIZE;

    uint32_t inBitIndex = 0;
    uint32_t outBitIndex = blockInfo.outBitStart;
    while (inBitIndex < inBitSize) {
        bool match = false;

        // Iterate through each Huffman code
        for (const auto& huffmanCode : huffmanCodes) {
            // Check if there are enough bits left in the input data for the current
            // Huffman code
            if (inBitIndex + huffmanCode.length <= inBitSize) {
                // Extract the code from the input data. It can span up to 3 bytes
                uint32_t byteIndex = (inBitStart + inBitIndex) / 8;
                uint32_t bitOffset = (inBitStart + inBitIndex) % 8;
                uint16_t mask = (1 << huffmanCode.length) - 1;
                uint16_t code = (((inData[byteIndex + 2] << (2 * BYTE_BITS)) | (inData[byteIndex + 1] << BYTE_BITS) |
                                  inData[byteIndex]) >>
                                 bitOffset) &
                                mask;

                // Check if the extracted code matches the current Huffman code
                if (code == huffmanCode.code) {
                    // Insert the decoded key into the output data
                    outBitIndex = InsertKey(huffmanCode.key, outData, outBitIndex);

                    // Update the input bit index by the length of the Huffman code
                    inBitIndex += huffmanCode.length;

                    // Set match flag to true and break the loop
                    match = true;
                    break;
                }
            }
        }
        if (!match) {
            outData.resize(inData.size());
            mDecodeError = true;
            return;
        }
    }
}

// Insert the decoded key into the output data at the specified bit index
uint32_t HuffmanDecodeFilter::InsertKey(uint8_t key, std::vector<uint8_t>& outData, uint32_t outBitIndex) {
    uint32_t outByteIndex = outBitIndex >> BYTE_SIZE_SHIFT;
    if (outBitIndex % 8 == 0) {
        // Insert the key into the lower nibble of a new byte
        outData.resize(outData.size() + 1);
        outData[outByteIndex] = (NIBBLE_MASK & key);
    } else {
        // Insert the key shifted to the higher nibble of the current byte
        outData[outByteIndex] |= (NIBBLE_MASK & key) << NIBBLE_BITS;
    }

    // Return the updated bit index, incremented by 4 bits (size of the key)
    return outBitIndex + NIBBLE_BITS;
}

// Get the decode error status
bool HuffmanDecodeFilter::GetDecodeError() const {
    return mDecodeError;
}
