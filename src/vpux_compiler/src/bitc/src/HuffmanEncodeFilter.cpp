//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "HuffmanEncodeFilter.h"
#include "HuffmanCodebook.h"
#include "commons.hpp"

using namespace vpux::bitc;
using namespace vpux::bitc::huffman;

bool HuffmanEncodeFilter::ProcessData(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData) {
    bool success = true;
    uint32_t totalOutBitSize = 0;
    outData.clear();

    // Generate the Huffman codebook
    std::vector<HuffmanCode> huffmanCodes = GenerateCodebook(inData);

    // Insert codebook
    totalOutBitSize = InsertCodebook(outData, huffmanCodes);

    // Compute the block information for each block and calculate the total output bit size
    uint32_t numBlocks = (inData.size() + HF_BLOCK_BYTE_SIZE - 1) / HF_BLOCK_BYTE_SIZE;
    std::vector<HuffmanBlockInfo> blockInfo(numBlocks);
    if (numBlocks > 0) {
        blockInfo[numBlocks - 1].lastBlock = true;
    }
    for (uint32_t block = 0; block < numBlocks; block++) {
        blockInfo[block] = ComputeBlockInfo(inData, block, totalOutBitSize, huffmanCodes);
        totalOutBitSize += blockInfo[block].outBitSize;
    }
    outData.resize(BYTE_ALIGN(totalOutBitSize) >> BYTE_SIZE_SHIFT);

    // Process each block
    for (uint32_t block = 0; block < numBlocks; block++) {
        ProcessBlock(inData, outData, blockInfo[block], block, huffmanCodes);
    }

    return success;
}

// Insert Huffman code lengths and the codes into the output data
uint32_t HuffmanEncodeFilter::InsertCodebook(std::vector<uint8_t>& outData,
                                             const std::vector<HuffmanCode>& huffmanCodes) {
    outData.clear();
    outData.resize(2 * (BYTE_ALIGN(HF_MAX_CODEBOOK_BIT_SIZE) >> BYTE_SIZE_SHIFT));

    // Insert 4bit code lengths (64 bits total)
    for (uint32_t key = 0; key < HF_CODEBOOK_KEYS; key += 2) {
        uint16_t index = key >> 1;  // 2 keys per byte
        outData[index] = ((huffmanCodes[key + 1].length << NIBBLE_BITS) & 0xF0) | (huffmanCodes[key].length & 0x0F);
    }
    uint32_t outBitIndex = HF_CODEBOOK_KEYS * NIBBLE_BITS;

    // Insert Huffman codes
    for (uint32_t key = 0; key < HF_CODEBOOK_KEYS; key++) {
        InsertCode(outData, outBitIndex, key, huffmanCodes);
        outBitIndex += huffmanCodes[key].length;
    }
    outData.resize(BYTE_ALIGN(outBitIndex) >> BYTE_SIZE_SHIFT);

    return outBitIndex;
}

// Compute the block information for the given data block
HuffmanBlockInfo HuffmanEncodeFilter::ComputeBlockInfo(const std::vector<uint8_t>& inData, uint32_t block,
                                                       uint32_t outBitStart,
                                                       const std::vector<HuffmanCode>& huffmanCodes) const {
    HuffmanBlockInfo blockInfo{0, 0, 0, 0, false, false};
    blockInfo.inBitStart = block * HF_BLOCK_BIT_SIZE;
    blockInfo.outBitStart = outBitStart;

    uint32_t remainingBits = (inData.size() << BYTE_SIZE_SHIFT) - blockInfo.inBitStart;

    // Compute the input size for the block
    uint32_t inByteSize;
    if (HF_BLOCK_BIT_SIZE < remainingBits) {
        // Full block
        blockInfo.inBitSize = HF_BLOCK_BIT_SIZE;
        inByteSize = HF_BLOCK_BYTE_SIZE;
    } else {
        // Partial last block
        blockInfo.inBitSize = remainingBits;
        inByteSize = BYTE_ALIGN(blockInfo.inBitSize) >> BYTE_SIZE_SHIFT;
    }

    // Compute the encoded bit size for the block
    // Iterate over the block data and add the bit size of the Huffman codes
    uint32_t encodedBitSize = 0;
    for (uint32_t offset = 0; offset < inByteSize; offset++) {
        uint32_t byteIndex = block * HF_BLOCK_BYTE_SIZE + offset;

        // Extract the upper and lower 4bit keys from the input data
        uint8_t upperKey = (inData[byteIndex] >> NIBBLE_BITS) & 0xF;
        uint8_t lowerKey = inData[byteIndex] & 0xF;
        encodedBitSize += huffmanCodes[upperKey].length + huffmanCodes[lowerKey].length;
    }

    // Check if block data should be copied as is. Add header size to the output bit size.
    if (encodedBitSize >= blockInfo.inBitSize) {
        blockInfo.copyBlock = true;
        blockInfo.outBitSize = blockInfo.inBitSize + HF_HEADER_BIT_SIZE;
    } else {
        blockInfo.copyBlock = false;
        blockInfo.outBitSize = encodedBitSize + HF_HEADER_BIT_SIZE;
    }

    return blockInfo;
}

// Process the input data block and encode it
void HuffmanEncodeFilter::ProcessBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                       const HuffmanBlockInfo& blockInfo, uint32_t block,
                                       const std::vector<HuffmanCode>& huffmanCodes) {
    // Insert the 9-bit header into the output buffer
    InsertHeader(outData, blockInfo);

    if (!blockInfo.copyBlock) {
        // Encode data block using the Huffman codes
        EncodeBlock(inData, outData, blockInfo, block, huffmanCodes);
    } else {
        // Copy the input data block as is
        CopyBlock(inData, outData, blockInfo, block);
    }
}

// Insert the 9-bit header into the output data at the specified bit index
void HuffmanEncodeFilter::InsertHeader(std::vector<uint8_t>& outData, const HuffmanBlockInfo& blockInfo) {
    uint32_t byteIndex = blockInfo.outBitStart >> BYTE_SIZE_SHIFT;
    uint32_t bitOffset = blockInfo.outBitStart % BYTE_BITS;

    uint16_t header = 0x0;                          // Header value for copy block
    uint16_t mask = (1 << HF_HEADER_BIT_SIZE) - 1;  // 9-bit mask
    if (!blockInfo.copyBlock) {
        // Compute header value as compressed size
        uint32_t compressedSize = blockInfo.outBitSize - HF_HEADER_BIT_SIZE;
        header = compressedSize & mask;
    }
    uint16_t shiftedHeader = header << bitOffset;
    uint16_t shiftedMask = mask << bitOffset;

    // Clear the bits where the header will be inserted
    outData[byteIndex + 1] &= ~(shiftedMask >> BYTE_BITS);
    outData[byteIndex] &= ~(shiftedMask & BYTE_MASK);

    // Insert the header bits
    outData[byteIndex + 1] |= (shiftedHeader >> BYTE_BITS);
    outData[byteIndex] |= (shiftedHeader & BYTE_MASK);
}

// Copy the input data block bits as is
void HuffmanEncodeFilter::CopyBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                    const HuffmanBlockInfo& blockInfo, uint32_t block) {
    const uint32_t inBlockStart = block * HF_BLOCK_BYTE_SIZE;
    const uint32_t inBlockEnd = inBlockStart + (BYTE_ALIGN(blockInfo.inBitSize) >> BYTE_SIZE_SHIFT);

    uint32_t outBitIndex = blockInfo.outBitStart + HF_HEADER_BIT_SIZE;
    uint32_t bitOffset = outBitIndex % BYTE_BITS;
    uint32_t outByteIndex = outBitIndex >> BYTE_SIZE_SHIFT;

    if (outBitIndex % BYTE_BITS == 0) {
        // Output data is byte aligned
        std::copy(inData.begin() + inBlockStart, inData.begin() + inBlockEnd, outData.begin() + outByteIndex);
    } else {
        // Output data is not byte aligned. Each byte has to be shifted with bitOffset
        for (uint32_t byteIndex = inBlockStart; (byteIndex < inBlockEnd); byteIndex++) {
            uint16_t shiftedData = inData[byteIndex] << bitOffset;
            uint16_t shiftedMask = BYTE_MASK << bitOffset;

            // Clear the bits where the data will be inserted
            outData[outByteIndex + 1] &= ~(shiftedMask >> BYTE_BITS);
            outData[outByteIndex] &= ~(shiftedMask & BYTE_MASK);

            // Insert the data bits
            outData[outByteIndex + 1] |= (shiftedData >> BYTE_BITS);
            outData[outByteIndex] |= (shiftedData & BYTE_MASK);
            outByteIndex++;
        }
    }
}

// Encode the input data using the Huffman codes
void HuffmanEncodeFilter::EncodeBlock(const std::vector<uint8_t>& inData, std::vector<uint8_t>& outData,
                                      const HuffmanBlockInfo& blockInfo, uint32_t block,
                                      const std::vector<HuffmanCode>& huffmanCodes) {
    const uint32_t inBlockStart = block * HF_BLOCK_BYTE_SIZE;
    const uint32_t inBlockEnd = inBlockStart + (BYTE_ALIGN(blockInfo.inBitSize) >> BYTE_SIZE_SHIFT);

    uint32_t outBitIndex = blockInfo.outBitStart + HF_HEADER_BIT_SIZE;

    for (uint32_t byteIndex = inBlockStart; (byteIndex < inBlockEnd) && (byteIndex < inData.size()); byteIndex++) {
        // Extract the upper and lower 4bit keys from the input data
        uint8_t upperKey = (inData[byteIndex] >> NIBBLE_BITS) & 0xF;
        uint8_t lowerKey = inData[byteIndex] & 0xF;

        // Insert the Huffman codes for the upper and lower keys
        InsertCode(outData, outBitIndex, lowerKey, huffmanCodes);
        outBitIndex += huffmanCodes[lowerKey].length;

        InsertCode(outData, outBitIndex, upperKey, huffmanCodes);
        outBitIndex += huffmanCodes[upperKey].length;
    }
}

// Insert the Huffman code for the given key into the output data at the specified bit index
void HuffmanEncodeFilter::InsertCode(std::vector<uint8_t>& outData, uint32_t outBitIndex, uint8_t key,
                                     const std::vector<HuffmanCode>& huffmanCodes) {
    // Calculate the byte index and bit offset within the byte
    uint32_t byteIndex = outBitIndex >> BYTE_SIZE_SHIFT;
    uint32_t bitOffset = outBitIndex % BYTE_BITS;

    // Retrieve the Huffman code and its length for the given key
    uint16_t code = huffmanCodes[key].code;
    uint16_t length = huffmanCodes[key].length;
    uint16_t mask = (1 << length) - 1;

    // Shift the code and mask to align with the bit offset. Some cases my require up to 3 bytes
    uint32_t shiftedCode = code << bitOffset;
    uint32_t shiftedMask = mask << bitOffset;

    // Insert the Huffman code into the output data
    uint32_t numBytes = 0;
    while (((numBytes * BYTE_BITS) < (bitOffset + length)) && ((byteIndex + numBytes) < outData.size())) {
        // Clear the bits where the Huffman code will be inserted
        outData[byteIndex + numBytes] &= ~((shiftedMask >> numBytes * BYTE_BITS) & BYTE_MASK);
        // Insert the Huffman code bits
        outData[byteIndex + numBytes] |= ((shiftedCode >> numBytes * BYTE_BITS) & BYTE_MASK);
        numBytes++;
    }
}

std::vector<HuffmanCode> HuffmanEncodeFilter::GenerateCodebook(const std::vector<uint8_t>& inData) {
    std::vector<uint64_t> frequencies(HF_CODEBOOK_KEYS, 0);

    // Calculate the frequency of each 4bit weight value (key) in the input data
    for (uint8_t byte : inData) {
        uint8_t upperKey = (byte >> NIBBLE_BITS) & 0xF;
        uint8_t lowerKey = byte & 0xF;
        frequencies[upperKey]++;
        frequencies[lowerKey]++;
    }

    // Generate the Huffman codes for the frequency values
    HuffmanCodebook huffmanCodebook(frequencies);
    std::vector<HuffmanCode> huffmanCodes = huffmanCodebook.GetHuffmanCodes();

    return huffmanCodes;
}
