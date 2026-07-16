//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpux::bitc::huffman {

// Huffman constants
constexpr size_t HF_CODEBOOK_KEYS = 16u;           // Number of Huffman codebook keys
constexpr size_t HF_MAX_CODEBOOK_BIT_SIZE = 135u;  // Maximum codebook bit size

class HuffmanCode {
public:
    uint16_t code;
    uint16_t length;
    uint8_t key;
    std::string GetCodeString() const;
};

class MinHeapNode {
public:
    MinHeapNode(uint8_t key, uint64_t frequency);
    uint8_t GetKey() const {
        return mKey;
    }
    uint64_t GetFrequency() const {
        return mFrequency;
    }
    std::unique_ptr<MinHeapNode>& GetLeft() {
        return mLeft;
    }
    std::unique_ptr<MinHeapNode>& GetRight() {
        return mRight;
    }
    void SetLeft(std::unique_ptr<MinHeapNode> left) {
        mLeft = std::move(left);
    }
    void SetRight(std::unique_ptr<MinHeapNode> right) {
        mRight = std::move(right);
    }
    bool IsLeaf() const {
        return (mLeft == nullptr) && (mRight == nullptr);
    }

private:
    uint8_t mKey{0};
    uint64_t mFrequency{0};
    std::unique_ptr<MinHeapNode> mLeft{nullptr};
    std::unique_ptr<MinHeapNode> mRight{nullptr};
};

class MinHeap {
public:
    MinHeap(const std::vector<uint64_t>& frequencies);
    std::unique_ptr<MinHeapNode> BuildHuffmanTree();

private:
    void BuildMinHeap();
    void MinHeapify(uint32_t index);
    std::unique_ptr<MinHeapNode> ExtractMin();
    void Insert(std::unique_ptr<MinHeapNode> node);
    std::vector<std::unique_ptr<MinHeapNode>> mNodes;
};

class HuffmanCodebook {
public:
    HuffmanCodebook(const std::vector<uint64_t>& frequencies);
    std::vector<HuffmanCode> GetHuffmanCodes() const {
        return mHuffmanCodes;
    }

private:
    void StoreCodes(const std::unique_ptr<MinHeapNode>& root, uint16_t code, uint16_t depth);
    std::vector<HuffmanCode> mHuffmanCodes;
};

}  // namespace vpux::bitc::huffman
