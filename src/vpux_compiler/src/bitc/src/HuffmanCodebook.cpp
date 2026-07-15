//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "HuffmanCodebook.h"
#include <algorithm>

using namespace vpux::bitc::huffman;

// Class HuffmanCode
std::string HuffmanCode::GetCodeString() const {
    char codeString[] = "                ";
    for (uint32_t i = 0; i < length; i++) {
        codeString[HF_CODEBOOK_KEYS - 2 - i] = (code & (1 << (length - i - 1))) ? '1' : '0';
    }
    return std::string(codeString);
}

// Class MinHeapNode
MinHeapNode::MinHeapNode(uint8_t key, uint64_t frequency) {
    mKey = key;
    mFrequency = frequency;
    mRight = nullptr;
    mLeft = nullptr;
}

// Class MinHeap
// The value of each node, or child, is greater than or equal to the value of its parent, with the minimum value at the
// root node)
MinHeap::MinHeap(const std::vector<uint64_t>& frequencies) {
    mNodes.resize(frequencies.size());

    // Create a vector of pairs to store frequency and corresponding key, then sort it in descending order of frequency
    std::vector<std::pair<uint64_t, uint8_t>> freqKeyPairs;
    for (uint8_t key = 0; key < HF_CODEBOOK_KEYS; key++) {
        freqKeyPairs.emplace_back(frequencies[key], key);
    }
    std::sort(freqKeyPairs.begin(), freqKeyPairs.end(), std::greater<>());

    // Create a MinHeapNode for each key
    for (uint32_t i = 0; i < mNodes.size(); i++) {
        mNodes[i] = std::make_unique<MinHeapNode>(freqKeyPairs[i].second, freqKeyPairs[i].first);
    }
    BuildMinHeap();
}

// Restore the min heap property
void MinHeap::MinHeapify(uint32_t index) {
    uint32_t smallest = index;
    uint32_t left = 2 * index + 1;
    uint32_t right = 2 * index + 2;

    // Check if the left child exists and is smaller than the current smallest
    if ((left < mNodes.size()) && (mNodes[left]->GetFrequency() < mNodes[smallest]->GetFrequency())) {
        smallest = left;
    }

    // Check if the right child exists and is smaller than the current smallest
    if ((right < mNodes.size() && mNodes[right]->GetFrequency() < mNodes[smallest]->GetFrequency())) {
        smallest = right;
    }

    // If the smallest is not the current index, swap and continue heapifying
    if (smallest != index) {
        std::swap(mNodes[index], mNodes[smallest]);
        MinHeapify(smallest);
    }
}

// Extract the node with the minimum frequency from the min heap
std::unique_ptr<MinHeapNode> MinHeap::ExtractMin() {
    // Move the minimum node (root of the heap) to a unique pointer
    std::unique_ptr<MinHeapNode> min = std::move(mNodes[0]);

    // Replace the root node with the last node
    mNodes[0] = std::move(mNodes.back());
    mNodes.pop_back();

    // Restore the min heap property
    MinHeapify(0);

    // Return the minimum node
    return min;
}

// Insert a new node into the min heap
void MinHeap::Insert(std::unique_ptr<MinHeapNode> node) {
    // Add the new node to the end of the vector
    mNodes.push_back(std::move(node));
    uint32_t i = mNodes.size() - 1;

    // Fix the min heap property if it is violated
    while (i > 0 && mNodes[(i - 1) / 2]->GetFrequency() > mNodes[i]->GetFrequency()) {
        // Swap the node with its parent
        std::swap(mNodes[i], mNodes[(i - 1) / 2]);
        // Move to the parent index
        i = (i - 1) / 2;
    }
}

// Build the min heap from the current nodes
void MinHeap::BuildMinHeap() {
    uint32_t n = mNodes.size() - 1;
    // Perform reverse level order traversal from the last non-leaf node to the root node
    for (int32_t i = (n - 1) / 2; i >= 0; i--) {
        // Restore the min heap property for each node
        MinHeapify(i);
    }
}

// Build the Huffman tree from the min heap
std::unique_ptr<MinHeapNode> MinHeap::BuildHuffmanTree() {
    while (mNodes.size() > 1) {
        // Extract the two nodes with the minimum frequency
        std::unique_ptr<MinHeapNode> left = ExtractMin();
        std::unique_ptr<MinHeapNode> right = ExtractMin();
        // Create a new node with the sum of the frequencies of the two nodes
        std::unique_ptr<MinHeapNode> parent =
                std::make_unique<MinHeapNode>(0xFF, left->GetFrequency() + right->GetFrequency());
        // Set the left and right children of the new node
        parent->SetLeft(std::move(left));
        parent->SetRight(std::move(right));
        // Insert the new node into the MinHeap
        Insert(std::move(parent));
    }
    // Return the root of the Huffman tree
    return std::move(mNodes[0]);
}
// Class HuffmanCodebook
HuffmanCodebook::HuffmanCodebook(const std::vector<uint64_t>& frequencies) {
    mHuffmanCodes.resize(HF_CODEBOOK_KEYS);

    // Build the Huffman tree from the min heap and get the root
    MinHeap minHeap(frequencies);
    std::unique_ptr<MinHeapNode> root = minHeap.BuildHuffmanTree();

    // Traverse Huffman tree and store resulting codes in mHuffmanCodes
    StoreCodes(root, 0, 0);
}

// Store the Huffman codes for each key in the codebook
void HuffmanCodebook::StoreCodes(const std::unique_ptr<MinHeapNode>& root, uint16_t code, uint16_t depth) {
    if (root->IsLeaf()) {
        // Store the code and length for the leaf node
        mHuffmanCodes[root->GetKey()].code = code;
        mHuffmanCodes[root->GetKey()].length = depth;
        mHuffmanCodes[root->GetKey()].key = root->GetKey();
    } else {
        // Add extra 0 bit to the left child
        StoreCodes(root->GetLeft(), code, depth + 1);
        // Add extra 1 bit to the right child
        StoreCodes(root->GetRight(), (code | 1 << depth), depth + 1);
    }
}
