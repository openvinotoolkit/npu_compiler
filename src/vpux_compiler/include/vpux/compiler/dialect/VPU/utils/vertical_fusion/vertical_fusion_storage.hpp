//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/dense_map.hpp"

namespace vpux {
namespace VPU {

/*
 Container for storage connection between object and information about it
 separated by VF tile

 VFKey - object (ex. block argument, operation)
 VFValue - additional information about object (ex. TileInfo)

 Compare - comparator for VFValue, storage keeps max element of info
*/
template <class VFKey, class VFValue, class Compare = std::less<VFValue>>
class VFContainer {
public:
    // constructor
    VFContainer() = default;

    // constructor with reserved size for each inner container
    VFContainer(ArrayRef<VFKey> vfKeys, int64_t reservedSize) {
        for (const auto& vfKey : vfKeys) {
            vfContainer[vfKey].reserve(reservedSize);
        }
    }

    // connection between number of tile and info
    using VFTileContainer = std::unordered_map<size_t, VFValue>;

    // pointer to container
    using UPtr = std::unique_ptr<VFContainer<VFKey, VFValue, Compare>>;

    // merge two containers, in case both containers have info
    // for same tile, max element is chosen based on comparator
    void merge(const VFContainer<VFKey, VFValue, Compare>& src);

    // insert new element in container, in case there is already
    // info for object and tile, max element is chosen based on comparator
    void insert(VFKey key, size_t tile, const VFValue& src);

    // get information about object for exact tile
    std::optional<VFValue> get(VFKey key, size_t tile);

    // function returns information gathered together for all tiles
    const VFTileContainer& gatherValue(VFKey key);

    // get whole inner container
    const std::unordered_map<VFKey, VFTileContainer>& getAll() const {
        return vfContainer;
    };

private:
    // inner container for storage connection
    std::unordered_map<VFKey, VFTileContainer> vfContainer;

    // comparator for elements of info
    Compare vfComparator;
};

template <class VFKey, class VFValue, class Compare>
void vpux::VPU::VFContainer<VFKey, VFValue, Compare>::merge(const VFContainer<VFKey, VFValue, Compare>& src) {
    for (auto& item : src.getAll()) {
        if (vfContainer.count(item.first) == 0) {
            vfContainer[item.first] = item.second;
        } else {
            for (auto& tileItem : item.second) {
                insert(item.first, tileItem.first, tileItem.second);
            }
        }
    }
}

template <class VFKey, class VFValue, class Compare>
void vpux::VPU::VFContainer<VFKey, VFValue, Compare>::insert(VFKey key, size_t tile, const VFValue& src) {
    auto& tileContainer = vfContainer[key];
    auto foundTileItem = tileContainer.find(tile);
    if (foundTileItem == tileContainer.end()) {
        tileContainer.insert({tile, src});
    } else {
        foundTileItem->second = std::max(foundTileItem->second, src, vfComparator);
    }
}

template <class VFKey, class VFValue, class Compare>
std::optional<VFValue> vpux::VPU::VFContainer<VFKey, VFValue, Compare>::get(VFKey key, size_t tile) {
    auto foundItem = vfContainer.find(key);

    if (foundItem == vfContainer.end()) {
        return std::nullopt;
    }

    auto foundTile = foundItem->second.find(tile);

    if (foundTile == foundItem->second.end()) {
        return std::nullopt;
    }

    return foundTile->second;
}

template <class VFKey, class VFValue, class Compare>
const typename vpux::VPU::VFContainer<VFKey, VFValue, Compare>::VFTileContainer&
vpux::VPU::VFContainer<VFKey, VFValue, Compare>::gatherValue(VFKey key) {
    return vfContainer[key];
}

}  // namespace VPU
}  // namespace vpux
