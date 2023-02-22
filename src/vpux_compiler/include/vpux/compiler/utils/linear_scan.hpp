//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Linear scan implementation.
// Manages allocation of the contiguous memory area amongst live ranges.
//
// Required methods in Handle class:
//
//   * bool isAlive(LiveRange) const;
//   * bool isFixedAlloc(LiveRange) const;
//   * AddressType getSize(LiveRange) const;
//   * AddressType getAlignment(LiveRange) const;
//   * AddressType getAddress(LiveRange) const;
//   * void allocated(LiveRange, AddressType) const;
//   * void freed(LiveRange) const;
//   * <type> getSpillWeight(LiveRange) const;
//   * bool spilled(LiveRange) const;
//

#pragma once

#include "vpux/compiler/utils/partitioner.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cassert>
#include <cstdint>

namespace vpux {

template <class LiveRange, class Handler>
class LinearScan final {
public:
    using Direction = Partitioner::Direction;
    using LiveRangeVector = SmallVector<LiveRange>;
    using LiveRangeIter = typename LiveRangeVector::iterator;
    using AllocatedAddrs = std::unordered_map<const LiveRange*, std::pair<AddressType, AddressType>>;
    using ReservedAddressAndSizeVector = SmallVector<std::pair<vpux::AddressType, vpux::AddressType>>;

public:
    explicit LinearScan(AddressType size): _par{size} {
    }
    template <typename... Args>
    explicit LinearScan(AddressType size, const ReservedAddressAndSizeVector& reservedVec, Args&&... args)
            : _par{size}, _handler{std::forward<Args>(args)...} {
        for (const auto& addressAndSize : reservedVec) {
            _par.allocFixed(addressAndSize.first, addressAndSize.second);
        }
    }

public:
    void freeNonAlive() {
        for (auto it = _liveRanges.begin(); it != _liveRanges.end();) {
            const auto& range = *it;

            if (!_handler.isAlive(range)) {
                it = freeLiveRange(it);
            } else {
                ++it;
            }
        }
    }

    //
    // Introduces new live ranges into scope and allocates memory for them.
    // Deprecates live ranges which became not alive.
    // Spills live ranges if there is not enough space for introducing new ones.
    // Returns true if can allocate memory for all passed ranges, false
    // otherwise.
    //
    template <class LiveRanges>
    bool alloc(const LiveRanges& newLiveRanges, bool allowSpills = true, Direction dir = Direction::Up) {
        AllocatedAddrs allocatedAddrs;

        //
        // First try to allocate fixed live ranges.
        //

        for (auto curIt = newLiveRanges.begin(); curIt != newLiveRanges.end(); ++curIt) {
            const auto& newRange = *curIt;

            if (!_handler.isFixedAlloc(newRange)) {
                continue;
            }

            if (!allocFixedRange(newRange, allowSpills, allocatedAddrs)) {
                return false;
            }
        }

        //
        // Find room for new non-fixed ranges.
        //

        for (auto curIt = newLiveRanges.begin(); curIt != newLiveRanges.end(); ++curIt) {
            const auto& newRange = *curIt;

            if (_handler.isFixedAlloc(newRange)) {
                continue;
            }

            if (!allocPlainRange(newRange, allowSpills, dir, allocatedAddrs)) {
                return false;
            }
        }

        //
        // All requested live ranges are allocated now.
        //

        for (auto curIt = newLiveRanges.begin(); curIt != newLiveRanges.end(); ++curIt) {
            const auto& newRange = *curIt;

            const auto addr = allocatedAddrs.at(&newRange).first;
            assert(addr != InvalidAddress);

            _handler.allocated(newRange, addr);
            _liveRanges.push_back(newRange);
        }

        return true;
    }

    template <class LiveRanges>
    bool canAlloc(const LiveRanges& newLiveRanges, Direction dir = Direction::Up) {
        auto gapCountBefore = _par.gaps().size();
        bool canAllocAll = true;
        SmallVector<std::pair<vpux::AddressType, vpux::AddressType>> tempAlloc;
        // temp allocation
        for (const auto& newRange : newLiveRanges) {
            const auto newRangeSize = _handler.getSize(newRange);
            const auto newRangeAlignment = _handler.getAlignment(newRange);
            auto allocAddr = _par.alloc(newRangeSize, newRangeAlignment, dir);
            if (allocAddr == InvalidAddress) {
                canAllocAll = false;
                break;
            }
            auto allocPair = std::make_pair(allocAddr, newRangeSize);
            tempAlloc.push_back(allocPair);
            // E#30278 case for exceeding NNCMX invariant with CMX-Concat
            if (_handler.checkInvariantExceedingNNCMX(newRange, allocAddr, _par.totalSize())) {
                canAllocAll = false;
                break;
            }
        }
        // deallocation
        for (auto curIt = tempAlloc.begin(); curIt != tempAlloc.end(); ++curIt) {
            vpux::AddressType address = curIt->first;
            vpux::AddressType size = curIt->second;
            _par.free(address, size);
        }
        VPUX_THROW_UNLESS(gapCountBefore == _par.gaps().size(), "Error new gaps created");
        return canAllocAll;
    }

    bool alloc(std::initializer_list<LiveRange> newLiveRanges, bool allowSpills = true, Direction dir = Direction::Up) {
        return this->alloc<std::initializer_list<LiveRange>>(newLiveRanges, allowSpills, dir);
    }

    bool canAlloc(std::initializer_list<LiveRange> newLiveRanges, Direction dir = Direction::Up) {
        return this->canAlloc<std::initializer_list<LiveRange>>(newLiveRanges, dir);
    }

public:
    const auto& liveRanges() const {
        return _liveRanges;
    }

    const auto& handler() const {
        return _handler;
    }
    auto& handler() {
        return _handler;
    }

    AddressType totalSize() const {
        return _par.totalSize();
    }

    AddressType totalFreeSize() const {
        return _par.totalFreeSize();
    }

    AddressType maxFreeSize() const {
        return _par.maxFreeSize();
    }

    const auto& gaps() const {
        return _par.gaps();
    }

private:
    bool allocFixedRange(const LiveRange& newRange, bool allowSpills, AllocatedAddrs& allocatedAddrs) {
        assert(_handler.isFixedAlloc(newRange));

        const auto newRangeAddr = _handler.getAddress(newRange);
        const auto newRangeSize = _handler.getSize(newRange);
        assert(newRangeAddr != InvalidAddress);

        const auto canAlloc = [this, newRangeAddr, newRangeSize]() {
            for (const auto& prevRange : _liveRanges) {
                const auto prevRangeAddr = _handler.getAddress(prevRange);
                const auto prevRangeSize = _handler.getSize(prevRange);

                if (Partitioner::intersects(newRangeAddr, newRangeSize, prevRangeAddr, prevRangeSize)) {
                    return false;
                }
            }

            return true;
        };

        LiveRangeVector spillCandidates;

        for (;;) {
            if (canAlloc()) {
                break;
            }

            if (!allowSpills) {
                rollback(allocatedAddrs);
                return false;
            }

            const auto spillCandidate = getSpillCandidate();

            if (!spillCandidate.hasValue()) {
                rollback(allocatedAddrs);
                return false;
            }

            spillCandidates.push_back(spillCandidate.getValue());
        }

        if (!performActualSpill(spillCandidates, newRangeAddr, newRangeSize)) {
            rollback(allocatedAddrs);
            return false;
        }

        _par.allocFixed(newRangeAddr, newRangeSize);
        allocatedAddrs.insert({&newRange, {newRangeAddr, newRangeSize}});

        return true;
    }

    bool allocPlainRange(const LiveRange& newRange, bool allowSpills, Direction dir, AllocatedAddrs& allocatedAddrs) {
        assert(!_handler.isFixedAlloc(newRange));

        const auto newRangeSize = _handler.getSize(newRange);
        const auto newRangeAlignment = _handler.getAlignment(newRange);

        auto newRangeAddr = InvalidAddress;

        LiveRangeVector spillCandidates;

        for (;;) {
            newRangeAddr = _par.alloc(newRangeSize, newRangeAlignment, dir);

            if (newRangeAddr != InvalidAddress) {
                break;
            }

            if (!allowSpills) {
                rollback(allocatedAddrs);
                return false;
            }

            const auto spillCandidate = getSpillCandidate();

            if (!spillCandidate.hasValue()) {
                rollback(allocatedAddrs);
                return false;
            }

            spillCandidates.push_back(spillCandidate.getValue());
        }

        if (!performActualSpill(spillCandidates, newRangeAddr, newRangeSize)) {
            rollback(allocatedAddrs);
            return false;
        }

        allocatedAddrs.insert({&newRange, {newRangeAddr, newRangeSize}});

        return true;
    }

    LiveRangeIter freeLiveRange(LiveRangeIter it) {
        assert(it != _liveRanges.end());
        const auto& range = *it;

        const auto addr = _handler.getAddress(range);
        const auto size = _handler.getSize(range);

        _par.free(addr, size);
        _handler.freed(range);

        *it = _liveRanges.back();
        _liveRanges.pop_back();

        return it;
    }

    Optional<LiveRange> getSpillCandidate() {
        const auto it = std::min_element(_liveRanges.begin(), _liveRanges.end(),
                                         [this](const LiveRange& r1, const LiveRange& r2) {
                                             return _handler.getSpillWeight(r1) < _handler.getSpillWeight(r2);
                                         });

        if (it == _liveRanges.end()) {
            return None;
        }

        const auto spillCandidate = *it;

        const auto spilledAddr = _handler.getAddress(spillCandidate);
        const auto spilledSize = _handler.getSize(spillCandidate);

        _par.free(spilledAddr, spilledSize);

        *it = _liveRanges.back();
        _liveRanges.pop_back();

        return spillCandidate;
    }

    bool performActualSpill(const LiveRangeVector& spillCandidates, AddressType newRangeAddr,
                            AddressType newRangeSize) {
        // Spill only those ranges which really conflict with the allocated one.
        for (const auto& spillCandidate : spillCandidates) {
            const auto spilledAddr = _handler.getAddress(spillCandidate);
            const auto spilledSize = _handler.getSize(spillCandidate);

            if (Partitioner::intersects(spilledAddr, spilledSize, newRangeAddr, newRangeSize)) {
                if (!_handler.spilled(spillCandidate)) {
                    return false;
                }

                _handler.freed(spillCandidate);
            } else {
                _par.allocFixed(spilledAddr, spilledSize);
                _handler.allocated(spillCandidate, spilledAddr);
                _liveRanges.push_back(spillCandidate);
            }
        }

        return true;
    }

    void rollback(const AllocatedAddrs& allocatedAddrs) {
        for (const auto& p : allocatedAddrs) {
            const auto addr = p.second.first;
            const auto size = p.second.second;

            if (addr == InvalidAddress) {
                continue;
            }

            _par.free(addr, size);
        }
    }

private:
    Partitioner _par;
    LiveRangeVector _liveRanges;
    Handler _handler;
};

}  // namespace vpux
