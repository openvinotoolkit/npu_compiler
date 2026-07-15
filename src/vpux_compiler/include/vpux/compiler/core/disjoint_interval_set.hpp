//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <cassert>
#include <limits>
#include <map>
#include <set>

namespace vpux {

template <typename T>
struct IntervalTraits {
    using UnitType = size_t;
    using IntervalType = int;

    static UnitType intervalBegin(const IntervalType&);
    static UnitType intervalEnd(const IntervalType&);
    static bool isIntervalProducer(const IntervalType&);
    static bool canProducersCoexist(const IntervalType& existingProd, const IntervalType& newProd);
    static bool sameIntervalAccess(const IntervalType& intA, const IntervalType& intB);
    static void setInterval(IntervalType&, const UnitType& beg, const UnitType& end);
};  // struct IntervalTraits //

// NOTE: these are only for intervals on an integer lattice //
template <typename T>
struct IntervalUtils {
    using Traits = IntervalTraits<T>;
    using IntervalType = typename Traits::IntervalType;
    using UnitType = typename Traits::UnitType;

    static bool intersects(const UnitType& abeg, const UnitType& aend, const UnitType& bbeg, const UnitType& bend) {
        assert((abeg <= aend) && (bbeg <= bend));
        return std::max(abeg, bbeg) <= std::min(aend, bend);
    }

    static bool intersects(const IntervalType& a, const IntervalType& b) {
        const auto abeg = Traits::intervalBegin(a);
        const auto aend = Traits::intervalEnd(a);

        const auto bbeg = Traits::intervalBegin(b);
        const auto bend = Traits::intervalEnd(b);

        assert((abeg <= aend) && (bbeg <= bend));

        return std::max(abeg, bbeg) <= std::min(aend, bend);
    }

    // Checks if interval 'a' is subset of interval 'b' //
    static bool isSubset(const IntervalType& a, const IntervalType& b) {
        if (!intersects(a, b)) {
            return false;
        }

        const auto abeg = Traits::intervalBegin(a);
        const auto aend = Traits::intervalEnd(a);

        const auto bbeg = Traits::intervalBegin(b);
        const auto bend = Traits::intervalEnd(b);

        const auto ibeg = std::max(abeg, bbeg);
        const auto iend = std::min(aend, bend);

        return (ibeg == abeg) && (iend == aend);
    }

    static bool isSubset(UnitType abeg, UnitType aend, UnitType bbeg, UnitType bend) {
        assert((abeg <= aend) && (bbeg <= bend));

        if (std::max(abeg, bbeg) > std::min(aend, bend)) {
            return false;
        }
        const auto ibeg = std::max(abeg, bbeg);
        const auto iend = std::min(aend, bend);

        return (ibeg == abeg) && (iend == aend);
    }

    template <typename BackInsertIterator>
    static size_t intervalXor(const IntervalType& a, const IntervalType& b, BackInsertIterator out_itr) {
        size_t ret_size = 0UL;
        if (!intersects(a, b)) {
            *out_itr = a;
            ++out_itr;
            *out_itr = b;
            ++out_itr;
            ret_size = 2UL;
        } else {
            const auto abeg = Traits::intervalBegin(a);
            const auto aend = Traits::intervalEnd(a);

            const auto bbeg = Traits::intervalBegin(b);
            const auto bend = Traits::intervalEnd(b);

            assert((abeg <= aend) && (bbeg <= bend));

            auto obeg = std::max(abeg, bbeg);
            auto oend = std::min(aend, bend);
            auto ubeg = std::min(abeg, bbeg);
            auto uend = std::max(aend, bend);

            // [ubeg,obeg] and [oend,uend] //
            --obeg;  // note: if its not integer lattice this is an open interval //
            if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
                Traits::setInterval(*out_itr, ubeg, obeg);
                ++out_itr;
                ++ret_size;
            }

            ++oend;
            if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
                Traits::setInterval(*out_itr, oend, uend);
                ++out_itr;
                ++ret_size;
            }
        }
        return ret_size;
    }

    static size_t intervalXor(const IntervalType& a, const IntervalType& b, UnitType (&out_begin)[2UL],
                              UnitType (&out_end)[2UL]) {
        const auto abeg = Traits::intervalBegin(a);
        const auto aend = Traits::intervalEnd(a);

        const auto bbeg = Traits::intervalBegin(b);
        const auto bend = Traits::intervalEnd(b);

        assert((abeg <= aend) && (bbeg <= bend));

        size_t ret_size = 0UL;

        if (!intersects(a, b)) {
            out_begin[0] = abeg;
            out_end[0] = aend;
            out_begin[1] = bbeg;
            out_end[1] = bend;
            ret_size = 2UL;
        } else {
            auto obeg = std::max(abeg, bbeg);
            auto oend = std::min(aend, bend);
            auto ubeg = std::min(abeg, bbeg);
            auto uend = std::max(aend, bend);

            // [ubeg,obeg] and [oend,uend] //
            --obeg;  // note: if its not integer lattice this is an open interval //
            if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
                out_begin[ret_size] = ubeg;
                out_end[ret_size] = obeg;
                ++ret_size;
            }

            ++oend;
            if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
                out_begin[ret_size] = oend;
                out_end[ret_size] = uend;
                ++ret_size;
            }
        }
        return ret_size;
    }

    // NOTE: the return value ensures out_end[0] < out_begin[1] //
    static size_t intervalXor(const UnitType& abeg, const UnitType& aend, const UnitType& bbeg, const UnitType& bend,
                              UnitType (&out_begin)[2UL], UnitType (&out_end)[2UL]) {
        size_t ret_size = 0UL;

        if (!intersects(abeg, aend, bbeg, bend)) {
            out_begin[0] = abeg;
            out_end[0] = aend;
            out_begin[1] = bbeg;
            out_end[1] = bend;
            ret_size = 2UL;
        } else {
            auto obeg = std::max(abeg, bbeg);
            auto oend = std::min(aend, bend);
            auto ubeg = std::min(abeg, bbeg);
            auto uend = std::max(aend, bend);

            // [ubeg,obeg] and [oend,uend] //
            --obeg;  // note: if its not integer lattice this is an open interval //
            if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
                out_begin[ret_size] = ubeg;
                out_end[ret_size] = obeg;
                ++ret_size;
            }

            ++oend;
            if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
                out_begin[ret_size] = oend;
                out_end[ret_size] = uend;
                ++ret_size;
            }
        }
        return ret_size;
    }

};  // struct IntervalUtils //

// Maintains a set of dynamic disjoint intervals and each interval is
// associated with a corresponding element. Supports the following operations:
//
// range_query: input [a, b] (a <= b) reports all the overlapping elements in
// the set which overlap [a,b]
//
// insert: inserts a interval [a,b] iff its disjoint with other elements in the
// set. Returns false if [a,b] is not disjoint.
//
// erase: erase an interval [a,b] from the data structure.
//
// NOTE: disjoint means no overlap and no touch.
template <typename Unit, typename Element>
class DisjointIntervalSet {
public:
    //            ----[*]---
    //                 x
    enum OrientType { LEFT_END = 0, RIGHT_END = 1 };  // enum OrientType //

    struct EndPointType {
        Unit _x;
        OrientType _orient;

        bool operator==(const EndPointType& o) const {
            return (_x == o._x) && (_orient == o._orient);
        }

        bool operator<(const EndPointType& o) const {
            return (_x != o._x) ? (_x < o._x) : _orient < o._orient;
        }

        bool isLeftEnd() const {
            return _orient == LEFT_END;
        }
        bool isRightEnd() const {
            return _orient == RIGHT_END;
        }

        EndPointType(Unit x, OrientType orient): _x(x), _orient(orient) {
        }
    };  // struct EndPointType //

    // Struct for storing ownership of interval
    // Each interval must have one producer and can have multiple users
    // _prevUsers caches the consumers that existed at the moment the current
    // producer(s) took ownership of the interval (set by newProducer). It lets
    // subsequent coexisting producers add control edges from those past consumers
    // without scanning the whole output dependency list.
    struct IntervalUsersType {
        std::set<Element> _producers;
        std::set<Element> _consumers;
        std::set<Element> _prevUsers;

        IntervalUsersType(const Element& prod): _producers({prod}) {
        }
        IntervalUsersType(const Element& prod, const std::set<Element>& cons): _producers({prod}), _consumers(cons) {
        }

        bool operator==(const IntervalUsersType& o) const {
            return (_producers == o._producers) && (_consumers == o._consumers);
        }

        void newProducer(const Element& prod) {
            // Preserve the consumers that the previous producer(s) served so that
            // future coexisting producers can derive their control-edge sources
            // directly from this set.
            if (!_consumers.empty()) {
                _prevUsers = std::move(_consumers);
            } else {
                _prevUsers = std::move(_producers);
            }

            _producers = {prod};
            _consumers.clear();
        }

        void addProducer(const Element& prod) {
            _producers.insert(prod);
        }

        void addConsumer(const Element& cons) {
            _consumers.insert(cons);
        }

        void addPrevUsers(const std::set<Element>& prevUsers) {
            _prevUsers.insert(prevUsers.begin(), prevUsers.end());
        }

    };  // struct IntervalUsersType //

    using EndPointTreeType = std::map<EndPointType, IntervalUsersType>;
    using ConstEndPointIteratorType = typename EndPointTreeType::const_iterator;

    // Invariant: itrBegin_ should always point to
    class IntervalIteratorType {
    public:
        IntervalIteratorType(ConstEndPointIteratorType begin, ConstEndPointIteratorType end)
                : itrBegin_(begin), itrEnd_(end) {
        }

        IntervalIteratorType(): itrBegin_(), itrEnd_() {
        }

        // only invalid iterators are equivalent //
        bool operator==(const IntervalIteratorType& o) const {
            return (itrBegin_ == itrEnd_) && (o.itrBegin_ == o.itrEnd_);
        }
        bool operator!=(const IntervalIteratorType& o) const {
            return !(*this == o);
        }

        // Precondition: itrBegin_ != itrEnd_ //
        const IntervalIteratorType& operator++() {
            // Invariant: itrBegin_ always points to a left end //
            ++itrBegin_;
            assert(itrBegin_ != itrEnd_);
            ++itrBegin_;
            return *this;
        }

        const std::set<Element>& getProducers() const {
            return itrBegin_->second._producers;
        }

        const std::set<Element>& getConsumers() const {
            return itrBegin_->second._consumers;
        }

        const std::set<Element>& getIntervalPrevUsers() const {
            return itrBegin_->second._prevUsers;
        }

        const IntervalUsersType& getIntervalUsers() const {
            return itrBegin_->second;
        }

        // Precondition: itrBegin_ != itrEnd_ //
        Unit intervalBegin() const {
            return (itrBegin_->first)._x;
        }

        // Precondition: itrBegin_ != itrEnd_ //
        Unit intervalEnd() const {
            auto itr_next = itrBegin_;
            ++itr_next;
            return (itr_next->first)._x;
        }

    private:
        ConstEndPointIteratorType itrBegin_;
        ConstEndPointIteratorType itrEnd_;
    };  // class IntervalIteratorType //

    bool insert(const Unit& ibeg, const Unit& iend, const Element& prod) {
        assert(ibeg <= iend);
        EndPointType lkey(ibeg, LEFT_END), rkey(iend, RIGHT_END);
        ConstEndPointIteratorType itr = _tree.lower_bound(lkey);

        if (!isIntervalDisjoint(lkey, rkey, itr)) {
            return false;
        }

        _tree.insert(itr /*hint*/, std::make_pair(lkey, prod));
        _tree.insert(itr /*hint*/, std::make_pair(rkey, prod));
        return true;
    }

    bool insert(const Unit& ibeg, const Unit& iend, const IntervalUsersType& prodCons) {
        assert(ibeg <= iend);
        EndPointType lkey(ibeg, LEFT_END), rkey(iend, RIGHT_END);
        ConstEndPointIteratorType itr = _tree.lower_bound(lkey);

        if (!isIntervalDisjoint(lkey, rkey, itr)) {
            return false;
        }

        _tree.insert(itr /*hint*/, std::make_pair(lkey, prodCons));
        _tree.insert(itr /*hint*/, std::make_pair(rkey, prodCons));
        return true;
    }

    bool erase(const Unit& ibeg, const Unit& iend) {
        ConstEndPointIteratorType litr = _tree.find(EndPointType(ibeg, LEFT_END));
        ConstEndPointIteratorType ritr = _tree.find(EndPointType(iend, RIGHT_END));
        if ((litr == _tree.end()) || (ritr == _tree.end())) {
            return false;
        }

        _tree.erase(litr);
        _tree.erase(ritr);
        return true;
    }

    bool overlaps(const Unit& ibeg, const Unit& iend) const {
        assert(ibeg <= iend);
        EndPointType lkey(ibeg, LEFT_END), rkey(iend, RIGHT_END);

        return !isIntervalDisjoint(lkey, rkey, _tree.lower_bound(lkey));
    }

    IntervalIteratorType query(const Unit& ibeg, const Unit& iend) const {
        ConstEndPointIteratorType litr = _tree.lower_bound(EndPointType(ibeg, LEFT_END));

        if ((litr != _tree.end()) && ((litr->first).isRightEnd())) {
            --litr;
            assert(litr != _tree.end());
        }

        ConstEndPointIteratorType ritr = _tree.lower_bound(EndPointType(iend, RIGHT_END));
        if ((ritr != _tree.end()) && ((ritr->first).isRightEnd())) {
            ++ritr;
        }
        return IntervalIteratorType(litr, ritr);
    }

    IntervalIteratorType begin() const {
        return IntervalIteratorType(_tree.begin(), _tree.end());
    }

    IntervalIteratorType end() const {
        return IntervalIteratorType(_tree.end(), _tree.end());
    }

    DisjointIntervalSet(): _tree() {
    }

    size_t size() const {
        assert(_tree.size() % 2 == 0);
        return _tree.size() / 2;
    }
    bool empty() const {
        return _tree.empty();
    }
    void clear() {
        _tree.clear();
    }

private:
    bool isLeftEnd(ConstEndPointIteratorType& o) const {
        return (o->first).isLeftEnd();
    }

    bool isIntervalDisjoint(const EndPointType& lkey, const EndPointType& rkey,
                            ConstEndPointIteratorType lkey_lower_bound) const {
        if (lkey_lower_bound == _tree.end()) {
            return true;
        }
        const EndPointType& lkey_lb = lkey_lower_bound->first;
        return (lkey_lb.isLeftEnd() && (lkey._x < lkey_lb._x) && (rkey._x < lkey_lb._x));
    }

    ////////////////////////////////////////////////////////////////////////////
    EndPointTreeType _tree;  // balanced search tree on EndPointType //
};  // class DisjointIntervalSet //

}  // namespace vpux
