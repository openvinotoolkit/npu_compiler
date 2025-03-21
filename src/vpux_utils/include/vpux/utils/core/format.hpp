//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// String format utilities.
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_os_ostream.h>

#include <array>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cstddef>

namespace vpux {

//
// formatv
//

using llvm::formatv;
using llvm::formatv_object_base;

//
// printToString
//

template <typename... Args>
std::string printToString(StringLiteral format, Args&&... args) {
    return formatv(format.data(), std::forward<Args>(args)...);
}

template <typename... Args>
std::string printToString(const char* format, Args&&... args) {
    return formatv(format, std::forward<Args>(args)...);
}

//
// printTo
//

template <typename... Args>
auto& printTo(llvm::raw_ostream& stream, StringLiteral format, Args&&... args) {
    return stream << formatv(format.data(), std::forward<Args>(args)...);
}

template <typename... Args>
auto& printTo(std::ostream& stream, StringLiteral format, Args&&... args) {
    llvm::raw_os_ostream(stream) << formatv(format.data(), std::forward<Args>(args)...);
    return stream;
}

template <class Stream, typename... Args>
auto&& printTo(Stream&& stream, StringLiteral format, Args&&... args) {
    return stream << printToString(format, std::forward<Args>(args)...);
}

//
// Various base classes for `llvm::format_provider` implementations.
//

struct ListFormatProvider {
    template <class List>
    static void format(const List& list, llvm::raw_ostream& stream, StringRef style) {
        using IterT = typename List::const_iterator;
        using IterRange = llvm::iterator_range<IterT>;

        stream << '[';
        llvm::format_provider<IterRange>::format(llvm::make_range(std::begin(list), std::end(list)), stream, style);
        stream << ']';
    }
};

struct MapFormatProvider {
    template <class Map>
    static void format(const Map& map, llvm::raw_ostream& stream, StringRef style) {
        stream << '<';

        size_t ind = 0;
        const auto size = static_cast<size_t>(map.size());
        for (const auto& p : map) {
            llvm::detail::build_format_adapter(p.first).format(stream, style);
            stream << " : ";
            llvm::detail::build_format_adapter(p.second).format(stream, style);

            if (++ind < size) {
                stream << ", ";
            }
        }

        stream << '>';
    }
};

//
// `printFormat` method handling.
//
// It allows to implement `printFormat` method in class without creating
// separate `llvm::format_provider` instance.
//

namespace details {

template <class Obj, typename = void>
struct HasPrintFormat {
    static constexpr bool value = false;
};

template <class Obj>
struct HasPrintFormat<Obj, decltype(std::declval<Obj>().printFormat(std::declval<llvm::raw_ostream&>()))> {
    static constexpr bool value = true;
};

}  // namespace details

}  // namespace vpux

//
// llvm::format_provider specialization
//

namespace llvm {

template <>
struct format_provider<std::error_code> final {
    static void format(const std::error_code& err, llvm::raw_ostream& stream, StringRef style) {
        llvm::detail::build_format_adapter(err.message()).format(stream, style);
    }
};

template <typename T1, typename T2>
struct format_provider<std::pair<T1, T2>> final {
    static void format(const std::pair<T1, T2>& val, llvm::raw_ostream& stream, StringRef style) {
        stream << '(';

        llvm::detail::build_format_adapter(val.first).format(stream, style);
        stream << ", ";
        llvm::detail::build_format_adapter(val.second).format(stream, style);

        stream << ')';
    }
};

template <typename T, size_t Count>
struct format_provider<std::array<T, Count>> final : vpux::ListFormatProvider {};

template <typename T, class A>
struct format_provider<std::vector<T, A>> final : vpux::ListFormatProvider {};

template <typename T, class A>
struct format_provider<std::list<T, A>> final : vpux::ListFormatProvider {};

template <typename T, class C, class A>
struct format_provider<std::set<T, C, A>> final : vpux::ListFormatProvider {};

template <typename T, class H, class P, class A>
struct format_provider<std::unordered_set<T, H, P, A>> final : vpux::ListFormatProvider {};

template <typename K, typename V, class C, class A>
struct format_provider<std::map<K, V, C, A>> final : vpux::MapFormatProvider {};

template <typename K, typename V, class H, class P, class A>
struct format_provider<std::unordered_map<K, V, H, P, A>> final : vpux::MapFormatProvider {};

template <class Obj>
struct format_provider<Obj, vpux::require_t<vpux::details::HasPrintFormat<Obj>>> final {
    static void format(const Obj& obj, llvm::raw_ostream& stream, StringRef) {
        obj.printFormat(stream);
    }
};

template <typename... Args>
struct format_provider<std::tuple<Args...>> final {
    static void format(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style) {
        stream << '<';
        printItems(val, stream, style);
        stream << '>';
    }

private:
    template <size_t Index = 0>
    static auto printItems(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style)
            -> std::enable_if_t<Index + 1 == sizeof...(Args)> {
        llvm::detail::build_format_adapter(std::get<Index>(val)).format(stream, style);
    }

    template <size_t Index = 0>
    static auto printItems(const std::tuple<Args...>& val, llvm::raw_ostream& stream, StringRef style)
            -> std::enable_if_t<Index + 1 < sizeof...(Args)> {
        llvm::detail::build_format_adapter(std::get<Index>(val)).format(stream, style);
        stream << ", ";

        printItems<Index + 1>(val, stream, style);
    }
};

}  // namespace llvm
