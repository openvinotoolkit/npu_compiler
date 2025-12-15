//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

//

//
// BF16, FP16, FP8E4M3, F8E5M2 and F4E2M1 implementation
//

#pragma once

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include "vpux/utils/core/type/bfloat16.hpp"
#include "vpux/utils/core/type/float16.hpp"
#include "vpux/utils/core/type/float4_e2m1.hpp"
#include "vpux/utils/core/type/float8_e4m3.hpp"
#include "vpux/utils/core/type/float8_e5m2.hpp"

namespace vpux {

template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float16, OutT>> checked_cast(vpux::type::bfloat16 val) {
    return vpux::type::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e4m3, OutT>> checked_cast(vpux::type::bfloat16 val) {
    return vpux::type::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e5m2, OutT>> checked_cast(vpux::type::bfloat16 val) {
    return vpux::type::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float4_e2m1, OutT>> checked_cast(vpux::type::bfloat16 val) {
    return vpux::type::float4_e2m1(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::bfloat16, OutT>> checked_cast(vpux::type::float16 val) {
    return vpux::type::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e4m3, OutT>> checked_cast(vpux::type::float16 val) {
    return vpux::type::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e5m2, OutT>> checked_cast(vpux::type::float16 val) {
    return vpux::type::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float4_e2m1, OutT>> checked_cast(vpux::type::float16 val) {
    return vpux::type::float4_e2m1(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e5m2, OutT>> checked_cast(vpux::type::float8_e4m3 val) {
    return vpux::type::float8_e5m2(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float16, OutT>> checked_cast(vpux::type::float8_e4m3 val) {
    return vpux::type::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::bfloat16, OutT>> checked_cast(vpux::type::float8_e4m3 val) {
    return vpux::type::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float4_e2m1, OutT>> checked_cast(vpux::type::float8_e4m3 val) {
    return vpux::type::float4_e2m1(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e4m3, OutT>> checked_cast(vpux::type::float8_e5m2 val) {
    return vpux::type::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float16, OutT>> checked_cast(vpux::type::float8_e5m2 val) {
    return vpux::type::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::bfloat16, OutT>> checked_cast(vpux::type::float8_e5m2 val) {
    return vpux::type::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float4_e2m1, OutT>> checked_cast(vpux::type::float8_e5m2 val) {
    return vpux::type::float4_e2m1(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::bfloat16, OutT>> checked_cast(vpux::type::float4_e2m1 val) {
    return vpux::type::bfloat16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float16, OutT>> checked_cast(vpux::type::float4_e2m1 val) {
    return vpux::type::float16(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e4m3, OutT>> checked_cast(vpux::type::float4_e2m1 val) {
    return vpux::type::float8_e4m3(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, std::is_same<vpux::type::float8_e5m2, OutT>> checked_cast(vpux::type::float4_e2m1 val) {
    return vpux::type::float8_e5m2(static_cast<float>(val));
}

template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<vpux::type::float8_e4m3, OutT>, std::is_same<vpux::type::float8_e5m2, OutT>,
                        std::is_same<vpux::type::float16, OutT>, std::is_same<vpux::type::float4_e2m1, OutT>>>>
checked_cast(vpux::type::bfloat16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<vpux::type::float8_e4m3, OutT>, std::is_same<vpux::type::float8_e5m2, OutT>,
                        std::is_same<vpux::type::bfloat16, OutT>, std::is_same<vpux::type::float4_e2m1, OutT>>>>
checked_cast(vpux::type::float16 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<vpux::type::float8_e5m2, OutT>, std::is_same<vpux::type::bfloat16, OutT>,
                        std::is_same<vpux::type::float16, OutT>, std::is_same<vpux::type::float4_e2m1, OutT>>>>
checked_cast(vpux::type::float8_e4m3 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<vpux::type::float8_e4m3, OutT>, std::is_same<vpux::type::bfloat16, OutT>,
                        std::is_same<vpux::type::float16, OutT>, std::is_same<vpux::type::float4_e2m1, OutT>>>>
checked_cast(vpux::type::float8_e5m2 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}
template <typename OutT>
enable_t<OutT, not_<or_<std::is_same<vpux::type::bfloat16, OutT>, std::is_same<vpux::type::float16, OutT>,
                        std::is_same<vpux::type::float8_e4m3, OutT>, std::is_same<vpux::type::float8_e5m2, OutT>>>>
checked_cast(vpux::type::float4_e2m1 val) {
    return checked_cast<OutT>(static_cast<float>(val));
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<vpux::type::bfloat16, OutT>> checked_cast(InT val) {
    return vpux::type::bfloat16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<vpux::type::float16, OutT>> checked_cast(InT val) {
    return vpux::type::float16(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<vpux::type::float8_e4m3, OutT>> checked_cast(InT val) {
    return vpux::type::float8_e4m3(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<vpux::type::float8_e5m2, OutT>> checked_cast(InT val) {
    return vpux::type::float8_e5m2(checked_cast<float>(val));
}
template <typename OutT, typename InT>
enable_t<OutT, std::is_same<vpux::type::float4_e2m1, OutT>> checked_cast(InT val) {
    return vpux::type::float4_e2m1(checked_cast<float>(val));
}

}  // namespace vpux
