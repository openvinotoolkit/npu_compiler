//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

//
// Dims3D
//

namespace Dims3D {

// Matmul3d activations

namespace Act {
inline constexpr Dim B(0);
inline constexpr Dim H(1);
inline constexpr Dim IC(2);

inline constexpr size_t numSpatialDims = 1;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 1, "Dims3D::Act: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 1);
}
}  // namespace Act

// Matmul3d filter

namespace Filter {
inline constexpr Dim B(0);
inline constexpr Dim IC(1);
inline constexpr Dim OC(2);

inline constexpr size_t numSpatialDims = 1;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 1, "Dims3D::Filter: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 1);
}
}  // namespace Filter

// Matmul3d output

namespace Output {
inline constexpr Dim B(0);
inline constexpr Dim H(1);
inline constexpr Dim OC(2);
}  // namespace Output

}  // namespace Dims3D

//
// Dims4D
//

namespace Dims4D {

// Convolution2D/Pooling2D activations

namespace Act {
inline constexpr Dim N(0);
inline constexpr Dim C(1);
inline constexpr Dim H(2);
inline constexpr Dim W(3);

inline constexpr size_t numDims = 4;
inline constexpr size_t numSpatialDims = 2;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 2, "Dims4D::Act: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 2);
}
}  // namespace Act

// Convolution2D filter

namespace Filter {
inline constexpr Dim OC(0);
inline constexpr Dim IC(1);
inline constexpr Dim KY(2);
inline constexpr Dim KX(3);

inline constexpr size_t numSpatialDims = 2;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 2, "Dims4D::Filter: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 2);
}
}  // namespace Filter

// Pooling2D kernel

namespace Kernel {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Kernel

// Convolution2D/Pooling2D strides

namespace Strides {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Strides

// Convolution2D dilations

namespace Dilation {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Dilation

// Convolution2D/Pooling2D paddings

namespace PadsBegin {
inline constexpr Dim Top(0);
inline constexpr Dim Left(1);
}  // namespace PadsBegin

namespace PadsEnd {
inline constexpr Dim Bottom(0);
inline constexpr Dim Right(1);
}  // namespace PadsEnd

// TransposedConvolution2D output paddings

namespace PadsOutput {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace PadsOutput

}  // namespace Dims4D

//
// Dims5D
//

namespace Dims5D {

// Convolution3D/Pooling3D activations

namespace Act {
inline constexpr Dim N(0);
inline constexpr Dim C(1);
inline constexpr Dim D(2);
inline constexpr Dim H(3);
inline constexpr Dim W(4);

inline constexpr size_t numSpatialDims = 3;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 3, "Dims5D::Act: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 2);
}
}  // namespace Act

// Convolution3D filter

namespace Filter {
inline constexpr Dim OC(0);
inline constexpr Dim IC(1);
inline constexpr Dim KZ(2);
inline constexpr Dim KY(3);
inline constexpr Dim KX(4);

inline constexpr size_t numSpatialDims = 3;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 3, "Dims5D::Filter: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 2);
}
}  // namespace Filter

// Pooling3D kernel

namespace Kernel {
inline constexpr Dim Z(0);
inline constexpr Dim Y(1);
inline constexpr Dim X(2);
}  // namespace Kernel

// Convolution3D/Pooling3D strides

namespace Strides {
inline constexpr Dim Z(0);
inline constexpr Dim Y(1);
inline constexpr Dim X(2);
}  // namespace Strides

// Convolution3D dilations

namespace Dilation {
inline constexpr Dim Z(0);
inline constexpr Dim Y(1);
inline constexpr Dim X(2);
}  // namespace Dilation

// Convolution3D/Pooling3D paddings
// refer to openvino/src/frontends/paddle/src/op/pad3d.cpp
namespace PadsBegin {
inline constexpr Dim Front(0);
inline constexpr Dim Top(1);
inline constexpr Dim Left(2);
}  // namespace PadsBegin

namespace PadsEnd {
inline constexpr Dim Back(0);
inline constexpr Dim Bottom(1);
inline constexpr Dim Right(2);
}  // namespace PadsEnd

// TransposedConvolution3D output paddings

namespace PadsOutput {
inline constexpr Dim Z(0);
inline constexpr Dim Y(1);
inline constexpr Dim X(2);
}  // namespace PadsOutput

}  // namespace Dims5D

// Layer itself is 2D, but several layers are grouped without sharing weights
namespace DimsGroups5D {

// Grouped layer activations
namespace Act {
inline constexpr Dim G(0);
inline constexpr Dim N(1);
inline constexpr Dim C(2);
inline constexpr Dim H(3);
inline constexpr Dim W(4);

inline constexpr size_t numDims = 5;
inline constexpr size_t numSpatialDims = 2;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 2, "DimsGroups5D::Act: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 3);
}
}  // namespace Act

// Grouped layer filter
namespace Filter {
inline constexpr Dim G(0);
inline constexpr Dim OC(1);
inline constexpr Dim IC(2);
inline constexpr Dim KY(3);
inline constexpr Dim KX(4);

inline constexpr size_t numDims = 5;
inline constexpr size_t numSpatialDims = 2;

inline Dim getSpatialDim(size_t index) {
    VPUX_THROW_UNLESS(index < 2, "DimsGroups5D::Filter: Wrong spatial dimension index '{0}'", index);
    return Dim(index + 3);
}
}  // namespace Filter

// Grouped layer kernel
namespace Kernel {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Kernel

// Grouped layer strides
namespace Strides {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Strides

// Grouped layer dilations
namespace Dilation {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace Dilation

// Grouped layer paddings
namespace PadsBegin {
inline constexpr Dim Top(0);
inline constexpr Dim Left(1);
}  // namespace PadsBegin

namespace PadsEnd {
inline constexpr Dim Bottom(0);
inline constexpr Dim Right(1);
}  // namespace PadsEnd

// Grouped layer output paddings
namespace PadsOutput {
inline constexpr Dim Y(0);
inline constexpr Dim X(1);
}  // namespace PadsOutput

}  // namespace DimsGroups5D

}  // namespace vpux
