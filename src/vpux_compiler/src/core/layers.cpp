//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/core/layers.hpp"

using namespace vpux;

//
// Dims4D
//

const Dim vpux::Dims4D::Act::N(0);
const Dim vpux::Dims4D::Act::C(1);
const Dim vpux::Dims4D::Act::H(2);
const Dim vpux::Dims4D::Act::W(3);

const Dim vpux::Dims4D::Filter::OC(0);
const Dim vpux::Dims4D::Filter::IC(1);
const Dim vpux::Dims4D::Filter::KY(2);
const Dim vpux::Dims4D::Filter::KX(3);

const Dim vpux::Dims4D::Kernel::Y(0);
const Dim vpux::Dims4D::Kernel::X(1);

const Dim vpux::Dims4D::Strides::Y(0);
const Dim vpux::Dims4D::Strides::X(1);

const Dim vpux::Dims4D::PadsBegin::Top(0);
const Dim vpux::Dims4D::PadsBegin::Left(1);

const Dim vpux::Dims4D::PadsEnd::Bottom(0);
const Dim vpux::Dims4D::PadsEnd::Right(1);