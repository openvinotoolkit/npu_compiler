#! /bin/bash
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

echo "#pragma once" | unix2dos
echo | unix2dos
find -name "*.obj" |
  xargs dumpbin.exe /symbols |
  grep "notype\s*External\s*| __MCM_REGISTER__" |
  cut -d \| -f2 |
  awk '{$1=$1;print "#pragma comment(linker, \"/include:" $1 "\")"}' |
  sort |
  uniq |
  unix2dos
