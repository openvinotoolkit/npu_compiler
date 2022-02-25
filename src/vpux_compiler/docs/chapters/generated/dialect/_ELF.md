<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'ELF' Dialect

[TOC]

## Type constraint definition

### ELF Section Type
This object represents closely a Section
### ELF Symbol Type
This object represents closely a Symbol
## Operation definition

### `ELF.CreateLogicalSection` (::vpux::ELF::CreateLogicalSectionOp)

Create an ELF Logical Section, with no actual binary content in the ELF file 


Syntax:

```
operation ::= `ELF.CreateLogicalSection` `secType` `(` $secType `)`
              `secFlags` `(` $secFlags `)`
              attr-dict
              `->` type(results)
              $declaredOps
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`secName` | ::mlir::StringAttr | string attribute
`secType` | vpux::ELF::SectionTypeAttrAttr | Enum for describing ELF section header types
`secFlags` | vpux::ELF::SectionFlagsAttrAttr | Enum for describing ELF section header flags (we can use also the | operator)
`secInfo` | mlir::IntegerAttr | Integer attribute
`secAddrAlign` | mlir::IntegerAttr | Integer attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`section` | ELF Section Type

### `ELF.CreateRelocationSection` (::vpux::ELF::CreateRelocationSectionOp)

Create ELF Relocation Section


Syntax:

```
operation ::= `ELF.CreateRelocationSection` `secName` `(` $secName `)`
              `sourceSymbolTableSection` `(` $sourceSymbolTableSection `)`
              `targetSection` `(` $targetSection `)`
              `secFlags` `(` $secFlags `)`
              attr-dict
              `->` type(results)
              $aRegion
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`secName` | ::mlir::StringAttr | string attribute
`secFlags` | vpux::ELF::SectionFlagsAttrAttr | Enum for describing ELF section header flags (we can use also the | operator)

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`sourceSymbolTableSection` | ELF Section Type
`targetSection` | ELF Section Type

#### Results:

| Result | Description |
| :----: | ----------- |
`section` | ELF Section Type

### `ELF.CreateSection` (::vpux::ELF::CreateSectionOp)

Create ELF Section


Syntax:

```
operation ::= `ELF.CreateSection` `secType` `(` $secType `)`
              `secFlags` `(` $secFlags `)`
              attr-dict
              `->` type(results)
              $aRegion
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`secName` | ::mlir::StringAttr | string attribute
`secType` | vpux::ELF::SectionTypeAttrAttr | Enum for describing ELF section header types
`secFlags` | vpux::ELF::SectionFlagsAttrAttr | Enum for describing ELF section header flags (we can use also the | operator)
`secInfo` | mlir::IntegerAttr | Integer attribute
`secAddrAlign` | mlir::IntegerAttr | Integer attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`section` | ELF Section Type

### `ELF.CreateSymbolTableSection` (::vpux::ELF::CreateSymbolTableSectionOp)

Create ELF Symbol Table Section


Syntax:

```
operation ::= `ELF.CreateSymbolTableSection` `secName` `(` $secName `)`
              `secFlags` `(` $secFlags `)`
              attr-dict
              `->` type(results)
              $aRegion
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`secName` | ::mlir::StringAttr | string attribute
`secFlags` | vpux::ELF::SectionFlagsAttrAttr | Enum for describing ELF section header flags (we can use also the | operator)
`isBuiltin` | ::mlir::BoolAttr | bool attribute

#### Results:

| Result | Description |
| :----: | ----------- |
`section` | ELF Section Type

### `ELF.PutOpInSection` (::vpux::ELF::PutOpInSectionOp)

Put the Argument Op in the ELF Section


Syntax:

```
operation ::= `ELF.PutOpInSection` $inputArg attr-dict `:` type($inputArg)
```


#### Operands:

| Operand | Description |
| :-----: | ----------- |
`inputArg` | any type

### `ELF.Reloc` (::vpux::ELF::RelocOp)

Reloc Op


Syntax:

```
operation ::= `ELF.Reloc` $offsetTargetField $relocationType $sourceSymbol $addend attr-dict
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`offsetTargetField` | mlir::IntegerAttr | Integer attribute
`relocationType` | vpux::ELF::RelocationTypeAttrAttr | Enum for describing ELF relocation types
`addend` | mlir::IntegerAttr | Integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`sourceSymbol` | ELF Symbol Type

### `ELF.Symbol` (::vpux::ELF::SymbolOp)

Create ELF Symbol Table Section


Syntax:

```
operation ::= `ELF.Symbol` $inputArg
              (`name` `(` $name^ `)`)?
              (`type` `(` $type^ `)`)?
              (`size` `(` $size^ `)`)?
              attr-dict
              `:` type($inputArg)
```


#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
`isBuiltin` | ::mlir::BoolAttr | bool attribute
`name` | ::mlir::StringAttr | string attribute
`type` | vpux::ELF::SymbolTypeAttrAttr | Enum to represent symbol types
`size` | ::mlir::IntegerAttr | 64-bit unsigned integer attribute
`value` | ::mlir::IntegerAttr | 64-bit unsigned integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
`inputArg` | any type

#### Results:

| Result | Description |
| :----: | ----------- |
`symbol` | ELF Symbol Type

## Type definition

### SectionType

ELF Section Type

This object represents closely a Section
### SymbolType

ELF Symbol Type

This object represents closely a Symbol