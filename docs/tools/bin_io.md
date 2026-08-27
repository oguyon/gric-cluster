# GRIC Binary Format & Conversion Tools

The GRIC suite includes a high-throughput, self-describing binary serialization format (`.bin`)
and two dedicated conversion utilities: `gric-ascii2bin` and `gric-bin2ascii`.

---

## 1. The GRIC Binary Format Specification

GRIC binary files use a 64-byte aligned header followed by contiguous row-major payload arrays.
They enable zero-copy memory mapping (`mmap`), SIMD alignment, and instant deserialization.

### Header Layout (64 Bytes)

* **Magic Bytes (4B)**: `0x47 0x52 0x49 0x43` (`"GRIC"`)
* **Version (2B)**: Version `1`
* **File Type (2B)**:
  * `0x0001` (`GRIC_BIN_TYPE_GENERIC`)
  * `0x0002` (`GRIC_BIN_TYPE_COORDS`)
  * `0x0003` (`GRIC_BIN_TYPE_ANCHORS`)
  * `0x0004` (`GRIC_BIN_TYPE_DCC`)
  * `0x0005` (`GRIC_BIN_TYPE_MEMBERSHIP`)
  * `0x0006` (`GRIC_BIN_TYPE_COUNTS`)
  * `0x0007` (`GRIC_BIN_TYPE_KNN`)
* **Data Type (2B)**: `FLOAT32`, `FLOAT64`, `INT32`, `UINT32`, `INT16`, `UINT16`, `UINT8`
* **Flags (2B)**: Byte ordering, row-major layout, compression flags
* **Dimensions (8B)**: Number of dimensions ($1 \le D \le 4$)
* **Shape Array (32B)**: Dimensions array `[dim0, dim1, dim2, dim3]`
* **Payload Byte Count (8B)**: Total byte count of the contiguous data buffer

---

## 2. Encoder: `gric-ascii2bin`

Converts ASCII coordinate tables, matrix outputs, or text lists into typed `.bin` files.

```bash
gric-ascii2bin <input.txt> <output.bin> [options]
```

### Options
* `-type <type>`: Semantic type (`anchors`, `dcc`, `membership`, `counts`, `coords`, `generic`)
* `-double`: Encode floating-point values as `float64` (default: `float32`)
* `-uint32`: Encode integers as unsigned 32-bit (`uint32`)
* `-int32`: Encode integers as signed 32-bit (`int32`)
* `-dim <D>`: Explicit column dimension count (default: auto-detected)
* `-comment <text>`: Embed descriptive metadata string into header
* `-v, --verbose`: Print detailed parsing and serialization statistics

---

## 3. Decoder: `gric-bin2ascii`

Decodes, inspects, and pipes GRIC `.bin` files to ASCII format or standard output.

```bash
gric-bin2ascii <input.bin> [output.txt] [options]
```

### Options
* `-info, -i`: Display header metadata summary without decoding payload
* `-fmt <spec>`: Custom printf formatting specifier (e.g. `'%.6f'`, `'%g'`, `'%d'`)
* `-v, --verbose`: Print decoding summary to stderr
* If `[output.txt]` is omitted or set to `'-'`, decoded data is written directly to stdout.
