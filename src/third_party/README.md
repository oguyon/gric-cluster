# Third-Party Dependencies (Level 0)

## Purpose
Vendored third-party libraries kept isolated from core GRIC modules to avoid code size,
formatting, and licensing entanglements.

## Architectural Layer
- **Level**: Level 0 (External / Vendored)
- **Vendored Libraries**:
  - `third_party/cjson/`: cJSON parser and serializer (MIT License).
- **Public Headers**:
  - `third_party/cjson/cJSON.h`
- **Permitted Dependencies**:
  - Standard C library only.
- **Constraints**:
  - Excluded from internal style checks, code size ratchets, and line length rules.
