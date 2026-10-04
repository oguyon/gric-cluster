#!/usr/bin/env python3
"""
Audit CLI options in source files against help documentation (docs/help/**/*.md).
"""

import os
import re
import sys

def parse_opts_table(filepath):
    """Parse struct gric_opt opts[] = { ... } table from C source."""
    flags = set()
    if not os.path.isfile(filepath):
        return flags

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    m = re.search(r"struct\s+gric_opt\s+opts\[\]\s*=\s*\{([^;]+)\};", text, re.DOTALL)
    if not m:
        return flags

    block = m.group(1)
    for line in block.split("\n"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        m_name = re.match(r'\{\s*"([^"]+)"', line)
        if m_name:
            flags.add(m_name.group(1))

    return flags

def parse_probe_flags(filepath):
    """Parse flags from strcmp(argv[...], "-...") in gric-probe/main.c."""
    flags = set()
    if not os.path.isfile(filepath):
        return flags

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    for m in re.finditer(r'strcmp\(argv\[[^\]]+\],\s*"(-{1,2}[a-zA-Z0-9_\-]+)"\)', text):
        flag = m.group(1).lstrip("-")
        if flag:
            flags.add(flag)

    return flags

def load_documentation_words(docs_dir, gen_help_py):
    """Load words and aliases from documentation and gen_help_c.py."""
    doc_words = set()
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    first_line = content.split("\n")[0] if content else ""
                    if first_line.startswith("# "):
                        topic_kw = first_line[2:].strip().lower()
                        doc_words.add(topic_kw)
                        doc_words.add(topic_kw.replace("-", "_"))
                        doc_words.add(topic_kw.replace("_", "-"))
                    base_kw = os.path.splitext(fname)[0].lower()
                    doc_words.add(base_kw)
                    doc_words.add(base_kw.replace("-", "_"))
                    doc_words.add(base_kw.replace("_", "-"))

                    tokens = re.findall(r'[a-zA-Z0-9_\-]+', content)
                    for tok in tokens:
                        t = tok.lower()
                        doc_words.add(t)
                        doc_words.add(t.replace("-", "_"))
                        doc_words.add(t.replace("_", "-"))

    # Also load ALIASES from gen_help_c.py
    if os.path.isfile(gen_help_py):
        with open(gen_help_py, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"ALIASES\s*=\s*\{([^}]+)\}", text, re.DOTALL)
        if m:
            for word in re.findall(r'"([^"]+)"', m.group(1)):
                w = word.lower()
                doc_words.add(w)
                doc_words.add(w.replace("-", "_"))
                doc_words.add(w.replace("_", "-"))

    return doc_words

def load_allowlist(allowlist_path):
    """Load allowed undocumented flags and their reasons."""
    allowed = {}
    if not os.path.isfile(allowlist_path):
        return allowed

    with open(allowlist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("#", 1)
            flag = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""
            if flag:
                allowed[flag] = reason
    return allowed

def main():
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <repo_root> <docs_help_dir> <allowlist_file> <gen_help_c_py>")
        sys.exit(1)

    repo_root = sys.argv[1]
    docs_help = sys.argv[2]
    allowlist_file = sys.argv[3]
    gen_help_py = sys.argv[4]

    cluster_c = os.path.join(repo_root, "src/gric-cluster/core/config_utils.c")
    knn_c = os.path.join(repo_root, "src/gric-knn/cli/knn_cli.c")
    probe_c = os.path.join(repo_root, "src/gric-probe/main.c")

    cluster_flags = parse_opts_table(cluster_c)
    knn_flags = parse_opts_table(knn_c)
    probe_flags = parse_probe_flags(probe_c)

    all_flags = sorted(cluster_flags | knn_flags | probe_flags)
    doc_words = load_documentation_words(docs_help, gen_help_py)
    allowlist = load_allowlist(allowlist_file)

    errors = []
    warnings = []

    for flag in all_flags:
        f_norm = flag.lower()
        f_alt = f_norm.replace("-", "_")
        f_dash = f_norm.replace("_", "-")

        is_doc = (f_norm in doc_words or f_alt in doc_words or f_dash in doc_words)

        if flag in allowlist:
            if is_doc:
                errors.append(
                    f"Allowlisted flag '{flag}' is now documented! Remove it from {allowlist_file}"
                )
        else:
            if not is_doc:
                errors.append(
                    f"Undocumented CLI flag '{flag}' detected! "
                    f"Document in {docs_help} or add to {allowlist_file}"
                )

    if errors:
        print("CLI Flag Documentation Drift Test FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print(
        f"PASS: All {len(all_flags)} CLI flags audited ({len(allowlist)} allowlisted, "
        f"{len(all_flags) - len(allowlist)} documented)."
    )

if __name__ == "__main__":
    main()
