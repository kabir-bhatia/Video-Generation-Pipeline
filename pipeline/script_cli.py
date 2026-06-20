"""Subprocess entry for the script stage.

The script LLM runs in its own process so its memory is fully returned to
the OS before the next model loads - in-process cleanup leaves the allocator
holding enough to crash a tight-RAM machine.

Usage:
    python -m pipeline.script_cli generate <args.json> <out.json>
"""

import json
import logging
import sys

from . import script_gen


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    op, in_path, out_path = sys.argv[1:4]
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    if op == "generate":
        result = script_gen.generate_script(**data)
    else:
        raise SystemExit(f"unknown op: {op}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
