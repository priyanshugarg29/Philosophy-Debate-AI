from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from philosophy_debate.runtime import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Build searchable indexes for the philosophy corpuses.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild even if cached indexes already exist.",
    )
    args = parser.parse_args()

    runtime = build_runtime(force_rebuild=args.rebuild)
    print("Knowledge bases are ready.")
    for report in runtime.reports.values():
        build_mode = "cache" if report.loaded_from_cache else "fresh build"
        print(
            f"- {report.display_name}: {report.source_count} sources, "
            f"{report.chunk_count} chunks, {build_mode}"
        )
        for warning in report.warnings:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
