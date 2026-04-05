from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_FULL_TARGETS,
    SOURCE_RAW_DIMER_PICKLE,
    ensure_dir,
    extract_full_dimer_targets,
    save_full_target_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract full DMC-DMC SAPT decomposition targets for MLFF training.")
    parser.add_argument("--raw-pickle", default=str(SOURCE_RAW_DIMER_PICKLE))
    parser.add_argument("--config-key", default="conf_001_DMC_DMC")
    parser.add_argument("--batch-key", default="000")
    parser.add_argument("--output-npz", default=str(DEFAULT_FULL_TARGETS))
    parser.add_argument("--output-csv", default=str(Path(DEFAULT_FULL_TARGETS).with_suffix(".csv")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = extract_full_dimer_targets(
        raw_pickle=Path(args.raw_pickle),
        config_key=args.config_key,
        batch_key=args.batch_key,
    )
    output_npz = Path(args.output_npz)
    output_csv = Path(args.output_csv)
    ensure_dir(output_npz.parent)
    save_full_target_bundle(targets, output_npz, output_csv)
    print(f"Wrote {output_npz}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
