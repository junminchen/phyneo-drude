import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter meta.txt to entries containing a token.")
    parser.add_argument("--input", type=Path, default=Path("example_data/meta.txt"))
    parser.add_argument("--output", type=Path, default=Path("example_data/meta_dmc_all.txt"))
    parser.add_argument("--token", type=str, default="DMC")
    args = parser.parse_args()

    lines = [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
    kept = [line for line in lines if args.token in line.split(",", 1)[1]]
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))
    print(f"wrote {len(kept)} lines to {args.output}")


if __name__ == "__main__":
    main()
