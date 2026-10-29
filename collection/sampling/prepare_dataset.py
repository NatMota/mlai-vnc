"""Prepare existing compressed episodes; does not collect or train a model."""
import argparse
from pathlib import Path
from dataset import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Directory containing episodes.jsonl')
    args = parser.parse_args()
    audit, rows = prepare(args.output)
    print(f'{len(audit)} audited executions; {len(rows)} prepared 30 Hz rows')
