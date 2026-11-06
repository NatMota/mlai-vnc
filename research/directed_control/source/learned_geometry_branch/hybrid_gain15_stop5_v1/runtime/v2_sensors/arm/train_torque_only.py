"""Train a torque-only arm; preserve the independent STOP checkpoint exactly.

Uses the existing full-episode loader, optimizer and export checks. Recovery
training may retrain driving; baseline driving remains frozen by default.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import train_golden as tg
import train_full_episode as tf
from pilot_torque_only import TorqueOnly, objective, infer_export


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frozen-stop', type=Path, required=True)
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original = (tg.ArmPolicy, tg.objective, tg.infer_export)
    original_train = tg.train_one

    def train(records, kind, seed, args):
        if kind == 'stop':
            for suffix in ('.npz', '_history.json', '_summary.json'):
                src = own.frozen_stop / f'stop_seed{seed}{suffix}'
                shutil.copy2(src, args.out / src.name)
            weights = dict(np.load(own.frozen_stop / f'stop_seed{seed}.npz'))
            meta = json.loads((own.frozen_stop / f'stop_seed{seed}_summary.json').read_text())
            meta.update(frozen=True, frozen_source=str(own.frozen_stop),
                        frozen_sha256=tg.sha(own.frozen_stop / f'stop_seed{seed}.npz'))
            tg.write_json(args.out / f'stop_seed{seed}_summary.json', meta)
            return weights, meta
        tg.ArmPolicy, tg.objective, tg.infer_export = ((TorqueOnly, objective, infer_export)
                                                     if kind == 'torque' else original)
        weights, meta = original_train(records, kind, seed, args)
        if kind == 'torque':
            assert not any(k.startswith('stop.') for k in weights)
            weights['info_auxiliary_loss_weight'] = np.float32(0)
            meta.update(loss_terms=['standardized_torque_mse'], auxiliary_loss_weight=0.,
                        architecture=f'18->{args.arm_hidden}->{args.arm_hidden}->7')
            np.savez(args.out / f'torque_seed{seed}.npz', **weights)
            tg.write_json(args.out / f'torque_seed{seed}_summary.json', meta)
        tg.ArmPolicy, tg.objective, tg.infer_export = original
        return weights, meta

    tf.train_one = train
    tf.main()


if __name__ == '__main__':
    main()
