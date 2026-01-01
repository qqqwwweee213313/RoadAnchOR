#!/usr/bin/env python3
"""Assert that the axis flags reach the *built* head instance.

``DETRHead.__init__`` accepts ``**kwargs`` and silently drops anything it does
not recognise, so a typo in a config key does not raise -- it just trains a
different model with the default value.  All three flags of this axis are
non-default, so a single missing key changes the experiment.

This script builds the model from a config and reads the attributes back off
the head object, which is the only thing that proves the value actually took
effect.

    python tools/check_config_flags.py adzoo/vad/configs/roadanchor/roadanchor_b2d.py

Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import sys

# Expected value -> (attribute on the head, config key, default when absent)
EXPECTED = {
    'score_target_use_cfar': (False, True),
    'plan_traj_cumsum': (True, False),
    'plan_col_v7': (True, False),
}

EXPECTED_LOSS = 'PlanCollisionLossV7'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='training config to check')
    ap.add_argument('--build', action='store_true', default=True,
                    help='build the model (default); the flags are read off the head instance')
    args = ap.parse_args()

    from mmcv import Config
    from mmcv.models import build_model

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_model(cfg.model,
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    head = model.pts_bbox_head

    ok = True
    print(f'config : {args.config}')
    print(f'model  : {type(model).__name__}')
    print(f'head   : {type(head).__name__}')
    print('-' * 68)
    for name, (want, default) in EXPECTED.items():
        if not hasattr(head, name):
            print(f'FAIL  {name}: the head has no such attribute')
            ok = False
            continue
        got = getattr(head, name)
        mark = 'ok  ' if bool(got) == bool(want) else 'FAIL'
        if bool(got) != bool(want):
            ok = False
        note = '' if bool(got) != bool(default) else '  <-- equals the default, flag had no effect'
        print(f'{mark}  {name}: want {want}, head reports {got} (default {default}){note}')

    loss_name = type(head.loss_plan_col).__name__
    if loss_name == EXPECTED_LOSS:
        print(f'ok    loss_plan_col: {loss_name}')
    else:
        print(f'FAIL  loss_plan_col: want {EXPECTED_LOSS}, built {loss_name}')
        ok = False

    print('-' * 68)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
