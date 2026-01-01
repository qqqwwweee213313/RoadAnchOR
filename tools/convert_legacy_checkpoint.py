#!/usr/bin/env python3
"""Strip the training metadata from a checkpoint before redistributing it.

The ``meta`` block of an mmcv checkpoint embeds the full training config and the
absolute paths of the machine that produced it, and ``optimizer`` carries the
optimiser state that a released checkpoint does not need.  ``--strip-meta``
drops both; the ``state_dict`` is passed through untouched.

Usage:
    python tools/convert_legacy_checkpoint.py old.pth new.pth --strip-meta
    python tools/convert_legacy_checkpoint.py --self-test
"""
import argparse
import sys

STRIPPED_KEYS = ('meta', 'optimizer')


def convert_checkpoint(ckpt, strip_meta=False):
    """Return (new_ckpt, n_dropped). Order is preserved."""
    if not (strip_meta and isinstance(ckpt, dict)):
        return ckpt, 0
    out = type(ckpt)((k, v) for k, v in ckpt.items() if k not in STRIPPED_KEYS)
    return out, len(ckpt) - len(out)


def self_test():
    sd = {
        'img_backbone.conv1.weight': 1,
        'pts_bbox_head.ra_planning_decoder.loc_head.0.weight': 2,
        'pts_bbox_head.transformer.level_embeds': 3,
    }

    ckpt = {'state_dict': dict(sd), 'meta': {'x': 1}, 'optimizer': {'y': 2}}
    out, n = convert_checkpoint(ckpt, strip_meta=True)
    assert out['state_dict'] == sd, out['state_dict']
    assert 'meta' not in out and 'optimizer' not in out
    assert n == 2, n

    # Without --strip-meta the checkpoint must come back untouched.
    out, n = convert_checkpoint(ckpt, strip_meta=False)
    assert out == ckpt and n == 0

    # A checkpoint that carries no metadata is already clean.
    bare = {'state_dict': dict(sd)}
    out, n = convert_checkpoint(bare, strip_meta=True)
    assert out == bare and n == 0

    # A bare state_dict (no 'state_dict' wrapper) has nothing to strip either.
    out, n = convert_checkpoint(dict(sd), strip_meta=True)
    assert out == sd and n == 0

    print('self-test PASS (4 cases)')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='?', help='input checkpoint (.pth)')
    ap.add_argument('dst', nargs='?', help='output checkpoint (.pth)')
    ap.add_argument('--strip-meta', action='store_true',
                    help="drop 'meta' and 'optimizer' from the checkpoint")
    ap.add_argument('--self-test', action='store_true', help='run the unit tests and exit')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.src or not args.dst:
        ap.error('src and dst are required unless --self-test is given')

    import torch
    ckpt = torch.load(args.src, map_location='cpu')
    out, n = convert_checkpoint(ckpt, strip_meta=args.strip_meta)
    torch.save(out, args.dst)
    total = len(out['state_dict']) if isinstance(out, dict) and 'state_dict' in out else len(out)
    print(f'{args.src} -> {args.dst}: {total} keys'
          + (f' ({n} metadata block(s) removed)' if args.strip_meta else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
