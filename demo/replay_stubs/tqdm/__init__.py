"""Minimal stand-in for tqdm (predict.py imports it; the demo never draws a progress bar)."""


def tqdm(iterable=None, *args, **kwargs):
    return iterable
