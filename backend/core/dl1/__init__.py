"""Deep Learning 1.0 — automatic modelling, pattern discovery and reporting.

Module layout (kept deliberately separable so a future 2.0 can swap pieces):

    config.py    AutoModelConfig — dataset → full hyperparameter set
    signals.py   one heavy numeric pass → SignalBundle
    patterns.py  SignalBundle → narrated, scored patterns
    pipeline.py  stage orchestration
    report.py    PDF sections
"""
