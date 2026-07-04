"""NovaVision: emotion-controllable text-to-image generation and an affect-recovery benchmark.

Public API. The lightweight pipeline is imported eagerly; the benchmark harness
(``run_experiment``) is exposed lazily so ``import novavision`` never pulls the
heavy eval stack until you actually score a system.

    from novavision import build_pipeline
    result = build_pipeline().auto_run("a quiet joy settled over the morning")

    import novavision
    novavision.run_experiment(backend="diffusers", contents=20, seeds=3, out="results/run")
"""

from novavision.pipeline import NovaVision, Result, build_pipeline

__version__ = "1.0.0"
__all__ = ["NovaVision", "Result", "build_pipeline", "run_experiment", "__version__"]


def __getattr__(name: str):
    # PEP 562: keep the benchmark harness lazy so the import stays cheap.
    if name == "run_experiment":
        from novavision.experiments.run import run_experiment

        return run_experiment
    raise AttributeError(f"module 'novavision' has no attribute {name!r}")
