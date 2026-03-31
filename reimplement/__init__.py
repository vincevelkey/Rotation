__all__ = [
    "CourvillePrior",
    "ParallelTemperingRJMCMC",
    "PosteriorSample",
    "RJMCMCSigmoidBeliefNetwork",
    "SBNState",
    "STIMULUS_NAMES",
    "conditioning_query",
    "generate_second_order_conditioning",
]


def __getattr__(name):
    if name in {"CourvillePrior", "ParallelTemperingRJMCMC", "PosteriorSample", "RJMCMCSigmoidBeliefNetwork", "SBNState"}:
        from .courville_sbn import (
            CourvillePrior,
            ParallelTemperingRJMCMC,
            PosteriorSample,
            RJMCMCSigmoidBeliefNetwork,
            SBNState,
        )

        return {
            "CourvillePrior": CourvillePrior,
            "ParallelTemperingRJMCMC": ParallelTemperingRJMCMC,
            "PosteriorSample": PosteriorSample,
            "RJMCMCSigmoidBeliefNetwork": RJMCMCSigmoidBeliefNetwork,
            "SBNState": SBNState,
        }[name]
    if name in {"STIMULUS_NAMES", "conditioning_query", "generate_second_order_conditioning"}:
        from .datasets import STIMULUS_NAMES, conditioning_query, generate_second_order_conditioning

        return {
            "STIMULUS_NAMES": STIMULUS_NAMES,
            "conditioning_query": conditioning_query,
            "generate_second_order_conditioning": generate_second_order_conditioning,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
