"""Human-readable failures for the intentionally small pipeline."""


class ExoSignalError(RuntimeError):
    """Base error that the command-line interface can present cleanly."""


class InvalidTicError(ExoSignalError):
    """The supplied identifier cannot be interpreted as a TIC ID."""


class NoTessDataError(ExoSignalError):
    """MAST has no matching SPOC TESS light-curve product for this target."""


class UnusableLightCurveError(ExoSignalError):
    """A product does not retain enough usable data after quality checks."""

