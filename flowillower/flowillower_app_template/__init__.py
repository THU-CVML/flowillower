import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "df",
    "indices",
    "num_points",
    "num_turns",
    "radius",
    "src",
    "streamlit_app",
    "theta",
    "x",
    "y",
]
