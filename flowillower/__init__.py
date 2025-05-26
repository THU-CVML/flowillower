__version__ = "0.0.1"
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ["core", "foo"]
