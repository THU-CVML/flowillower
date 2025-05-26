__version__ = "0.0.1"
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "author_name_en_us",
    "author_name_zh_cn",
    "core",
    "foo",
    "github_repo",
    "github_user",
    "help",
    "import_name",
    "lib_name",
    "lib_name_en_us",
    "lib_name_zh_cn",
    "lib_paths",
    "nucleus",
    "pretty_name",
    "upgrade_command_pip",
]
