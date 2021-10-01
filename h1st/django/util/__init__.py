__all__ = 'PGSQL_IDENTIFIER_MAX_LEN', 'dir_path_with_slash', 'import_obj'


from importlib import import_module


PGSQL_IDENTIFIER_MAX_LEN = 63


def dir_path_with_slash(path: str) -> str:
    return path if path.endswith('/') else f'{path}/'


def import_obj(module_and_qualname: str):
    module_name, qualname = module_and_qualname.rsplit(sep='.', maxsplit=1)
    return getattr(import_module(module_name), qualname)
