__all__ = (
    'PGSQL_IDENTIFIER_MAX_LEN',
    'dir_path_with_slash',
    'enable_dict_io',
    'import_obj',
)


from functools import wraps
from importlib import import_module
from inspect import getfullargspec, isfunction, ismethod
from typing import Any, Callable, TypeVar


PGSQL_IDENTIFIER_MAX_LEN: int = 63


def dir_path_with_slash(path: str) -> str:
    return path if path.endswith('/') else f'{path}/'


CallableTypeVar = TypeVar('CallableTypeVar', bound=Callable[..., Any])


def enable_dict_io(f: CallableTypeVar) -> CallableTypeVar:
    assert isfunction(f) and (not ismethod(f))

    arg_spec = getfullargspec(func=f)
    is_method_with_self = (arg_spec.args[0] == 'self')

    @wraps(f)
    def decor_func_w_dict_io(*args, **kwargs):
        if is_method_with_self and (len(args) == 2) and (not kwargs):
            self, d = args

            assert isinstance(d, dict), \
                TypeError(f'*** ONLY POSITIONAL ARG {d} NOT A DICT ***')

            d['result'] = f(self, **{k: v for k, v in d.items()
                                     if k in arg_spec.args})

            return d

        elif (len(args) == 1) and (not kwargs):
            d = args[0]

            assert isinstance(d, dict), \
                TypeError(f'*** ONLY POSITIONAL ARG {d} NOT A DICT ***')

            d['result'] = f(**{k: v for k, v in d.items()
                               if k in arg_spec.args})

            return d

        else:
            return f(*args, **kwargs)

    return decor_func_w_dict_io


def import_obj(module_and_qualname: str):
    module_name, qualname = module_and_qualname.rsplit(sep='.', maxsplit=1)
    return getattr(import_module(module_name), qualname)
