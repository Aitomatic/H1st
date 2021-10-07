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
        from ..model.models import Workflow

        if is_method_with_self:
            self = args[0]

            if isinstance(self, Workflow):
                assert not kwargs, \
                    ValueError('*** H1ST GRAPHS DO NOT TAKE KWARGS ***')

                if (len(args) == 2) and isinstance(args[1], dict):
                    d = args[1]

                else:
                    non_self_args = args[1:]
                    n_non_self_args = len(non_self_args)

                    assert n_non_self_args <= len(self.args), \
                        ValueError(
                            f'*** NO. OF PROVIDED ARGS {non_self_args}'
                            f'EXCEEDS NO. OF EXPECTED ARGS {self.args} ***')

                    d = dict(zip(self.args[:n_non_self_args], non_self_args))

                print(f'*** {self}({d}) ***')
                return f(self, d)

            elif (len(args) == 2) and (not kwargs):
                d = args[1]

                if isinstance(d, dict):
                    d['result'] = f(self, **{k: v for k, v in d.items()
                                             if k in arg_spec.args})
                    return d

                else:
                    return f(self, d)

            else:
                return f(*args, **kwargs)

        elif (len(args) == 1) and (not kwargs):
            d = args[0]

            if isinstance(d, dict):
                d['result'] = f(**{k: v for k, v in d.items()
                                   if k in arg_spec.args})
                return d

            else:
                return f(d)

        else:
            return f(*args, **kwargs)

    return decor_func_w_dict_io


def import_obj(module_and_qualname: str):
    module_name, qualname = module_and_qualname.rsplit(sep='.', maxsplit=1)
    return getattr(import_module(module_name), qualname)
