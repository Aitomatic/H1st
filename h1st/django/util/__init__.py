__all__: tuple[str] = (
    'PGSQL_IDENTIFIER_MAX_LEN',
    'LOGGER',
    'dir_path_with_slash',
    'enable_dict_io',
    'full_qual_name',
    'import_obj',
)


from functools import wraps
from importlib import import_module
from inspect import FullArgSpec, getfullargspec, isfunction, ismethod
from logging import Formatter, getLogger, Logger, StreamHandler, DEBUG
from sys import stdout
from typing import Any, Callable, TypeVar


PGSQL_IDENTIFIER_MAX_LEN: int = 63


STDOUT_HANDLER: StreamHandler = StreamHandler(stream=stdout)

STDOUT_HANDLER.setFormatter(
    fmt=Formatter(fmt='%(asctime)s   %(levelname)s   %(name)s:\n%(message)s\n',
                  datefmt='%Y-%m-%d %H:%M'))

LOGGER: Logger = getLogger(name=__name__)
LOGGER.setLevel(level=DEBUG)
LOGGER.addHandler(STDOUT_HANDLER)


def dir_path_with_slash(path: str) -> str:
    return path if path.endswith('/') else f'{path}/'


CallableTypeVar = TypeVar(name='CallableTypeVar', bound=Callable[..., Any],
                          covariant=False, contravariant=False)


def enable_dict_io(f: CallableTypeVar) -> CallableTypeVar:
    assert isfunction(f) and (not ismethod(f))

    arg_spec: FullArgSpec = getfullargspec(func=f)
    is_method_with_self: bool = (arg_spec.args[0] == 'self')

    @wraps(f)
    def decor_func_w_dict_io(*args, **kwargs):
        from ..modeling.models import Workflow

        if is_method_with_self:
            self: Any = args[0]

            if isinstance(self, Workflow):
                assert not kwargs, \
                    ValueError('*** H1ST GRAPHS DO NOT TAKE KWARGS ***')

                if (len(args) == 2) and isinstance(args[1], dict):
                    d: dict[str, Any] = args[1]

                else:
                    non_self_args: tuple[Any] = args[1:]
                    n_non_self_args: int = len(non_self_args)

                    assert n_non_self_args <= len(self.args), \
                        ValueError(
                            f'*** NO. OF PROVIDED ARGS {non_self_args}'
                            f'EXCEEDS NO. OF EXPECTED ARGS {self.args} ***')

                    d: dict[str, Any] = dict(zip(self.args[:n_non_self_args],
                                                 non_self_args))

                LOGGER.info(msg=f'{self} .__call__( {d} )')

                return f(self, d)

            if (len(args) == 2) and (not kwargs):
                d: Any = args[1]

                if isinstance(d, dict):
                    _d: dict[str, Any] = {k: v
                                          for k, v in d.items()
                                          if k in arg_spec.args}

                    LOGGER.info(f'{self} .__call__( ' +
                                ', '.join(f'{k}={v}' for k, v in _d.items()) +
                                ' )')

                    d['result'] = f(self, **_d)

                    return d

                LOGGER.info(msg=f'{self} .__call__( {d} )')

                return f(self, d)

            return f(*args, **kwargs)

        if (len(args) == 1) and (not kwargs):
            d: Any = args[0]

            if isinstance(d, dict):
                d['result'] = f(**{k: v
                                   for k, v in d.items()
                                   if k in arg_spec.args})
                return d

            return f(d)

        return f(*args, **kwargs)

    return decor_func_w_dict_io


def full_qual_name(obj: Any) -> str:
    return f'{obj.__module__}.{obj.__qualname__}'


def import_obj(module_and_qualname: str) -> Any:
    module_name, qualname = module_and_qualname.rsplit(sep='.', maxsplit=1)
    return getattr(import_module(module_name), qualname)
