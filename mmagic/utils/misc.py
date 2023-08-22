# Copyright (c) OpenMMLab. All rights reserved.
import functools
import logging
import re
import textwrap
from typing import Callable

from mmcv import print_log


def deprecated_function(since: str, removed_in: str,
                        instructions: str) -> Callable:
    """Marks functions as deprecated.

    Throw a warning when a deprecated function is called, and add a note in the
    docstring. Modified from https://github.com/pytorch/pytorch/blob/master/torch/onnx/_deprecation.py
    Args:
        since (str): The version when the function was first deprecated.
        removed_in (str): The version when the function will be removed.
        instructions (str): The action users should take.
    Returns:
        Callable: A new function, which will be deprecated soon.
    """  # noqa: E501

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            print_log(
                f"'{function.__module__}.{function.__name__}' "
                f'is deprecated in version {since} and will be '
                f'removed in version {removed_in}. Please {instructions}.',
                # logger='current',
                level=logging.WARNING,
            )
            return function(*args, **kwargs)

        indent = '    '
        # Add a deprecation note to the docstring.
        docstring = function.__doc__ or ''
        # Add a note to the docstring.
        deprecation_note = textwrap.dedent(f"""\
            .. deprecated:: {since}
                Deprecated and will be removed in version {removed_in}.
                Please {instructions}.
            """)
        # Split docstring at first occurrence of newline
        pattern = '\n\n'
        summary_and_body = re.split(pattern, docstring, 1)

        if len(summary_and_body) > 1:
            summary, body = summary_and_body
            body = textwrap.indent(textwrap.dedent(body), indent)
            summary = '\n'.join(
                [textwrap.dedent(string) for string in summary.split('\n')])
            summary = textwrap.indent(summary, prefix=indent)
            # Dedent the body. We cannot do this with the presence of the
            # summary because the body contains leading whitespaces when the
            # summary does not.
            new_docstring_parts = [
                deprecation_note, '\n\n', summary, '\n\n', body
            ]
        else:
            summary = summary_and_body[0]
            summary = '\n'.join(
                [textwrap.dedent(string) for string in summary.split('\n')])
            summary = textwrap.indent(summary, prefix=indent)
            new_docstring_parts = [deprecation_note, '\n\n', summary]

        wrapper.__doc__ = ''.join(new_docstring_parts)

        return wrapper

    return decorator
