# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.utils import deprecated_function


def test_deprecated_function():

    @deprecated_function('0.2.0', '0.3.0', 'toy instruction')
    def deprecated_demo(arg1: int, arg2: int) -> tuple:
        """This is a long summary. This is a long summary. This is a long
        summary. This is a long summary.

        Args:
            arg1 (int): Long description with a line break. Long description
                with a line break.
            arg2 (int): short description.

        Returns:
            Long description without a line break. Long description without
            a line break.
        """

        return arg1, arg2

    # MMLogger.get_instance('test_deprecated_function')
    deprecated_demo(1, 2)
    # out, _ = capsys.readouterr()
    # assert "'test_misc.deprecated_demo' is deprecated" in out
    assert (1, 2) == deprecated_demo(1, 2)

    expected_docstring = \
    """.. deprecated:: 0.2.0
    Deprecated and will be removed in version 0.3.0.
    Please toy instruction.


    This is a long summary. This is a long summary. This is a long
    summary. This is a long summary.

    Args:
        arg1 (int): Long description with a line break. Long description
            with a line break.
        arg2 (int): short description.

    Returns:
        Long description without a line break. Long description without
        a line break.
    """  # noqa: E122
    assert expected_docstring.strip(' ') == deprecated_demo.__doc__
    # MMLogger._instance_dict.clear()

    # Test with short summary without args.
    @deprecated_function('0.2.0', '0.3.0', 'toy instruction')
    def deprecated_demo1():
        """Short summary."""

    expected_docstring = \
    """.. deprecated:: 0.2.0
    Deprecated and will be removed in version 0.3.0.
    Please toy instruction.


    Short summary."""  # noqa: E122
    assert expected_docstring.strip(' ') == deprecated_demo1.__doc__
