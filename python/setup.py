#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# This was borrowed heavily form https://github.com/RUrlus/diptest/
import sys

import pybind11
from setuptools import find_packages
from skbuild import setup

import io
import re
from os.path import dirname
from os.path import join

PACKAGE_NAME = 'l0learn'

MAJOR = 0
MINOR = 4
MICRO = 2
DEVELOPMENT = False

# note: also update README.rst

VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
FULL_VERSION = VERSION
if DEVELOPMENT:
    FULL_VERSION += '.dev'


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


def write_version_py(filename: str = 'src/l0learn/version.py') -> None:
    """Write package version to version.py.
    This will ensure that the version in version.py is in sync with us.
    Parameters
    ----------
    filename : str
        the path the file to write the version.py
    """
    # Do not modify the indentation of version_str!
    version_str = """\"\"\"THIS FILE IS AUTO-GENERATED BY diptest SETUP.PY.\"\"\"
name = '{name!s}'
version = '{version!s}'
full_version = '{full_version!s}'
release = {is_release!s}
"""

    with open(filename, 'w') as version_file:
        version_file.write(
            version_str.format(name=PACKAGE_NAME.lower(),
                               version=VERSION,
                               full_version=FULL_VERSION,
                               is_release=not DEVELOPMENT)
        )


if __name__ == '__main__':
    write_version_py()

    setup(
        name=PACKAGE_NAME,
        version=FULL_VERSION,
        package_dir={"": "src"},
        long_description="%s\n%s"
                         % (
                             re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
                                 "", read("README.md")
                             ),
                             re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.md")),
                         ),
        packages=find_packages(),
        setup_requires=["setuptools",
                        "wheel",
                        "scikit-build",
                        "cmake",
                        "ninja"],
        cmake_source_dir="src",
        cmake_args=[
                f"-DL0LEARN_VERSION_INFO:STRING={VERSION}",
                f"-DPython3_EXECUTABLE:STRING={sys.executable}",
        ]
    )
