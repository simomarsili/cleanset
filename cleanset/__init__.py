# -*- coding: utf-8 -*-
# Copyright (C) 2019 Simone Marsili
# All rights reserved.
# License: BSD 3 clause
"""
========
cleanset
========
**cleanset** is a Python package for the educated removal of invalid/undesired
entries from data matrices.
"""
import pkg_resources
from cleanset.cleaner import Cleaner, clean

project_name = 'cleanset'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili (simo.marsili@gmail.com)'
__all__ = [Cleaner, clean]
