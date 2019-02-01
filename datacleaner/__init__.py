import pkg_resources


# the name of the project from the setup.py
project_name = 'datacleaner'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili (simo.marsili@gmail.com)'
__all__ = []

from datacleaner.cleaner import Cleaner
