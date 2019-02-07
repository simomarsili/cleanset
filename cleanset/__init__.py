import pkg_resources
from cleanset.cleaner import Cleaner, clean

# the name of the project from the setup.py
project_name = 'cleanset'
__version__ = pkg_resources.require(project_name)[0].version
__copyright__ = 'Copyright (C) 2019 Simone Marsili'
__license__ = 'BSD 3 clause'
__author__ = 'Simone Marsili (simo.marsili@gmail.com)'
__all__ = []
