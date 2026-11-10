Installation
============

The package is published on `PyPI <https://pypi.org/project/lime-stratified/>`_
as ``lime-stratified`` and requires Python 3.8 or newer.

.. code-block:: sh

   pip install lime-stratified

Install the latest development version from GitHub:

.. code-block:: sh

   pip install git+https://github.com/rashidrao-pk/lime_stratified.git

Or install it from a local clone:

.. code-block:: sh

   git clone https://github.com/rashidrao-pk/lime_stratified
   cd lime_stratified
   python -m pip install .

.. note::

   The distribution is called ``lime-stratified``, but the importable package
   is still ``lime``, the same as the original LIME. Do not install both
   ``lime`` and ``lime-stratified`` in the same environment: they would
   overwrite each other.

Development and documentation extras
------------------------------------

.. code-block:: sh

   python -m pip install -e ".[dev]"    # tests, linters
   python -m pip install -e ".[docs]"   # Sphinx + theme
   make html                            # builds into build/html
