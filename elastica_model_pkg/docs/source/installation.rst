Installation
============

Requirements
------------

* Python 3.11+
* AUTO continuation software

Download the Auto-07p Installation Guide here:
:download:`Auto-07p installation steps for Windows <files/Auto-07p installation steps for WIndows.docx>`


Install from source
---------------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/Vamsi2496/elastica-model.git
   cd elastica_model_pkg
   py -3.11 -m pip install -e .
.. note::

   The ``.`` at the end of the command is required.

Configure paths
---------------

After installation, run the setup wizard to point to your
AUTO directory and Python 2.6 executable (one time setup):

.. code-block:: bash

   py -3.11 -m elastica_model.setup_config


