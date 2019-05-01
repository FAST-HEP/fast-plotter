.. _ref-cli:

Command-line Usage
==================
The command-line tools are the primary way to use fast-carpenter and friends at this point.
All of the FAST commands provide built-in help by providing the ``--help`` option.

.. _ref-cli_fast_plotter:

``fast_plotter``
----------------
Take a list of fast-carpenter output binned dataframe tables and turns these into plots.

To configure how these plots are made, use either the command-line options, or
provide these in a `YAML <https://en.wikipedia.org/wiki/YAML>`_ configuration.
If an option is provided to both, then the command-line value will take
precedence.

.. command-output:: fast_plotter --help
