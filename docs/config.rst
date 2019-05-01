
.. _ref-plotting_config:

The Plotting Config
===================
Use the config file to fine-tune your plots.

Valid options:

	+------------------+--------------+--------------------------------------------------------------------------+
	| Option           | Default      | Description                                                              |
	+==================+==============+==========================================================================+
	| ``data``         | ``"data"``   | | Regular expression for values in dataset_col to                        |
	|                  |              | | plot as scatter markers                                                |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``weights``      | ``None``     |  Which weight columns to use for plotting.                               |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``yscale``       | ``"linear"`` | Should y-axis be on a ``linear`` or a ``log`` scale.                     |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``lumi``         | ``1``        | Multiply all simulated datasets by this value.                           |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``ylabel``       |              | Give the Y-axis a title.                                                 |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``legend``       |              | | Control the legend placement options.  A dictionary of                 |
	|                  |              | | kwarg pairs passed directly to :py:func:`matplotlib.pyplot.legend`     |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``limits``       |              | | Set the axis ranges.  Takes a dictionary with ``x`` and / or ``y`` as  |
	|                  |              | | the keys, and a list with upper and lower bounds.                      |
	+------------------+--------------+--------------------------------------------------------------------------+
	| ``annotations``  |              | | Add text to the plot.  Should be a list of text labels, each described |
	|                  |              | | by a dictionary containing the ``text`` and the ``position``.  All     |
	|                  |              | | other paramers are passed as keyword-argument pairs to                 |
	|                  |              | | :py:func:`matplotlib.pyplot.annotate`                                  |
	+------------------+--------------+--------------------------------------------------------------------------+

.. todo::
   Describe the ``bin_variable_replacements`` and ``value_replacements`` options for the config.

.. seealso::
  An example of a plotting config cms_public_tutorial demo repository:
  https://gitlab.cern.ch/fast-hep/public/fast_cms_public_tutorial/blob/master/plot_config.yml

