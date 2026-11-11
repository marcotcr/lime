API reference
=============

The stratified sampling lives in :mod:`lime.lime_image` (sampling and
adjustment factors) and :mod:`lime.lime_base` (weighted regression). The other
modules are inherited unchanged from the original LIME.

Key entry points
----------------

* :class:`lime.lime_image.LimeImageExplainer`: see the
  ``use_stratification`` argument of
  :meth:`~lime.lime_image.LimeImageExplainer.explain_instance`.
* :meth:`lime.lime_image.LimeImageExplainer.data_labels`: generates the
  synthetic neighborhood and the adjustment factors.
* :meth:`lime.lime_base.LimeBase.explain_instance_with_data`: applies the
  adjustment factors to the kernel weights and fits the surrogate model.

.. toctree::
   :maxdepth: 3

   lime
