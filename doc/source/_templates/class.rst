{{ fullname | escape | underline}}

{% set exclude_methods = ['add_traits', 'class_own_trait_events', 'class_own_traits', 'class_trait_names', 'class_traits',
                          'has_trait', 'hold_trait_notifications', 'notify_change', 'observe', 'on_trait_change',
                          'set_trait', 'setup_instance', 'trait_events', 'trait_metadata', 'trait_names', 'traits',
                          'unobserve', 'unobserve_all'] %}

{% set exclude_attributes = ['cross_validation_lock'] %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}


   ----------------

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      {% if item not in exclude_methods %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      {% if item not in exclude_attributes %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
