{{ fullname | escape | underline}}

{% set exclude_methods = ['add_traits', 'class_own_trait_events', 'class_own_traits', 'class_trait_names', 'class_traits',
                          'has_trait', 'hold_trait_notifications', 'notify_change', 'observe', 'on_trait_change',
                          'set_trait', 'setup_instance', 'trait_events', 'trait_metadata', 'trait_names', 'traits',
                          'unobserve', 'unobserve_all'] %}

{% set exclude_attributes = ['cross_validation_lock'] %}

{% if methods %}
{% set classmethods = methods|select('in', ['from_json', 'from_xarray', 'from_definition', 'points', 'grid'])|list %}
{% set methods = methods|reject('in', exclude_methods)|reject('in', classmethods) %}
{% endif %}

{% if attributes %}
{% set attributes = attributes|reject('in', exclude_attributes) %}
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block classmethods %}

   {% if classmethods %}
   .. rubric:: Class Methods

   .. autosummary::
   {% for item in classmethods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
