{%- if obj.display %}
.. py:method:: {{ obj.short_name }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}

{% for (args, return_annotation) in obj.overloads %}
            {{ obj.short_name }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}

{% endfor %}
   {% if obj.properties %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}

   {% else %}

   {% endif %}
   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
