{%- if obj.display %}
.. py:property:: {{ obj.short_name }}
   {% if obj.annotation %}
   :type: {{ obj.annotation }}
   {% endif %}
   {% if obj.properties %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}
   {% endif %}

   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
