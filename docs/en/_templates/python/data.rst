{% if obj.display %}
.. py:{{ obj.type }}:: {{ obj.name }}
   {%+ if obj.value is not none or obj.annotation is not none -%}
   :annotation:
        {%- if obj.annotation %} :{{ obj.annotation }}
        {%- endif %}
        {%- if obj.value is not none %} = {%
            if obj.value is string and obj.value.splitlines()|count > 1 -%}
                Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text
        :linenos:

        {{ obj.value|indent(width=8) }}

    .. raw:: html

        </details>

            {%- else -%}
                {{ obj.value|string|truncate(100) }}
            {%- endif %}
        {%- endif %}
    {% endif %}


   {{ obj.docstring|indent(3) }}
{% endif %}
