---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.category.study %}
    <ul>
        <li>{{ post[1].title }}</li>
    </ul>
{% endfor %}
