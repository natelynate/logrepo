---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.category.study %}
    <ul>
        <li>{{ post.title }}</li>
    </ul>
{% endfor %}
