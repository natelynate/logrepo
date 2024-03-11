---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.categories.study %}
    <ul>
        <li>{{ post.title }}</li>
    </ul>
{% endfor %}
