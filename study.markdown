---
layout: page
title: ~/study
permalink: /study/
---

<ul>
{% for post in site.categories.study %}
        <li>{{ post.title }}</li>
{% endfor %}
</ul>
