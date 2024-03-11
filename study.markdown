---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.category.study %}
    <li>{{post.title}}</li>
{% endfor %}
