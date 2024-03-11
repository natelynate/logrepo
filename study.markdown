---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.posts %}
    {% if post.category == "projects" %}
        <li>{{post.title}}</li>
    {% endif %}
{% endfor %}
