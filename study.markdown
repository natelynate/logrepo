---
layout: page
title: ~/study
permalink: /study/
---

{% for post in site.posts %}
    <ul>
        {% for category in post.category %}
        <li>{{post.category}}/{{post.title}}/{{post.tag}}</li>
        {% if post.category == projects %}
            Hello, World!
        {% endif %}
        {% endfor %}
    </ul>

{% endfor %}
