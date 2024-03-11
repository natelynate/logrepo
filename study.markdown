---
layout: page
title: ~/study
permalink: /study/
---

<h1>Machine Learning</h1>
<ul>
{% for post in site.categories.study %}
    {% if post.tags contain "machinelearning" %}
        <li><a href="{{ post.url | prepend: site.baseurl}}"></a>{{ post.title }}</li>
    {% endif %}
{% endfor %}
</ul>

<h1>Coding Test</h1>
<ul>
{% for post in site.categories.study %}
    {% if post.tags contain "codingtest" %}
        <li><a href="{{ post.url | prepend: site.baseurl}}"></a>{{ post.title }}</li>
    {% endif %}
{% endfor %}
</ul>