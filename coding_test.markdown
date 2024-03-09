---
layout: page
title: ~/coding_test
permalink: /codingtest/
---

{post.title}

{% for post in site.coding_test %}
 <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}