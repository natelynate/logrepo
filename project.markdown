---
layout: page
title: ~/projects
permalink: /projects/
food: potato
---
<h1>{{page.food}}</h1>

{% for category in site.categories %}
  <h3>{{ category[0] }}</h3>
  <ul>
    {% for post in categories[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}