---
layout: page
title: ~/projects
permalink: /projects/
published: True
food: potato
---
<h1>{page.food}</h1>

<ul>
  {% for post in site.posts %}
    <li>
      <a>{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

<ul>
  {% for post in site.posts %}
    <li>
      <a>{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>