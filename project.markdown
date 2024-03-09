---
layout: page
title: ~/projects
permalink: /projects/
published: True
food: blue cheese
---
Hello, World!
<h1>{{ page.food }}</h1>

{{% for post in site.categories.projects %}
 <li>post.title</li>
{% endfor %}}