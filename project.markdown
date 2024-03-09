---
layout: page
title: ~/projects
permalink: /projects/
published: True
food: potato
---
Hello, World!
<h1>{{ page.food }}</h1>

{{% for post in site.category.projects %}}
 <li>post.title</li>
{{% endfor %}}