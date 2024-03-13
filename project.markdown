---
layout: page
title: ~/projects
permalink: /projects/
---
<h2>개발 일지<h2>
<h3>Predicting Consumer Score through Webcam-based GazeTracking</h3>
<ul>
{% for post in site.categories.projects %}
  {% if post.tags contains "gamcheugi" %}
    <li><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></li>
  {% endif %}
{% endfor %}
</ul>

