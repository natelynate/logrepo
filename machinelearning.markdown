---
layout: page
title: ~/machinelearning
permalink: /machinelearning/
published: True

---
Hello, machine learning

{{% for post in site.category.projects %}}
 <li>post.title</li>
{{% endfor %}}


{{% for post in site.category.machine_learning %}}
 <li>post.title</li>
{% endfor %}