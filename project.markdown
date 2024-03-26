---
layout: page
title: ~/projects
permalink: /projects/
---
<style>
  .custom-font-size {
    font-size: 14px;
  }
</style>
<br>
<br>
<br>
<h5>현재 진행중이거나 과거에 진행했던 프로젝트에 대해서 정리한 페이지입니다.<h4>

<h4><b>Predicting Consumer Score through Webcam-based GazeTracking</b></h4>
Feb 2024 - Ongoing
<font color=blue> #Gaze-Estimation #Webservice #Neuromarketing #Market-Research </font>

The following project aims to create a web-based service that aids users to conveniently create and share cognitive tests for recording recipient's latent biological responses in response to sample product images, using built-in test generation protocols and webcam-based eyetracking algorithm. The service provides detailed analytical report of observed average gaze patterns and indicators which could be used for business decision making.

<font color=blue> #시선추적 #웹서비스 #뉴로마케팅 #사전시장조사 </font>   
해당 프로젝트는 자체적인 테스트 생성 절차와 웹캠 기반 시선추적 알고리즘을 통해 샘플 시안 이미지에 대한 피험자의 잠재적 생체 반응을 측정하는 인지 테스트를 간편하게 생성 및 공유할 수 있는 웹 기반 서비스입니다. 본 서비스는 비즈니스 의사결정에 활용될 수 있도록 관측된 평균적인 시선 패턴과 다양한 관측 지표를 포함하는 세부 분석 리포트를 제공합니다. 
<br> 
<h4><b>개발 일지</b><h4>
<ul>
{% for post in site.categories.projects %}
  {% if post.tags contains "gamcheugi" %}
    <li class="custom-font-size"><a href="{{ post.url | prepend: site.baseurl }}">{{ post.date | date:"%m-%d-%Y" }} || {{ post.title }}</a></li>
  {% endif %}
{% endfor %}
</ul>

