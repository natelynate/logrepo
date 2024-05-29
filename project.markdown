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

<h4><b>Appearance-Based Gaze Tracking Analysis</b></h4>
Feb 2024 - Ongoing  
<font color=blue> #Gaze-Estimation #Web-service #Digital-biomarker #Medical-Screening #Market-Research #Neuromarketing </font><br>  
The following project aims to create a web-based AI service that aids users to conveniently create and distribute online cognitive tests for recording recipient’s gaze movement in response to image stimuli. It aims to record and analyze latent metrics from gaze movement patterns, to meet needs from various medical and marketing domains, from medical screening of depression disorders to neuromarketing-inspired consumer research. The service provides a detailed analytical report of observed average gaze patterns and indicators which could be used for business decision making. 
<br>
<br>
<font color=blue> #시선추적 #웹서비스 #디지털 바이오마커 #사전의료진단 #시장조사 #뉴로마케팅</font><br>  
해당 프로젝트는 시각 자극에 대한 피험자의 시선 움직임 데이터를 수집할 수 있는 온라인 인지 테스트를 손쉽게 생성하고 배포할 수 있는 웹 기반 AI 서비스입니다. 본 프로젝트는 우울증 사전 진단이나 소비자 분석 등, 의료 분야부터 뉴로마케팅 등 포괄적인 영역에서의 필요를 충족시키기 위해 시선 움직임 데이터에서 다양한 잠재적인 지표를 기록하고 분석합니다. 본 서비스는 비즈니스 의사결정에 활용될 수 있도록 관측된 평균적인 시선 패턴과 다양한 관측 지표를 포함하는 세부 분석 리포트를 제공합니다. 
<br> 
<h4><b>개발 일지</b><h4>
<ul>
{% for post in site.categories.projects %}
  {% if post.tags contains "gamcheugi" %}
    <li class="custom-font-size"><a href="{{ post.url | prepend: site.baseurl }}">{{ post.date | date:"%m-%d-%Y" }} || {{ post.title }}</a></li>
  {% endif %}
{% endfor %}
</ul>

