---
layout: post
title:  "EAR를 통한 Eye Closure Detection"
date:   2024-04-13 01:15:16 +0900
categories: projects
tags: gamcheugi computervision
---

Eye Aspect Ratio는 눈의 가로 세로 비율을 말한다. 눈이 감기면 가로 방향(x)의 비율은 변함이 없지만, 세로 방향의 비율만 감소하므로 가로 세로 비율의 변화를 관찰하면 눈을 감았는지 여부를 정량적으로 판단할 수 있다. 

<h4>EAR Formula</h4>  
![alt text]({{"/assets/images/2024-04-13-EyeClosureDetection/0.PNG" | relative_url}})  


![alt text]({{"/assets/images/2024-04-13-EyeClosureDetection/1.png" | relative_url}})  

공식 및 예시 이미지 출처: <a href="https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706">Link</a>

<h4>Samples</h4>  
![alt text]({{"/assets/images/2024-04-13-EyeClosureDetection/2.png" | relative_url}})  

관찰해보면 통상적으로 눈을 뜨고 있는 경우 0.2~0.3 정도의 EAR이 나오고, 눈을 완전히 감은 경우 0.1이 나온다. 
따라서 이전 샘플 추출 스크립트 제작 글에서도 얘기했듯이, 0.15 정도를 임계값으로 사용하였다. 