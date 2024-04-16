---
layout: post
title:  Chatterjee's ξ Correlation Coefficient
date:   2024-04-16 13:15:16 +0900
categories: study
tags: statistics dataanalysis
---
<link rel="stylesheet" href="{{ '/assets/styles/styles.css' | relative_url }}">


The three most influential correlation coefficients are:

1. Pearson’s correlation coefficient 
2. Spearman’s ρ
3. Kendall’s τ
.
.
.
.
<h4><b>Pearson’s Correlation Coefficient (r): <b><h4>  

<font color='green'>Type of Data:</font> Quantitative.
<br>
<font color='green'>Nature of Association:</font> Linear.  
<br>
<font color='green'>Calculation:</font> Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. It is calculated as the covariance of the variables divided by the product of their standard deviations.  
<br>
<font color='green'>Value Range:</font> The coefficient values range from -1 to 1. A value of 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.  
<br>
<font color='green'>Limitation:</font> Pearson's coefficient only captures linear relationships and can be heavily influenced by outliers.  
<hr style="border: none; border-top: 1px solid #000;">

<h4><b>Spearman’s Rank Correlation Coefficient (ρ, rho):<b><h4>   
<font color='green'>Type of Data:</font> Ordinal or non-normally distributed interval data.  
<br>
<font color='green'>Nature of Association:</font> Monotonic.  
<br>
<font color='green'>Calculation:</font> Spearman’s rho is a measure of the monotonic relationship between two variables. It assesses how well the relationship between two variables can be described using a monotonic function. To calculate it, each value is replaced by its rank, and Pearson's formula is then applied to these ranks.  
<br>
<font color='green'>Value Range:</font> Like Pearson’s, Spearman’s coefficient ranges from -1 to 1, with -1, 1, and 0 having similar interpretations but for monotonic, rather than strictly linear, relationships.  
<br>
<font color='green'>Advantage:</font> It is less sensitive to outliers than Pearson’s coefficient and can be used with non-parametric data.
<br>
<br>
<hr style="border: none; border-top: 1px solid #000;">
<h4><b>Kendall’s Tau Coefficient (τ, tau):<b><h4> 

<font color='green'>Type of Data:</font> Ordinal.  
<br>
<font color='green'>Nature of Association:</font> Monotonic.  
<br>
<font color='green'>Calculation:</font> Kendall’s tau measures the strength of the monotonic relationship by comparing the number of concordant pairs to the number of discordant pairs in the data set. A pair of observations is concordant if the ranks for both elements agree (i.e., both ranks are either higher or lower than the other pair’s ranks).  
<br>
<font color='green'>Value Range:</font> It ranges from -1 (perfect negative association) to 1 (perfect positive association). A value of 0 indicates the absence of association.  
<br>
<font color='green'>Advantage:</font> It is often used when the data set is small, as it is a more robust measure of correlation in the presence of outliers and errors than Pearson’s r.
<br>
<br>
<br>
<hr style="border: none; border-top: 1px solid #000;">

<h4><b>4. Chatterjee's ξ</b></h4> 

Chatterjee's correlation coefficient aims to measure the strength of relationship that exists between $X$ and $Y$ even when 
the relationship is <font color='red'> not monotonic </font>, and <font color='red'> non-linear </font>.

The generalized formulas are as follows:

<div id='mathjax'>
\[
    \xi(X, Y) = 1 - \frac{3\Sigma^{n-1}_{i-1}|r_{i+1}-r_i}{n^2 - 1}  
\]
</div>
<br>
<br>

<div id='mathjax'>
\[
    \xi(X, Y) = 1 - \frac{n\Sigma^{n-1}_{i-1}|r_{i+1}-r_i}{2\Sigma^{n}_{i=1}l_i(n-l_i)}
\]
</div>
<br>
<br>
<br>
<br>
<br>
<br>

<h5>Bibliography:<h5>

<a href="https://towardsdatascience.com/a-new-coefficient-of-correlation-64ae4f260310">Medium: New Coefficient of Correlation</a>  
<br>
<a href="https://arxiv.org/pdf/1909.10140.pdf">ORIGINAL PAPER</a>  
<br>
<a href="https://souravchatterjee.su.domains">ABOUT AUTHOR</a>  

