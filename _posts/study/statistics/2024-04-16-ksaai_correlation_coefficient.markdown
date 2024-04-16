---
layout: post
title:  Chatterjee's ξ Correlation Coefficient
date:   2024-04-14 13:15:16 +0900
categories: study
tags: statistics
---

The three most influential correlation coefficients are:

1. Pearson’s correlation coefficient 
2. Spearman’s ρ
3. Kendall’s τ
.
.
.
.
<h4> Pearson’s Correlation Coefficient (r): <h4>   
Type of Data: Quantitative.  
Nature of Association: Linear.  

<font color='green'>Calculation:</font> Pearson's correlation coefficient measures the strength and direction of the linear relationship between two continuous variables. It is calculated as the covariance of the variables divided by the product of their standard deviations.  
<font color='green'>Value Range:</font> The coefficient values range from -1 to 1. A value of 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative linear relationship, and 0 indicates no linear relationship.  

<font color='green'>Limitation:</font> Pearson's coefficient only captures linear relationships and can be heavily influenced by outliers.  
<br>
<h4> Spearman’s Rank Correlation Coefficient (ρ, rho):<h4>   
<font color='green'>Type of Data:</font> Ordinal or non-normally distributed interval data.  
<font color='green'>Nature of Association:</font> Monotonic.  
<br>
<font color='green'>Calculation:</font> Spearman’s rho is a measure of the monotonic relationship between two variables. It assesses how well the relationship between two variables can be described using a monotonic function. To calculate it, each value is replaced by its rank, and Pearson's formula is then applied to these ranks.  

<font color='green'>Value Range:</font> Like Pearson’s, Spearman’s coefficient ranges from -1 to 1, with -1, 1, and 0 having similar interpretations but for monotonic, rather than strictly linear, relationships.  

<font color='green'>Advantage:</font> It is less sensitive to outliers than Pearson’s coefficient and can be used with non-parametric data.
<br>
<h4> Kendall’s Tau Coefficient (τ, tau):<h4> 
<font color='green'>Type of Data:</font> Ordinal.  
<font color='green'>Nature of Association:</font> Monotonic.  
<font color='green'>Calculation:</font> Kendall’s tau measures the strength of the monotonic relationship by comparing the number of concordant pairs to the number of discordant pairs in the data set. A pair of observations is concordant if the ranks for both elements agree (i.e., both ranks are either higher or lower than the other pair’s ranks).  

<font color='green'>Value Range:</font> It ranges from -1 (perfect negative association) to 1 (perfect positive association). A value of 0 indicates the absence of association.  

<font color='green'>Advantage:</font> It is often used when the data set is small, as it is a more robust measure of correlation in the presence of outliers and errors than Pearson’s r.
<br>
<br>
<br>
4. Chatterjee's ξ 

Chatterjee's correlation coefficient aims to measure the strength of relationship that exists between $X$ and $Y$ even when 
the relationship is <font color='red'> not monotonic </font>, and <font color='red'> non-linear </font>.

The generalized formulas are as follows:

<div style="font-size: 20px;">
\[
    $\xi(X, Y) = 1 - \frac{3\Sigma^{n-1}_{i-1}|r_{i+1}-r_i}{n^2 - 1}$  
\]
</div>

<br>
<br>



$\xi(X, Y) = 1 - \frac{n\Sigma^{n-1}_{i-1}|r_{i+1}-r_i}{2\Sigma^{n}_{i=1}l_i(n-l_i)}$

<br>
<br>
<br>
<br>
<br>
<br>

<h5>Bibliography:<h5>


<a href=https://towardsdatascience.com/a-new-coefficient-of-correlation-64ae4f260310>Medium: New Coefficient of Correlation</a>

<a href=https://arxiv.org/pdf/1909.10140.pdf>ORIGINAL PAPER</a>

<a href=https://souravchatterjee.su.domains>ABOUT AUTHOR</a>

