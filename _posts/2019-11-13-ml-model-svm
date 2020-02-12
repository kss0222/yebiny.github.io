---
layout: post
title:  "[SVM]Support vector machine"
subtitle:  "SVM 개념과 수식원리를 알아봅시다."
categories: ml
tags: model
---

오늘은 딥러닝이 생기기 이전 기계학습 분야에서 가장 널리, 유명하게 쓰였던 `Support Vector Machine`에 대해서 알아보도록 하겠습니다. `SVM`을 제대로 이해하기 위해서는 이와 관련된 수학적인 기법들 (Lagrangian, Daul problrem, KKT) 을 우선적으로 이해하여야 합니다. 하지만 이것들을 모르더라도 전체적인 `SVM`에 대한 개략적인 방법을 설명하도록 하겠습니다. 

## Idea

`SVM`이 가장 큰 성능을 내는 것은 분류문제 입니다. 어떻게 분류할 것인가 하는것은 기계학습에 가장 기본적인 문제이기도 하죠. 여기 두 집단의 데이터가 있습니다. $$+$$와 $$-$$ 집단입니다. 두 집단을 잘 구분하는 선은 둘 중 어느 선일까요?
![graph]({{ site.url }}/img/svm-1.PNG)

대다수의 분들이 1번이라고 할 것입니다. 그 이유는 1번선이 2번선에 비해서 두 집단을 좀 더 여유롭게 분류하고 있기 때문이죠. 이 때 이 여유로운 공간을  `Margin` 이라고 합니다. 그리고 `SVM`은 `margin`을 최대화 하는 분류 경계면을 찾는 기법이라고 할 수 있습니다. 

![graph]({{ site.url }}/img/svm-2.PNG)

## Decision rule

개념적으로 `SVM`이 어떻게 작동하는지 알아보았습니다. 하지만 늘 그렇듯, 언어로 이해하기는 쉽지만 수학적으로 풀어내기는 쉽지 않죠. `Margin`을 최대화하는 분류 경계면을 찾기 위해 우리는 수학적인 식들을 찾아야 합니다. 먼저 우리가 찾는 분류 경계선이 있을 때 이 경계선이 $$+$$와 $$-$$를 어떻게 구분하는지 `desision rule`을 정해봅시다. 

먼저 분류 경계면에 대해 수직인 벡터 $$\vec{w}$$ 를 정의 합니다. 그리고 각 데이터에 대한 벡터 $$\vec{x}$$에 대해 $$\vec{w}$$를 내적하고 이 값이 어떤 상수보다 크면 $$+$$, 작으면 $$-$$로 분류할 수 있습니다. 여기서 내적을 함으로써 $$\vec{w}$$ 방향으로 $$\vec{x}$$ 를 `projection` 한다는 의미가 있습니다. 이를 수식으로 쓰면 다음과 같이 쓸 수 있겠죠

![graph]({{ site.url }}/img/svm-3.PNG)


$$ \vec{w} \cdot \vec{x} > b \quad then \;+ $$

$$ \vec{w} \cdot \vec{x} < b \quad then \;- $$

사실 이것은 아직 `SVM`의 식은 아닙니다. `SVM`의 특징인 `margin`를 첨가해야 겠죠. 또한 이렇게 만들어지는 식의 미지수인 $$\vec{w}$$와 $$b$$도 정하지 못했습니다. 우리가 아는 것은 $$\vec{w}$$가 분류경계면과 수직이라는 사실 뿐입니다. 기본적인 식에 조건이나 제약을 추가하여 $$\vec{w}$$와 $$b$$를 찾아가는 것이 이번 시간에 해야할 목표가 되겠습니다. 

## Boundary 정의

이제부터 `SVM`에 필요한 식을 만들어 봅시다. 먼저 `decision rule`을 참고하여 $$\vec{w}$$와 $$\vec{x}$$를 내적하고 상수 $$b$$를 더한 값이 0일때를 `분류경계면`으로 설정합시다. 

$$ \vec{w} \cdot \vec{x} + b = 0 \quad then \;Boundary $$

그럼 수식의 값이 0보다 크면 $$\vec{x}$$를 $$+$$로 분류해야 할까요? 아직은 판단하기 이릅니다. `SVM`에서는 `margin` 영역이 있습니다. $$\vec{x_+} $$와 $$\vec{x_-} $$ 어느 데이터도 들어 있지 않은 구간이죠. 이 구간을 정해줍시다.

$$\vec{w} \cdot \vec{x} + b = +1 \quad then \quad Plus \; boundary$$

$$\vec{w} \cdot \vec{x} + b = -1 \quad then \quad Minus \; boundary$$

위 식에서 경계선을 각각 $$1$$과 $$-1$$로 둔 것은 경계면 길이를 $$1$$로 `normalization` 했다고 볼 수 있습니다. 이 세개의 경계면에 대한 식을 그림으로 나타내면 아래와 같습니다. 그림을 보시면 한결 이해하기 편할 겁니다. 

![graph]({{ site.url }}/img/svm-4.PNG)

## 제약식의 정의

앞에서 `boundary`를 정의하는 식을 보았습니다. 이제 실제 데이터가 있는 공간 `Plust plane`과 `Minus plane`에 속해있는 데이터 벡터들 $$\vec{x_+}$$와 $$\vec{x_-}$$를 가지고 제약식을 정의해 보도록 하겠습니다. 참고로 우리는 `Lagrange` 식을 이용하여 `SVM`을 풀어갈 것입니다. Lagrange 문제에서는 `목적식`과 `제약식`이 필요합니다. 따라서 이 중에서 `제약식`을 먼저 정의하도록 하겠습니다. 

위에서 정한 boundary를 기준으로 생각하면  plus data vector $$\vec{x_+}$$ 들은 plus boundary 위에 있어야 할 것이고 minus data vector $$\vec{x_-}$$들이 있는 있는 공간은 minus boundary 아래에 위치해야 겠죠. 따라서 이것을 식으로 그대로 쓰면 다음과 같이 쓸 수 있겠습니다. 

$$ \left\{\begin{matrix}
 & \vec{w} \cdot \vec{x_+} + b \geq +1 & \\ 
 & \vec{w} \cdot \vec{x_+} + b \leq -1 & 
\end{matrix}\right. $$

그런데 이렇게 데이터 x에 관한 식을 쓰고 나니 두개의 식이 되었습니다. 계산의 편의를 위해서 두개의 식을 하나의 식으로 표현하기 위해서 아래와 같이 정의된 $$y_i$$를 도입하도록 하겠습니다. 

$$ y_i = \left\{\begin{matrix}
 &1  & for  +\\ 
 &-1 & for  -
\end{matrix}\right.$$

앞선 두 식의 양 변에 $$y_i$$를 곱해줍니다. 그럼 $$y_i$$가 $$+$$ 식에서는 $$+1$$, $$-$$ 식에서는 $$-1$$이기 떄문에 아래와 같이 식을 정리할 수 있습니다. 

$$ \begin{align*}
\left\{\begin{matrix}
 & \vec{w} \cdot \vec{x_+} + b \geq +1 & \\ 
 & \vec{w} \cdot \vec{x_+} + b \leq -1 & 
\end{matrix}\right. 
 &\leftrightarrow \left\{\begin{matrix}
 & y_i(\vec{w} \cdot \vec{x_+} + b) \geq \;\;y_i & \\ 
 & y_i(\vec{w} \cdot \vec{x_-} + b) \geq -y_i & 
\end{matrix}\right.\\ 
 &\leftrightarrow  \; y_i(\vec{w}\cdot\vec{x}+b)\geq 1 \\ 
 &\leftrightarrow \; y_i(\vec{w}\cdot\vec{x}+b)-1\geq 0
\end{align*} $$


이렇게 해서 하나의 식으로 정리가 되었습니다. 이 식이 나타내는 것은 데이터들이 어떤 공간에 존재하는 가를 나타냅니다. 이 식이 바로 제약식이 됩니다. 

$$y_i(\vec{w}\cdot\vec{x}+b)-1\leq0$$

$$\quad\quad y_i = \left\{\begin{matrix}
 &1  & for  +\\ 
 &-1 & for  -
\end{matrix}\right.$$

그렇다면 제약식이 0이 될때는 $$\vec{x}$$ 가 어디에 존재할까요? 바로 boundary 위에 존재한다는 것을 알 수 있습니다. 

![graph]({{ site.url }}/img/svm-5.PNG)

이 점을 잘 생각하면서 다음으로 목적식을 정의해 보도록 하겠습니다. 

## 목적식의 정의

`SVM`의 목적은 무엇이였는지 생각해 봅시다. `SVM`의 목적은 `margin`을 최대화 하는 것이였죠. 그렇다면 이 `margin`이란 것을 수식으로 표현해서 `maximize` 하면 될 것 같군요. Margin을 계산하기 위해 아래 그림에서 처럼 `boundary` 위에 있는 $$\vec{x_+}$$와 $$\vec{x_-}$$를 사용하겠습니다.

![graph]({{ site.url }}/img/svm-6.PNG)

위의 그림대로 margin의 식을 쓰면 다음과 같이 쓸 수 있습니다. 참고로 
$$
\frac{\vec{w}}
{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
$$는 단위 벡터입니다. 

$$
\begin{align*}
margin &= (\vec{x_+}-\vec{x_-})\cdot 
\frac{\vec{w}}
{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
\\ 
 &=\frac{\vec{x_+}\cdot \vec{w}-\vec{x_-}\cdot \vec{w}}
{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
\end{align*}
$$

여기서 $$\vec{x_+}\cdot \vec{w}$$와 $$\vec{x_-}\cdot \vec{w}$$ 계산하기 위해 앞서 구한 제약식을 사용하도록 합니다. Margin을 구하기 위해 사용하는 $$x_+$$와 $$x_-$$는 `boundary`위의 데이터죠. 그런데 제약식 $$ y_i (\vec{w}\cdot\vec{x_i}+b)-1 $$ 에서 $$x_i$$ 가 `plust boundary` or `minus boundary`위의 데이터라면 제약식이 0이 된다는 것을 이미 알고 있습니다. 따라서 이 식을 이용하여 $$\vec{x_+}\cdot \vec{w}$$와 $$\vec{x_-}\cdot \vec{w}$$를 다음과 같이 구할 수 있습니다. 


$$
\begin{align*}
y_i(\vec{w}\cdot\vec{x}_i + b)-1=0 &\leftrightarrow y_i(\vec{w}\cdot\vec{x}_i + b) = 1 \\
&\leftrightarrow \vec{w}\cdot\vec{x}_+ + b = 1 \\
&\leftrightarrow \vec{w}\cdot\vec{x}_+ = 1 -b \\
\end{align*}
$$

$$
\begin{align*}
y_i(\vec{w}\cdot\vec{x}_i + b)-1=0 &\leftrightarrow y_i(\vec{w}\cdot\vec{x}_i + b) = 1 \\
&\leftrightarrow \vec{w}\cdot\vec{x}_- + b = -1 \\
&\leftrightarrow \vec{w}\cdot\vec{x}_- = -1 -b \\
\end{align*}
$$

이제 이 값을 $$\vec{x_+}\cdot \vec{w}$$와 $$\vec{x_-}\cdot \vec{w}$$에 각각 대입하기만 하면 되겠죠.

$$
\begin{align*}
margin &= (\vec{x_+}-\vec{x_-})\cdot 
\frac{\vec{w}}
{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
\\ 
 &=\frac{\vec{x_+}\cdot \vec{w}-\vec{x_+}\cdot \vec{w}}
{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
\\
&= \frac{((1-b)-(-1-b))}{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}\\
&= \frac{2}{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
\end{align*}
$$

이렇게 해서 `margin`을 값으로 나타냈습니다. `margin`을 최대화하는 SVM의 목적은 
$$
\frac{2}{
\begin{Vmatrix}
\vec{w}
\end{Vmatrix}_2
}
$$
을 최대화 하는것과 같은 말이 되겠습니다. 그런데 우리는 수학적 계산의 편의성을 위해서 이 값을 살짝 바꾸기로 하겠습니다. 

![graph]({{ site.url }}/img/svm-9.PNG)

굳이 이렇게 바꿔서 푸는 이유는 오로지 계산의 편의성을 위해서입니다. 분모에 들어가는 값이 L2 norm 인데 이것이 정의된 값을 생각하면 루트값이기 때문에 제곱해주면 계산의 편리해지겠고, 앞에 $$\frac{1}[2}$$ 을 곱해준다면 미분할 때 편리할 것임을 짐작할 수 있습니다.

## Lagrange 문제로의 변환

![graph]({{ site.url }}/img/svm-10.PNG)

앞서 우리는 제약식과 목적식을 구해보았습니다. 이렇게 제약식과 목적식을 구하고 나면 이제 문제를 Lagrange 문제로 변환하여 풀 수 있게 됩니다. Lagrange에 대한 글은 차후에 자세히 쓰도록 하겠습니다. 우선 라그랑지안 승수법에 대해서 간략히 소개하자면

> Lagrange multiplier method
>: 제약식에 형식적인 라그랑지안 승수를 곱하고 이를 최적화하려는 목적식에 더하여 제약된 문제를 제약이 없는 문제로 바꾸는 기법.

언뜻봐서는 잘 와닿지 않죠. 한마디로 하면 제약식과 목적식 두개로 나누어진 식을 하나의 식으로 풀겠다는 것입니다. 

![graph]({{ site.url }}/img/svm-11.PNG)


## KKT 조건

![graph]({{ site.url }}/img/svm-12.PNG)

최적해를 찾는 과정에서 KKT Stationarity 조건에 따라서 우리가 찾는 $$\vec{w}$$가 어떻게 구성되어 있는지 알 수 있습니다. $$\alpha$$, $$y$$, $$x$$를 곱하고 이를 모두 합한 것으로 정해집니다. 그런데 여기서 $$x$$와 $$y$$는 데이터의 값이고 그렇다면 $$\alpha$$가 어떻게 결정되는지 보아야 하는데 4번의 Complementary slackness 조건에 의해 $$\alpha_i$$와 조건식 $$y_i(\vec{w} \cdot x_i +b)-1$$의 곱은 0이 되어야 합니다. 두 조건을 같이 생각해보면 $$\vec{w}$$에 기여를 하는 데이터 $$x_i$$라면, $$\alpha$$는 0이 아니여야 하고, $$\alpha$$가 0이 아니려면 조건식이 무조건 0이여야 하는데 조건식이 0일때, 즉 $$ y_i(\vec{w}\cdot\vec{x}_i)-1=0 $$ 이면 `plus boundary` `or` `minus boundar` 위에 있는 점이였다는걸 알 수 있습니다. 


![graph]({{ site.url }}/img/svm-13.PNG)

## Dual 문제로의 변환 
> Duality:
> 최적화이론에서 쌍대성(duality)이란 어떤 최적화 문제가 원초 문제(the primal problem)와 쌍대 문제(the dual problem)의 두 가지 관점에서 볼 수 있다는 원칙입니다.  목적 함수의 최솟값을 찾는 문제는 최대값을 찾는 문제로 바뀝니다. 

앞선 KKT stationarity condtition을 이용하여 라그랑지 식에 미분한 값을 대입해줍니다. 



## Lagrangian Dual

따라서 Lagrangian dual 은 다음과 같은 quadratic programming formulation 으로 나타낼 수 잇습니다. 

![graph]({{ site.url }}/img/svm-14.PNG)

* Original problem formulation (primal formulation) 보다 풀기 쉬운 형태
* Objective function is quadratic and constraint is linear -> quadratic programming -> convex optimization-> globally optimal solution exists (전역최적해 존재)
* Optimization 문제가 x들의 inner product 만으로 표현됨(nonlinear case로 확장했을 때 좋은 성질)
* Lagrangian dual의 decision variable은 𝛼이며, quadratic optimization을 풀어 𝛼에 대한 solution을 얻을 수 있음 


## SVM의 한계

SVM은 두 범주를 잘 분류하면서 margin이 최대화된 초평면(hyperplne)을 찾는 기법입니다. 하지만 아래 그림처럼 어떤 직선을 그어도 두 범주를 완벽하게 분류하기 어려운 경우도 많습니다. 따라서 이를 해결하기 위해 보완적인 방법이 있습니다. 


![graph]({{ site.url }}/img/svm-15.PNG)
