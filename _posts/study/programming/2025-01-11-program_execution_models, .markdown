---
layout: post
title:  Program Execution Models
date:   2025-01-11 19:15:16 +0900
categories: study
tags: programming python
---

<h4> Program Execution Models </h4> 

소프트웨어가 어떻게 실행되고, 어떻게 상태가 적용 및 관리되는지 여러가지지 관점에서 서술한 용어들이 많다. 

대표적인 용어들을 살펴보면:

1. 프로세스

2. 스레드 

3. 루틴

4. 런타임

5. 컨텍스트

6. 스코프

7. 세션

8. 상태(state)

9. 

등이 있다.

해당 글의 목표는

"세션" 이나 "스코프" 등, 유사한 의미를 가진 용어들을 순수하게 "현재 실행 중인 환경" 이나 "실행 조건"의 애매한 유의어로서 활용하는 경우가 있다.

목표는 이러한 용어들의 정확한 기술적인 의미를 짚고, 그리고 각 용어들이 지목하고 있는 소프트웨어 실행 조건의 맥락을 자체적으로 구분해서, 비록 어느 정도는 자체적인 면이 있더라도, 

추후 소프트웨어 실행 환경에 대한 이해를 높이기 위함이다.


### 프로세스
프로세스는 low-level의 관점이다. OS에서 현재 실행 중인 프로그램의 인스턴스를 직접적으로 나타낸다. 즉 일반적인 의미로서의 "과정" 보다는 실제 동작 중인 프로그램의 실체를 지칭하는 것에 가깝다.

(물론 이는 프로그램 == 과정 이라는 점을 깔고 이야기한다면 조금의 어긋남이 있는 서술일 수도 있지만)

이러한 독립된 실행 단위로서의 프로세스의 큰 특징은 메모리 공간을 독립적으로 점유한다는 점이다. 즉 이는 다른 프로세스들과 고립되어 실행하도록 한다. 

<b>고유의 메모리 공간과 리소스를 배정 받은 프로그램의 인스턴스</b>를 프로세스라 한다.


### 스레드
핵심은 `스레드는 프로세스의 일부` 라는 것이다. 항상 프로세스의 컴포넌트나 서브유닛의 관계이다. 

스레드는 프로세스 내의 더 작은 규모의 실행 단위(Unit of Execution)이다. 

동일한 프로세스 내의 다른 스레드들은 메모리 공간을 공유한다.

스레드는 좀 더 Light-weight하다. 생성과 삭제가 좀 더 쉽고, 여러 스레드는 같은 메모리 공간을 점유하기 때문에, 서로 간의 데이터를 공유하기가 훨씬 편하다. 이에 비해 프로세스는 완전히 격리되어 실행하기 때문에 프로세스 간 통신을 위해서는 IPC를 통해야 하므로, 더 복잡하다.

예컨대 특정 서버는 Concurrent Request를 처리하기 위해 멀티프로세스 구조를 택할 수도 있고, 멀티스레드 구조를 택할 수도 있다.

멀티 프로세스의 경우, Request 1개에 process를 하나씩 독립적으로 생성한다.

멀티 스레드의 경우, 프로세스 아래에 여러 개의 Thread를 생성해서 Request 별로 할당한다고 생각하면 된다. 

멀티 프로세스/멀티 스레드라고 해서 꼭 Async Operation이나 Parallel processing을 가정하지 않는다. 


### 루틴

루틴은 함수나 Procedure처럼, 어떠한 Sequence of instructions이다. 즉 어떠한 스레드든, 프로세스에 의해 실행되는 코드 단위이다. 
Thread는 "실행하는 것" 이고, 루틴은 "실행되는 것" 이라고 생각하면 좋겠다.

특히 "절차 지향 프로그래밍에서 특정한 절차 일부를 구현한 단위"가 맞다.

왜냐하면 모든 함수는 루틴이라고 하기에는, 객체기반프로그래밍의 함수나, 혹은 Event handler 따위의 함수들은 원본의 그 절차적인 의미를 

알아차리기 조금 힘들기 때문이다. 

```
// More traditional concept of a routine
void calculatePayroll() {
    // A sequence of steps in specific order 
    // 밑의 함수들이 calculatePayroll의 '루틴'이라고 보면 된다. 
    readEmployeeData();
    calculateHours();
    applyTaxRules();
    generatePayslips();
}
```
즉 아주 명확한 sequential step을 통해 특정 태스크를 수행하는 코드 단위를 부르는 말이라고 알면 되겠다.

### 런타임 + 컨텍스트

런타임은 "환경" 이라는 단어로 치환하면 편하다. 환경이라는 단어도 꽤나 모호하지만, 여기서는 소프트웨어가 실행되기 위한 서비스를 제공해주는 조건이라고 보면 된다. 

컨텍스트는 런타임의 현재 상태(스냅샷)이다.

예컨대 파이썬 기반 소프트웨어가 실행되려면 파이썬 인터프리터와 엔진이 필요하다. 이들이 파이썬의 "런타임"을 구성한다.

코드를 실행하고, 메모리 관리나 Garbage collection을 지원하는 것이 목적이다. 

따라서 "가상 환경"은 '환경' 이지만 런타임은 아니다. 가상 환경은 단순히 런타임 내의 일종의 컨테이너라고 보면 된다. 

내가 사용할 특정을 파이썬 인터프리터와 패키지 관리를 독립시켜서 관리할 수 있게 해주는 행정적인 단위다.

즉 파이썬 런타임 내에서는 다수의 가상 환경이 존재할 수 있고, 다수의 컨텍스트 또한 존재할 수 있다. 

예컨대 

```
# This runs in the '__main__' module context
x = 1

def function():
    # New function execution context
    y = 2
    print(x)  # Can access main context

# Each module import creates its own context
import math  # math module has its own context

익숙한 용어인 파이썬의 네임스페이스는 "컨텍스트"의 일부이다. 즉 실행 컨텍스트를 정의하기 위해 필요한 컴포넌트 중 하나로
네임스페이스가 포함된다. 
```

### 스코프
스코프는 코드 내의 변수나 네임 접근을 통제한다.  

```
x = 1

def function():
    # This creates a new local scope
    try:
        print(x)  # Can read from global scope
        x += 1    # But can't modify without 'global'
    except UnboundLocalError:
        print("Can't modify global without declaring")

    global x     # Now we can modify global x
    x += 1      # Works now
```

스코프는 현재 상태에서 내가 접근할 수 있는 것을 정의하고, 네임스페이스는 매핑을 제공하며, 이 모든 것들은 현재 속한 컨텍스트라는 환경 내에서 이루어진다.

환경은 넓은 의미에서, `코드를 실행하기 위해 필요한 것들의 모음` 이라고 생각한다. 

런타임도 환경이고 (인터프리터와 엔진),
실행 상태도 환경이다 (네트워크 연결, 빌드 환경, DB 커넥션 풀 등),
시스템 리소스나 환경 설정
Ex:
```
config = {
    'database': {
        'host': 'localhost',
        'port': 5432
    },
    'api_keys': {
        'service_a': 'key1',
        'service_b': 'key2'
    }
}
```
도 넓은 범주에서 환경으로 포함된다. 


### 세션
두 개의 컴포넌트 간의 Period of interaction을 나타내는 개념이다. 즉 `두 개의 컴포넌트 간 Interaction이라는 맥락 속에서 어떠한 state나 data를 온존하는 기간`이다. 

가장 쉬운 예시는 "로그인 세션"이다. 즉 사용자가 서버와 interaction을 하고 있는 기간 그 자체이면서, 동시에 토큰 등의 객체로

해당 세션을 물리적으로 현현한다.