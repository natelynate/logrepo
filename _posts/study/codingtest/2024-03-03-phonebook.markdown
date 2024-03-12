---
layout: post
title:  "[Programmers Lv.2] 전화번호부"
date:   2024-03-03 19:15:16 +0900
categories: study
tags: codingtest programmers trie
---
https://school.programmers.co.kr/learn/courses/30/lessons/42577  
Level 2  
Nested Dictionary로 구현  
초반에 node와 trie는 포인터 역할을 함. trie는 매번 trie의 최상단을 가리키고, node는 갱신될 때마다 단계적으로 내려감.  
초기에 node = trie를 통해 두 개의 포인터 위치를 유지하는 방식을 택함.   

```
def solution(phone_book):
    def insert_items(phone_book):
        trie = {}
        for phone_number in phone_book:
            node = trie
            for i in phone_number: # 전화번호의 각 자릿수 확인
                if i not in node: # dict key에 i가 없으면
                    node[i] = {} # 노드를 새로 생성
                node = node[i] # 다음 단계 노드를 포인팅 하도록 이동
            node['end_of_word'] = True
        return trie

    trie = insert_items(phone_book)
    for number in sorted(phone_book, key=len):
        node = trie
        for n in number:
            if 'end_of_word' in node:
                return False
            else:
                node = node[n]
    return True
 
if __name__ == '__main__':
    case1 = ["119", "97674223", "1195524421"]
    case2 = ["123","456","789"]
    trie = solution(case1)
    print(trie)
```