---
layout: post
title:  "[Programmers Lv.3] 아이템 줍기"
date:   2024-03-24 19:15:16 +0900
categories: study
tags: codingtest programmers bfs
---

```python
"""
문제 팁: 
1. 카테지안 평면과 리스트로 만든 파이썬의 2차원 Array는 1:1 호환이 될 수 없다. 이를 극복하기 위해 카테지안 평면의 좌표를 *2 해서 표현하는 방식이 있다.
2. 2차원 Array로 표현한 다각형에서 다각형의 가장자리에 있는지에 대한 여부는 해당 위치에서 모든 방향을 다 탐색했을 때 외부에 있는 칸(0)이 하나라도 있으면 가장자리에 있다고 경험적으로 판단할 수 있다.
3. 0,0이 좌측하단이 아니라 좌측상단이나 다른 곳에 있을 경우 간단하게 맵의 크기-(좌표)를 통해 바꿀 수 있다.
"""
from collections import deque

def solution(rectangles, characterX, characterY, itemX, itemY):
    max_x, max_y = -1, -1
    for rectangle in rectangles:
        up_x, up_y = rectangle[2], rectangle[3]
        max_x, max_y = max(max_x, up_x), max(max_y, up_y)
    
    lim_x, lim_y = max_x*2+2, max_y*2+2 # Set the maximum map size based on the farthest polygon point
    worldmap = [[0 for _ in range(lim_x)] for j in range(lim_y)]
    dx8 = [-1, 0, 1, -1, 1, -1, 0, 1] 
    dy8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx4 = [-1, 1, 0, 0]
    dy4 = [0, 0, 1, -1]
    
    for rectangle in rectangles:
        low_x, low_y, up_x, up_y = rectangle
        low_x, low_y, up_x, up_y = low_x * 2, low_y * 2, up_x * 2, up_y * 2 
        for h in range(up_y-low_y+1):
            for w in range(up_x-low_x+1):
                row = low_y + h 
                col = low_x + w  
                worldmap[lim_y-row][col] = 1
    
    for row in range(lim_y):
        for col in range(lim_x):
            if worldmap[row][col] == 0:
                continue
            flag = True
            for i in range(8):
                nx, ny = col + dx8[i], row + dy8[i]
                if worldmap[ny][nx] == 0:
                    flag = False
                    break
            if flag: # if all surrounding columns are 1
                worldmap[row][col] = 2
    
    
    queue = deque([(characterX*2, lim_y-(characterY*2), 0)])
    worldmap[lim_y-(characterY*2)][characterX*2] = 3 # Mark the Starting spot
    worldmap[lim_y-(itemY*2)][itemX*2] = 4 # Mark the item location
    answer = []

    while queue:
        x, y, n = queue.popleft()
        for i in range(4):
            nx, ny = x+dx4[i], y+dy4[i]
            if worldmap[ny][nx] == 1:
                queue.append((nx, ny, n+1))
                worldmap[ny][nx] = 3 # mark as visited
            elif worldmap[ny][nx] == 4:
                answer.append(n+1)
    return worldmap, answer 
    

if __name__ == "__main__":
    worldmap, answer = solution([[1,1,7,4],[3,2,5,5],[4,3,6,9],[2,6,8,8]], 1, 3, 7, 8)

    for i in worldmap:
        print(i)

    print(min(answer)//2)
```