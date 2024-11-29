import math
import numpy as np

def check_location(boxes_person, boxes_helmet):
    helmeted_person = 0
    # xywh
    print(len(boxes_person))
    print(len(boxes_helmet))
    if len(boxes_person)==0 or len(boxes_helmet)==0:
        return helmeted_person
    
    # 헬멧의 중점이 어떤 사람의 bbox에 들어가 있는지를 판단
    helmet_owner = {}
    print('@',boxes_person, boxes_helmet)
    for h, box_h in enumerate(boxes_helmet):
        for p, box_p in enumerate(boxes_person):
            # 헬멧의 중점 x좌표가 사람의 가로 범위 내에 있고
            # 헬멧의 중점 y좌표가 사람의 세로 범위 내에 있다면
            if box_p[0][0]-box_p[0][2]//2 <= box_h[0][0] and \
               box_p[0][0]+box_p[0][2]//2 >= box_h[0][0] and \
               box_p[0][1]-box_p[0][3]//2 <= box_h[0][1] and \
               box_p[0][1]+box_p[0][3]//2 >= box_h[0][1]:
                # 그렇다면 이 헬멧은 이 사람의 것이다.
                helmet_owner[h] = p
    print('@@', helmet_owner)
    for h_idx in list(helmet_owner.keys()):
        p_idx = helmet_owner[h_idx]
        person_xywh = boxes_person[p_idx][0]
        helmet_xywh = [boxes_helmet[h_idx][0][0], boxes_helmet[h_idx][0][1]]

        top_xy = [person_xywh[0], person_xywh[1]-person_xywh[3]//2]
        left_xy = [person_xywh[0]-person_xywh[2]//2 , person_xywh[1]]
        right_xy = [person_xywh[0]+person_xywh[2]//2 , person_xywh[1]]
        bottom_xy = [person_xywh[0], person_xywh[1]+person_xywh[3]//2]

        dist_list = [math.dist(helmet_xywh, top_xy), math.dist(helmet_xywh, left_xy), math.dist(helmet_xywh, right_xy), math.dist(helmet_xywh, bottom_xy)]
        print('@@@@', np.argmin(dist_list))
        if np.argmin(dist_list)==0:
            helmeted_person+=1

    print('------')
    print(helmet_owner)
    return helmeted_person
