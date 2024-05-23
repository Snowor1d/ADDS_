    def random_agent_distribute(self, agent_num):
        case = random.randint(1,2) 
        # case1 -> 방에 사람이 있는 경우
        # case2 -> 밖에 주로 사람이 있는 경우
        only_space = []
        for sp in self.space_list:
            if (not sp in self.room_list and sp != [[0,0], [10, 10]] and sp != [[]]):
                only_space.append(sp)
        space_num = len(only_space)
        
        if(case==1 or space_num<7):
            room_agent = random.randint(int(agent_num*7/10), int(agent_num*9/10))
        else:
            room_agent = random.randint(int(agent_num*1/10), int(agent_num*4/10))
        
        space_agent = agent_num-room_agent

        room_num = len(self.room_list)
  
        random_list = [0] * room_num

        # 총합이 room_agent가 되도록 할당
        for i in range(room_num - 1):
            random_num = random.randint(1, room_agent - sum(random_list) - (room_num - i - 1))
            random_list[i] = random_num

        # 마지막 숫자는 나머지 값으로 설정
        random_list[-1] = room_agent - sum(random_list)

        for j in range(len(self.room_list)):
            self.agent_place(self.room_list[j][0], self.room_list[j][1], random_list[j])

        space_random_list = [0] * space_num
        #print(only_space)

        for k in range(space_num - 1):
            if(only_space[k] == [[0,5], [5, 49]] or only_space[k] == [[5, 0], [49, 5]] or only_space[k] == [[5, 45], [49, 49]] or only_space[k] == [[45, 5], [49, 45]]):
                random_space_num = random.randint(1,max(min(space_agent-sum(space_random_list)- (space_num-k-1), 5),1))

            else:
                random_space_num = random.randint(1, space_agent - sum(space_random_list) - (space_num - k - 1))
            space_random_list[k] = random_space_num 
        space_random_list[-1] = space_agent - sum(space_random_list)


        for l in range(len(only_space)):
            self.agent_place(only_space[l][0], only_space[l][1], space_random_list[l])
    

    def map_recur_divider(self, xy, x_unit, y_unit, num, space_list, room_list, is_room): # ex) xy = [[2,3], [4,5]]
        x_diff = xy[1][0] - xy[0][0]
        y_diff = xy[1][1] - xy[0][1]

        real_xy =  [ [xy[0][0]*x_unit, xy[0][1]*y_unit], [xy[1][0]*x_unit, xy[1][1]*y_unit]]
        if(is_room==0):
            space_list.append(real_xy)
            return
                   
        if(x_diff<4 or y_diff<4):
            space_list.append(real_xy)
            room_list.append(real_xy)
            return
        
            
        if(num==1): 
            a = random.randint(1,20)
            if(a<4):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return 
        elif(num==2):
            a = random.randint(1,10)
            if(a<12):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return
        elif(num==3):
            a = random.randint(1,10)
            if(a<15):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==4):
            a = random.randint(1,10)
            if(a<20):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
            
        divide_num_y = random.randint(1, y_diff-1)
        divide_num_x = random.randint(1, x_diff-1)

        random_exist_room1 = random.randint(0,1)
        random_exist_room2 = random.randint(0,1)
        random_exist_room3 = random.randint(0,1)
        random_exist_room4 = random.randint(0,1)

        if (random_exist_room1 == 0):
            random_exist_room2 = 1
        if (random_exist_room3 == 0):
            random_exist_room4 = 1
        

        special_hallway = random.randint(1, 4)
        if(special_hallway < 3 and num<2):
            if (num%2==0): #가로로 나눈다
                left = int(x_diff/2)
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[0][0]+left, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0]+left, xy[0][1]], [xy[0][0]+left+1, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 0)
                self.map_recur_divider([[xy[0][0]+left+1, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
            else: #세로로 나눈다
                up = int(y_diff/2)
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+up]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0], xy[0][1]+up], [xy[1][0], xy[0][1]+up+1]], x_unit, y_unit, num+1, space_list, room_list, 0)
                self.map_recur_divider([[xy[0][0], xy[0][1]+up+1], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)

        else:
            if(num<1):
                random_exist_room1 = random_exist_room2 = random_exist_room3 = random_exist_room4 = 1
            if (num%2==0): #가로로 나눈다
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[0][0]+divide_num_x, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room1)
                self.map_recur_divider([[xy[0][0]+divide_num_x, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room2)
        
            else: #세로로 나눈다
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+divide_num_y]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room3)
                self.map_recur_divider([[xy[0][0], xy[0][1]+divide_num_y], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room4) 
    
    