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
    