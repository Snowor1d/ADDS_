from mesa import Model
from agent_renew import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_renew
from agent_renew import WallAgent
import random
import copy
import math

# goal_list = [[(80, 119), (79, 119), (78, 119), (77, 119)], #gate1 
#              [(198, 119), (197, 119), (196, 119)], #gate2
#              [(119, 19), (119, 20), (119, 21), (119, 22)]] #gate3 

# goal_list = [[(79, 119), (78, 119)], #gate1 
#              [(198, 119), (197, 119)], #gate2
#              [(119, 19), (119, 20)]] #gate3 

goal_list = [[0,50], [49, 50]]
hazard_id = 5000

def make_plane(xy1, xy2):
    new_plane = []
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]

    if(xy1[0]>xy2[0]):
        temp = x1
        x1 = x2
        x2 = temp
    if(xy1[1]>xy2[1]):
        temp = y1
        y1 = y2
        y2 = temp

    for i in range(x1, x2+1):
        for j in range(y1, y2+1):
            new_plane.append((i, j))

    return new_plane

def make_room(xy1, xy2):
    new_plane = []
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]

    # if(xy1[0]>xy2[0]):
    #     temp = x1
    #     x1 = x2
    #     x2 = temp
    # if(xy1[1]>xy2[1]):
    #     temp = y1
    #     y1 = y2
    #     y2 = temp
    
    rooms = []
    rooms = rooms + make_plane([x1, y1], [x2, y1])
    rooms = rooms + make_plane([x1, y1], [x1, y2])
    rooms = rooms + make_plane([x2, y1], [x2, y2])
    rooms = rooms + make_plane([x1, y2], [x2, y2])

    return rooms

def make_door(xy1, xy2, door_size):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    
    door = []

    if(xy1[0]>xy2[0]):
        temp = x1
        x1 = x2
        x2 = temp
    if(xy1[1]>xy2[1]):
        temp = y1
        y1 = y2
        y2 = temp
    if(x1 == x2):
        if((y2-y1)<=door_size):
            return door
        door_start = random.randint(y1, y2-door_size)
        for i in range(door_size):
            door.append((x1, door_start+i))
        return door 
    elif(y1==y2):
        if((x2-x1)<=door_size):
            return door
        door_start = random.randint(x1, x2-door_size)
        for i in range(door_size):
            door.append((door_start+i, y1))
        return door
def goal_average(xys):
    middle_x = 0
    middle_y = 0
    for i in xys:
        middle_x += i[0]
        middle_y += i[1]
    middle_x = middle_x/len(xys)
    middle_y = middle_y/len(xys)
    return [middle_x, middle_y]
    
def make_door2(xy1, xy2, door_size):
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    
    door = []

    if(xy1[0]>xy2[0]):
        temp = x1
        x1 = x2
        x2 = temp
    if(xy1[1]>xy2[1]):
        temp = y1
        y1 = y2
        y2 = temp
    if(x1 == x2):
        if((y2-y1-2)<=door_size):
            return door
        door_start = random.randint(y1+1, y2-door_size-1)
        for i in range(door_size):
            door.append((x1, door_start+i))
        return door 
    elif(y1==y2):
        if((x2-x1-2)<=door_size):
            return door
        door_start = random.randint(x1+1, x2-door_size-1)
        for i in range(door_size):
            door.append((door_start+i, y1))
        return door 
    
def goal_extend(xy_space, goal):
    xy = [0, 0]
    xy[0] = (xy_space[0][0] + xy_space[1][0])/2
    xy[1] = (xy_space[1][0] + xy_space[1][1])/2
    d = math.sqrt(pow(goal[0]-xy[0],2) + pow(goal[1]-xy[1], 2))
    return [goal[0] + 2*(goal[0]-xy[0])/d, goal[1] + 2*(goal[1]-xy[1])/d]
    
def make_door_to_outdoor(door_list, space_list):
    for i in space_list: #외부와 연결된 문 만들기
        if(i[0][0] == 10 and i[0][1]==10):
            x=random.randint(0,1)
            if(x):
                door_list = door_list + make_door(i[0], [i[0][0], i[1][1]], 4)
            else:
                door_list = door_list + make_door(i[0], [i[1][0], i[0][1]], 4)

        elif (i[0][0] == 10):
            door_list = door_list + make_door(i[0], [i[0][0], i[1][1]], 4)
        elif (i[0][1] == 10):
            door_list = door_list + make_door(i[0], [i[1][0], i[0][1]], 4)
        elif(i[1][0] == 90 and i[1][1]==90):
            x = random.randint(0,1)
            if(x):
                door_list = door_list + make_door([i[1][0], i[0][1]], i[1], 4)
            else:
                door_list = door_list + make_door([i[0][0], i[1][1]], i[1], 4)

        elif (i[1][0] == 90):
            door_list = door_list + make_door([i[1][0], i[0][1]], i[1], 4)
        elif (i[1][1] == 90):
            door_list = door_list + make_door([i[0][0], i[1][1]], i[1], 4) 

# class space_graph():
#     def __init__(self, space_list):


    


class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int):
        self.simulation_type = 0 #0->outdoor, #1->indoor
        self.room_list = []
        self.map_divide =  [[[0,0], [0,0]]*10 for _ in range(10)]
        self.map_repre = [[0]*11 for _ in range(11)]
        self.num_agents = number_agents
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.schedule = RandomActivation(self)
        self.schedule_e = RandomActivation(self)
        self.schedule_w = RandomActivation(self)
        self.schedule_h = RandomActivation(self)
        self.running = (
            True  # required by the MESA Model Class to start and stop the simulation
        )
        self.agent_id = 1000

        self.datacollector_currents = DataCollector(
            {
                "Remained Agents": FightingModel.current_healthy_agents,
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        # self.agent_place((40,20), (70, 80), 100)
        # self.make_hazard((40,20), (70,80), 4)


        # self.space_goal_dict[((0,0), (5, 99))] = [[0,0]]
        # self.space_goal_dict[((5,5), (99, 5))] = [[0,0]]
        # self.space_goal_dict[((5,95), (99, 99))] = [[5, 95]]
        # self.space_goal_dict[((95, 5), (99, 95))] = [[95, 5]]

        exit_rec = self.make_exit()


    

        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기


        wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        space = []
        self.wall_matrix = list()
        self.only_one_wall = list()
        self.indoor_connect = list() # 방과 방 사이를 연결하는 문을 만들기 위한 리스트 
        for i in range(101):
            tmp = []
            for j in range(101):
                tmp.append(0)
            self.wall_matrix.append(tmp)
            self.only_one_wall.append(tmp)
            self.indoor_connect.append(tmp)
        
        from server_renew import NUMBER_OF_CELLS

        for i in range(int(NUMBER_OF_CELLS)):
            wall.append((i, 0))
            wall.append((0, i))
            wall.append((i, int(NUMBER_OF_CELLS)-1))
            wall.append((int(NUMBER_OF_CELLS)-1, i))
           
            self.wall_matrix[i][0] = 1
            self.wall_matrix[0][i] = 1
            self.wall_matrix[i][int(NUMBER_OF_CELLS)-1] = 1
            self.wall_matrix[int(NUMBER_OF_CELLS)-1][0] = 1

        #wall = wall+self.random_map_generator2(3, 5, 2, 100)
        self.space_list = []
        self.room_list = []
        
        self.room_goal_dict = {}
        self.space_goal_dict = {} #각 space가 가지는 gaol을 표현하기 위함
        self.space_index = {} #각 space의 index를 마크하기 위함 
        self.space_graph = {} #각 space의 인접 space를 표현하기 위함
        self.space_type = {} #space type이 0이면 빈 공간, 1이면 room

        self.init_outside() #외곽지대 탈출로 구현 
        
        self.door_list = []
        index = 5
        self.map_recur_divider_fine([[1, 1], [19, 19]], 5, 5, 0, self.space_list, self.room_list, 1)

        for j in self.space_list: 
            self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = [] # 모든 space에 대한 goal을 설정할 것임
            self.space_index[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = index
            index = index+1
            self.space_graph[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = []
        for k in self.room_list:
            self.space_type[((k[0][0], k[0][1]), (k[1][0], k[1][1]))] = 1

        self.connect_space() #space 그래프 연결 
        
        if(self.simulation_type):
            self.make_door_between_room() #방 사이에 room 만들기
            self.make_door_to_outside()
            self.space_connect_via_door()
            for r in self.room_list:
                result = self.check_bridge(r, ((0,0), (10,10))) #방이 고립되어 있지는 않은가 확인 
                if (result == 0):
                    self.make_one_door_in_room(r)

            for r in self.room_list:
                result = self.check_bridge(r, ((0,0), (10,10))) #방이 고립되어 있지는 않은가 확인 
                if (result == 0):
                    self.make_one_door_in_room(r)
            self.space_connect_via_door()

        #print(self.space_graph)

        #self.random_agent_distribute(100)
        #self.random_hazard_placement(random.randint(1,3))
        if(self.is_left_exit):
            self.space_goal_dict[((0,0), (5, 95))] = [self.left_exit_goal]
        else:
            self.space_goal_dict[((0,0), (5, 95))]
        self.space_goal_dict[((0,95), (5, 99))] = [[0,0]]
        self.space_goal_dict[((95,5), (99, 99))] = [[5, 95]]
        self.space_goal_dict[((5, 0), (99, 5))] = [[95, 99]]

        #print(self.space_goal_dict)


        # 이제 고립된 방들 문 만들어주기 
        print(self.room_list)


        
        #make wall
                
        for i in self.room_list:
            wall = wall+make_room(i[0], i[1])
        for j in self.space_list:
             space = space+make_room(j[0], j[1])
        #print(self.space_list)
        #print(self.room_list)
        #print(len(self.door_list)/4)

        set_transform = set(wall)
        wall = list(set_transform)
        for i in goal_list:
            for j in i:
                if j in wall:    
                    wall.remove(j)
                    self.wall_matrix[j[0]][j[1]] = 0
    
        for i in self.door_list:
                if i in wall:    
                    wall.remove(i)
                    self.wall_matrix[i[0]][i[1]] = 0

        for i in range(len(wall)):
            if (self.only_one_wall[wall[i][0]][wall[i][1]] == 1 and wall[i][0]!=0 and wall[i][1]!=0 and wall[i][1]!=99):
                continue
            c = FightingAgent(i, self, wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, wall[i])
            self.only_one_wall[wall[i][0]][wall[i][1]] = 1
        for i in range(len(space)):
            if (self.only_one_wall[space[i][0]][space[i][1]] == 1 and space[i][0]!=0 and space[i][1]!=0 and space[i][1]!=99):
                continue
            c = FightingAgent(10000+i, self, space[i], 12)
            self.schedule_w.add(c)
            self.grid.place_agent(c, space[i])
            self.only_one_wall[space[i][0]][space[i][1]] = 1

    def make_exit(self):
        exit_rec = []
        self.is_down_exit = random.randint(0,1)
        self.is_left_exit = random.randint(0,1)
        self.is_up_exit = random.randint(0,1)
        self.is_right_exit = 0
        if (self.is_down_exit==0 and self.is_left_exit==0 and self.is_up_exit==0):
            self.is_right_exit = 1 
        else:
            self.is_right_exit = 0

        left_exit_num = 0
        self.left_exit_goal = [0,0]
        if(self.is_left_exit):
            exit_size = random.randint(30, 70)
            start_exit_cell = random.randint(0, 99-exit_size)
            for i in range(0, 5):
                for j in range(start_exit_cell, start_exit_cell+exit_size):
                    exit_rec.append((i,j))
                    self.left_exit_goal[0] += i
                    self.left_exit_goal[1] += j
                    left_exit_num +=1
            self.left_exit_goal[0] = self.left_exit_goal[0]/left_exit_num
            self.left_exit_goal[1] = self.left_exit_goal[1]/left_exit_num

        right_exit_num = 0    
        self.right_exit_goal = [0,0]
        if(self.is_right_exit):
            exit_size = random.randint(30, 70)
            start_exit_cell = random.randint(0, 99-exit_size)
            for i in range(95, 100):
                for j in range(start_exit_cell, start_exit_cell+exit_size):
                    exit_rec.append((i,j))
                    self.right_exit_goal[0] += i
                    self.right_exit_goal[1] += j
                    right_exit_num +=1
            self.right_exit_goal[0] = self.right_exit_goal[0]/right_exit_num
            self.right_exit_goal[1] = self.right_exit_goal[1]/right_exit_num

        down_exit_num = 0    
        self.down_exit_goal = [0,0]
        if(self.is_down_exit):
            exit_size = random.randint(30, 70)
            start_exit_cell = random.randint(0, 99-exit_size)
            for i in range(start_exit_cell, start_exit_cell+exit_size):
                for j in range(0, 5):
                    exit_rec.append((i,j))
                    self.down_exit_goal[0] += i
                    self.down_exit_goal[1] += j
                    down_exit_num +=1
            self.down_exit_goal[0] = self.down_exit_goal[0]/down_exit_num
            self.down_exit_goal[1] = self.down_exit_goal[1]/down_exit_num

        up_exit_num = 0    
        up_exit_goal = [0,0]
        if(self.is_up_exit):
            exit_size = random.randint(30, 70)
            start_exit_cell = random.randint(0, 99-exit_size)
            for i in range(start_exit_cell, start_exit_cell+exit_size):
                for j in range(95, 100):
                    exit_rec.append((i,j))
                    up_exit_goal[0] += i
                    up_exit_goal[1] += j
                    up_exit_num +=1
            up_exit_goal[0] = up_exit_goal[0]/up_exit_num
            up_exit_goal[1] = up_exit_goal[1]/up_exit_num
        return exit_rec



    def check_bridge(self, space1, space2):
        visited = {}
        for i in self.space_graph.keys():
            visited[i] = 0
        
        stack = [space1]
        while(stack):
            node = stack.pop()
            if(visited[((node[0][0], node[0][1]), (node[1][0], node[1][1]))] == 0):
                visited[((node[0][0], node[0][1]), (node[1][0], node[1][1]))] = 1
                stack.extend(self.space_graph[((node[0][0], node[0][1]), (node[1][0], node[1][1]))])
        if (visited[space2] == 0):
            return 0
        else:
            return 1            
        
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
            if(only_space[k] == [[0,5], [5, 99]] or only_space[k] == [[5, 0], [99, 5]] or only_space[k] == [[5, 95], [99, 99]] or only_space[k] == [[95, 5], [99, 95]]):
                random_space_num = random.randint(1,max(min(space_agent-sum(space_random_list)- (space_num-k-1), 5),1))

            else:
                random_space_num = random.randint(1, space_agent - sum(space_random_list) - (space_num - k - 1))
            space_random_list[k] = random_space_num 
        space_random_list[-1] = space_agent - sum(space_random_list)


        for l in range(len(only_space)):
            self.agent_place(only_space[l][0], only_space[l][1], space_random_list[l])

    def random_hazard_placement(self, hazard_num):
        min_size = 4
        max_size = 5
        only_space = [] #바깥쪽 비상탈출구 제외
        for sp in self.space_list:
            if (sp != [[0,0], [5, 5]] and sp != [[0,5], [5, 99]] and sp != [[5, 0], [95, 5]] and sp != [[5, 90], [99, 99]] and sp != [[95, 5], [99, 95]]):
                only_space.append(sp)
        hazard_num = min(hazard_num, len(only_space))

        hazard_visited = [0] * len(only_space)

        while(1):
            hazard_index = random.randint(0, len(only_space)-1)
            if(hazard_visited[hazard_index] == 1):
                continue
            hazard_visited[hazard_index] = 1
            self.make_hazard(only_space[hazard_index][0], only_space[hazard_index][1], random.randint(min_size, max_size))
            hazard_num = hazard_num - 1
            if(hazard_num == 0):
                break

    def init_outside(self): #외곽 탈출로 구현 

        self.space_goal_dict[((0,0), (5, 95))] = [[0,0]]
        self.space_type[((0,0), (5, 95))] = 0
        self.space_index[((0,0), (5, 95))] = 1
        self.space_list.append([[0,0], [5,95]])

        self.space_goal_dict[((0,95), (5, 99))] = [[0,0]]
        self.space_type[((0,95), (5, 99))] = 0
        self.space_index[((0,95), (5, 99))] = 2
        self.space_list.append([[0, 95], [5, 99]]) 

        self.space_goal_dict[((95,5), (99, 99))] = [[5, 95]]
        self.space_type[((95,5), (99, 99))] = 0
        self.space_index[((95,5), (99, 99))] = 3
        self.space_list.append([[95, 5], [99, 99]])

        self.space_goal_dict[((5, 0), (99, 5))] = [[95, 5]] #외곽지대 골 설정
        self.space_type[((5, 0), (99, 5))] = 0
        self.space_index[((5, 0), (99, 5))] = 4
        self.space_list.append([[5, 0], [99, 5]])

    def connect_space(self):
        check_connection = []
        for i in range(101):
            tmp = []
            for j in range(101):
                tmp.append(0)
            check_connection.append(tmp)

        for space in self.space_list: #space끼리 연결 #space 그래프 만들기
            if space in self.room_list:
                continue
            check_connection = []
            for i1 in range(101):
                tmp = []
                for j1 in range(101):
                    tmp.append(0)
                check_connection.append(tmp)
            
            for y in range(space[0][1]+1, space[1][1]):
                check_connection[space[0][0]][y] = 1 #left 
            for y in range(space[0][1]+1, space[1][1]):
                check_connection[space[1][0]][y] = 1 #right
            for x in range(space[0][0]+1, space[1][0]):
                check_connection[x][space[0][1]] = 1 #down
            for x in range(space[0][0]+1, space[1][0]):
                check_connection[x][space[1][1]] = 1 #up

            for space2 in self.space_list:
                if space2 in self.room_list:
                    continue
                check_connection2 = copy.deepcopy(check_connection)
                checking = 0
                if(space == space2):
                    continue

                left_goal = [0, 0]
                left_goal_num = 0

                right_goal = [0, 0]
                right_goal_num = 0

                down_goal = [0, 0]
                down_goal_num = 0

                up_goal = [0, 0]
                up_goal_num = 0

                for y2 in range(space2[0][1]+1, space2[1][1]):
                    check_connection2[space2[0][0]][y2] += 1 #left 
                    if(check_connection2[space2[0][0]][y2] == 2):
                        #print(space, space2, "가 LEFT에서 만남")
                        left_goal[0] += space2[0][0]
                        left_goal[1] += y2
                        left_goal_num = left_goal_num + 1
                        checking = 1
                for y3 in range(space2[0][1]+1, space2[1][1]):
                    check_connection2[space2[1][0]][y3] += 1 #right
                    if(check_connection2[space2[1][0]][y3] == 2):
                        #print(space, space2, "가 RIGHT에서 만남")
                        right_goal[0] += space2[1][0]
                        right_goal[1] += y3
                        right_goal_num = right_goal_num + 1
                        checking = 1
                for x2 in range(space2[0][0]+1, space2[1][0]):
                    check_connection2[x2][space2[0][1]] += 1 #down
                    if(check_connection2[x2][space2[0][1]] == 2):
                        #print(space, space2, "가 DOWN에서 만남")
                        down_goal[0] += x2
                        down_goal[1] += space2[0][1]
                        down_goal_num = down_goal_num + 1
                        checking = 1
                for x3 in range(space2[0][0]+1, space2[1][0]):
                    check_connection2[x3][space2[1][1]] += 1 #up
                    if(check_connection2[x3][space2[1][1]] == 2):
                        #print(space, space2, "가 UP에서 만남")
                        up_goal[0] += x3
                        up_goal[1] += space2[1][1]
                        up_goal_num = up_goal_num + 1
                        checking = 1
                if (checking==1 and space != space2):
                    self.space_graph[((space[0][0], space[0][1]), (space[1][0], space[1][1]))].append(space2)
                
                if(left_goal[0] != 0 and left_goal[1] != 0):
                    first_left_goal = [0, 0]
                    first_left_goal[0] = (left_goal[0]/left_goal_num)
                    first_left_goal[1] = (left_goal[1]/left_goal_num)-3
                    second_left_goal = [0, 0]
                    second_left_goal[0] = (left_goal[0]/left_goal_num)
                    second_left_goal[1] = (left_goal[1]/left_goal_num)+3
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_left_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_left_goal))
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(first_left_goal)
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(second_left_goal)
                    
                elif(right_goal[0] != 0 and right_goal[1] != 0):
                    first_right_goal = [0, 0]
                    first_right_goal[0] = (right_goal[0]/right_goal_num)
                    first_right_goal[1] = (right_goal[1]/right_goal_num)-3
                    second_right_goal = [0, 0]
                    second_right_goal[0] = (second_right_goal[0]/right_goal_num)
                    second_right_goal[1] = (right_goal[1]/right_goal_num)+3
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_right_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_right_goal))
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(first_right_goal)
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(second_right_goal)

                elif(down_goal[0] != 0 and down_goal[1] != 0):
                    first_down_goal = [0, 0]
                    first_down_goal[0] = (down_goal[0]/down_goal_num)+3
                    first_down_goal[1] = (down_goal[1]/down_goal_num)
                    second_down_goal = [0, 0]
                    second_down_goal[0] = (down_goal[0]/down_goal_num)-3
                    second_down_goal[1] = (down_goal[1]/down_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_down_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_down_goal))
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(first_down_goal)
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(second_down_goal)

                elif(up_goal[0] != 0 and up_goal[1] != 0):
                    first_up_goal = [0, 0]
                    first_up_goal[0] = (up_goal[0]/up_goal_num)+3
                    first_up_goal[1] = (up_goal[1]/up_goal_num)
                    second_up_goal = [0, 0]
                    second_up_goal[0] = (up_goal[0]/up_goal_num)-3
                    second_up_goal[1] = (up_goal[1]/up_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_up_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_up_goal))
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(first_up_goal)
                    #self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(second_up_goal)


    def make_door_between_room(self):
        for i in self.room_list: #방과 방 사이에 문 만들기
            left_down = i[0]
            right_down = [i[1][0], i[0][1]]
            left_up = [i[0][0], i[1][1]]
            right_up = i[1]

            x_len = i[1][0] - i[0][0]
            y_len = i[1][1] - i[0][1]
            target = []
            for j in self.room_list:
                left_down_j = j[0]
                right_down_j = [j[1][0], j[0][1]]
                left_up_j = [j[0][0], j[1][1]]
                right_up_j = j[1]

                x_len_j = j[1][0] - j[0][0]
                y_len_j = j[1][1] - j[0][1]

                if(right_down == left_down_j):
                    if(y_len>y_len_j or (y_len==y_len_j and i[0]>j[0])):
                        target = [j[0], [j[0][0], j[1][1]]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(right_up == left_up_j):
                    if(y_len>y_len_j or (y_len==y_len_j and i[0]>j[0])):
                        target = [j[0], [j[0][0], j[1][1]]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(right_up == right_down_j):
                    if(x_len>x_len_j or (x_len==x_len_j and i[0]>j[0])):
                        target = [j[0], [j[1][0], j[0][1]]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(left_up == left_down_j):
                    if(x_len>x_len_j or (x_len==x_len_j and i[0]>j[0])):
                        target = [j[0], [j[1][0], j[0][1]]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(left_up == right_up_j):
                    if(y_len>y_len_j or (y_len==y_len_j and i[0]>j[0])):
                        target = [[j[1][0], j[0][1]], j[1]] #check
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(left_down == right_down_j):
                    if(y_len>y_len_j or (y_len==y_len_j and i[0]>j[0])):
                        target = [[j[1][0], j[0][1]], j[1]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(left_down == left_up_j):
                    if(x_len>x_len_j or (x_len==x_len_j and i[0]>j[0])):
                        target = [[j[0][0], j[1][1]], j[1]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list
                elif(right_down == right_up_j):
                    if(x_len>x_len_j or (x_len==x_len_j and i[0]>j[0])):
                        target = [[j[0][0], j[1][1]], j[1]]
                        new_door_list = make_door2(target[0], target[1], 4)
                        self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list))) #새로운 goal 넣어주기 (문)
                        self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))].append(goal_extend(((j[0][0], j[0][1]), (j[1][0], j[1][1])), goal_average(new_door_list)))
                        self.door_list = self.door_list + new_door_list

    def make_door_to_outside(self):
                
        door_to_outdoor = random.randint(1,6) #외부와 연결된 문 몇개 이하로 제한할건지 
        if(door_to_outdoor<3):
            door_to_outdoor = 1
        elif(door_to_outdoor<5):
            door_to_outdoor = 2
        else:
            door_to_outdoor = 3

        now_door_to_outdoor = 0

        for i in self.room_list: #외부와 연결된 문 만들기
            if(now_door_to_outdoor == door_to_outdoor):
                break

            if(i[0][0] == 10 and i[0][1]==10):
                x=random.randint(0,1)
                if(x):
                    new_door_list = make_door(i[0], [i[0][0], i[1][1]], 4)
                    self.door_list = self.door_list + new_door_list
                    self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                else:
                    new_door_list = make_door(i[0], [i[1][0], i[0][1]], 4)
                    self.door_list = self.door_list + new_door_list
                    self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1

            elif (i[0][0] == 10):
                new_door_list = make_door(i[0], [i[0][0], i[1][1]], 4)
                self.door_list = self.door_list + new_door_list
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1

            elif (i[0][1] == 10):
                new_door_list = make_door(i[0], [i[1][0], i[0][1]], 4)
                self.door_list = self.door_list + new_door_list
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1
    
            elif(i[1][0] == 90 and i[1][1]==90):
                x = random.randint(0,1)
                if(x):
                    new_door_list = make_door([i[1][0], i[0][1]], i[1], 4)
                    self.door_list = self.door_list + new_door_list
                    self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                else:
                    new_door_list = make_door([i[0][0], i[1][1]], i[1], 4)
                    self.door_list = self.door_list + new_door_list
                    self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1

            elif (i[1][0] == 90):
                new_door_list = make_door([i[1][0], i[0][1]], i[1], 4)
                self.door_list = self.door_list + new_door_list
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1

            elif (i[1][1] == 90):
                new_door_list = make_door([i[0][0], i[1][1]], i[1], 4)
                self.door_list = self.door_list + new_door_list
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))].append(goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), goal_average(new_door_list)))
                now_door_to_outdoor = now_door_to_outdoor + 1
    
    def make_one_door_in_room(self, r):

        check_door = []
        for i in range(101):
            tmp = []
            for j in range(101):
                tmp.append(0)
            check_door.append(tmp)

        for door in self.door_list:
            check_door[door[0]][door[1]] = 1
        
        left = 1 #같은 쪽 벽에 문이 두개 생기는 걸 막기 위함 
        right = 1
        up = 1
        down = 1 
        for y in range(r[0][1]+1, r[1][1]):
            if(check_door[r[0][0]][y] == 1): #left 
                left = 0
                break

        for y in range(r[0][1]+1, r[1][1]):
            if(check_door[r[1][0]][y] == 1): #right
                right = 0
        
        for x in range(r[0][0]+1, r[1][0]):
            if(check_door[x][r[0][1]] == 1): #down
                down = 0


        for x in range(r[0][0]+1, r[1][0]):
            if(check_door[x][r[1][1]] == 1): #down
                up = 0
        direction_list = []
        if(left==1):
            direction_list.append(0)
        if(right==1):
            direction_list.append(1)
        if(down==1):
            direction_list.append(2)
        if(up==1):
            direction_list.append(3)

        random_door = random.randint(0, len(direction_list)-1) #left, right, down, up쪽의 벽 중에 문이 없는 부분만 골라서 
                                                             #문이 없는 벽 중 랜덤으로 하나 골라서 문을 만들거야 
            
        if (direction_list[random_door] == 0): #left
            new_door_list = make_door(r[0], [r[0][0], r[1][1]], 4)
            self.door_list = self.door_list + new_door_list
            self.space_goal_dict[((r[0][0], r[0][1]), (r[1][0], r[1][1]))].append(goal_extend(((r[0][0], r[0][1]), (r[1][0], r[1][1])), goal_average(new_door_list)))
        elif (direction_list[random_door] == 1): #right
            new_door_list = make_door([r[1][0], r[0][1]], r[1], 4)
            self.door_list = self.door_list + new_door_list
            self.space_goal_dict[((r[0][0], r[0][1]), (r[1][0], r[1][1]))].append(goal_extend(((r[0][0], r[0][1]), (r[1][0], r[1][1])), goal_average(new_door_list)))
        elif (direction_list[random_door] == 2): #down
            new_door_list = make_door(r[0], [r[1][0], r[0][1]], 4)
            self.door_list = self.door_list + new_door_list
            self.space_goal_dict[((r[0][0], r[0][1]), (r[1][0], r[1][1]))].append(goal_extend(((r[0][0], r[0][1]), (r[1][0], r[1][1])), goal_average(new_door_list)))
        else: #up
            new_door_list = make_door([r[0][0], r[1][1]], r[1], 4)
            self.door_list = self.door_list + new_door_list
            self.space_goal_dict[((r[0][0], r[0][1]), (r[1][0], r[1][1]))].append(goal_extend(((r[0][0], r[0][1]), (r[1][0], r[1][1])), goal_average(new_door_list)))


        # if(which_wall !=0 and which_wall != 1 and which_wall != 2 and which_wall !=3)
        #     for x in range(r[0][0]+1, r[1][0]):
        #     check_door[x][r[1][1]] = 1 #up

        
    def space_connect_via_door(self):
        check_door = []
        for i in range(101):
            tmp = []
            for j in range(101):
                tmp.append(0)
            check_door.append(tmp)

        for door in self.door_list:
            check_door[door[0]][door[1]] = 1


        for space in self.space_list: #문 있는 곳 끼리 연결 #space 그래프 만들기
            door_connection = []
            for i1 in range(101):
                tmp = []
                for j1 in range(101):
                    tmp.append(0)
                door_connection.append(tmp)
            
            for y in range(space[0][1]+1, space[1][1]):
                door_connection[space[0][0]][y] = 1 #left 
            for y in range(space[0][1]+1, space[1][1]):
                door_connection[space[1][0]][y] = 1 #right
            for x in range(space[0][0]+1, space[1][0]):
                door_connection[x][space[0][1]] = 1 #down
            for x in range(space[0][0]+1, space[1][0]):
                door_connection[x][space[1][1]] = 1 #up

            for space2 in self.space_list:
                door_connection2 = copy.deepcopy(door_connection)
                checking = 0
                if(space == space2):
                    continue
                for y2 in range(space2[0][1]+1, space2[1][1]):
                    #check_connection2[space2[0][0]][y2] += 1 #left 
                    if(door_connection2[space2[0][0]][y2] == 1 and check_door[space2[0][0]][y2]==1):
                        #print(space, space2, "가 LEFT에서 만남")
                        checking = 1
                for y3 in range(space2[0][1]+1, space2[1][1]):
                    #check_connection2[space2[1][0]][y3] += 1 #right
                    if(door_connection2[space2[1][0]][y3] == 1 and check_door[space2[1][0]][y3]==1):
                        #print(space, space2, "가 RIGHT에서 만남")
                        checking = 1
                for x2 in range(space2[0][0]+1, space2[1][0]):
                    #check_connection2[x2][space2[0][1]] += 1 #down
                    if(door_connection2[x2][space2[0][1]] == 1 and check_door[x2][space2[0][1]]):
                        #print(space, space2, "가 DOWN에서 만남")
                        checking = 1
                for x3 in range(space2[0][0]+1, space2[1][0]):
                    #check_connection2[x3][space2[1][1]] += 1 #up
                    if(door_connection2[x3][space2[1][1]] == 1 and check_door[x3][space2[1][1]]):
                        #print(space, space2, "가 UP에서 만남")
                        checking = 1
                if (checking==1 and space != space2):
                    if(space2 not in self.space_graph[((space[0][0], space[0][1]), (space[1][0], space[1][1]))]): #인접 리스트에 들어있지 않으면
                        self.space_graph[((space[0][0], space[0][1]), (space[1][0], space[1][1]))].append(space2)
        





    def make_hazard(self, xy1, xy2, depth):
        new_plane = []
    
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

        x1 = xy1[0]
        y1 = xy1[1]
        x2 = xy2[0]
        y2 = xy2[1]
        x_len = x2-x1
        y_len = y2-y1
        # check_list = [[0]*y_len for _ in range(x_len)]


        # hazard_size = random.randint(min_area, max_area)
        hazard_start = (random.randint(x1, x2)+1, random.randint(y1, y2)-1)
        self.hazard_recur(hazard_start[0], hazard_start[1], depth, [x1, x2], [y1, y2])

    
    def hazard_recur(self, x, y, depth, x_range, y_range):
        global hazard_id
        if (x<(x_range[0]+1) or x>(x_range[1]-1) or y<(y_range[0]+1) or y>(y_range[1]-1) or depth==0):
            return 
        a = FightingAgent(hazard_id, self,[x,y], 1)
        self.schedule_h.add(a)
        self.grid.place_agent(a, (x, y))
        hazard_id = hazard_id + 1
        self.hazard_recur(x-1, y, depth-1, x_range, y_range)
        self.hazard_recur(x+1, y, depth-1, x_range, y_range)
        self.hazard_recur(x, y+1, depth-1, x_range, y_range)
        self.hazard_recur(x, y-1, depth-1, x_range, y_range)
       
    # def random_map_generator2(self, room_min, room_max, exit_num, map_size):
    #     rooms = []
    #     x_unit = int(map_size/10)
    #     y_unit = int(map_size/10)
    #     for i in range(10): 
    #         for j in range(10):
    #             #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit])
    #             #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit, j*y_unit+y_unit-1])
    #             #rooms = rooms + make_plane([i*x_unit+x_unit-1, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
    #             #rooms = rooms + make_plane([i*x_unit, j*y_unit+y_unit-1], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
    #             #rooms = rooms + make_room([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
    #             self.map_divide[i][j] = [[i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1]]
        
    #     room_num = random.randint(room_min, room_max) #how many rooms?
    #     room_type = random.randint(1, 3) # what type the room is? 

    #     now_room_num = 0
    #     while_breaker = 0

    #     while(now_room_num != room_num):
    #         while_breaker = while_breaker+1
    #         if (while_breaker>1000):
    #             break
    #         continue_while = 0    
    #         room_x_len = random.randint(2,5)
    #         room_y_len = random.randint(2,5)
    #         room_start_xy = [random.randint(1, 10-room_x_len), random.randint(1, 10-room_y_len)]

    #         for i in range(room_start_xy[0], room_start_xy[0]+room_x_len+1):
    #             for j in range(room_start_xy[1], room_start_xy[1]+room_y_len+1):
    #                 if (self.map_repre[i][j] == 1):
    #                     continue_while = 1
    #                     break 
    #                 else :
    #                     self.map_repre[i][j] = 1
    #             if(continue_while==1):
    #                 break
    #         if(continue_while):
    #             continue
    #         rooms = rooms+make_room([room_start_xy[0]*x_unit, room_start_xy[1]*y_unit], [(room_start_xy[0]+room_x_len)*x_unit-1, (room_start_xy[1]+room_y_len)*y_unit-1])
    #         now_room_num = now_room_num + 1
    #         self.room_list.append([[room_start_xy[0]*x_unit, room_start_xy[1]*y_unit], [(room_start_xy[0]+room_x_len)*x_unit-1, (room_start_xy[1]+room_y_len)*y_unit-1]])
            

    #     #room_type = random.radint()
    #     return rooms

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
        
        

        # if (num%2==0): #가로로 나눈다
        #     self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[0][0]+int(x_diff*divide_num/6), xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        #     self.map_recur_divider([[xy[0][0]+int(x_diff*divide_num/6)+1, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        
        # else: #세로로 나눈다
        #     self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+int(y_diff*divide_num/6)]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        #     self.map_recur_divider([[xy[0][0], xy[0][1]+int(y_diff*divide_num/6)+1], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room) 
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
    

    def map_recur_divider_fine(self, xy, x_unit, y_unit, num, space_list, room_list, is_room): # ex) xy = [[2,3], [4,5]]
        x_diff = xy[1][0] - xy[0][0]
        y_diff = xy[1][1] - xy[0][1]
        print("num : ", num)

        real_xy =  [ [xy[0][0]*x_unit, xy[0][1]*y_unit], [xy[1][0]*x_unit, xy[1][1]*y_unit]]
        if(is_room==0):
            space_list.append(real_xy)
            return
                   
        if(x_diff<3 or y_diff<3):
            space_list.append(real_xy)
            room_list.append(real_xy)
            return
        
            
        if(num==1): 
            a = random.randint(1,20)
            if(a<1):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return 
        elif(num==2):
            a = random.randint(1,20)
            if(a<3):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return
        elif(num==3):
            a = random.randint(1,20)
            if(a<4):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==4):
            a = random.randint(1,20)
            if(a<10):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==5):
            a = random.randint(1,20)
            if(a<20):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return
        # elif(num==6):
        #     a = random.randint(1,20)
        #     if(a<8):
        #         space_list.append(real_xy)
        #         room_list.append(real_xy)
        #         return
        # elif(num==7):
        #     a = random.randint(1,20)
        #     if(a<10):
        #         space_list.append(real_xy)
        #         room_list.append(real_xy)
        #         return
        # elif(num==8):
        #     a = random.randint(1,20)
        #     if(a<20):
        #         space_list.append(real_xy)
        #         room_list.append(real_xy)
        #         return
            
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
        
        

        # if (num%2==0): #가로로 나눈다
        #     self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[0][0]+int(x_diff*divide_num/6), xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        #     self.map_recur_divider([[xy[0][0]+int(x_diff*divide_num/6)+1, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        
        # else: #세로로 나눈다
        #     self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+int(y_diff*divide_num/6)]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room)
        #     self.map_recur_divider([[xy[0][0], xy[0][1]+int(y_diff*divide_num/6)+1], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room) 
        special_hallway = random.randint(1, 2)
        if(num<3):
            print(num, "번째 쪼갭니다")
            if (num%2==0): #가로로 나눈다
                left = int(x_diff*random.randint(1,3)/4)
                hallway_size = random.randint(1,2)
                if(xy[0][0]+left+hallway_size >= (xy[1][0]-2)):
                    self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                    return
                # if(x_diff<13):
                #     left = int(x_diff*(1/2))
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[0][0]+left, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider_fine([[xy[0][0]+left, xy[0][1]], [xy[0][0]+left+hallway_size, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 0)
                self.map_recur_divider_fine([[xy[0][0]+left+hallway_size, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
            else: #세로로 나눈다
                up = int(y_diff*random.randint(1,3)/4)
                hallway_size = random.randint(1,2)
                if(xy[0][1]+up+hallway_size >= (xy[1][1]-2)):
                    self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                    return
                # if(y_diff<13):
                #     up = int(y_diff*(1/2))
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+up]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]+up], [xy[1][0], xy[0][1]+up+hallway_size]], x_unit, y_unit, num+1, space_list, room_list, 0)
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]+up+hallway_size], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)

        else:
            if(num<1):
                random_exist_room1 = random_exist_room2 = random_exist_room3 = random_exist_room4 = 1
            if (num%2==0): #가로로 나눈다
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[0][0]+divide_num_x, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room1)
                self.map_recur_divider_fine([[xy[0][0]+divide_num_x, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room2)
        
            else: #세로로 나눈다
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+divide_num_y]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room3)
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]+divide_num_y], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room4) 
    
    

    def agent_place(self, xy1, xy2, num):
    
        agent_list = []
        x_len = xy2[0]-xy1[0]
        y_len = xy2[1]-xy1[1]

        check_list = [[0]*y_len for _ in range(x_len)]
        while(num>0):
            x = random.randint(xy1[0]+1, xy2[0]-1)
            y = random.randint(xy1[1]+1, xy2[1]-1)
            #print(x, xy1[0])
            #print(y, xy1[1])
            while(check_list[x-xy1[0]][y-xy1[1]] == 1):
                x = random.randint(xy1[0]+1, xy2[0]-1)
                y = random.randint(xy1[1]+1, xy2[1]-1)
            check_list[x-xy1[0]][y-xy1[1]] = 1
            num = num-1
            a = FightingAgent(self.agent_id, self, [x,y], 0)
            self.agent_id = self.agent_id + 1
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))


    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model

        # Checking if there is a champion
        if FightingModel.current_healthy_agents(self) == 0:
            self.running = False

    @staticmethod
    def current_healthy_agents(model) -> int:
        """Returns the total number of healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule.agents if agent.health > 0]) ### agent의 health가 0이어야 cureent_healthy_agents 수에 안 들어감
                                                                               ### agent.py 에서 exit area 도착했을 때 health를 0으로 바꿈

    @staticmethod
    def current_non_healthy_agents(model) -> int:
        """Returns the total number of non healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule.agents if agent.health == 0])
