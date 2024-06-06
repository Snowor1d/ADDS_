#this source code requires Mesa==2.2.1 
#^__^
from mesa import Model
from agent import FightingAgent
from agent2 import FightingAgent2
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

import agent
from agent import WallAgent
import random
import copy
import math

# goal_list = [[0,50], [49, 50]]
hazard_id = 5000
total_crowd = 10 
max_specification = [20, 20]

number_of_cases = 0 # 난이도 함수 ; 경우의 수
started = 1

def make_plane(xy1: int, xy2: int): # 두 좌표를 받고, (이를 모서리로 하는 평면)을 구성하는 점들의 집합을 도출
    '''
    히히히 xy1 =[x1, y1], xy2 =[x2, y2]
    '''
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

def make_room(xy1, xy2): # 두 좌표를 양쪽 꼭짓점으로 하는 방을 만듦(벽을 구상하고 있는 점들을 return)
    new_plane = []
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    
    rooms = []
    rooms = rooms + make_plane([x1, y1], [x2, y1]) #아래벽
    rooms = rooms + make_plane([x1, y1], [x1, y2]) #왼벽
    rooms = rooms + make_plane([x2, y1], [x2, y2]) #오른벽
    rooms = rooms + make_plane([x1, y2], [x2, y2]) #욋벽 

    return rooms

def make_door(xy1, xy2, door_size): #xy1의 좌표~xy2의 좌표 사이 임의의 위치에 door_size 크기의 문을 만듦 
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
def goal_average(xys): #좌표들의 평균 도출 
    middle_x = 0
    middle_y = 0
    for i in xys:
        middle_x += i[0]
        middle_y += i[1]
    middle_x = middle_x/len(xys)
    middle_y = middle_y/len(xys)
    return [middle_x, middle_y]

def space_connected_linear(xy1, xy2): # 두 공간 사이에 겸치는 지점들의 중앙값을 return 
    # ex) xy1, xy2는 [(int, int), (int, int)] 형태
    check_connection = [] #어느 점이 겹치는지 체크 

    for i1 in range(51):
        tmp = []
        for j1 in range(51):
            tmp.append(0)
        check_connection.append(tmp)
    
    for y in range(xy1[0][1]+1, xy1[1][1]):
        check_connection[xy1[0][0]][y] = 1 #left 
    for y in range(xy1[0][1]+1, xy1[1][1]):
        check_connection[xy1[1][0]][y] = 1 #right
    for x in range(xy1[0][0]+1, xy1[1][0]):
        check_connection[x][xy1[0][1]] = 1 #down
    for x in range(xy1[0][0]+1, xy1[1][0]):
        check_connection[x][xy1[1][1]] = 1 #up


    check_connection2 = copy.deepcopy(check_connection)
    checking = 0

    left_goal = [0, 0]
    left_goal_num = 0 #left_goal -> 왼쪽 벽이 연결되어 있으면 활성화 

    right_goal = [0, 0]
    right_goal_num = 0

    down_goal = [0, 0]
    down_goal_num = 0

    up_goal = [0, 0]
    up_goal_num = 0

    for y2 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[0][0]][y2] += 1 #left 
        if(check_connection2[xy2[0][0]][y2] == 2): #space와 space2는 접한다 

            left_goal[0] += xy2[0][0]
            left_goal[1] += y2
            left_goal_num = left_goal_num + 1
            checking = 1 #이을거다~ (space와 space2를 )
    for y3 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[1][0]][y3] += 1 #right
        if(check_connection2[xy2[1][0]][y3] == 2):

            right_goal[0] += xy2[1][0]
            right_goal[1] += y3
            right_goal_num = right_goal_num + 1
            checking = 1
    for x2 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x2][xy2[0][1]] += 1 #down
        if(check_connection2[x2][xy2[0][1]] == 2):
 
            down_goal[0] += x2
            down_goal[1] += xy2[0][1]
            down_goal_num = down_goal_num + 1
            checking = 1
    for x3 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x3][xy2[1][1]] += 1 #up
        if(check_connection2[x3][xy2[1][1]] == 2):

            up_goal[0] += x3
            up_goal[1] += xy2[1][1]
            up_goal_num = up_goal_num + 1
            checking = 1

    if(left_goal[0] != 0 and left_goal[1] != 0):
        first_left_goal = [0, 0]
        first_left_goal[0] = (left_goal[0]/left_goal_num)
        first_left_goal[1] = (left_goal[1]/left_goal_num)
        return first_left_goal
        
    elif(right_goal[0] != 0 and right_goal[1] != 0):
        first_right_goal = [0, 0]
        first_right_goal[0] = (right_goal[0]/right_goal_num)
        first_right_goal[1] = (right_goal[1]/right_goal_num)
        return first_right_goal
    
    elif(down_goal[0] != 0 and down_goal[1] != 0):
        first_down_goal = [0, 0]
        first_down_goal[0] = (down_goal[0]/down_goal_num)
        first_down_goal[1] = (down_goal[1]/down_goal_num)
        return first_down_goal
 

    elif(up_goal[0] != 0 and up_goal[1] != 0):
        first_up_goal = [0, 0]
        first_up_goal[0] = (up_goal[0]/up_goal_num)
        first_up_goal[1] = (up_goal[1]/up_goal_num)
        return first_up_goal

    return [0, 0]
 
    
    
def make_door2(xy1, xy2, door_size): #두 room 사이에 벽 뚫기 (문 만들기)
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
    
def goal_extend(xy_space, goal): #goal의 목적은 space를 벗어나게 하려는 건데, goal이 space 벽에 있다면 agent가 잘 벗어나려 하지 않음. 그래서 goal을 space 밖으로 좀 연장하기 위한 함수 (return : 연장된 goal 좌표)
    xy = [0, 0]
    xy[0] = (xy_space[0][0] + xy_space[1][0])/2
    xy[1] = (xy_space[0][1] + xy_space[1][1])/2
    d = math.sqrt(pow(goal[0]-xy[0],2) + pow(goal[1]-xy[1], 2))
    if (d!=0):
        return [goal[0] + 1*(goal[0]-xy[0])/d, goal[1] + 1*(goal[1]-xy[1])/d]
    return [goal[0], goal[1]]
    
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


class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int):
        self.exit_way_rec =  [[0]*51 for _ in range(51)]
        self.robot = None 
        self.simulation_type = 0 #0->outdoor, #1->indoor
        self.room_list = [] # ex) [((1, 2), (3,4)), ((4,5), (5,6))]
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
        

        exit_rec = self.make_exit() 
        self.exit_rec = exit_rec

        # 벽을 agent로 표현하게 됨. agent 10이 벽이다.


        wall = [] 
        space = []
        self.wall_matrix = list()
        self.only_one_wall = list()
        self.indoor_connect = list() # 방과 방 사이를 연결하는 문을 만들기 위한 리스트
        self.valid_space = list() 
        for i in range(51):
            tmp = []
            tmp2 = []
            for j in range(51):
                tmp.append(0)
                tmp2.append(1)
            self.wall_matrix.append(tmp)
            self.only_one_wall.append(tmp)
            self.indoor_connect.append(tmp) #50x50 맵 초기화 
            self.valid_space.append(tmp2)
        
        NUMBER_OF_CELLS = 50

        for i in range(int(NUMBER_OF_CELLS)): #최외각 벽 세우기
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
        self.space_goal_dict = {} #각 space가 가지는 goal을 표현하기 위함
        self.space_graph = {} #각 space의 인접 space를 표현하기 위함
        self.space_type = {} #space type이 0이면 빈 공간, 1이면 room


        self.init_outside() #외곽지대 탈출로 구현 
        
        self.door_list = [] #일단 무시
        self.map_recur_divider_fine([[1, 1], [9, 9]], 5, 5, 0, self.space_list, self.room_list, 1) # recursion을 이용해 랜덤으로 맵을 나눔 
        #print(self.space_list)
        notvalid_list = []
        for i in self.room_list:
            notvalid_list.extend(make_plane(i[0], i[1]))
        notvalid_list.extend(make_plane((0,0), (49,0)))
        notvalid_list.extend(make_plane((0,0), (0, 49)))
        notvalid_list.extend(make_plane((49, 0), (49, 49)))
        notvalid_list.extend(make_plane((0, 49), (49, 49)))
        for j in notvalid_list :
            self.valid_space[j[0]][j[1]] =0
        for j in self.space_list: 
            self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = [] # 모든 space에 대한 goal을 설정할 것임
            self.space_graph[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = []
        for k in self.room_list:
            self.space_type[((k[0][0], k[0][1]), (k[1][0], k[1][1]))] = 1 # self.space_type에서는 room 이면 value의 값을 1로 설정 

        self.connect_space_with_one_goal() #space 그래프 연결 
        
        if(self.simulation_type): # 만약 room이 있는 시뮬레이션이라면..
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
        global total_crowd
        #self.random_agent_distribute_outdoor(total_crowd)
        #self.random_hazard_placement(random.randint(1,3))
        
        self.exit_compartment = ((0,0), (0, 0)) ##출구 위치 저장

        if(self.is_left_exit):
            self.space_goal_dict[((0,0), (5, 45))] = [self.left_exit_goal]
            self.exit_compartment = ((0,0), (5, 45))

        if(self.is_up_exit):
            self.space_goal_dict[((0,45), (45, 49))] = [self.up_exit_goal]
            self.exit_compartment = ((0,45), (45, 49))

        if(self.is_right_exit):
            self.space_goal_dict[((45,5), (49, 49))] = [self.right_exit_goal]
            self.exit_compartment = ((45,5), (49, 49))

        if(self.is_down_exit):
            self.space_goal_dict[((5,0), (49, 5))] = [self.down_exit_goal]
            self.exit_compartment = ((5,0), (49, 5))

        #exit 구역의 goal 재정의
#---------------------------------------------------------------------------------------------------------------------
        self.space_agent_num = {} #각 space에 agent가 몇명 있는가..
        for i in self.space_list:
            self.space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0

        self.outdoor_space = [] #outdoor_space (방 아닌 것들)
        for i in self.space_list: 
            if i in self.room_list:
                continue 
            self.outdoor_space.append(i)

        self.grid_to_space = list() # self.grid_to_space 존재 이유 : grid 좌표를 입력하면 곧 바로 해당하는 space를 내보내기 위함 
        for i in range(51):
            tmp = []
            for j in range(51):
                tmp.append([])
            self.grid_to_space.append(tmp)
        for space in self.space_list:
            x1 = space[0][0]
            x2 = space[1][0]
            y1 = space[0][1]
            y2 = space[1][1]
            for x in range(x1, x2+1):
                for y in range(y1, y2+1):
                    self.grid_to_space[x][y] = space 


        #print(self.space_goal_dict)

        for i in self.room_list:
            wall = wall + make_room(i[0], i[1])
        for j in self.space_list:
            space = space + make_room(j[0], j[1])

        set_transform = set(wall)
        wall = list(set_transform)
        # for i in goal_list:
        #     for j in i:
        #         if j in wall:    
        #             wall.remove(j)
        #             self.wall_matrix[j[0]][j[1]] = 0
    
        for i in self.door_list:
                if i in wall:    
                    wall.remove(i)
                    self.wall_matrix[i[0]][i[1]] = 0

        

        self.way_to_exit() #탈출구와 연결된 space들은 탈출구로 향하게 하기  
        
        ###@@ 아래 고치고 싶었으나 left_exit_goal 까지 바꿔야 해서 일단 pass.. 
        if(self.is_left_exit):
            self.space_goal_dict[((0,0), (5, 45))] = [self.left_exit_goal]

        if(self.is_up_exit):
            self.space_goal_dict[((0,45), (45, 49))] = [self.up_exit_goal]

        if(self.is_right_exit):
            self.space_goal_dict[((45,5), (49, 49))] = [self.right_exit_goal]

        if(self.is_down_exit):
            self.space_goal_dict[((5,0), (49, 5))] = [self.down_exit_goal]
        

        self.floyd_warshall_matrix = self.floyd_warshall() 
        #floyd_warshall() 함수는 두 개의 이중 딕셔너리를 리턴함
        # 첫 번째 이중 딕셔너리는 start space 부터 end space까지 경로
        # 두 번째 이중 딕셔너리는 start space 부터 end space까지의 거리 
        
        self.floyd_path = self.floyd_warshall_matrix[0]
        self.floyd_distance = self.floyd_warshall_matrix[1]

        vertices = list(self.space_graph.keys()) # space_graph에서 key를 추출 (모든 공간이 담김)
        goal_matrix = {start: {end: float('infinity') for end in vertices} for start in vertices}

        for i in vertices:
            for j in vertices:
                if (i==j):
                    continue
                goal_matrix[i][j] = space_connected_linear(i, j) # 공간 i 와 공간 j 사이에 골 찍기 
        self.dict_NoC = {}
        self.difficulty_f()
        
        self.wall = wall 
        self.space = space
        self.exit_rec = exit_rec

        # self.make_agents()
        # self.random_agent_distribute_outdoor(10)
        # self.make_robot()

        

    def make_robot(self):
        self.robot_placement() #로봇 배치 
    def make_agents(self):
        
        self.wall = [list(t) for t in self.wall]
        self.exit_rec = [list(t) for t in self.exit_rec]
        self.exit_way_rec =  [list(t) for t in self.exit_way_rec]
        self.space = [list(t) for t in self.space]

        for i in range(len(self.exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, self.exit_rec[i]) ##exit_rec 에 agents 채우기
        for i in range(len(self.wall)):
            if (self.only_one_wall[self.wall[i][0]][self.wall[i][1]] == 1 and self.wall[i][0]!=0 and self.wall[i][1]!=0 and self.wall[i][1]!=49):
                continue
            c = FightingAgent(i, self, self.wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, self.wall[i])
            self.only_one_wall[self.wall[i][0]][self.wall[i][1]] = 1
        for i in range(len(self.space)):
            if (self.only_one_wall[self.space[i][0]][self.space[i][1]] == 1 ):
                continue
            c = FightingAgent(10000+i, self, self.space[i], 12)
            self.schedule_w.add(c)
            self.grid.place_agent(c, self.space[i])
            self.only_one_wall[self.space[i][0]][self.space[i][1]] = 1 

        count = 0
        for i in range(50):
            for j in range(50):
                if(self.exit_way_rec[i][j]==1):
                    b = FightingAgent(count+20300, self, [0,0], 20)
                    count += 1
                    self.schedule_e.add(b) 
                    self.grid.place_agent(b, [i, j])


    def make_agents2(self):
        self.wall = [list(t) for t in self.wall]
        self.exit_rec = [list(t) for t in self.exit_rec]
        self.space = [list(t) for t in self.space]
        for i in range(len(self.exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent2(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, self.exit_rec[i]) ##exit_rec 에 agents 채우기
        for i in range(len(self.wall)):
            if (self.only_one_wall[self.wall[i][0]][self.wall[i][1]] == 1 and self.wall[i][0]!=0 and self.wall[i][1]!=0 and self.wall[i][1]!=49):
                continue
            c = FightingAgent2(i, self, self.wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, self.wall[i])
            self.only_one_wall[self.wall[i][0]][self.wall[i][1]] = 1
        for i in range(len(self.space)):
            if (self.only_one_wall[self.space[i][0]][self.space[i][1]] == 1 ):
                continue
            c = FightingAgent2(10000+i, self, self.space[i], 12)
            self.schedule_w.add(c)
            self.grid.place_agent(c, self.space[i])
            self.only_one_wall[self.space[i][0]][self.space[i][1]] = 1  
        count = 0
        for i in range(50):
            for j in range(50):
                if(self.exit_way_rec[i][j]==1):
                    b = FightingAgent(count+20300, self, [0,0], 20)
                    count += 1
                    self.schedule_e.add(b) 
                    self.grid.place_agent(b, [i, j])
          

    def make_exit(self):
        exit_rec = []
        only_one_exit = random.randint(1,4) #현재는 출구가 하나만 있게 함 
        self.exit_goal = [0,0]
        self.is_down_exit = 0
        self.is_left_exit = 0
        self.is_up_exit = 0
        self.is_right_exit = 0

        if(only_one_exit == 1):
            self.is_down_exit = 1
        elif(only_one_exit == 2):
            self.is_left_exit = 1
        elif(only_one_exit == 3):
            self.is_up_exit = 1
        elif(only_one_exit == 4):
            self.is_right_exit = 1
        # self.is_down_exit = random.randint(0,1)
        # self.is_left_exit = random.randint(0,1) #0이면 출구없음 #1이면 출구있음
        # self.is_up_exit = random.randint(0,1)
        # self.is_right_exit = 0
        if (self.is_down_exit==0 and self.is_left_exit==0 and self.is_up_exit==0):
            self.is_right_exit = 1  #출구 넷 중에 하나 이상은 되게 한다~! 
        else:
            self.is_right_exit = 0

        left_exit_num = 0
        self.left_exit_goal = [0,0]
        if(self.is_left_exit): #left에 존재하면?
            exit_size = random.randint(10, 30) #출구 사이즈를 30~70 정한다 
            start_exit_cell = random.randint(0, 49-exit_size) #출구가 어디부터 시작되는가? #넘어갈까봐
            for i in range(0, 5): 
                for j in range(start_exit_cell, start_exit_cell+exit_size): #채운다~
                    exit_rec.append((i,j)) #exit_rec에 떄려 넣는다~
                    self.left_exit_goal[0] += i
                    self.left_exit_goal[1] += j
                    left_exit_num +=1
            self.left_exit_goal[0] = self.left_exit_goal[0]/left_exit_num #출구 좌표의 평균 
            self.left_exit_goal[1] = self.left_exit_goal[1]/left_exit_num
            self.left_exit_area = [[0, start_exit_cell], [5, start_exit_cell+exit_size]]
            self.exit_goal = [self.left_exit_goal[0], self.left_exit_goal[1]]
        right_exit_num = 0    
        self.right_exit_goal = [0,0]
        if(self.is_right_exit):
            exit_size = random.randint(10, 30)
            start_exit_cell = random.randint(0, 49-exit_size)
            for i in range(45, 50):
                for j in range(start_exit_cell, start_exit_cell+exit_size):
                    exit_rec.append((i,j))
                    self.right_exit_goal[0] += i
                    self.right_exit_goal[1] += j
                    right_exit_num +=1
            self.right_exit_goal[0] = self.right_exit_goal[0]/right_exit_num
            self.right_exit_goal[1] = self.right_exit_goal[1]/right_exit_num
            self.right_exit_area = [[45, start_exit_cell], [49, start_exit_cell+exit_size]]
            self.exit_goal = [self.right_exit_goal[0], self.right_exit_goal[1]]

        down_exit_num = 0    
        self.down_exit_goal = [0,0]
        if(self.is_down_exit):
            exit_size = random.randint(10, 30)
            start_exit_cell = random.randint(0, 49-exit_size)
            for i in range(start_exit_cell, start_exit_cell+exit_size):
                for j in range(0, 5):
                    exit_rec.append((i,j))
                    self.down_exit_goal[0] += i
                    self.down_exit_goal[1] += j
                    down_exit_num +=1
            self.down_exit_goal[0] = self.down_exit_goal[0]/down_exit_num
            self.down_exit_goal[1] = self.down_exit_goal[1]/down_exit_num
            self.down_exit_area = [[start_exit_cell, 0], [start_exit_cell+exit_size, 5]]
            self.exit_goal = [self.down_exit_goal[0], self.down_exit_goal[1]]

        up_exit_num = 0    
        self.up_exit_goal = [0,0]
        if(self.is_up_exit):
            exit_size = random.randint(10, 30)
            start_exit_cell = random.randint(0, 49-exit_size)
            for i in range(start_exit_cell, start_exit_cell+exit_size):
                for j in range(45, 50):
                    exit_rec.append((i,j))
                    self.up_exit_goal[0] += i
                    self.up_exit_goal[1] += j
                    up_exit_num = up_exit_num + 1
            self.up_exit_goal[0] = self.up_exit_goal[0]/up_exit_num
            self.up_exit_goal[1] = self.up_exit_goal[1]/up_exit_num
            self.up_exit_area = [[start_exit_cell, 45], [start_exit_cell+exit_size, 49]]
            self.exit_goal = [self.up_exit_goal[0], self.up_exit_goal[1]]
        #exit_rec에는 탈출 점들의 좌표가 쌓임
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
    def way_to_exit(self):
        visible_distance = 6
        x1=100 
        x2= 0
        y1= 100
        y2=0
        for i in self.exit_rec:
            if i[0]>x2:
                x2 = i[0]
            if i[0]<x1:
                x1 = i[0]
            if i[1]>y2:
                y2 = i[1]
            if i[1]<y1:
                y1 = i[1]
        for i in range(y1, y2+1):
            #self.recur_exit(x1, i, visible_distance, "ul")
            self.recur_exit(x1, i, visible_distance, "l")
            #self.recur_exit(x1, i, visible_distance, "dl")
        for i in range(x1, x2+1):
            #self.recur_exit(i, y2, visible_distance, "lu")
            #self.recur_exit(i, y2, visible_distance, "ru")
            self.recur_exit(i, y2, visible_distance, "u")
        for i in range(x1, x2+1):
            #self.recur_exit(i, y1, visible_distance, "ld")
            #self.recur_exit(i, y1, visible_distance, "rd")
            self.recur_exit(i, y1, visible_distance, "d")
        for i in range(y1, y2+1):
            #self.recur_exit(x2, i, visible_distance, "ur")
            self.recur_exit(x2, i, visible_distance, "r")
            #self.recur_exit(x2, i, visible_distance, "dr")
        # for i in range(0, 50):
        #     for j in range(0, 50):
        #         if(self.exit_way_rec[i][j] == 1):
        #             print("x, y :", i, j)
                    

    def recur_exit(self, x, y, visible_distance, direction):
        if(visible_distance < 1):
            return 
        if(x==0 or y==0 or x==49 or y==49):
            return
        if(self.grid_to_space[x][y] in self.room_list):
            return
        self.exit_way_rec[x][y] = 1         
        if direction=="l":
            self.recur_exit(x-1, y-1, visible_distance-2, "l")
            self.recur_exit(x-1, y, visible_distance-1, "l")
            self.recur_exit(x-1, y+1, visible_distance-2, "l")
        elif direction =="r":
            self.recur_exit(x+1, y+1, visible_distance-2, "r")
            self.recur_exit(x+1, y, visible_distance-1, "r")
            self.recur_exit(x+1, y-1, visible_distance-2, "r")
        elif direction =="u":
            self.recur_exit(x-1, y+1, visible_distance-2, "u")
            self.recur_exit(x, y+1, visible_distance-1, "u")
            self.recur_exit(x+1, y+1, visible_distance-2, "u")
        else :
            self.recur_exit(x+1, y-1, visible_distance-2, "d")
            self.recur_exit(x, y-1, visible_distance-1, "d")
            self.recur_exit(x-1, y-1, visible_distance-2, "d")
        



    def robot_placement(self): # 야외 공간에 무작위로 로봇 배치 
        inner_space = []
        for i in self.outdoor_space:
            if (i!=[[0,0], [5, 45]] and i!=[[45,5], [49, 49]] and i != [[0,45], [45, 49]] and i !=[[5,0], [49, 5]]):
                inner_space.append(i)
        space_index = 0 
        if(len(inner_space) > 1):
            space_index = random.randint(0, len(inner_space)-1)
        else :
            space_index = 0
        if(len(inner_space) == 0):
            return
        xy = inner_space[space_index]

    
        x_len = xy[0][0] - xy[1][0]
        y_len = xy[1][0] - xy[1][1]


        x = random.randint(xy[0][0]+1, xy[1][0]-1)
        y = random.randint(xy[0][1]+1, xy[1][1]-1)


        self.robot = FightingAgent(self.agent_id, self, [x,y], 3)
        self.agent_id = self.agent_id + 1
        self.schedule.add(self.robot)
        self.grid.place_agent(self.robot, (x, y))

    def robot_respawn(self):
        inner_space = []
        for i in self.outdoor_space:
            if (i!=[[0,0], [5, 45]] and i!=[[45,5], [49, 49]] and i != [[0,45], [45, 49]] and i !=[[5,0], [49, 5]]):
                inner_space.append(i)
        space_index = 0 
        if(len(inner_space) > 1):
            space_index = random.randint(0, len(inner_space)-1)
        else :
            space_index = 0
        if(len(inner_space) == 0):
            return
        xy = inner_space[space_index]

    
        x_len = xy[0][0] - xy[1][0]
        y_len = xy[1][0] - xy[1][1]


        x = random.randint(xy[0][0]+1, xy[1][0]-1)
        y = random.randint(xy[0][1]+1, xy[1][1]-1)


        return [x, y]

    
    
    def random_agent_distribute_outdoor(self, agent_num, ran):
        
        # case1 -> 방에 사람이 있는 경우
        # case2 -> 밖에 주로 사람이 있는 경우
        only_space = []
        for sp in self.space_list:
            if (not sp in self.room_list and sp != [[0,0], [5, 45]] and sp != [[0, 45], [45, 49]] and sp != [[45, 5], [49, 49]] and sp != [[5,0], [49,5]]):
                only_space.append(sp)
        space_num = len(only_space)
        
        
        space_agent = agent_num

        random_list = [0] * space_num

        # 총합이 agent num이 되도록 할당
        for i in range(space_num - 1):
            #random_num = random.randint(1, space_agent - sum(random_list) - (space_num - i - 1))
            random_num = ran % (space_agent - sum(random_list) - (space_num - i - 1)) + 1
            random_num = ran % (space_agent - sum(random_list) - (space_num - i - 1)) + 1
            ran += (ran*ran-int(12343/34)+3435*ran%(23))
            random_list[i] = random_num

        # 마지막 숫자는 나머지 값으로 설정
        if(space_num != 0):
            random_list[-1] = space_agent - sum(random_list)

        for j in range(len(only_space)):
            self.agent_place(only_space[j][0], only_space[j][1], random_list[j],ran)

    def random_agent_distribute_outdoor2(self, agent_num, ran):
        case = random.randint(1,2) 
        # case1 -> 방에 사람이 있는 경우
        # case2 -> 밖에 주로 사람이 있는 경우
        only_space = []
        for sp in self.space_list:
            if (not sp in self.room_list and sp != [[0,0], [5, 45]] and sp != [[0, 45], [45, 49]] and sp != [[45, 5], [49, 49]] and sp != [[5,0], [49,5]]):
                only_space.append(sp)
        space_num = len(only_space)
        
        
        space_agent = agent_num

        random_list = [0] * space_num

        # 총합이 agent num이 되도록 할당
        for i in range(space_num - 1):
            #random_num = random.randint(1, space_agent - sum(random_list) - (space_num - i - 1))
            random_num = ran % (space_agent - sum(random_list) - (space_num - i - 1)) + 1

                #random_num = random.randint(1, space_agent - sum(random_list) - (space_num - i - 1))
            random_num = ran % (space_agent - sum(random_list) - (space_num - i - 1)) + 1
            ran += (ran*ran-int(12343/34)+3435*ran%(23))
            random_list[i] = random_num

        # 마지막 숫자는 나머지 값으로 설정
        if(space_num != 0):
            random_list[-1] = space_agent - sum(random_list)

        for j in range(len(only_space)):
            self.agent_place2(only_space[j][0], only_space[j][1], random_list[j],ran)


    def random_hazard_placement(self, hazard_num):
        min_size = 4
        max_size = 5
        only_space = [] #바깥쪽 비상탈출구 제외
        for sp in self.space_list:
            if (sp != [[0,0], [5, 5]] and sp != [[0,5], [5, 49]] and sp != [[5, 0], [45, 5]] and sp != [[5, 45], [49, 49]] and sp != [[45, 5], [49, 45]]):
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

        #self.space_goal_dict[((0,0), (5, 95))] = [[0,0]]
        self.space_type[((0,0), (5, 49))] = 0
        self.space_list.append([[0,0], [5,45]])

        self.space_type[((0,45), (45, 49))] = 0
        self.space_list.append([[0, 45], [45, 49]]) 

        self.space_type[((45,5), (49, 49))] = 0
        self.space_list.append([[45, 5], [49, 49]])

        self.space_type[((5, 0), (49, 5))] = 0
        self.space_list.append([[5, 0], [49, 5]])

    def connect_space(self): #space끼리 이어주는 함수 
                            #dict[(space_key)] = [space1, space2, space3 ] -> self.space_graph
        check_connection = []
        for i in range(51):
            tmp = []
            for j in range(51):
                tmp.append(0)
            check_connection.append(tmp) #이중 리스트로 겹치는지 확인할거임

        for space in self.space_list: #space끼리 연결 #space 그래프 만들기
            if space in self.room_list: #방이면 건너뛰기
                continue
            check_connection = []
            for i1 in range(51):
                tmp = []
                for j1 in range(51):
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
                    if(check_connection2[space2[0][0]][y2] == 2): #space와 space2는 접한다 
                        left_goal[0] += space2[0][0]
                        left_goal[1] += y2
                        left_goal_num = left_goal_num + 1
                        checking = 1 #이을거다~ (space와 space2를 )
                for y3 in range(space2[0][1]+1, space2[1][1]):
                    check_connection2[space2[1][0]][y3] += 1 #right
                    if(check_connection2[space2[1][0]][y3] == 2):
                        right_goal[0] += space2[1][0]
                        right_goal[1] += y3
                        right_goal_num = right_goal_num + 1
                        checking = 1
                for x2 in range(space2[0][0]+1, space2[1][0]):
                    check_connection2[x2][space2[0][1]] += 1 #down
                    if(check_connection2[x2][space2[0][1]] == 2):
                        down_goal[0] += x2
                        down_goal[1] += space2[0][1]
                        down_goal_num = down_goal_num + 1
                        checking = 1
                for x3 in range(space2[0][0]+1, space2[1][0]):
                    check_connection2[x3][space2[1][1]] += 1 #up
                    if(check_connection2[x3][space2[1][1]] == 2):
                        up_goal[0] += x3
                        up_goal[1] += space2[1][1]
                        up_goal_num = up_goal_num + 1
                        checking = 1
                if (checking==1 and space != space2):
                    self.space_graph[((space[0][0], space[0][1]), (space[1][0], space[1][1]))].append(space2)
                                              #왼쪽아래모서리              #오른쪽위모서리
                #위에까지가 space graph 만들기
                    

                if(left_goal[0] != 0 and left_goal[1] != 0):
                    first_left_goal = [0, 0]
                    first_left_goal[0] = (left_goal[0]/left_goal_num)
                    first_left_goal[1] = (left_goal[1]/left_goal_num)-1.5
                    second_left_goal = [0, 0]
                    second_left_goal[0] = (left_goal[0]/left_goal_num)
                    second_left_goal[1] = (left_goal[1]/left_goal_num)+1.5
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_left_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_left_goal))
                    self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(first_left_goal)
                    self.space_goal_dict[((space2[0][0],space2[0][1]), (space2[1][0], space2[1][1]))].append(second_left_goal)
                    
                elif(right_goal[0] != 0 and right_goal[1] != 0):
                    first_right_goal = [0, 0]
                    first_right_goal[0] = (right_goal[0]/right_goal_num)
                    first_right_goal[1] = (right_goal[1]/right_goal_num)-1.5
                    second_right_goal = [0, 0]
                    second_right_goal[0] = (second_right_goal[0]/right_goal_num)
                    second_right_goal[1] = (right_goal[1]/right_goal_num)+1.5
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_right_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_right_goal))

                elif(down_goal[0] != 0 and down_goal[1] != 0):
                    first_down_goal = [0, 0]
                    first_down_goal[0] = (down_goal[0]/down_goal_num)+1.5
                    first_down_goal[1] = (down_goal[1]/down_goal_num)
                    second_down_goal = [0, 0]
                    second_down_goal[0] = (down_goal[0]/down_goal_num)-1.5
                    second_down_goal[1] = (down_goal[1]/down_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_down_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_down_goal))

                elif(up_goal[0] != 0 and up_goal[1] != 0):
                    first_up_goal = [0, 0]
                    first_up_goal[0] = (up_goal[0]/up_goal_num)+1.5
                    first_up_goal[1] = (up_goal[1]/up_goal_num)
                    second_up_goal = [0, 0]
                    second_up_goal[0] = (up_goal[0]/up_goal_num)-1.5
                    second_up_goal[1] = (up_goal[1]/up_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_up_goal))
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), second_up_goal))

    def connect_space_with_one_goal(self): #space끼리 이어주는 함수 
                            #dict[(space_key)] = [space1, space2, space3 ] -> self.space_graph
        check_connection = []
        for i in range(51):
            tmp = []
            for j in range(51):
                tmp.append(0)
            check_connection.append(tmp) #이중 리스트로 겹치는지 확인할거임

        for space in self.space_list: #space끼리 연결 #space 그래프 만들기
            if space in self.room_list: #방이면 건너뛰기
                continue
            check_connection = []
            for i1 in range(51):
                tmp = []
                for j1 in range(51):
                    tmp.append(0)
                check_connection.append(tmp)
            # 모든 회색 1로 채우기
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
                    if(check_connection2[space2[0][0]][y2] == 2): #space와 space2는 접한다 
                        #print(space, space2, "가 LEFT에서 만남")
                        left_goal[0] += space2[0][0]
                        left_goal[1] += y2
                        left_goal_num = left_goal_num + 1
                        checking = 1 #이을거다~ (space와 space2를 )
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
                                              #왼쪽아래모서리              #오른쪽위모서리
                #위에까지가 space graph 만들기
                    

                if(left_goal[0] != 0 and left_goal[1] != 0):
                    first_left_goal = [0, 0]
                    first_left_goal[0] = (left_goal[0]/left_goal_num)
                    first_left_goal[1] = (left_goal[1]/left_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_left_goal))
         
                    
                elif(right_goal[0] != 0 and right_goal[1] != 0):
                    first_right_goal = [0, 0]
                    first_right_goal[0] = (right_goal[0]/right_goal_num)
                    first_right_goal[1] = (right_goal[1]/right_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_right_goal))
        

                elif(down_goal[0] != 0 and down_goal[1] != 0):
                    first_down_goal = [0, 0]
                    first_down_goal[0] = (down_goal[0]/down_goal_num)
                    first_down_goal[1] = (down_goal[1]/down_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_down_goal))
           


                elif(up_goal[0] != 0 and up_goal[1] != 0):
                    first_up_goal = [0, 0]
                    first_up_goal[0] = (up_goal[0]/up_goal_num)
                    first_up_goal[1] = (up_goal[1]/up_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_up_goal))



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
        for i in range(51):
            tmp = []
            for j in range(51):
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
        for i in range(51):
            tmp = []
            for j in range(51):
                tmp.append(0)
            check_door.append(tmp)

        for door in self.door_list:
            check_door[door[0]][door[1]] = 1


        for space in self.space_list: #문 있는 곳 끼리 연결 #space 그래프 만들기
            door_connection = []
            for i1 in range(51):
                tmp = []
                for j1 in range(51):
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
       

    def map_recur_divider_fine(self, xy, x_unit, y_unit, num, space_list, room_list, is_room): # ex) xy = [[2,3], [4,5]] # space 나누는 것. 나누고 방을 선택함
        x_diff = xy[1][0] - xy[0][0] # a점의 튜플은 (xy[0][0],xy[0][1])  b점의 튜플은 (xy[1][0],xy[1][1]) 
        y_diff = xy[1][1] - xy[0][1]

        real_xy =  [ [xy[0][0]*x_unit, xy[0][1]*y_unit], [xy[1][0]*x_unit, xy[1][1]*y_unit]] # 실제 좌표 
        if(is_room==0):
            space_list.append(real_xy)
            return
                   
        if(x_diff<3 or y_diff<3): #방이 일정크기 이하면 방 만들고 회귀 빠져나오기
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
            # if(a<5):
            if(a<5):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==4):
            a = random.randint(1,20)
            # if(a<8):
            if(a<17):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==5):
            a = random.randint(1,20)
            if(a<20):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return
            
        divide_num_y = random.randint(1, y_diff-1)
        divide_num_x = random.randint(1, x_diff-1)

        random_exist_room1 = random.randint(0,1)
        #random_exist_room2 = random.randint(0,1)
        random_exist_room3 = random.randint(0,1)
        #random_exist_room4 = random.randint(0,1)

        if (random_exist_room1 == 0):
            random_exist_room2 = 1
        else:
            random_exist_room2 = 0
        if (random_exist_room3 == 0):
            random_exist_room4 = 1
        else:
            random_exist_room4 = 0
    
        special_hallway = random.randint(1, 2) #가운데 나눠지는 길을 만들기 위함(일반적인 건물배치도 생성 유도)
        if(num<3):
            if (num%2==0): #가로로 나눈다
                left = int(x_diff*random.randint(1,3)/4)
                #hallway_size = random.randint(1,2)
                hallway_size = 1
                if(xy[0][0]+left+hallway_size >= (xy[1][0]-2)):
                    self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                    return
                # if(x_diff<13):5
                #     left = int(x_diff*(1/2))
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[0][0]+left, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider_fine([[xy[0][0]+left, xy[0][1]], [xy[0][0]+left+hallway_size, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 0)
                self.map_recur_divider_fine([[xy[0][0]+left+hallway_size, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
            else: #세로로 나눈다
                up = int(y_diff*random.randint(1,3)/4)
                hallway_size = random.randint(1,2)
                hallway_size = 1
                # if(xy[0][1]+up+hallway_size >= (xy[1][1]-2)):
                #     print("xy[0][1]+up+hallway_size :",xy[0][1]+up+hallway_size)
                #     print("xy[1][1]-2 :", xy[1][1]-2)
                #     self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                #     return
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
    
    def floyd_warshall(self): #공간과 공간사이의 최단 경로를 구하는 알고리즘 

        vertices = list(self.space_graph.keys())
        n = len(vertices)
        distance_matrix = {start: {end: float('infinity') for end in vertices} for start in vertices}  
        next_vertex_matrix = {start: {end: None for end in vertices} for start in vertices}

        for start in self.space_graph.keys():
            for end in self.space_graph[start]:
                end_t = ((end[0][0], end[0][1]), (end[1][0],end[1][1]))
                start_xy = [(start[0][0]+start[1][0])/2, (start[0][1]+start[1][1])/2]
                end_xy = [(end[0][0]+end[1][0])/2, (end[0][1]+end[1][1])/2]
                distance_matrix[start][end_t] = math.sqrt(pow(start_xy[0]-end_xy[0],2)+pow(start_xy[1]-end_xy[1], 2))
                next_vertex_matrix[start][end_t] = end_t

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                        next_vertex_matrix[i][j] = next_vertex_matrix[i][k]
        return [next_vertex_matrix, distance_matrix]

    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 
        start = ((start[0][0], start[0][1]), (start[1][0], start[1][1]))
        end = ((end[0][0], end[0][1]), (end[1][0], end[1][1]))
        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path
        
    

    def agent_place(self, xy1, xy2, num, ran):
    
        agent_list = []
        x_len = xy2[0]-xy1[0]
        y_len = xy2[1]-xy1[1]

        check_list = [[0]*y_len for _ in range(x_len)]
        while(num>0):
            #x = random.randint(xy1[0]+1, xy2[0]-1)
            x = ran % ((xy2[0]-1) - (xy1[0]+1) + 1) + xy1[0]+1
            #y = random.randint(xy1[1]+1, xy2[1]-1)
            y = ran % ((xy2[1]-1) - (xy1[1]+1) + 1) + xy1[1]+1
            ran += (ran*ran-int(12343/34)+343*ran%(25))
            #print(x, xy1[0])
            #print(y, xy1[1])
            while(check_list[x-xy1[0]][y-xy1[1]] == 1):
                # x = random.randint(xy1[0]+1, xy2[0]-1)
                # y = random.randint(xy1[1]+1, xy2[1]-1)
                x = ran % ((xy2[0]-1) - (xy1[0]+1) + 1) + xy1[0]+1
                y = ran % ((xy2[1]-1) - (xy1[1]+1) + 1) + xy1[1]+1
                ran += (ran*ran-int(12343/34)+32343*ran%(21))
            check_list[x-xy1[0]][y-xy1[1]] = 1
            num = num-1
            a = FightingAgent(self.agent_id, self, [x,y], 0)
            self.agent_id = self.agent_id + 1
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
            #self.agents.append(a)

    def agent_place2(self, xy1, xy2, num, ran):
    
        agent_list = []
        x_len = xy2[0]-xy1[0]
        y_len = xy2[1]-xy1[1]

        check_list = [[0]*y_len for _ in range(x_len)]
        while(num>0):
            # x = random.randint(xy1[0]+1, xy2[0]-1)
            # y = random.randint(xy1[1]+1, xy2[1]-1)
            x = ran % ((xy2[0]-1) - (xy1[0]+1) + 1) + xy1[0]+1
            y = ran % ((xy2[1]-1) - (xy1[1]+1) + 1) + xy1[1]+1
            ran += (ran*ran-int(12343/34)+343*ran%(25))
            #print(x, xy1[0])
            #print(y, xy1[1])
            while(check_list[x-xy1[0]][y-xy1[1]] == 1):
                # x = random.randint(xy1[0]+1, xy2[0]-1)
                # y = random.randint(xy1[1]+1, xy2[1]-1)
                x = ran % ((xy2[0]-1) - (xy1[0]+1) + 1) + xy1[0]+1
                y = ran % ((xy2[1]-1) - (xy1[1]+1) + 1) + xy1[1]+1
                ran += (ran*ran-int(12343/34)+32343*ran%(21))
            check_list[x-xy1[0]][y-xy1[1]] = 1
            num = num-1
            a = FightingAgent2(self.agent_id, self, [x,y], 0)
            self.agent_id = self.agent_id + 1
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))

    def dfs(self, key, m_list, p_list): #튜플, 리스트, 리스트 속 튜플
        global number_of_cases

        if self.dict_NoC[key] == 1:
            number_of_cases += 1
            return
        
        for i in m_list:
            i = tuple(map(tuple, i)) # 이중리스트 형태인 i를 튜플 형태로 변환
            pp_list = copy.deepcopy(p_list)
            if i not in pp_list:
                pp_list.append(i)
                self.dfs(i, self.space_graph[i], pp_list)
                pp_list.pop()


    def step(self):
        """Advance the model by one step."""
        global started
        if(started):
            #self.difficulty_f()
            started = 0
    
        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model
        # if(self.robot != None):
        #     print(self.robot.robot_xy)
        
    
        # Checking if there is a champion
        if FightingModel.current_healthy_agents(self) == 0:
            self.running = False
        self.num_remained_agents()

    def difficulty_f(self): # 공간을 넣으면 해당 공간의 난이도 출력
        global number_of_cases


        for key, val in self.space_graph.items():
            if len(val) != 0 : #닫힌 공간 제외 val 0으로 초기화
                self.dict_NoC[key] = 0

        for key in self.dict_NoC.keys(): # key 공간이 출구와 맞닿아 있으면 value 1
            self.dict_NoC[self.exit_compartment] = -1 # 출구는 -1
            if list(map(list, self.exit_compartment)) in self.space_graph[key]: 
                self.dict_NoC[key] = 1

        for key, val in self.dict_NoC.items():
            number_of_cases = 0
            if val == 0:
                p_list = [key]
                self.dfs(key, self.space_graph[key], p_list) #튜플, 리스트, 리스트 속 튜플
                self.dict_NoC[key] = number_of_cases
        

    def space_specification(self):

        global max_specification
        new_space_list = []

        for i in self.space_list:
            x_size = i[1][0] - i[0][0]
            y_size = i[1][1] - i[0][1]

            if(x_size>max_specification[0]):
                middle = int((i[0][0] + i[1][0])/2)
                new_x = [[i[0][0], i[0][1]], [middle, i[1][1]]]
                new_x2 = [[middle+1, i[0][1]], [i[1][0], i[1][1]]]
                new_space_list.append(new_x)
                new_space_list.append(new_x2)
            else:
                new_space_list.append(i)
        
        new_space_list2 = []
        
        for i in new_space_list:
            y_size = i[1][1] - i[0][1]

            if(y_size>max_specification[0]):
                middle = int((i[0][1] + i[1][1])/2)
                new_y = [[i[0][0], i[0][1]], [i[1][0],middle]]
                new_y2 = [[i[0][0], middle+1], [i[1][0], i[1][1]]]
                new_space_list2.append(new_y)
                new_space_list2.append(new_y2)
            else:
                new_space_list2.append(i)


        return new_space_list2
    
    def reward_distance_difficulty(self): # 모든 agent 각각의 거리 총합, 난이도 총합 고려 reward 산출
        s_distance = 0 # 거리합
        for i in self.agents:
            if(i.dead == False and (i.type == 0 or i.type == 1)):
                agent_space = self.grid_to_space[int(round(i.xy[0]))][int(round(i.xy[1]))]
                next_goal = space_connected_linear(tuple(map(tuple, agent_space)), self.floyd_warshall()[0][tuple(map(tuple, agent_space))][self.exit_compartment])
                agent_space_x_center = (agent_space[0][0] + agent_space[1][0])/2
                agent_space_y_center = (agent_space[1][0] + agent_space[1][1])/2
                a = (self.floyd_distance[tuple(map(tuple, agent_space))][self.exit_compartment] #agent가 있는 공간-출구 공간 까지의 거리
                - math.sqrt(pow(agent_space_x_center-next_goal[0],2) + pow(agent_space_y_center-next_goal[1],2)) #-(agent 공간 외부점..- agent 공간 중심점)
                + math.sqrt(pow(next_goal[0]-i.xy[0],2) + pow(next_goal[1]-i.xy[1],2))) #+(agent공간 외부점-agent 위치)
                
                s_distance += a

        s_difficulty = 0
        for i in self.agents:
            if(i.dead == False and (i.type == 0 or i.type == 1)):
                agent_space = self.grid_to_space[int(round(i.xy[0]))][int(round(i.xy[1]))] # 각 agent 가 있는 공간
                if self.dict_NoC[tuple(map(tuple, agent_space))] != -1:
                    s_difficulty += self.dict_NoC[tuple(map(tuple, agent_space))] #그 공간의 난이도 합산

        a = 0.1
        b = 1
        
        #print("reward_distance_difficulty (", a*s_distance + b*s_difficulty, ") = a(", a, ") * s_distance(", s_distance, ") + b(", b, ") * s_difficulty(", s_difficulty, ")\n")
        return a*s_distance + b*s_difficulty
    
    def return_robot(self):
        return self.robot


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

    def num_remained_agents(self):
        #from model import Model
        self.num_remained_agent = 0
        space_agent_num = {}
        for i in self.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.agents:
            space_xy = self.grid_to_space[int(round((i.xy)[0]))][int(round((i.xy)[1]))]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        
        for j in space_agent_num.keys():
            self.num_remained_agent += space_agent_num[j]
        return self.num_remained_agent