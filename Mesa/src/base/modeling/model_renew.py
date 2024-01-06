from mesa import Model
from agent_renew import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_renew
from agent_renew import WallAgent
import random

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
    


class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int):
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

        self.datacollector_currents = DataCollector(
            {
                "Healthy Agents": FightingModel.current_healthy_agents,
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        self.agent_place((40,20), (70, 80), 100)
        self.make_hazard((40,20), (70,80), 4)





        exit_rec = [] ## exit_rec list : exit_w * exit_h 크기 안에 (0,0)~(exit_w, exit_h) 토플 채워짐
        for i in range(0, agent_renew.exit_w):
            for j in range(0, agent_renew.exit_h):
                exit_rec.append((i,j))

        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기


        wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        self.wall_matrix = list()
        self.only_one_wall = list()
        for i in range(100):
            tmp = []
            for j in range(100):
                tmp.append(0)
            self.wall_matrix.append(tmp)
            self.only_one_wall.append(tmp)
        
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
        self.map_recur_divider([[1, 1], [9, 9]], 10, 10, 0, self.space_list, self.room_list, 1)
        for i in self.room_list:
            wall = wall+make_room(i[0], i[1])
        print(self.space_list)
        print(self.room_list)


        for i in goal_list:
            for j in i:
                if j in wall:    
                    wall.remove(j)
                    self.wall_matrix[j[0]][j[1]] = 0

        for i in range(len(wall)):
            if (self.only_one_wall[wall[i][0]][wall[i][1]] == 1 and wall[i][0]!=0 and wall[i][1]!=0 and wall[i][1]!=99):
                continue
            c = FightingAgent(i, self, wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, wall[i])
            self.only_one_wall[wall[i][0]][wall[i][1]] = 1
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

        # new_plane.append(hazard_start)
        # check_list[hazard_start[0]-x1][hazard_start[1]-y1] = 1
        # area_size = 0
        # while(hazard_size != area_size):
        #     UDLR = random.randint(1, 4)
        #     if (UDLR==1):
        #         hazard_start = (hazard_start[0], hazard_start[1]+1)
        #     elif (UDLR==2):
        #         hazard_start = (hazard_start[0], hazard_start[1]-1)
        #     elif (UDLR==3):
        #         hazard_start = (hazard_start[0]-1, hazard_start[1])
        #     elif (UDLR==4):
        #         hazard_start = (hazard_start[0]+1, hazard_start[1])
        #     while not(hazard_start[0]>x1 and hazard_start[0]<x2 and hazard_start[1]>y1 and hazard_start[1]<y2 and check_list[hazard_start[0]-x1][hazard_start[1]-y1] == 0):
        #         UDLR = random.randint(1, 4)
        #         if (UDLR==1):
        #             hazard_start = (hazard_start[0], hazard_start[1]+1)
        #         elif (UDLR==2):
        #             hazard_start = (hazard_start[0], hazard_start[1]-1)
        #         elif (UDLR==3):
        #             hazard_start = (hazard_start[0]-1, hazard_start[1])
        #         elif (UDLR==4):
        #             hazard_start = (hazard_start[0]+1, hazard_start[1])
        #     new_plane.append(hazard_start)
        #     check_list[hazard_start[0]-x1][hazard_start[1]-y1] = 1
        #     area_size = area_size+1
        #     a = FightingAgent(300+area_size, self, [hazard_start[0],hazard_start[1]], 1)
        #     self.schedule_h.add(a)
        #     self.grid.place_agent(a, (hazard_start[0], hazard_start[1]))
    
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
       
    # def random_map_generator(self, room_min, room_max, exit_num, map_size):
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

    #     while(now_room_num != room_num):
    #         continue_while = 0    
    #         room_x_len = random.randint(1,5)
    #         room_y_len = random.randint(1,5)
    #         room_start_xy = [random.randint(0, 10-room_x_len), random.randint(0, 10-room_y_len)]

    #         for i in range(room_start_xy[0], room_start_xy[0]+room_x_len):
    #             for j in range(room_start_xy[1], room_start_xy[1]+room_y_len):
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

            

    #     #room_type = random.radint()
    #     return rooms
    
    def random_map_generator2(self, room_min, room_max, exit_num, map_size):
        rooms = []
        x_unit = int(map_size/10)
        y_unit = int(map_size/10)
        for i in range(10): 
            for j in range(10):
                #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit])
                #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit, j*y_unit+y_unit-1])
                #rooms = rooms + make_plane([i*x_unit+x_unit-1, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
                #rooms = rooms + make_plane([i*x_unit, j*y_unit+y_unit-1], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
                #rooms = rooms + make_room([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
                self.map_divide[i][j] = [[i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1]]
        
        room_num = random.randint(room_min, room_max) #how many rooms?
        room_type = random.randint(1, 3) # what type the room is? 

        now_room_num = 0
        while_breaker = 0

        while(now_room_num != room_num):
            while_breaker = while_breaker+1
            if (while_breaker>1000):
                break
            continue_while = 0    
            room_x_len = random.randint(2,5)
            room_y_len = random.randint(2,5)
            room_start_xy = [random.randint(1, 10-room_x_len), random.randint(1, 10-room_y_len)]

            for i in range(room_start_xy[0], room_start_xy[0]+room_x_len+1):
                for j in range(room_start_xy[1], room_start_xy[1]+room_y_len+1):
                    if (self.map_repre[i][j] == 1):
                        continue_while = 1
                        break 
                    else :
                        self.map_repre[i][j] = 1
                if(continue_while==1):
                    break
            if(continue_while):
                continue
            rooms = rooms+make_room([room_start_xy[0]*x_unit, room_start_xy[1]*y_unit], [(room_start_xy[0]+room_x_len)*x_unit-1, (room_start_xy[1]+room_y_len)*y_unit-1])
            now_room_num = now_room_num + 1
            self.room_list.append([[room_start_xy[0]*x_unit, room_start_xy[1]*y_unit], [(room_start_xy[0]+room_x_len)*x_unit-1, (room_start_xy[1]+room_y_len)*y_unit-1]])
            

        #room_type = random.radint()
        return rooms
    def random_map_generator3(self, room_min, room_max, exit_num, map_size):
        rooms = []
        x_unit = int(map_size/10)
        y_unit = int(map_size/10)
        # for i in range(10): 
        #     for j in range(10):
        #         #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit])
        #         #rooms = rooms + make_plane([i*x_unit, j*y_unit], [i*x_unit, j*y_unit+y_unit-1])
        #         #rooms = rooms + make_plane([i*x_unit+x_unit-1, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
        #         #rooms = rooms + make_plane([i*x_unit, j*y_unit+y_unit-1], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
        #         #rooms = rooms + make_room([i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1])
        #         self.map_divide[i][j] = [[i*x_unit, j*y_unit], [i*x_unit+x_unit-1, j*y_unit+y_unit-1]]
        

        self.map_recur_divider([[0, 0], [10, 10]], 10, 10, 0, self.space_list, self.room_list, 1)
        for i in self.room_list:
            wall = wall+make_room(i[0], i[1])
        return rooms
    
    def map_recur_divider(self, xy, x_unit, y_unit, num, space_list, room_list, is_room): # ex) xy = [[2,3], [4,5]]
        x_diff = xy[1][0] - xy[0][0]
        y_diff = xy[1][1] - xy[0][1]

        real_xy =  [ [xy[0][0]*x_unit, xy[0][1]*y_unit], [xy[1][0]*x_unit, xy[1][1]*y_unit]]
                   
        if(x_diff<2 or y_diff<2):
            space_list.append(real_xy)
            return
        
        if(is_room==0):
            space_list.append(real_xy)
            return
            
            
        if(num==1): 
            a = random.randint(1,20)
            if(a<2):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return 
        elif(num==2):
            a = random.randint(1,10)
            if(a<4):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return
        elif(num==3):
            a = random.randint(1,10)
            if(a<6):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
        elif(num==4):
            a = random.randint(1,10)
            if(a<8):
                space_list.append(real_xy)
                room_list.append(real_xy)
                return #여기에 걸리면 방 만들고 종료.
            
        divide_num_y = random.randint(1, y_diff-1)
        divide_num_x = random.randint(1, x_diff-1)
        random_exist_room = random.randint(1,10)
        if(random_exist_room<5):
            random_exist_room_x = 1
        else:
            random_exist_room_x = 1
        
        

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
                self.map_recur_divider([[xy[0][0]+left, xy[0][1]], [xy[0][0]+left+1, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0]+left+1, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
            else: #세로로 나눈다
                up = int(y_diff/2)
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+up]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0], xy[0][1]+up], [xy[1][0], xy[0][1]+up+1]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0], xy[0][1]+up+1], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)

        else:
            if (num%2==0): #가로로 나눈다
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[0][0]+divide_num_x, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0]+divide_num_x, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1)
        
            else: #세로로 나눈다
                self.map_recur_divider([[xy[0][0], xy[0][1]], [xy[1][0], xy[0][1]+divide_num_y]], x_unit, y_unit, num+1, space_list, room_list, 1)
                self.map_recur_divider([[xy[0][0], xy[0][1]+divide_num_y], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, 1) 
    
    
    
    

    def agent_place(self, xy1, xy2, num):
    
        agent_list = []
        x_len = xy2[0]-xy1[0]
        y_len = xy2[1]-xy1[1]

        check_list = [[0]*y_len for _ in range(x_len)]
        while(num>0):
            x = random.randint(xy1[0]+1, xy2[0]-1)
            y = random.randint(xy1[1]+1, xy2[1]-1)
            print(x, xy1[0])
            print(y, xy1[1])
            while(check_list[x-xy1[0]][y-xy1[1]] == 1):
                x = random.randint(xy1[0]+1, xy2[0]-1)
                y = random.randint(xy1[1]+1, xy2[1]-1)
            check_list[x-xy1[0]][y-xy1[1]] = 1
            num = num-1
            a = FightingAgent(num, self, [x,y], 0)
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
