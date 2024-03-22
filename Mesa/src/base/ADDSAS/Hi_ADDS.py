
from mesa import Model
from mesa import Agent
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
import numpy as np
import time
num_remained_agent = 0

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20
exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]
STRATEGY = 1
random_disperse = 1

theta_1 = random.randint(1,10)
theta_2 = random.randint(1,10)
theta_3 = random.randint(1,10)

exit_area = [[0,exit_w], [0,exit_h]]

robot_xy = [2, 2]
robot_radius = 20 #로봇 반경 -> 10미터 
robot_status = 0
robot_ringing = 0
robot_goal = [0, 0]
past_target = ((0,0), (0,0))

def Multiple_linear_regresssion(distance_ratio, remained_ratio, now_affected_agents_ratio, v_min, v_max):
    global theta_1, theta_2, theta_3
    v = distance_ratio*theta_1 + remained_ratio*theta_2 + now_affected_agents_ratio*theta_3
    if (v>v_max):
        return v_max
    elif (v<v_min):
        return v_min
    else:
        return v


def space_connected_linear(xy1, xy2):
    check_connection = []
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
    left_goal_num = 0

    right_goal = [0, 0]
    right_goal_num = 0

    down_goal = [0, 0]
    down_goal_num = 0

    up_goal = [0, 0]
    up_goal_num = 0

    for y2 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[0][0]][y2] += 1 #left 
        if(check_connection2[xy2[0][0]][y2] == 2): #space와 space2는 접한다 
            #print(space, space2, "가 LEFT에서 만남")
            left_goal[0] += xy2[0][0]
            left_goal[1] += y2
            left_goal_num = left_goal_num + 1
            checking = 1 #이을거다~ (space와 space2를 )
    for y3 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[1][0]][y3] += 1 #right
        if(check_connection2[xy2[1][0]][y3] == 2):
            #print(space, space2, "가 RIGHT에서 만남")
            right_goal[0] += xy2[1][0]
            right_goal[1] += y3
            right_goal_num = right_goal_num + 1
            checking = 1
    for x2 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x2][xy2[0][1]] += 1 #down
        if(check_connection2[x2][xy2[0][1]] == 2):
            #print(space, space2, "가 DOWN에서 만남")
            down_goal[0] += x2
            down_goal[1] += xy2[0][1]
            down_goal_num = down_goal_num + 1
            checking = 1
    for x3 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x3][xy2[1][1]] += 1 #up
        if(check_connection2[x3][xy2[1][1]] == 2):
            #print(space, space2, "가 UP에서 만남")
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


goal_list = [[(71, 52)], [(89, 52)]]

def central_of_goal(goals):
    real_goal = [0, 0]
    for i in goals:
        real_goal[0] += i[0]
        real_goal[1] += i[1]
    real_goal[0] /= len(goals)
    real_goal[1] /= len(goals) 
    return real_goal

def check_departure(pose, goals):
    for i in goals:
        if (i[0]>pose[0] and i[1]>pose[1]):
            return True
    return False

 # goals의 가운데를 가져오는 함수
 # 어디로 향하게 할 것인가? -> goals의 가운데 

class WallAgent(Agent): ## wall .. 탈출구 범위 내에 agents를 채워넣어서 탈출구라는 것을 보여주고 싶었음.. 
    def __init__(self, pos, model, agent_type):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type


def set_agent_type_settings(agent, type):
    """Updates the agent's instance variables according to its type.

    Args:
        agent (FightingAgent): The agent instance.
        type (int): The type of the agent.
    """
    if type == 1:
        agent.health = 2 * INITIAL_HEALTH ## 200
        agent.attack_damage = 2 * ATTACK_DAMAGE ## 100
    if type == 2:
        agent.health = math.ceil(INITIAL_HEALTH / 2) ## 50
        agent.attack_damage = math.ceil(ATTACK_DAMAGE / 2) ## 25
    if type == 3:
        agent.health = math.ceil(INITIAL_HEALTH / 4) ## 25
        agent.attack_damage = ATTACK_DAMAGE * 4 ## 80
    if type == 10: ## 구분하려고 아무 숫자 함, exit_rec 채우는 agent type
        agent.health = 500 ## ''
        agent.attack_damage = 0 ## ''
    if type == 11: ## 마찬가지.. 이건 wall list 채우는 agent의 type
        agent.health = 500
        agent.attack_damage = 0

class FightingAgent(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, pos, type): 
        super().__init__(unique_id, model)
        global robot_xy
        robot_xy = pos
        self.goal_init = 0
        self.type = type
        self.health = INITIAL_HEALTH
        self.attack_damage = ATTACK_DAMAGE
        self.attacked = False
        self.dead = False
        self.robot_guide = 0
        self.drag = 0
        self.dead_count = 0
        self.buried = False
        self.which_goal = 0
        self.previous_stage = []
        self.now_goal = [0,0]
        
        #self.robot_xy = [2,2]
        #self.robot_status = 0

        self.xy = pos
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.mass = 3
        self.previous_goal = [0,0]

        #for robot 
        self.robot_space = ((0,0), (5,95))
        self.mission_complete = 1
        self.going = 0
        self.guide = 0
        self.save_target = 0
        self.save_point = 0
        self.robot_now_path = []
        self.robot_waypoint_index = 0

        self.go_path_num= 0
        self.back_path_num = 0


        # self.xy[0] = self.random.randrange(self.model.grid.width)
        # self.xy[1] = self.random.randrange(self.model.grid.height)
        
        set_agent_type_settings(self, type)

    def __repr__(self) -> str:
        return f"{self.unique_id} -> {self.health}"

    def step(self) -> None: # 각 step마다 agent가 어떤 행동을 할지에 대한 내용이 적혀있음
        global exit_area
        global goal_list

        """Handles the step of the model dor each agent.
        Sets the flags of each agent during the simulation.
        """

        # buried agents do not move (Do they???? :))
        if self.buried:
            return

        # dead for too long it is buried not being displayed 
        if self.dead_count > 4:
            self.buried = True
            return

        # no health and not buried increment the count
        if self.dead and not self.buried:
            self.dead_count += 1
            return

        # when attacked needs one turn until be able to attack
        if self.attacked:
            self.attacked = False
            return
        if(self.type != 3): #robot은 죽지 않는다  # 탈출구에 가면 agent가 죽는 코드
            if (self.model.is_left_exit):
                if (self.xy[0] > self.model.left_exit_area[0][0] and self.xy[0] < self.model.left_exit_area[1][0] and self.xy[1] > self.model.left_exit_area[0][1] and self.xy[1]<self.model.left_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 

            if (self.model.is_right_exit):
                if (self.xy[0] > self.model.right_exit_area[0][0] and self.xy[0] < self.model.right_exit_area[1][0] and self.xy[1] > self.model.right_exit_area[0][1] and self.xy[1]<self.model.right_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        
            if (self.model.is_up_exit):
                if (self.xy[0] > self.model.up_exit_area[0][0] and self.xy[0] < self.model.up_exit_area[1][0] and self.xy[1] > self.model.up_exit_area[0][1] and self.xy[1]<self.model.up_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        
            if (self.model.is_down_exit):
                if (self.xy[0] > self.model.down_exit_area[0][0] and self.xy[0] < self.model.down_exit_area[1][0] and self.xy[1] > self.model.down_exit_area[0][1] and self.xy[1]<self.model.down_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        self.move()

    def check_stage_agent(self):
        x = self.xy[0]
        y = self.xy[1]
        now_stage = []
        for i in self.model.space_list:
            if (x>i[0][0] and x<i[1][0] and y>i[0][1] and y<i[1][1]):
                now_stage = i
                break
        if(len(now_stage) != 0):
            now_stage = ((now_stage[0][0], now_stage[0][1]), (now_stage[1][0], now_stage[1][1]))
        else:
            now_stage = ((0,0), (5, 45))
        return now_stage

    def which_goal_agent_want(self):
        

        if(self.goal_init == 0):
            now_stage = self.check_stage_agent() 
            goal_candiate = self.model.space_goal_dict[now_stage] #1개에서 4개까지 있겠지 #space goal dict는 space넣으면 extend까지한 goal의 좌표가 나옴
            if(len(goal_candiate)==1):
                goal_index = 0
            else:
                goal_index = random.randint(0, len(goal_candiate)-1)
            self.now_goal = goal_candiate[goal_index]
            self.goal_init = 1
            self.previous_stage = now_stage
        now_stage = self.check_stage_agent() #now_stage -> agent가 현재 어느 space에 있는가 
        if(self.previous_stage != self.check_stage_agent()): # 다음 space로 넘어온 상태
            goal_candiate = self.model.space_goal_dict[now_stage] # ex) [[2,0], [3,5],[4,1]] 
            goal_candiate2 = []
            if(len(goal_candiate)>1): # 여기부터가 이전 space로 다시 돌아가지는 않게 하는 장치
                min_d = 1000
                min_i = goal_candiate[0]
                for i in goal_candiate:
                    d = math.sqrt(pow(self.xy[0]-i[0], 2)+pow(self.xy[1]-i[1], 2))
                    if (d<min_d):
                        min_d = d
                        min_i = i #가장 가까운 골 찾기
                for j in goal_candiate: 
                    if (j==min_i): #goal 후보에서 빼버림
                        continue
                    else:
                        goal_candiate2.append(j)
                if(len(goal_candiate2)==1):
                    goal_index = 0
                else:
                    goal_index = random.randint(0, len(goal_candiate2)-1)
                self.now_goal = goal_candiate2[goal_index]
                self.previous_stage = now_stage
                return
            elif(len(goal_candiate)==0):
                self.now_goal = self.previous_goal
            else :
                goal_index = 0
                self.now_goal = goal_candiate[goal_index]
                self.previous_stage = now_stage
            self.previous_goal = self.now_goal

            


    def attackOrMove(self, cells_with_agents, possible_steps) -> None:
        """Decides if the user is going to attack or just move.
        Acts randomly.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
            possible_steps (list[Coordinates]): The list of available cell where to go.
        """
        should_attack = self.random.randint(0, 1) ## 50% 확률로 attack
        if should_attack:
            self.attack(cells_with_agents)
            return
        new_position = self.random.choice(possible_steps) ## 다음 step에 이동할 위치 설정
        self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def attack(self, cells_with_agents) -> None:
        """Handles the attack of the agent.
        Gets the list of cells with the agents the agent can attack.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
        """
        agentToAttack = self.random.choice(cells_with_agents) ## agent끼리 마주쳤을 때 맞을 애는 랜덤으로 고름
        agentToAttack.attacked = True ## 맞은 애 attacked 됐다~ 
        if agentToAttack.health <= 0: ## health 가 0보다 작으면 dead
            agentToAttack.dead = True

    def move(self) -> None:
        global goal_list
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""

        cells_with_agents = []

        if (self.type == 3): # type이 3이면 로봇
            new_position = self.robot_policy2() # robot_policy2는 로봇이 행동하는 코드
            self.model.grid.move_agent(self, new_position)
            return

        new_position = self.test_modeling() # test_modeling은 일반 agent가 행동하는 코드
        if(self.type ==0 or self.type==1):
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
    
    def robot_policy(self): # 이거는 무시해도됨
        time_step = 0.2
        from model_renew import Model
        global random_disperse
        global robot_status
        global robot_xy 
        global robot_radius
        global robot_ringing
        self.drag = 1
        robot_status = 1
        space_agent_num = self.agents_in_each_space() #어느 stage에 몇명이 있는지
        floyd_distance = self.model.floyd_distance # floyd_distance[stage1][stage2] = 최단거리 
        floyd_path = self.model.floyd_path #floyd_path[stage1][stage2] = stage_x 
                                           #floyd_path[stage_x][stage_2] = stage_y 
                                           #floyd_path[stage_y][stage_2] = ste... 
                                           # s1 -> sx -> sy -> .. stage
    
        

        self.robot_space = self.model.grid_to_space[int(robot_xy[0])][int(robot_xy[1])] #로봇이 어느 stage에 있는지 나온다 

        if(self.mission_complete == 1): #새로운 탈출 path를 찾는다
            self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
            agent_max = 0 #agent가 가장 많은 stage 
            for i in space_agent_num.keys(): 
                if (space_agent_num[i]>agent_max):
                    self.save_target = i #현재 가장 인구가 많이 있는 stage
                    agent_max = space_agent_num[self.save_target] 
            
            evacuation_points = []
            if(self.model.is_left_exit): 
                evacuation_points.append(((0,0), (5, 45)))
            if(self.model.is_up_exit):
                evacuation_points.append(((0,45), (45, 49)))
            if(self.model.is_right_exit):
                evacuation_points.append(((45,5), (49, 49)))
            if(self.model.is_down_exit):
                evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

            min_distance = 1000
            for i in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
                if(floyd_distance[self.save_target][i]<min_distance):
                    self.save_point = i 
            go_path = self.model.get_path(floyd_path, self.robot_space, self.save_target) #로봇의 초기 위치 -> save_target까지 가는데 최단 경로 stage 리스트 
            
            back_path = self.model.get_path(floyd_path, self.save_target, self.save_point) # save_target(인구가 가장 많은 곳)에서 save_point(safe zone) 까지의 최단 경로 
            self.go_path_num = len(go_path) #guide를 하기 위해서 ~ 

            for i in range(len(go_path)-1):
                self.robot_now_path.append(space_connected_linear(go_path[i], go_path[i+1])) #(stage1 stage2) -> 중간 goal을 알려준다  
            self.robot_now_path.append([(self.save_target[0][0]+self.save_target[1][0])/2, (self.save_target[0][1]+self.save_target[1][1])/2]) #save target 중점까지 간다 
            for i in range(len(back_path)-1):
                self.robot_now_path.append(space_connected_linear(back_path[i], back_path[i+1]))  #back path도 넣는다 
            self.mission_complete = 0 
        #print(self.robot_now_path)
        
        if(self.robot_waypoint_index > self.go_path_num-1): # 돌아오는 상황 
            robot_status = 1 #robot_status가 1일때 -> guide함, 로봇 색깔바뀜(빨간색), 로봇에 영향받는 agent 색깔 바뀜(주황색) 
            self.drag = 1 
        else:
            robot_status = 0
            self.drag = 0
        #print("현재 골 : ", self.robot_now_path[self.robot_waypoint_index])
        robot_goal = self.robot_now_path[self.robot_waypoint_index]
        

        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<3):
            self.robot_waypoint_index = self.robot_waypoint_index + 1

        if(self.robot_waypoint_index == len(self.robot_now_path)):
            self.mission_complete = 1 #미션을 새로 만들어야해 (끝났으니까)
            self.robot_waypoint_index = 0
            return [int(robot_xy[0]), int(robot_xy[1])]

        goal_x = self.robot_now_path[self.robot_waypoint_index][0] - robot_xy[0] #역학을 위한.. 
        goal_y = self.robot_now_path[self.robot_waypoint_index][1] - robot_xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        
        intend_force = 2
        desired_speed = 1.5

        if(self.drag == 0):
            desired_speed = 5
        else:
            desired_speed = 5

        if(goal_d != 0):
            desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
            desired_force = [0, 0]
    
        
        x=int(round(robot_xy[0]))
        y=int(round(robot_xy[1]))

        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i)
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i])
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]

        k=4

        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = robot_xy[0] - n_x
            d_y = robot_xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))


            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 2*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 2*k*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1
        
        F_x = 0
        F_y = 0
        
        F_x += desired_force[0]
        F_y += desired_force[1]

        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass

        robot_xy[0] += vel[0] * time_step
        robot_xy[1] += vel[1] * time_step
        
        next_x = int(round(robot_xy[0]))
        next_y = int(round(robot_xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49

        return (next_x, next_y)
    
    def robot_policy2(self): # 어딘가에 들리고 safe 존으로 가는 걸 반복
        time_step = 0.2
        from model_renew import Model
        global random_disperse
        global robot_status
        global robot_xy 
        global robot_radius
        global robot_ringing
        global robot_goal
        global past_target
        self.drag = 1
        robot_status = 1
        space_agent_num = self.agents_in_each_space() #어느 stage에 몇명이 있는가
        floyd_distance = self.model.floyd_distance # floyd_distance[stage1][stage2] = 최단거리 
        floyd_path = self.model.floyd_path #floyd_path[stage1][stage2] = stage_x 
                                           #floyd_path[stage_x][stage_2] = stage_y 
                                           #floyd_path[stage_y][stage_2] = ste... 
                                           # s1 -> sx -> sy -> .. stage
        
    
        

        self.robot_space = self.model.grid_to_space[int(robot_xy[0])][int(robot_xy[1])] #로봇이 어느 space에 있는지 나온다 

        if(self.mission_complete == 1): #새로운 탈출 path를 찾는다 # safe zone에 들렀다.
            self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
            agent_max = 0 #agent가 가장 많은 stage 
            self.find_target(space_agent_num, floyd_distance) # 공간의 꼭짓점 좌표가 나옴

            past_target = self.save_target # 전에 갔던 공간을 가지 않기 위해 past_target으로 저장해둔다

            go_path = self.model.get_path(floyd_path, self.robot_space, self.save_target) #로봇의 초기 위치 -> save_target까지 가는데 최단 경로 stage 리스트 
            
            back_path = self.model.get_path(floyd_path, self.save_target, self.save_point) # save_target(인구가 가장 많은 곳)에서 save_point(safe zone) 까지의 최단 경로 
            self.go_path_num = len(go_path) #guide를 하기 위해서 ~ 

            for i in range(len(go_path)-1):
                self.robot_now_path.append(space_connected_linear(go_path[i], go_path[i+1])) #(stage1 stage2) -> 중간 goal을 알려준다  
            self.robot_now_path.append([(self.save_target[0][0]+self.save_target[1][0])/2, (self.save_target[0][1]+self.save_target[1][1])/2]) #save target 중점까지 간다 
            for i in range(len(back_path)-1):
                self.robot_now_path.append(space_connected_linear(back_path[i], back_path[i+1]))  #back path도 넣는다 
            self.mission_complete = 0 
        #print(self.robot_now_path)
        
        if(self.robot_waypoint_index > self.go_path_num-1): # 돌아오는 상황 
            robot_status = 1 #robot_status가 1일때 -> guide함, 로봇 색깔바뀜(빨간색), 로봇에 영향받는 agent 색깔 바뀜(주황색) 
            self.drag = 1 # 사람들이 이끌리게 만드는 term 
        else:
            robot_status = 0
            self.drag = 0
        print("현재 골 : ", self.robot_now_path[self.robot_waypoint_index])
        print(self.robot_waypoint_index)
        print(robot_xy)
        print(self.save_target)
        print(self.robot_now_path)
        robot_goal = self.robot_now_path[self.robot_waypoint_index]
        
        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<3):  # d는 실제 좌표의 거리  이게 작아지면 인덱스 하나 크게 해서 다음 목표로
            self.robot_waypoint_index = self.robot_waypoint_index + 1

        if(self.robot_waypoint_index == len(self.robot_now_path)):
            self.mission_complete = 1 #미션을 새로 만들어야해 (끝났으니까)
            self.robot_waypoint_index = 0
            return [int(robot_xy[0]), int(robot_xy[1])]

        goal_x = self.robot_now_path[self.robot_waypoint_index][0] - robot_xy[0] #역학을 위한.. 
        goal_y = self.robot_now_path[self.robot_waypoint_index][1] - robot_xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        
        intend_force = 2
        desired_speed = 1.5

        if(self.drag == 0):
            desired_speed = 10
        else:
            desired_speed = 10

        if(goal_d != 0):
            desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
            desired_force = [0, 0]
    
        
        x=int(round(robot_xy[0]))
        y=int(round(robot_xy[1]))

        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i)
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i])
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]

        k=4

        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = robot_xy[0] - n_x
            d_y = robot_xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))


            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 5*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 5*k*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1
        
        F_x = 0
        F_y = 0
        
        F_x += desired_force[0]
        F_y += desired_force[1]

        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass

        robot_xy[0] += vel[0] * time_step
        robot_xy[1] += vel[1] * time_step
        
        next_x = int(round(robot_xy[0]))
        next_y = int(round(robot_xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        #self.robot_guide = 0
        return (next_x, next_y)

    def agents_in_each_space(self):
        global num_remained_agent
        from model_renew import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int((i.xy)[0])][int((i.xy)[1])]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        for j in space_agent_num.keys():
            print(j, "공간에 ", space_agent_num[j], "명이 있음")
            num_remained_agent += space_agent_num[j]
        return space_agent_num

    def find_target(self, space_agent_num, floyd_distance):
        global past_target
        self.robot_now_path = []
        agent_max = 0
        space_priority = {}
        distance_to_safe = {}
        

        evacuation_points = []
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

        for i in space_agent_num.keys():
            min_d = 10000
            distance_to_safe[i] = min_d
            for j in evacuation_points:
                if min_d>floyd_distance[i][j] :
                    min_d = floyd_distance[i][j]
                    distance_to_safe[i] = min_d
        for i2 in space_agent_num.keys():
            if (distance_to_safe[i2]>9999):
                distance_to_safe[i2] = -1

        #print("distance_to_safe :", distance_to_safe)
                
        
        for l in space_agent_num.keys():
            space_priority[l] = distance_to_safe[l] * space_agent_num[l]
            if(l==past_target):
                space_priority[l] -= 10000
        #(space_priority)
        agent_max = 0
        print(space_priority)
        for k in space_priority.keys():
            if (space_priority[k]>agent_max):
                self.save_target = k
                agent_max = space_priority[self.save_target]
        min_distance = 1000
        for m in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[self.save_target][m]<min_distance):
                self.save_point = m
                min_distance = floyd_distance[self.save_target]
        print(self.save_target)
        
    def test_modeling(self):
        global robot_radius
        global robot_xy
        global robot_status
        from model_renew import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # 주변 12칸
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i) # near은 실제로 현실적인 주변 좌표고.
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i]) # 이걸 하면 near에 있는 fighting agent 객체가 나옴
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일 # 결국 near_agents_list에는 주변 그리드의 agent가 쌓임

        F_x = 0
        F_y = 0
        k = 3
        r_0 = 0.3
        valid_distance = 3
        intend_force = 2
        time_step = 0.2 #time step... 작게하면? 현실의 연속적인 시간과 비슷해져 현실적인 결과를 얻을 수 있음. 그러나 속도가 느려짐
                        # 크게하면? 속도가 빨라지나 비현실적.. (agent가 튕기는 등..)
        #time_step마다 desired_speed로 가고, desired speed의 단위는 1픽셀, 1픽셀은 0.5m
        #만약 time_step가 0.1이고, desired_speed가 2면.. 0.1초 x 2x0.5m = 한번에 최대 0.1m 이동 가능..
        desired_speed = 2 # agent가 갈 수 있는 최대 속도, 나중에는 정규분포화 시킬 것
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]
        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = self.xy[0] - n_x
            d_y = self.xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))
            if(valid_distance<d):
                continue    # d가 3보다 크면 이 아래가 안 돌고 다음 for문이 돌아.

            
            # print("F : ", F)
            # if(d>0 and near_agent.dead == False):
            #     F_x += (F*(d_x/d))
            #     F_y += (F*(d_y/d))
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽 / 회색 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 1/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1/4*np.exp(-(d/2))*(d_y/d) # 뒤에 거는 벡터고, 앞에 거가 힘의 크기다.

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 2*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 2*k*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1

             
        self.which_goal_agent_want() # agent의 다음 골 목표 설정 

        goal_x = self.now_goal[0] - self.xy[0]
        goal_y = self.now_goal[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))

        robot_x = robot_xy[0] - self.xy[0]
        robot_y = robot_xy[1] - self.xy[1]
        robot_d = math.sqrt(pow(robot_x,2)+pow(robot_y,2))
        if(robot_d<robot_radius and robot_status == 1):
            goal_x = robot_x
            goal_y = robot_y
            goal_d = robot_d
            self.type = 1 # 가이드 당하고 있는 agent는 type=1
            self.now_goal = robot_goal
            
        else :
            self.type = 0


        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :                      # intend_force랑 desired_speed는 상수
          desired_force = [0, 0]
        

        F_x += desired_force[0]
        F_y += desired_force[1]
        
        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        

        self.acc[0] = F_x/self.mass
        self.acc[1] = F_y/self.mass

        self.vel[0] = self.acc[0]
        self.vel[1] = self.acc[1]

        self.xy[0] += self.vel[0] * time_step
        self.xy[1] += self.vel[1] * time_step
        
        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        self.robot_guide = 0
        return (next_x, next_y)




#-------------------------------------------------------------------------


hazard_id = 5000

def make_plane(xy1, xy2): # 두 좌표를 받고, (이를 모서리로 하는 평면)을 구성하는 점들의 집합을 도출
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

        # 벽을 agent로 표현하게 됨. agent 10이 벽이다.
        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기


        wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        space = []
        self.wall_matrix = list()
        self.only_one_wall = list()
        self.indoor_connect = list() # 방과 방 사이를 연결하는 문을 만들기 위한 리스트 
        for i in range(51):
            tmp = []
            for j in range(51):
                tmp.append(0)
            self.wall_matrix.append(tmp)
            self.only_one_wall.append(tmp)
            self.indoor_connect.append(tmp) #50x50 맵 초기화 
        
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
        self.space_goal_dict = {} #각 space가 가지는 goal을 표현하기 위함 # 딕셔너리임
        self.space_graph = {} #각 space의 인접 space를 표현하기 위함 # 딕셔너리임
        self.space_type = {} #space type이 0이면 빈 공간, 1이면 room



        self.init_outside() #외곽지대 탈출로 구현 # 탈출로에서는 탈출로로만 다님. 다시 안으로 들어가지 않음.
        
        self.door_list = [] #일단 무시
        self.map_recur_divider_fine([[1, 1], [9, 9]], 5, 5, 0, self.space_list, self.room_list, 1) # recursion을 이용해 랜덤으로 맵을 나눔 

        for j in self.space_list: 
            self.space_goal_dict[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = [] # 모든 space에 대한 goal을 설정할 것임 # 초기화 해 놓은거
            self.space_graph[((j[0][0], j[0][1]), (j[1][0], j[1][1]))] = []
        for k in self.room_list:
            self.space_type[((k[0][0], k[0][1]), (k[1][0], k[1][1]))] = 1 # self.space_type에서는 room 이면 value의 값을 1로 설정 

        self.connect_space_with_one_goal() #space 그래프 연결 
        
        if(self.simulation_type): # 만약 room이 있는 시뮬레이션이라면.. # 야외면 무시해도 되는 부분
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

        self.random_agent_distribute_outdoor(20) # agent를 space에 랜덤 배치. 각 space에 1명씩은 할당되도록 함.
        #self.random_hazard_placement(random.randint(1,3))
        if(self.is_left_exit):
            self.space_goal_dict[((0,0), (5, 45))] = [self.left_exit_goal] # 딕셔너리에 출구의 평균 좌표를 할당

        if(self.is_up_exit):
            self.space_goal_dict[((0,45), (45, 49))] = [self.up_exit_goal]

        if(self.is_right_exit):
            self.space_goal_dict[((45,5), (49, 49))] = [self.right_exit_goal]

        if(self.is_down_exit):
            self.space_goal_dict[((5,0), (49, 5))] = [self.down_exit_goal]

        #exit 구역의 goal 재정의
#---------------------------------------------------------------------------------------------------------------------
        self.space_agent_num = {} #각 space에 agent가 몇명 있는가..
        for i in self.space_list:
            self.space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0 # 공간에 좌표에 해당하는 그 공간의 agent 수 딕셔너리에 할당
            # value는 나중에
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
                    self.grid_to_space[x][y] = space # 어떤 좌표든 찍으면 어느 space에 해당하는 지 나오겠네


        #print(self.space_goal_dict)

        for i in self.room_list: # 실제로 plane을 만드는 부분인 것 같은데?
            wall = wall + make_room(i[0], i[1]) # 여태까지의 room_list는 꼭짓점의 좌표만 담고 있는데, 이거를 통해 wall에 벽의 모든 좌표를 담음
        for j in self.space_list:
            space = space + make_room(j[0], j[1])

        set_transform = set(wall)
        wall = list(set_transform) # wall의 요소 중 겹치는 걸 제외하는 표현
        # for i in goal_list:
        #     for j in i:
        #         if j in wall:    
        #             wall.remove(j)
        #             self.wall_matrix[j[0]][j[1]] = 0
    
        for i in self.door_list: # 일단 무시
                if i in wall:    
                    wall.remove(i)
                    self.wall_matrix[i[0]][i[1]] = 0

        for i in range(len(wall)): # 실제로 agent를 넣는 부분이라고 이해
            if (self.only_one_wall[wall[i][0]][wall[i][1]] == 1 and wall[i][0]!=0 and wall[i][1]!=0 and wall[i][1]!=49): # 벽이 겹치지 않도록 하는 이중장치로 생각
                continue
            c = FightingAgent(i, self, wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, wall[i])
            self.only_one_wall[wall[i][0]][wall[i][1]] = 1
        for i in range(len(space)):
            if (self.only_one_wall[space[i][0]][space[i][1]] == 1 and space[i][0]!=0 and space[i][1]!=0 and space[i][1]!=49):
                continue
            c = FightingAgent(10000+i, self, space[i], 12)
            self.schedule_w.add(c)
            self.grid.place_agent(c, space[i])
            self.only_one_wall[space[i][0]][space[i][1]] = 1
        #print(self.space_graph)
        

        self.way_to_exit() #탈출구와 연결된 space들은 탈출구로 향하게 하기
        if(self.is_left_exit):
            self.space_goal_dict[((0,0), (5, 45))] = [self.left_exit_goal] # make_exit에서부터 할당됩니다~
            # space_goal_dict란 공간을 넣으면 그 공간에 있을 때 agent가 어디로 가야할 지 알려주는 딕셔너리
        if(self.is_up_exit):
            self.space_goal_dict[((0,45), (45, 49))] = [self.up_exit_goal]

        if(self.is_right_exit):
            self.space_goal_dict[((45,5), (49, 49))] = [self.right_exit_goal]

        if(self.is_down_exit):
            self.space_goal_dict[((5,0), (49, 5))] = [self.down_exit_goal]
        
        self.robot_placement() #로봇 배치 

        self.floyd_warshall_matrix = self.floyd_warshall() 
        #floyd_warshall() 함수는 두 개의 이중 딕셔너리를 리턴함
        # 첫번째 이중 딕셔너리는 start space 부터 end space까지 경로
        # 두번째 이중 딕셔너리는 start space 부터 end space까지의 거리 
        
        self.floyd_path = self.floyd_warshall_matrix[0]   
        self.floyd_distance = self.floyd_warshall_matrix[1]
        # 여기부터 안쓰입니다.
        vertices = list(self.space_graph.keys()) # space_graph에서 key를 추출 (모든 공간이 담김)
        goal_matrix = {start: {end: float('infinity') for end in vertices} for start in vertices}

        for i in vertices:
            for j in vertices:
                if (i==j):
                    continue
                goal_matrix[i][j] = space_connected_linear(i, j) # 공간 i 와 공간 j 사이에 골 찍기 
                
    def make_exit(self):
        exit_rec = []
        only_one_exit = random.randint(1,4) #현재는 출구가 하나만 있게 함 
        
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
            start_exit_cell = random.randint(0, 45-exit_size) #출구가 어디부터 시작되는가? #넘어갈까봐
            for i in range(0, 5): 
                for j in range(start_exit_cell, start_exit_cell+exit_size): #채운다~
                    exit_rec.append((i,j)) #exit_rec에 떄려 넣는다~
                    self.left_exit_goal[0] += i
                    self.left_exit_goal[1] += j
                    left_exit_num +=1
            self.left_exit_goal[0] = self.left_exit_goal[0]/left_exit_num #출구 좌표의 평균 
            self.left_exit_goal[1] = self.left_exit_goal[1]/left_exit_num
            self.left_exit_area = [[0, start_exit_cell], [5, start_exit_cell+exit_size]]

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
    
    def way_to_exit(self):
        if(self.is_left_exit): 
            for i in self.space_graph[((0,0), (5, 45))]: # i에는 left exit과 접하는 space가 찍힐 것.
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))] = [goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), space_connected_linear(i, [[0,0], [5, 45]]))] # space_connected_linear는 왼쪽 출구와 접하는 공간인 i와 왼쪽 탈출구의 좌표를 넣으면, 접하는 부분의 가운데 좌표를 뱉음
        if(self.is_right_exit):                                                 
            for i in self.space_graph[((45,5), (49, 49))]:
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))] = [goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), space_connected_linear(i, [[45,5], [49, 49]]))]
        if(self.is_up_exit):
            for i in self.space_graph[((0,45), (45, 49))]:
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))] = [goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), space_connected_linear(i, [[0,45], [45, 49]]))]
        if(self.is_down_exit):
            for i in self.space_graph[((5,0), (49, 5))]:
                self.space_goal_dict[((i[0][0], i[0][1]), (i[1][0], i[1][1]))] = [goal_extend(((i[0][0], i[0][1]), (i[1][0], i[1][1])), space_connected_linear(i, [[5,0], [49, 5]]))]

    def robot_placement(self): # 야외 공간에 무작위로 로봇 배치 
        inner_space = []
        for i in self.outdoor_space:
            if (i!=[[0,0], [5, 45]] and i!=[[45,5], [49, 49]] and i != [[0,45], [45, 49]] and i !=[[5,0], [49, 5]]):
                inner_space.append(i) # 방이 아니면서, 탈출로도 아닌 공간을 inner_space로 저장
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


        x = random.randint(xy[0][0]+1, xy[1][0]-1) # 무작위로 뽑힌 inner_space 안의 좌표 중 무작위로 선택
        y = random.randint(xy[0][1]+1, xy[1][1]-1)


        a = FightingAgent(self.agent_id, self, [x,y], 3)
        self.agent_id = self.agent_id + 1 # agent id는 모두 달라야해서 벽이랑 다르게 하려고 안전빵으로 1000으로 해놓고 1씩 추가해가면서 쓴다
        self.schedule.add(a)
        self.grid.place_agent(a, (x, y))
        #self.agents.append(a)
        
        
                    
        
    
    
    def random_agent_distribute_outdoor(self, agent_num):
        case = random.randint(1,2) 
        # case1 -> 방에 사람이 있는 경우
        # case2 -> 밖에 주로 사람이 있는 경우
        only_space = []
        for sp in self.space_list:
            if (not sp in self.room_list and sp != [[0,0], [5, 45]] and sp != [[0, 45], [45, 49]] and sp != [[45, 5], [49, 49]] and sp != [[5,0], [49,5]]):
                only_space.append(sp) # 탈출로가 아닌 space를 only space에 할당
        space_num = len(only_space)
        
        
        space_agent = agent_num # 위치시킬 agent 개수

        random_list = [0] * space_num # space_num 만큼의 0 을 가진 list 생성

        # 총합이 agent num이 되도록 할당
        for i in range(space_num - 1): # 각 space에 최소한 1명씩은 들어가게 설정
            random_num = random.randint(1, space_agent - sum(random_list) - (space_num - i - 1))
            while(random_num>space_agent*(1/3)):
                random_num = random.randint(1, space_agent - sum(random_list) - (space_num - i - 1))

            random_list[i] = random_num

        # 마지막 숫자는 나머지 값으로 설정해서 총 agent 수 맞춰주기
        if(space_num != 0):
            random_list[-1] = space_agent - sum(random_list)

        for j in range(len(only_space)):
            self.agent_place(only_space[j][0], only_space[j][1], random_list[j])
            # agent_place 함수는 space와 num 주면, 그 space 안에 num만큼 agent 랜덤 할당하는 함수


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
            check_connection.append(tmp) #이중 리스트로 space들 겹치는 부분 확인할거임

        for space in self.space_list: # space끼리 연결 #space 그래프 만들기
            if space in self.room_list: # 방이면 건너뛰기
                continue
            check_connection = []
            for i1 in range(51):
                tmp = []
                for j1 in range(51):
                    tmp.append(0)
                check_connection.append(tmp)

            # space의 바운더리 1로 채우기
            for y in range(space[0][1]+1, space[1][1]):
                check_connection[space[0][0]][y] = 1 #left 
            for y in range(space[0][1]+1, space[1][1]):
                check_connection[space[1][0]][y] = 1 #right
            for x in range(space[0][0]+1, space[1][0]):
                check_connection[x][space[0][1]] = 1 #down
            for x in range(space[0][0]+1, space[1][0]):
                check_connection[x][space[1][1]] = 1 #up
            # 다른 space들 겹치는 지 체크 하는 부분
            for space2 in self.space_list: 
                if space2 in self.room_list: # 방이면 패스
                    continue
                check_connection2 = copy.deepcopy(check_connection)
                checking = 0
                if(space == space2): # 지금 방이랑 겹치면 패스
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
                        left_goal[0] += space2[0][0] # 겹치는 부분 x좌표 다 더해
                        left_goal[1] += y2 # 겹치는 부분 y좌표 다 더해
                        left_goal_num = left_goal_num + 1 # 나중에 이거로 나눠서 평균 구할 것임
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
                    self.space_graph[((space[0][0], space[0][1]), (space[1][0], space[1][1]))].append(space2) # 딕셔너리에 접하는 space인 space2 추가
                                              #왼쪽아래모서리              #오른쪽위모서리
                #위에까지가 space graph 만들기
                    

                if(left_goal[0] != 0 and left_goal[1] != 0):
                    first_left_goal = [0, 0]
                    first_left_goal[0] = (left_goal[0]/left_goal_num)
                    first_left_goal[1] = (left_goal[1]/left_goal_num)
                    self.space_goal_dict[((space[0][0],space[0][1]), (space[1][0], space[1][1]))].append(goal_extend(((space[0][0],space[0][1]), (space[1][0], space[1][1])), first_left_goal))
                    # goal 딕셔너리에 추가하긴 하는데, 그냥 goal 추가하면 다음 방의 goal을 못 찾아서 goal_extend 함수 정의해서 다음 방의 goal을 안전히 파악할
                    # 수 있게 한 뒤에 딕셔너리에 추가
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
    

    def map_recur_divider_fine(self, xy, x_unit, y_unit, num, space_list, room_list, is_room): # ex) xy = [[2,3], [4,5]] # space 나누는 것. 나누고 방을 선택함
        x_diff = xy[1][0] - xy[0][0] # a점의 튜플은 (xy[0][0],xy[0][1])  b점의 튜플은 (xy[1][0],xy[1][1]) 
        y_diff = xy[1][1] - xy[0][1]

        real_xy =  [ [xy[0][0]*x_unit, xy[0][1]*y_unit], [xy[1][0]*x_unit, xy[1][1]*y_unit]] # 실제 좌표 
        if(is_room==0): # 빈 공간이면 space list에 넣어
            space_list.append(real_xy)
            return
                   
        if(x_diff<3 or y_diff<3): #방이 일정크기 이하면 방 만들고 회귀 빠져나오기
            space_list.append(real_xy)
            room_list.append(real_xy)
            return
        
            
        if(num==1): # 첫번째 획 그은 거. 점차 획 많이 그을수록, room 만들고 탈출할 확률 증가함.
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
            
        divide_num_y = random.randint(1, y_diff-1) # 2개로 나눌 때, 몇 칸 띄고 나눌건지
        divide_num_x = random.randint(1, x_diff-1)

        random_exist_room1 = random.randint(0,1)
        random_exist_room2 = random.randint(0,1)
        random_exist_room3 = random.randint(0,1)
        random_exist_room4 = random.randint(0,1)

        if (random_exist_room1 == 0):
            random_exist_room2 = 1
        if (random_exist_room3 == 0):
            random_exist_room4 = 1
    
        special_hallway = random.randint(1, 2) #가운데 나눠지는 길을 만들기 위함(일반적인 건물배치도 생성 유도)
        if(num<3): # 3개로 나눔
            if (num%2==0): # 세로선을 긋는다
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
            else: # 가로선을 긋는다
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

        else: # 2개로 나눔
            if(num<1):
                random_exist_room1 = random_exist_room2 = random_exist_room3 = random_exist_room4 = 1
            if (num%2==0): #세로선을 긋는다
                self.map_recur_divider_fine([[xy[0][0], xy[0][1]], [xy[0][0]+divide_num_x, xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room1)
                self.map_recur_divider_fine([[xy[0][0]+divide_num_x, xy[0][1]], [xy[1][0], xy[1][1]]], x_unit, y_unit, num+1, space_list, room_list, random_exist_room2)
        
            else: #가로선을 긋는다
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
            #self.agents.append(a)


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

#-----------------------------------------------------------------------------------------------------------------------
# 여기서부터 시각화 없이 시뮬
'''
def agents_in_each_space(self):
        global num_remained_agent
        from model_renew import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int((i.xy)[0])][int((i.xy)[1])]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        for j in space_agent_num.keys():
            print(j, "공간에 ", space_agent_num[j], "명이 있음")
            num_remained_agent += space_agent_num[j]
        return space_agent_num
'''
model = FightingModel(5,50,50)

n = 500  # n을 반복하려는 횟수로 설정


num_escaped_episodes = {
    "50%": 0,
    "80%": 0,
    "100%": 0
}
start_time = time.time()
for i in range(n): # 에피소드 n번 돌린다
    model.step()
    print('에피소드 수',i+1)
    print('남은 agent 수', num_remained_agent)
#-----------------------------------------------------------------------------------------------------------------------
    if i == 0: # 처음 생성된 agent 수 저장
        num_assigned_agent = num_remained_agent

    if num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
        if num_escaped_episodes["50%"] == 0:
            num_escaped_episodes["50%"] = i+1

    if num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
        if num_escaped_episodes["80%"] == 0:
            num_escaped_episodes["80%"] = i+1
    
    if num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
        if num_escaped_episodes["100%"] == 0:
            num_escaped_episodes["100%"] = i+1
            print(num_escaped_episodes)
            with open("example.txt", "a") as f:
                f.write("{}, {}, {}\n".format(num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))

        break
    else:
        num_remained_agent = 0 # 초기화
#-----------------------------------------------------------------------------------------------------------------------

end_time = time.time()
execution_time = end_time - start_time
print("코드 실행 시간:", execution_time, "초")