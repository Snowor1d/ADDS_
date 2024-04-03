#this source code requires Mesa==2.2.1 
#^__^
from mesa import Agent
import math
import numpy as np
import random
import copy
num_remained_agent = 0

feature_weights = [1,1]


one_foot = 1
SumList = [0, 0, 0, 0, 0]

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
robot_radius = 5 #로봇 반경 -> 10미터 
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

    def step(self) -> None:
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
        if(self.type != 3): #robot은 죽지 않는다
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

    def check_stage_agent(self): ## 이건 언제 쓰이나??? agent 움직일 때 현재 자기가 있는 위치 알 때
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
            goal_candiate = self.model.space_goal_dict[now_stage]
            if(len(goal_candiate)==1):
                goal_index = 0
            else:
                goal_index = random.randint(0, len(goal_candiate)-1)
            self.now_goal = goal_candiate[goal_index]
            self.goal_init = 1
            self.previous_stage = now_stage
        now_stage = self.check_stage_agent() #now_stage -> agent가 현재 어느 space에 있는가 
        if(self.previous_stage != self.check_stage_agent()):
            goal_candiate = self.model.space_goal_dict[now_stage] # ex) [[2,0], [3,5],[4,1]] 
            goal_candiate2 = []
            if(len(goal_candiate)>1):
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

        if (self.type == 3):
            new_position = self.robot_policy2()
            print("선택된 action : ",  self.select_Q(robot_xy))
            reward = self.reward_distance(robot_xy, "none", "none")
            #print("reward : ", reward)
            self.model.grid.move_agent(self, new_position)
            return

        new_position = self.test_modeling()
        if(self.type ==0 or self.type==1):
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
    
    # def robot_policy(self):
    #     time_step = 0.2
    #     from model_renew import Model
    #     global random_disperse
    #     global robot_status
    #     global robot_xy 
    #     global robot_radius
    #     global robot_ringing
    #     self.drag = 1
    #     robot_status = 1
    #     space_agent_num = self.agents_in_each_space() #어느 stage에 몇명이 있는지
    #     floyd_distance = self.model.floyd_distance # floyd_distance[stage1][stage2] = 최단거리 
    #     floyd_path = self.model.floyd_path #floyd_path[stage1][stage2] = stage_x 
    #                                        #floyd_path[stage_x][stage_2] = stage_y 
    #                                        #floyd_path[stage_y][stage_2] = ste... 
    #                                        # s1 -> sx -> sy -> .. stage
    
        

    #     self.robot_space = self.model.grid_to_space[int(robot_xy[0])][int(robot_xy[1])] #로봇이 어느 stage에 있는지 나온다 

    #     if(self.mission_complete == 1): #새로운 탈출 path를 찾는다
    #         self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
    #         agent_max = 0 #agent가 가장 많은 stage 
    #         for i in space_agent_num.keys(): 
    #             if (space_agent_num[i]>agent_max):
    #                 self.save_target = i #현재 가장 인구가 많이 있는 stage
    #                 agent_max = space_agent_num[self.save_target] 
            
    #         evacuation_points = []
    #         if(self.model.is_left_exit): 
    #             evacuation_points.append(((0,0), (5, 45)))
    #         if(self.model.is_up_exit):
    #             evacuation_points.append(((0,45), (45, 49)))
    #         if(self.model.is_right_exit):
    #             evacuation_points.append(((45,5), (49, 49)))
    #         if(self.model.is_down_exit):
    #             evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

    #         min_distance = 1000
    #         for i in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
    #             if(floyd_distance[self.save_target][i]<min_distance):
    #                 self.save_point = i 
    #         go_path = self.model.get_path(floyd_path, self.robot_space, self.save_target) #로봇의 초기 위치 -> save_target까지 가는데 최단 경로 stage 리스트 
            
    #         back_path = self.model.get_path(floyd_path, self.save_target, self.save_point) # save_target(인구가 가장 많은 곳)에서 save_point(safe zone) 까지의 최단 경로 
    #         self.go_path_num = len(go_path) #guide를 하기 위해서 ~ 

    #         for i in range(len(go_path)-1):
    #             self.robot_now_path.append(space_connected_linear(go_path[i], go_path[i+1])) #(stage1 stage2) -> 중간 goal을 알려준다  
    #         self.robot_now_path.append([(self.save_target[0][0]+self.save_target[1][0])/2, (self.save_target[0][1]+self.save_target[1][1])/2]) #save target 중점까지 간다 
    #         for i in range(len(back_path)-1):
    #             self.robot_now_path.append(space_connected_linear(back_path[i], back_path[i+1]))  #back path도 넣는다 
    #         self.mission_complete = 0 
        
    #     if(self.robot_waypoint_index > self.go_path_num-1): # 돌아오는 상황 
    #         robot_status = 1 #robot_status가 1일때 -> guide함, 로봇 색깔바뀜(빨간색), 로봇에 영향받는 agent 색깔 바뀜(주황색) 
    #         self.drag = 1 
    #     else:
    #         robot_status = 0
    #         self.drag = 0
    #     robot_goal = self.robot_now_path[self.robot_waypoint_index]
        

    #     d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
    #     if (d<3):
    #         self.robot_waypoint_index = self.robot_waypoint_index + 1

    #     if(self.robot_waypoint_index == len(self.robot_now_path)):
    #         self.mission_complete = 1 #미션을 새로 만들어야해 (끝났으니까)
    #         self.robot_waypoint_index = 0
    #         return [int(robot_xy[0]), int(robot_xy[1])]

    #     goal_x = self.robot_now_path[self.robot_waypoint_index][0] - robot_xy[0] #역학을 위한.. 
    #     goal_y = self.robot_now_path[self.robot_waypoint_index][1] - robot_xy[1]
    #     goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        
    #     intend_force = 2
    #     desired_speed = 1.5

    #     if(self.drag == 0):
    #         desired_speed = 5
    #     else:
    #         desired_speed = 5

    #     if(goal_d != 0):
    #         desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
    #     else :
    #         desired_force = [0, 0]
    
        
    #     x=int(round(robot_xy[0]))
    #     y=int(round(robot_xy[1]))

    #     #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
    #     temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
    #     near_loc = []
    #     for i in temp_loc:
    #         if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
    #             near_loc.append(i)
    #     near_agents_list = []
    #     for i in near_loc:
    #         near_agents = self.model.grid.get_cell_list_contents([i])
    #         if len(near_agents):
    #             for near_agent in near_agents:
    #                 near_agents_list.append(near_agent) #kinetic 모델과 동일
    #     repulsive_force = [0, 0]
    #     obstacle_force = [0, 0]

    #     k=4

    #     for near_agent in near_agents_list:
    #         n_x = near_agent.xy[0]
    #         n_y = near_agent.xy[1]
    #         d_x = robot_xy[0] - n_x
    #         d_y = robot_xy[1] - n_y
    #         d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))


    #         if(near_agent.dead == True):
    #             continue
                
    #         if(d!=0):
    #             if(near_agent.type == 12): ## 가상 벽
    #                 repulsive_force[0] += 0
    #                 repulsive_force[1] += 0

    #             elif(near_agent.type == 1): ## agents
    #                 repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
    #                 repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

    #             elif(near_agent.type == 11):## 검정벽 
    #                 repulsive_force[0] += 2*k*np.exp(-(d/2))*(d_x/d)
    #                 repulsive_force[1] += 2*k*np.exp(-(d/2))*(d_y/d)
    #         else :
    #             if(random_disperse):
    #                 repulsive_force = [1, -1]
    #                 random_disperse = 0
    #             else:
    #                 repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
    #                 random_disperse = 1
        
    #     F_x = 0
    #     F_y = 0
        
    #     F_x += desired_force[0]
    #     F_y += desired_force[1]

    #     F_x += repulsive_force[0]
    #     F_y += repulsive_force[1]
    #     vel = [0,0]
    #     vel[0] = F_x/self.mass
    #     vel[1] = F_y/self.mass

    #     robot_xy[0] += vel[0] * time_step
    #     robot_xy[1] += vel[1] * time_step
        
    #     next_x = int(round(robot_xy[0]))
    #     next_y = int(round(robot_xy[1]))

    #     if(next_x<0):
    #         next_x = 0
    #     if(next_y<0):
    #         next_y = 0
    #     if(next_x>49):
    #         next_x = 49
    #     if(next_y>49):
    #         next_y = 49

    #     return (next_x, next_y)
    
    def robot_policy2(self):
        time_step = 0.2
        from model import Model
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
        
    
        

        self.robot_space = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))] #로봇이 어느 stage에 있는지 나온다 

        if(self.mission_complete == 1): #새로운 탈출 path를 찾는다
            self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
            agent_max = 0 #agent가 가장 많은 stage 
            self.find_target(space_agent_num, floyd_distance)

            past_target = self.save_target

            go_path = self.model.get_path(floyd_path, self.robot_space, self.save_target) #로봇의 초기 위치 -> save_target까지 가는데 최단 경로 stage 리스트 
            
            back_path = self.model.get_path(floyd_path, self.save_target, self.save_point) # save_target(인구가 가장 많은 곳)에서 save_point(safe zone) 까지의 최단 경로 
            self.go_path_num = len(go_path) #guide를 하기 위해서 ~ 

            for i in range(len(go_path)-1):
                self.robot_now_path.append(space_connected_linear(go_path[i], go_path[i+1])) #(stage1 stage2) -> 중간 goal을 알려준다  
            self.robot_now_path.append([(self.save_target[0][0]+self.save_target[1][0])/2, (self.save_target[0][1]+self.save_target[1][1])/2]) #save target 중점까지 간다 
            for i in range(len(back_path)-1):
                self.robot_now_path.append(space_connected_linear(back_path[i], back_path[i+1]))  #back path도 넣는다 
            self.mission_complete = 0 
        
        if(self.robot_waypoint_index > self.go_path_num-1): # 돌아오는 상황 
            robot_status = 1 #robot_status가 1일때 -> guide함, 로봇 색깔바뀜(빨간색), 로봇에 영향받는 agent 색깔 바뀜(주황색) 
            self.drag = 1 
        else:
            robot_status = 0
            self.drag = 0

        
        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<1):
            self.robot_waypoint_index = self.robot_waypoint_index + 1

        if(self.robot_waypoint_index == len(self.robot_now_path)):
            self.mission_complete = 1 #미션을 새로 만들어야해 (끝났으니까)
            self.robot_waypoint_index = 0
            return [int(round(robot_xy[0])), int(round(robot_xy[1]))]

        goal_x = self.robot_now_path[self.robot_waypoint_index][0] - robot_xy[0] #역학을 위한.. 
        goal_y = self.robot_now_path[self.robot_waypoint_index][1] - robot_xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        
        intend_force = 2
        desired_speed = 1.5

        if(self.drag == 0):
            desired_speed = 4
        else:
            desired_speed = 4

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
                    repulsive_force[0] += 5*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 5*np.exp(-(d/2))*(d_y/d)
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
        self.move_to_valid_robot()
        
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

            
        #self.robot_guide = 0
        robot_goal = self.robot_now_path[self.robot_waypoint_index]
        return (next_x, next_y)

    def agents_in_each_space(self):
        global num_remained_agent
        from model import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int(round((i.xy)[0]))][int(round((i.xy)[1]))]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        
        for j in space_agent_num.keys():
            num_remained_agent += space_agent_num[j]
        return space_agent_num
    

    def agents_in_robot_area(self, robot_xyP):
        from model import Model
        number_a = 0
        for i in self.model.agents:
            if(i.dead == False and (i.type == 0 or i.type == 1)): ##  agent가 살아있을 때 / 끌려가는 agent 일 때
                if pow(robot_xyP[0]-i.xy[0], 2) + pow(robot_xyP[1]-i.xy[1], 2) < pow(robot_radius, 2) : ## 로봇 반경 내에 agent가 있다면
                    number_a += 1

        return number_a

        

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

                
        
        for l in space_agent_num.keys():
            space_priority[l] = distance_to_safe[l] * space_agent_num[l]
            if(l==past_target):
                space_priority[l] -= 10000
        #(space_priority)
        agent_max = 0
        for k in space_priority.keys():
            if (space_priority[k]>agent_max):
                self.save_target = k
                agent_max = space_priority[self.save_target]
        min_distance = 1000
        for m in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[self.save_target][m]<min_distance):
                self.save_point = m
                min_distance = floyd_distance[self.save_target]

        
    def test_modeling(self):
        global robot_radius
        global robot_xy
        global robot_status
        from model import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
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
                continue    

            F = k * (valid_distance-d)
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 1*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1*np.exp(-(d/2))*(d_y/d) 

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
            self.type = 1
            self.now_goal = robot_goal
            
        else :
            self.type = 0


        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
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
        #self.xy = self.move_to_valid(self.xy)
        
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


        self.robot_guide = 0
        return (next_x, next_y)
    
    def move_to_valid(self, loc):
        original_loc = [0, 0]
        original_loc[0] = loc[0]
        original_loc[1] = loc[1]
        count = 0
        #print("함수 내 : ", self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))])
        while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
            loc[0] = (original_loc[0] - 0.5)
            loc[1] = (original_loc[1] - 0.5)
            loc[0] += (random.randint(0, 5)/5)
            loc[1] += (random.randint(0, 5)/5)
            count += 1
            if (count>=40):
                break 
        if (count>=40):
            while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
                #print("두번째 루프")
                loc[0] = (original_loc[0] - 1)
                loc[1] = (original_loc[1] - 1)
                loc[0] += (random.randint(0, 10)/5)
                loc[1] += (random.randint(0, 10)/5)
                count += 1 
                if (count>=100):
                    break 
        if (count>=100):
            while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
                loc[0] = (original_loc - 2)
                loc[1] = (original_loc - 2)
                loc[0] += (random.randint(0, 20)/5)
                loc[1] += (random.randint(0, 20)/5)
                count += 1
                if (count>=200):
                    break 
        return loc 

    
    def move_to_valid_robot(self):
        global robot_xy
        original_loc = [0, 0]
        original_loc[0] = robot_xy[0]
        original_loc[1] = robot_xy[1]
        count = 0
        #print("함수 내 : ", self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))])
        while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
            robot_xy[0] = (original_loc[0] - 0.5)
            robot_xy[1] = (original_loc[1] - 0.5)
            robot_xy[0] += (random.randint(0, 5)/5)
            robot_xy[1] += (random.randint(0, 5)/5)
            count += 1
            if (count>=40):
                break 
        if (count>=40):
            while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
                robot_xy[0] = (original_loc[0] - 1)
                robot_xy[1] = (original_loc[1] - 1)
                robot_xy[0] += (random.randint(0, 10)/5)
                robot_xy[1] += (random.randint(0, 10)/5)
                count += 1 
                if (count>=100):
                    break 
        if (count>=100):
            while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
                robot_xy[0] = (original_loc - 2)
                robot_xy[1] = (original_loc - 2)
                robot_xy[0] += (random.randint(0, 20)/5)
                robot_xy[1] += (random.randint(0, 20)/5)
                count += 1
                if (count>=200):
                    break 

        



    def F0_distance(self, state, action, mode):
        global robot_xy
        global one_foot

        now_space = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]
        evacuation_points = []
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
            exit = ((0,0), (5, 45))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
            exit = ((0,45), (45, 49))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
            exit = ((45,5), (49, 49))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 ##?? 근데 왜 초록 구역이 아니고 사이드 구역 전체일까? 그것은 여기로 오면 탈출하기 때문이지요?
            exit = ((5,0), (49, 5))
        min_distance = 1000
        floyd_distance = self.model.floyd_distance 

        
        next_vertex_matrix = self.model.floyd_warshall()[0] ## 이중 딕셔너리, start space 부터 end space 까지의 경로
        for i in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][i] < min_distance): ##floyd_distance[start(now space)][end(i;출구)] 
                exit = i ##지금은 출구가 하나라서 exit이 evacuation_points이지만 출구가 많아진다면 ~
        
        if(exit!=0):
            next_goal = space_connected_linear(((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1])), next_vertex_matrix[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][exit])
        else :
            next_goal = robot_xy
        now_space_x_center = (now_space[0][0] + now_space[1][0])/2
        now_space_y_center = (now_space[1][0] + now_space[1][1])/2

        next_robot_position = [0, 0]
        next_robot_position[0] += robot_xy[0]
        next_robot_position[1] += robot_xy[1]

        if (action=="UP"):
            next_robot_position[1] += one_foot
        elif (action=="DOWN"):
            next_robot_position[1] -= one_foot
        elif (action=="LEFT"):
            next_robot_position[0] -= one_foot
        elif (action=="RIGHT"):
            next_robot_position[0] += one_foot
         
        return floyd_distance[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][exit] - math.sqrt(pow(now_space_x_center-next_goal[0],2)+pow(now_space_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-next_robot_position[0],2)+pow(next_goal[1]-next_robot_position[1],2))
    
    def F1_near_agents(self, state, action, mode):
        global one_foot
        robot_xyP = [0, 0]
        robot_xyP[0] = state[0] ## robot_xyP : action 이후 로봇의 위치
        robot_xyP[1] = state[1]

        if action == "UP": ## action이 UP 이면 로봇의 y좌표에 one_foot을 더함
            robot_xyP[1] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP) ##action 이후 로봇 반경 내 agents 수 구함
        elif action == "DOWN":
            robot_xyP[1] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "RIGHT":
            robot_xyP[0] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "LEFT":
            robot_xyP[0] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)

        return NumberOfAgents


    def reward_distance(self, state, action, mode):
        global SumList
        SumOfDistances = 0 ##agent 하나로부터 출구까지의 거리의 합
        floyd_distance = self.model.floyd_distance

        evacuation_points = [] ## 출구 찾기~
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5)))

        for i in self.model.agents: ##SumOfDistaces 구하는 과정
            if(i.dead == False and (i.type==0 or i.type==1)):
                agent_space = self.model.grid_to_space[int(round(i.xy[0]))][int(round(i.xy[1]))]
                
                next_goal = space_connected_linear(((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1])), self.model.floyd_warshall()[0][((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]])
                agent_space_x_center = (agent_space[0][0] + agent_space[1][0])/2
                agent_space_y_center = (agent_space[1][0] + agent_space[1][1])/2
                a = (floyd_distance[((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]] 
                - math.sqrt(pow(agent_space_x_center-next_goal[0],2) + pow(agent_space_y_center-next_goal[1],2)) 
                + math.sqrt(pow(next_goal[0]-i.xy[0],2) + pow(next_goal[1]-i.xy[1],2)))
                
                ###준아야 너는 아래 코드를 수정해야 하며, 문제는 같은 space 내에서 agents가 움직이는 걸 반영하지 못하는 것에 있단다. 위 코드를 보며 수정하도록 야호^^
                # SumOfDistances += floyd_distance[(agent_space[0][0], agent_space[0][1]), (agent_space[1][0], agent_space[1][1])][evacuation_points[0]]
                SumOfDistances += a

        t = SumList[4]
        SumList[4] = SumList[3]
        SumList[3] = SumList[2]
        SumList[2] = SumList[1]
        SumList[1] = SumList[0]
        SumList[0] = SumOfDistances

        reward = (SumList[1]+SumList[2]+SumList[3]+SumList[4])/4 - SumOfDistances

        return reward

    def select_Q(self, state) :
        global feature_weights
        action_list = [["UP", "GUIDE"], ["UP", "NOGUIDE"], ["DOWN", "GUIDE"], ["DOWN", "NOGUIDE"], ["LEFT", "GUIDE"], ["LEFT", "NOGUIDE"], ["RIGHT", "GUIDE"], ["RIGHT", "NOGUIDE"]]
        Q_list = []
        for i in range(8):
            Q_list.append(i)
        MAX_Q = -9999999
        selected = ["UP", "GUIDE"]
        for j in range(len(Q_list)):
            Q_list[j] = (self.F0_distance(state, action_list[j][0], action_list[j][1]) * feature_weights[0] + self.F1_near_agents(state, action_list[j][0], action_list[j][1])*feature_weights[1])
            if (Q_list[j]>MAX_Q):
                MAX_Q= Q_list[j]
                selected = action_list[j]
        return selected 

        
        # for i in self.model.agents:
        #     i_xyP = [i.xy[0], i.xy[1]] ## action
        #     if action == "UP":
        #         i_xyP[0] += one_foot
        #     elif action == "DOWN":
        #         i_xyP[0] -= one_foot
        #     elif action == "RIGHT":
        #         i_xyP[1] += one_foot
        #     elif action == "LEFT":
        #         i_xyP[1] -= one_foot
            
        #     agent_spaceP = self.model.grid_to_space[int(i_xyP[0])][int(i_xyP[1])]
        #     SumOfDistancesP += floyd_distance[((agent_space[0][0], agent_space[0][1]), evacuation_points[0])]


































