#this source code requires Mesa==2.2.1 
#^__^
from mesa import Model
from agent import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

import agent
from agent import WallAgent
import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import DBSCAN
from matplotlib.path import Path
import triangle as tr

import cv2

def are_meshes_adjacent(mesh1, mesh2):
    # 두 mesh의 공통 꼭짓점의 개수를 센다
    common_vertices = set(mesh1) & set(mesh2)
    return len(common_vertices) >= 2  # 공통 꼭짓점이 두 개 이상일 때 인접하다고 판단

# goal_list = [[0,50], [49, 50]]
hazard_id = 5000
total_crowd = 10
max_specification = [20, 20]

number_of_cases = 0 # 난이도 함수 ; 경우의 수
started = 1

def get_points_within_polygon(vertices, grid_size=1):
    polygon_path = Path(vertices)
    
    # 다각형의 bounding box 설정
    min_x = int(np.min([v[0] for v in vertices]))
    max_x = int(np.max([v[0] for v in vertices]))
    min_y = int(np.min([v[1] for v in vertices]))
    max_y = int(np.max([v[1] for v in vertices]))
    
    # 그리드 점 생성
    x_grid = np.arange(min_x, max_x + grid_size, grid_size)
    y_grid = np.arange(min_y, max_y + grid_size, grid_size)
    grid_points = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)
    
    # 다각형 내부 점 필터링
    inside_points = grid_points[polygon_path.contains_points(grid_points)]
    
    return inside_points.tolist()

def bresenham(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to find all grid points that a line passes through.
    
    Args:
    x0, y0: Starting point of the line.
    x1, y1: Ending point of the line.
    
    Returns:
    A list of grid coordinates that the line passes through.
    """
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    while True:
        points.append([x0, y0])
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x0 += sx
        
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

def find_triangle_lines(v0, v1, v2):
    """
    Finds all grid coordinates that the triangle's edges pass through.
    
    Args:
    v0, v1, v2: The three vertices of the triangle, each as [x, y].
    
    Returns:
    A list of unique grid coordinates that the triangle's edges pass through.
    """
    line_points = set()  # Using a set to avoid duplicates
    
    # Get the points for each edge of the triangle
    line_points.update(tuple(pt) for pt in bresenham(v0[0], v0[1], v1[0], v1[1]))
    line_points.update(tuple(pt) for pt in bresenham(v1[0], v1[1], v2[0], v2[1]))
    line_points.update(tuple(pt) for pt in bresenham(v2[0], v2[1], v0[0], v0[1]))
    
    return list(line_points)

# # Example usage
# v0 = [10, 10]
# v1 = [20, 15]
# v2 = [15, 25]

# # Find grid coordinates for the triangle's edges
# line_coords = find_triangle_lines(v0, v1, v2)
# print("Grid coordinates that the triangle's edges pass through:", line_coords)

def is_point_in_triangle(p, v0, v1, v2):
    """
    Determines if a point p is inside the triangle formed by v0, v1, v2 using barycentric coordinates.
    
    Args:
    p: The point to check, as [x, y].
    v0, v1, v2: The triangle's vertices, each as [x, y].
    
    Returns:
    True if the point is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def calculate_internal_coordinates_in_triangle(width, height, v0, v1, v2, D):
    """
    Finds grid points inside the triangle formed by v0, v1, v2. 
    A point is included if more than half of the grid square overlaps with the triangle.
    
    Args:
    grid: The grid of points, a 2D array where each point is a coordinate [x, y].
    v0, v1, v2: The triangle's vertices, each as [x, y].
    D: The distance between grid points (grid resolution).
    
    Returns:
    A list of grid points inside the triangle.
    """
    grid_points_in_triangle = []
    
    # Loop through all grid points
    for x in range(width):
        for y in range(height):
            grid_point = [x, y]
            
            # Check if the center of the grid point is inside the triangle
            if is_point_in_triangle(grid_point, v0, v1, v2):
                grid_points_in_triangle.append(grid_point)
            # else:
            #     # If the center is not inside, check the neighboring points (for partial inclusion)
            #     # Check the four corner points of the grid square
            #     corners = [
            #         [x - D/2, y - D/2],
            #         [x + D/2, y - D/2],
            #         [x - D/2, y + D/2],
            #         [x + D/2, y + D/2]
            #     ]
                
            #     inside_corners = sum(is_point_in_triangle(corner, v0, v1, v2) for corner in corners)
                
            #     # Include grid point if more than half of its corners are inside the triangle
            #     if inside_corners >= 2:
            #         grid_points_in_triangle.append(grid_point)

    return grid_points_in_triangle

def add_intermediate_points(p1, p2, D):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    if dist > D:
        num_points = int(dist // D) + 1
        return np.linspace(p1, p2, num=num_points+1, endpoint = False)[1:].tolist()
    return []

def generate_segments_with_points(vertices, segments, D):
    new_vertices = vertices.copy()
    new_segments = []
    for seg in segments:
        p1 = vertices[seg[0]]
        p2 = vertices[seg[1]]
        new_points = add_intermediate_points(p1, p2, D)
        last_index = seg[0]
        for point in new_points:
            new_vertices.append(point)
            new_index = len(new_vertices) - 1
            new_segments.append([last_index, new_index])
            last_index = new_index
        new_segments.append([last_index, seg[1]])
    return new_vertices, new_segments

 
class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int, model_num = -1):
        if (model_num == -1):
            model_num = random.randint(1,5)

        self.running = (
            True  # required by the MESA Model Class to start and stop the simulation
        )
        self.agent_id = 1000
        self.agent_num = 0
        self.datacollector_currents = DataCollector(
            {
                "Remained Agents": FightingModel.current_healthy_agents,
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        self.width = width
        self.height = height      
        self.obstacle_mesh = []
        # map_ran_num = 2
        self.walls = list()
        self.obstacles = list()
        self.mesh = list()
        self.mesh_list = list()
        self.extract_map()     
        self.distance = {}  
        self.schedule = RandomActivation(self)
        self.schedule_e = RandomActivation(self)
        self.running = (
            True
        )
        self.next_vertex_matrix = {}
        self.exit_grid = np.zeros((self.width, self.height))
        self.pure_mesh = []
        self.mesh_danger = {}
        self.match_grid_to_mesh = {}
        self.match_mesh_to_grid = {}
        self.exit_goal_point = []
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.fill_outwalls(width, height)
        self.mesh_map()
        self.make_exit()
        self.construct_map()
        self.calculate_mesh_danger()
        self.exit_list = []
        a = FightingAgent(self.agent_num, self, [0,0], 1)
        self.random_agent_distribute_outdoor(30, 1)
        self.make_robot()
        self.visualize_danger()
        self.robot_xy = [0, 0]
        self.robot_mode = 0

    def fill_outwalls(self, w, h):
        for i in range(w):
            self.walls.append((i, 0))
            self.walls.append((i, h-1))
        for j in range(h):
            self.walls.append((0, j))
            self.walls.append((w-1, j))
    def choice_safe_mesh_visualize(self, point):
        point_grid = (int(point[0]), int(point[1]))
        x = point_grid[0]
        y = point_grid[1]
        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1)]
        for c in candidates:
            if (self.match_grid_to_mesh[c] in self.pure_mesh):
                return c

        return False

        return self.match_grid_to_mesh[point_grid]
    def visualize_danger(self):
        for mesh in self.mesh:
            for i in range(len(mesh)):
                a = FightingAgent(self.agent_num, self, [mesh[i][0], mesh[i][1]], 99)
                
                corresponding_mesh = self.match_grid_to_mesh[(mesh[i][0], mesh[i][1])]
                
                if (corresponding_mesh not in self.pure_mesh):
                    check = self.choice_safe_mesh_visualize([mesh[i][0], mesh[i][1]])
                    if (check == False):
                        continue
                    corresponding_mesh = self.match_grid_to_mesh[check]

                a.danger = self.mesh_danger[corresponding_mesh]
                self.agent_num+=1
                self.schedule_e.add(a)
                self.grid.place_agent(a, [mesh[i][0], mesh[i][1]])
    
    def calculate_mesh_danger(self):
        for mesh in self.pure_mesh:
            shortest_distance = 9999999999
            near_mesh = None
            for e in self.exit_point:
                distance = math.sqrt(pow(mesh[0][0]-e[0], 2) + pow(mesh[0][1]-e[1], 2))
                if distance < shortest_distance:
                    shortest_distance = distance
                    near_mesh = e 
            self.mesh_danger[mesh] = shortest_distance
        return 0

    def mesh_map(self):

        D = 10

        map_boundary = [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        obstacle_hulls = []

        for obstacle in self.obstacles:
            if len(obstacle) == 3 or len(obstacle) == 4:
                hull = ConvexHull(obstacle)
                hull_points = np.array(obstacle)[hull.vertices]
                obstacle_hulls.append(hull_points)
            else:
                raise ValueError("Each obstacle must have either 3 or 4 points.")

        # 경계점 및 장애물의 모서리 점 추가
        vertices = map_boundary.copy()
        for hull_points in obstacle_hulls:
            vertices.extend(hull_points.tolist())

        segments = [[i, (i + 1) % 4] for i in range(4)]  # 맵의 경계
        offset = 4  # 맵 경계 포인트를 위한 오프셋

        # 장애물의 모서리 추가
        for hull_points in obstacle_hulls:
            n = len(hull_points)
            segments.extend([[i + offset, (i + 1) % n + offset] for i in range(n)])
            offset += n

        # 세그먼트 및 포인트로 메쉬화
        vertices_with_points, segments_with_points = generate_segments_with_points(vertices, segments, D)

        # 삼각형화를 위한 데이터 생성
        triangulation_data = {'vertices': np.array(vertices_with_points), 'segments': np.array(segments_with_points)}

        # 삼각형화
        t = tr.triangulate(triangulation_data, 'p')
        boundary_coords = []

        for tri in t['triangles']:
            v0, v1, v2 = t['vertices'][tri[0]], t['vertices'][tri[1]], t['vertices'][tri[2]]
            vertices_tuple = tuple(sorted([tuple(v0), tuple(v1), tuple(v2)]))
            self.mesh_list.append(vertices_tuple)
            
            # 삼각형의 내부 좌표 계산
            internal_coords = calculate_internal_coordinates_in_triangle(self.width, self.height, v0, v1, v2, D)
            # 내부 좌표 저장
            self.mesh.append(internal_coords)
            

        for mesh in self.mesh_list:
            internal_coords = calculate_internal_coordinates_in_triangle(self.width, self.height, mesh[0], mesh[1], mesh[2], D)
            for i in internal_coords:
                if not (i[0], i[1]) in self.match_grid_to_mesh.keys():
                    self.match_grid_to_mesh[(i[0], i[1])] = (mesh[0], mesh[1], mesh[2])


        for mesh in self.mesh_list:
            middle_point = ((mesh[0][0]+mesh[1][0]+mesh[2][0])/3, (mesh[0][1]+mesh[1][1]+mesh[2][1])/3)
            
            for obstacle in self.obstacles:
                if len(obstacle) == 4: # 사각형 obstacle
                    if is_point_in_triangle(middle_point, obstacle[0], obstacle[1], obstacle[2]) or is_point_in_triangle(middle_point, obstacle[0], obstacle[2], obstacle[3]) :
                        self.obstacle_mesh.append(mesh)
                elif len(obstacle) == 3:
                    if is_point_in_triangle(middle_point, obstacle[0], obstacle[1], obstacle[2]):
                        self.obstacle_mesh.append(mesh)


        path = {}
        
        self.next_vertex_matrix = {start: {end: None for end in self.mesh_list} for start in self.mesh_list}
        for i, mesh1 in enumerate(self.mesh_list):
            self.distance[mesh1] = {}
            path[mesh1] = {}
            for j, mesh2 in enumerate(self.mesh_list):
                self.distance[mesh1][mesh2] = 9999999999
                if i == j:
                    self.distance[mesh1][mesh2] = 0
                    self.next_vertex_matrix[mesh1][mesh2] = mesh1
                elif (mesh1 in self.obstacle_mesh or mesh2 in self.obstacle_mesh):
                    # if mesh1 in self.obstacle_mesh:
                    #     print(mesh1, "이 obstacle_mesh에 있음")
                    # elif mesh2 in self.obstacle_mesh:
                    #     print("mesh2가 obstacle_mesh에 있음")
                    self.distance[mesh1][mesh2] = math.inf
                    path[mesh1][mesh2] = None
                elif are_meshes_adjacent(mesh1, mesh2):  # 인접한 경우에만 거리 계산
                    # print("인접함!")
                    mesh1_center = ((mesh1[0][0] + mesh1[1][0] + mesh1[2][0])/3, (mesh1[0][1]+mesh1[1][1]+mesh1[2][1])/3)
                    mesh2_center = ((mesh2[0][0] + mesh2[1][0] + mesh2[2][0])/3, (mesh2[0][1]+mesh2[1][1]+mesh2[2][1])/3)        
                    dist = math.sqrt(pow(mesh1_center[0]-mesh2_center[0], 2) + pow(mesh1_center[1]-mesh2_center[1],2))
                    self.distance[mesh1][mesh2] = dist
                    self.next_vertex_matrix[mesh1][mesh2] = mesh2 
                    #path[mesh1][mesh2] = [i, j] if dist < math.inf else None
                else:
                    self.distance[mesh1][mesh2] = math.inf
                    self.next_vertex_matrix[mesh1][mesh2] = None
        
        n = len(mesh)
        

        for mesh1 in self.mesh_list:
            for mesh2 in self.mesh_list:
                for mesh3 in self.mesh_list:
                    i = mesh2
                    k = mesh1
                    j = mesh3
                    if mesh1 in self.obstacle_mesh or mesh3 in self.obstacle_mesh:
                        continue
                    if self.distance[i][k] + self.distance[k][j] < self.distance[i][j]:
                        self.distance[i][j] = self.distance[i][k] + self.distance[k][j]
                        self.next_vertex_matrix[i][j] = self.next_vertex_matrix[i][k]
        for mesh in self.mesh_list:
            if mesh not in self.obstacle_mesh:
                self.pure_mesh.append(mesh)
        

        boundary_coords = list(set(map(tuple, boundary_coords)))
        
        for i in range(self.width):
            for j in range(self.height):
                for mesh in self.pure_mesh:
                    if is_point_in_triangle([i, j], mesh[0], mesh[1], mesh[2]):
                        if mesh not in self.match_mesh_to_grid.keys():
                            self.match_mesh_to_grid[mesh] = []
                        self.match_mesh_to_grid[mesh].append([i, j])

    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 

        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path

    def extract_map(self):
        width = 70
        height = 70 

        #좌하단 #우하단 #우상단 #좌상단 순으로 입력해주기
        self.obstacles.append([[10, 10], [20, 20], [10, 20]])
        self.obstacles.append([[10, 20], [20, 20], [20,50], [10, 50]])
        self.obstacles.append([[20, 40], [50, 40], [50, 50], [20, 50]])
        self.obstacles.append([[40, 10], [60, 20], [40, 20]])
        

    def construct_map(self):
        for i in range(len(self.walls)):
            a = FightingAgent(self.agent_num, self, self.walls[i], 9)
            self.agent_num+=1
            self.schedule_e.add(a)
            self.grid.place_agent(a, self.walls[i])
        for i in range(len(self.obstacles)):
            for each_point in  get_points_within_polygon(self.obstacles[i], 1):
                a = FightingAgent(self.agent_num, self, each_point, 9)
                self.agent_num+=1
                self.schedule_e.add(a)
                self.grid.place_agent(a, each_point)
        num = 0
        exit_grid = []
        for e in self.exit_list:
            exit_grid.append(get_points_within_polygon(e, 1))
            for each_point in get_points_within_polygon(e, 1):
                self.exit_grid[each_point[0]][each_point[1]] = 1
        for i in range(len(exit_grid)):
            a = FightingAgent(self.agent_num, self, self.exit_list[i][0], 10)
            self.agent_num+=1
            self.schedule_e.add(a)
            for each_point in exit_grid[i]:
                self.grid.place_agent(a, each_point)

        # for mesh in self.mesh:
        #     num +=1 
            # for i in range(len(mesh)):
            #     a = FightingAgent(self.agent_num, self, [mesh[i][0], mesh[i][1]], num%11+1)
            #     self.agent_num+=1
            #     self.schedule_e.add(a)
            #     self.grid.place_agent(a, [mesh[i][0], mesh[i][1]])


                                  
    def make_robot(self):
        self.robot_placement() #로봇 배치 


    def reward_distance_sum(self):
        result = 0
        for i in self.agents:
            if(i.dead == False and (i.type==0 or i.type==1)):
                result += i.danger
        return result 
          

    def make_exit(self):
        exit_width = 5
        exit_height = 5
        self.exit_list = [[(0,0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],
                         [(self.width-exit_width-1,0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],
                         [(0, self.height-exit_height-2), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],
                         [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)]
                        ]
        self.exit_point = [[(exit_width)/2, (exit_height)/2],
                           [(self.width-exit_width-1+self.width-1)/2, (exit_height)/2],
                           [(exit_width)/2, (self.height-exit_height-1+self.height-1)/2],
                           [(self.width-exit_width-1+self.width-1)/2, (self.height-exit_height-1+self.height-1)/2]
                           ]
        return 0




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

        # 출구를 순회하면서 각 출구에 대한 x1, x2, y1, y2를 구합니다.
        for exit_rec in self.exit_recs:
            x1, x2 = float('inf'), float('-inf')
            y1, y2 = float('inf'), float('-inf')
            
            # 출구의 경계좌표를 찾습니다.
            for i in exit_rec:
                if i[0] > x2:
                    x2 = i[0]
                if i[0] < x1:
                    x1 = i[0]
                if i[1] > y2:
                    y2 = i[1]
                if i[1] < y1:
                    y1 = i[1]

            # 좌표 범위에 대해 탐색
            for j in range(y1, y2 + 1):
                self.recur_exit(x1, j, visible_distance, "l")
                self.recur_exit(x2, j, visible_distance, "r")

            for j in range(x1, x2 + 1):
                self.recur_exit(j, y1, visible_distance, "d")
                self.recur_exit(j, y2, visible_distance, "u")

    def recur_exit(self, x, y, visible_distance, direction):
        # 기저 조건 확인
        if visible_distance < 1:
            return
        
        # 경계값 확인
        max_index = len(self.grid_to_space) - 1
        if x < 0 or y < 0 or x > max_index or y > max_index:
            return
        
        # 방문한 위치가 방 내부라면 반환
        if self.grid_to_space[x][y] in self.room_list:
            return

        # 현재 위치를 경로로 설정
        self.exit_way_rec[x][y] = 1
        
        # 방향에 따른 재귀 호출
        if direction == "l":
            self.recur_exit(x - 1, y - 1, visible_distance - 2, "l")
            self.recur_exit(x - 1, y, visible_distance - 1, "l")
            self.recur_exit(x - 1, y + 1, visible_distance - 2, "l")
        elif direction == "r":
            self.recur_exit(x + 1, y - 1, visible_distance - 2, "r")
            self.recur_exit(x + 1, y, visible_distance - 1, "r")
            self.recur_exit(x + 1, y + 1, visible_distance - 2, "r")
        elif direction == "u":
            self.recur_exit(x - 1, y + 1, visible_distance - 2, "u")
            self.recur_exit(x, y + 1, visible_distance - 1, "u")
            self.recur_exit(x + 1, y + 1, visible_distance - 2, "u")
        else:  # direction == "d"
            self.recur_exit(x + 1, y - 1, visible_distance - 2, "d")
            self.recur_exit(x, y - 1, visible_distance - 1, "d")
            self.recur_exit(x - 1, y - 1, visible_distance - 2, "d")



    def robot_placement(self): # 야외 공간에 무작위로 로봇 배치 
        get_point = self.exit_point[random.randint(0, 3)]
        get_point = (int(round(get_point[0])), int(round(get_point[1])))
        self.agent_id = self.agent_id + 10
        self.robot = FightingAgent(self.agent_id, self, [get_point[0],get_point[1]], 3)
        self.agent_id = self.agent_id + 10
        self.schedule.add(self.robot)
        self.grid.place_agent(self.robot, (get_point[0], get_point[1]))

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
        

        space_num = len(self.pure_mesh)
        
        
        space_agent = agent_num
        agent_location = []

        for i in range(agent_num):
            assign_mesh_num = random.randint(0, space_num-1)
            assigned_mesh = self.pure_mesh[assign_mesh_num]
            assigned_coordinates = self.match_mesh_to_grid[assigned_mesh]

            assigned = assigned_coordinates[random.randint(0, len(assigned_coordinates)-1)]
            assigned = [int(assigned[0]), int(assigned[1])]
            if not assigned in agent_location:
                agent_location.append(assigned)
                a = FightingAgent(self.agent_num, self, assigned, 1)
                self.agent_num += 1
                self.schedule.add(a)
                self.grid.place_agent(a, assigned)


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
        


    def step(self):
        """Advance the model by one step."""
        global started
        max_id = 1
        if(started):
            for agent in self.agents:
                if(agent.type==1 or agent.type==0):
                    if(agent.unique_id > max_id):
                        max_id = agent.unique_id
            #self.difficulty_f()
            for agent in self.agents:
                if(max_id == agent.unique_id):
                    agent.dead = True
            started = 0
            max_id = 1
            for agent in self.agents:
                if (agent.unique_id > max_id and (agent.type== 0 or agent.type==1)):
                    max_id = agent.unique_id
            for agent in self.agents:
                if(max_id == agent.unique_id):
                    agent.dead = True 
        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model



    

    
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
        return sum([1 for agent in model.schedule_e.agents if agent.health > 0]) ### agent의 health가 0이어야 cureent_healthy_agents 수에 안 들어감
                                                                               ### agent.py 에서 exit area 도착했을 때 health를 0으로 바꿈


    @staticmethod
    def current_non_healthy_agents(model) -> int:
        """Returns the total number of non healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule_e.agents if agent.health == 0])

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
