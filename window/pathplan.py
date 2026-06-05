import numpy as np
import random
import heapq
from collections import defaultdict
import math
import pyiges
from scipy.interpolate import CubicSpline
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def pathplan(routes,start,goal):
    def moving_average(data, window_size=5):
        # 确保原始数据长度不小于窗口大小
        if len(data) < window_size:
            return data

        # 初始化平滑结果
        smoothed = np.zeros_like(data)

        # 保留起点和终点
        smoothed[0] = data[0]
        smoothed[-1] = data[-1]

        # 对中间段进行滑动平均
        for i in range(1, len(data) - 1):
            start = max(i - window_size // 2, 1)  # 确保窗口在数据范围内
            end = min(i + window_size // 2 + 1, len(data) - 1)
            smoothed[i] = np.mean(data[start:end])

        return smoothed
    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2 +
            (p1[2] - p2[2]) ** 2) ** 0.5

    def build_graph(routes, threshold=0.001):
        graph = defaultdict(list)
        point_route_map = defaultdict(set)
        all_points = set()
        # 构建 point_route_map 和添加双向路线内的边
        for route_index, route in enumerate(routes):
            for i in range(len(route)):
                point = tuple(route[i])
                all_points.add(point)
                point_route_map[point].add(route_index)
                if i > 0:
                    prev_point = tuple(route[i - 1])
                    dist = distance(prev_point, point)
                    # 添加双向边
                    graph[prev_point].append((point, dist))
                    graph[point].append((prev_point, dist))
        # 连接共享公共点的路线
        for point, routes_set in point_route_map.items():
            if len(routes_set) > 1:
                for route_index in routes_set:
                    route = routes[route_index]
                    indices = [i for i, p in enumerate(route) if tuple(p) == point]
                    for idx in indices:
                        # 前一个点
                        if idx > 0:
                            prev_point = tuple(route[idx - 1])
                            if prev_point not in [n for n, _ in graph[point]]:
                                dist = distance(point, prev_point)
                                graph[point].append((prev_point, dist))
                                graph[prev_point].append((point, dist))
                        # 后一个点
                        if idx < len(route) - 1:
                            next_point = tuple(route[idx + 1])
                            if next_point not in [n for n, _ in graph[point]]:
                                dist = distance(point, next_point)
                                graph[point].append((next_point, dist))
                                graph[next_point].append((point, dist))
        return graph, all_points

    def find_nearest_point(point, points_set):
        min_dist = float('inf')
        nearest_point = None
        for p in points_set:
            dist = distance(point, p)
            if dist < min_dist:
                min_dist = dist
                nearest_point = p
        return nearest_point

    def a_star(graph, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        costs = {start: 0}
        parents = {start: None}
        visited = set()

        while open_set:
            current_cost, current_point = heapq.heappop(open_set)
            #print(f"正在处理节点：{current_point}，当前代价：{current_cost}")
            if current_point == goal:
                #print("已到达目标点。")
                break
            if current_point in visited:
                continue
            visited.add(current_point)
            for neighbor, weight in graph.get(current_point, []):
                new_cost = costs[current_point] + weight
                heuristic = distance(neighbor, goal)
                total_cost = new_cost + heuristic
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    parents[neighbor] = current_point
                    heapq.heappush(open_set, (total_cost, neighbor))
        else:
        # 没有找到路径
            return None

    # 重建路径
        path = []
        point = goal
        while point is not None:
            path.append(point)
            point = parents[point]
        path.reverse()
        return path

    graph, all_points = build_graph(routes, threshold=0.0001)
    start = find_nearest_point(start, all_points)
    goal = find_nearest_point(goal, all_points)
    path = a_star(graph, start, goal)
    if path:
        path_points = np.array(path)

        # 对中间点平滑，保留起点和终点
        smooth_x = moving_average(path_points[:, 0], window_size=5)
        smooth_y = moving_average(path_points[:, 1], window_size=5)
        smooth_z = moving_average(path_points[:, 2], window_size=5)

        smooth_path = list(zip(smooth_x, smooth_y, smooth_z))

    return path, smooth_path, graph
    #return smooth_path, graph
if __name__ == "__main__":
    # ---- 确保使用交互式窗口（能旋转）----
    import matplotlib
    matplotlib.use("Qt5Agg")  # 要在 pyplot 导入之前调用

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 激活 3D 支持
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from stl import mesh
    import numpy as np
    import pyiges

    # ========= 1. 读取 IGES 中心线 =========
    iges = pyiges.read('老模型中心线.igs')  # 换成你的 iges 文件名
    all_points = []

    for entity in iges:
        parameters = entity.parameters
        num_points = int(parameters[0][2])      # 点的数量
        coords_strings = parameters[0][3:]      # 坐标字符串
        coords_floats = [float(value.strip()) for value in coords_strings]

        pts = []
        for i in range(num_points):
            x = coords_floats[i * 3]
            y = coords_floats[i * 3 + 1]
            z = coords_floats[i * 3 + 2]
            pts.append([x, y, z])

        pts = np.array(pts)
        all_points.append(pts)

    # ========= 2. 读取 STL 模型 =========
    # 把 'airway_model.stl' 换成你的气道 STL 文件名
    airway_mesh = mesh.Mesh.from_file('支气管.stl')

    # ========= 3. 设置起点和终点，路径规划 =========
    start = [9.1551647, -94.828995, 43.316994]
    goal  = [50.776901, 144.23911, 39.526905]

    path, smooth_path, graph = pathplan(all_points, start, goal)
    print("raw path length:", len(path))

    # ========= 4. 绘图 =========
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=90)
    # ---- 4.1 画 STL 表面（半透明）----
    stl_collection = Poly3DCollection(
        airway_mesh.vectors,
        alpha=0.3,              # 透明度
        facecolor='lightgray',   # 面颜色
        edgecolor='none'         # 不画边线
    )
    ax.add_collection3d(stl_collection)

    # 根据 STL 自动设置坐标范围
    xs = airway_mesh.x.flatten()
    ys = airway_mesh.y.flatten()
    zs = airway_mesh.z.flatten()
    ax.auto_scale_xyz(xs, ys, zs)

    # ---- 4.2 画中心线（graph）----
    for node in graph:
        x1, y1, z1 = node
        for neighbor, _ in graph[node]:
            x2, y2, z2 = neighbor
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', alpha=0.4)

    # ---- 4.3 画规划路径 ----
    #if path:
        #xs_p, ys_p, zs_p = zip(*path)
        #ax.plot(xs_p, ys_p, zs_p, 'r-', linewidth=2, label='Planned path')

    # 如需画平滑路径，可以打开下面注释
    # if smooth_path:
    #     xs_s, ys_s, zs_s = zip(*smooth_path)
    #     ax.plot(xs_s, ys_s, zs_s, 'y--', linewidth=2, label='Smoothed path')

    # ---- 4.4 起点和终点 ----
    #ax.scatter(*start, color='green', s=60, label='Start')
    #ax.scatter(*goal,  color='red',   s=60, label='Goal')

    # 关掉坐标轴（不显示轴线、刻度、标签）
    ax.set_axis_off()

    # 如果你还想要图例，可以留着；不想要就注释掉
    plt.legend()
    plt.tight_layout()
    plt.show()   # 在弹出的窗口里用鼠标左键拖动即可自由旋转

    # 打印整条路径坐标（可选）
    print("path points:")
    for p in path:
        print(p)

