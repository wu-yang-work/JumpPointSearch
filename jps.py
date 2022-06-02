from queue import PriorityQueue
import math


class Node(object):
    def __init__(self, x, y, parent=None, force_neighbours=None):
        force_neighbours = force_neighbours if force_neighbours else []
        parent = parent if parent else None
        self.x = x
        self.y = y
        self.parent = parent
        self.force_neighbours = force_neighbours


class JPS(object):
    def __init__(self, map_data, start, end):
        self.data = map_data
        self.start = Node(start[0], start[1])
        self.end = Node(end[0], end[1])
        # self.open = PriorityQueue()
        self.close = []
        self.width = len(self.data)
        self.high = len(self.data[0])

    def is_walkable(self, x, y):
        # 1表示障碍物
        if 0 <= x < self.width and 0 <= y < self.high and self.data[x][y] == 0:
            return True
        return False

    def dir(self, v1, v2):
        v = v1 - v2
        if v > 0:
            return 1
        elif v == 0:
            return 0
        return -1

    def calc_euclidean(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def calc_Manhattan(self, node1, node2):
        return abs(node1.x - node2.x) + abs(node1.y - node2.y)

    def is_in_close(self, node):
        for temp in self.close:
            if node.x == temp.x and node.y == temp.y:
                return True
        return False

    def is_in_open(self, node, open):
        for temp in open:
            if node.x == temp[1].x and node.y == temp[1].y:
                return True
        return False

    def search_hv(self, node, temp, li):
        if not self.is_walkable(temp.x, temp.y):
            return
        hori_dir = self.dir(temp.x, node.x)
        vert_dir = self.dir(temp.y, node.y)
        # 判断temp是否是跳点
        jump_node = self.get_jump_point(node, temp, hori_dir, vert_dir)
        if jump_node is not None:
            if not self.is_walkable(jump_node.x, jump_node.y):
                return
            f = self.calc_euclidean(jump_node, self.start) + self.calc_Manhattan(jump_node, self.end)
            jump_node.parent = node
            print('jump points ({0},{1})'.format(jump_node.x, jump_node.y))
            print('插入前 open size: {}'.format(len(li)))
            li.append((f, jump_node))
            print('插入后 open size: {}'.format(len(li)))

        jump_node = self.jump_search_hv(temp, hori_dir, vert_dir)
        if jump_node is not None:
            if not self.is_walkable(jump_node.x, jump_node.y):
                return
            f = self.calc_euclidean(jump_node, self.start) + self.calc_Manhattan(jump_node, self.end)
            jump_node.parent = node
            print('插入前 open size: {}'.format(len(li)))
            li.append((f, jump_node))
            print('jump points:({0},{1})'.format(jump_node.x, jump_node.y))
            print('插入后 open size: {}'.format(len(li)))

    def search_diag(self, node, temp, li):
        print('search diag start>>')
        if temp is None or not self.is_walkable(temp.x, temp.y):
            return
        hori_dir = self.dir(temp.x, node.x)
        vert_dir = self.dir(temp.y, node.y)
        pre_node = node

        while True:
            if self.is_in_close(temp) or self.is_in_open(temp, li) or not self.is_walkable(temp.x, temp.y):
                return
            if self.get_jump_point(pre_node, temp, hori_dir, vert_dir):
                f = self.calc_euclidean(self.start, temp) + self.calc_Manhattan(self.end, temp)
                temp.parent = node
                print('插入前 open size: {}'.format(len(li)))
                li.append((f, temp))
                print('jump points:({0},{1})'.format(temp.x, temp.y))
                print('插入后 open size: {}'.format(len(li)))
                return
            pre_node = temp
            temp = Node(temp.x + hori_dir, temp.y + vert_dir)

    def get_jump_point(self, node, temp, hori_dir, vert_dir):
        if not self.is_walkable(temp.x, temp.y):
            return
        # 一、起点和终点是跳点
        if (temp.x == self.start.x and temp.y == self.start.y) or (temp.x == self.end.x and temp.y == self.end.y):
            return temp
        # 二、如果temp有强迫邻居，则temp是跳点
        if self.has_force_neighbour(node, temp):
            print('has force neighbours!!')
            return temp
        # 如果父节点在对角方向，节点node水平或垂直满足一、二
        if hori_dir != 0 and vert_dir != 0:
            return self.jump_search_hv(temp, hori_dir, vert_dir)
        return

    def jump_search_hv(self, node, hori_dir, vert_dir):
        i = node.x
        print('    x轴搜索jump point')
        while hori_dir != 0:
            i += hori_dir
            temp = Node(i, node.y)
            if not self.is_walkable(temp.x, temp.y):
                break
            if self.get_jump_point(node, temp, hori_dir, 0) is not None:
                return temp

        j = node.y
        print('    y轴搜索jump point')
        while vert_dir != 0:
            j += vert_dir
            temp = Node(node.x, j)
            if temp is None or not self.is_walkable(temp.x, temp.y):
                break
            if self.get_jump_point(node, temp, 0, vert_dir) is not None:
                return temp
        return

    def has_force_neighbour(self, node, temp):
        if not self.is_walkable(temp.x, temp.y):
            return False
        dirction = self.dir(temp.x, node.x), self.dir(temp.y, node.y)
        temp.force_neighbours = []
        # 水平、垂直
        if dirction[0] == 0 or dirction[1] == 0:
            result1 = self.check_hv_force_neighbour(temp, dirction, 1)
            result2 = self.check_hv_force_neighbour(temp, dirction, -1)
        else:
            result1 = self.check_diag_force_neighbour(temp, dirction, 1)
            result2 = self.check_diag_force_neighbour(temp, dirction, -1)
        return result2 or result1

    def check_hv_force_neighbour(self, node, dirction, sign):
        # 方向
        obstacle_dir = abs(dirction[1]) * sign, abs(dirction[0]) * sign
        obstacle = Node(node.x + obstacle_dir[0], node.y + obstacle_dir[1])
        neighbour = Node(obstacle.x + dirction[0], obstacle.y + dirction[1])

        if not self.is_walkable(neighbour.x, neighbour.y):
            return False
        if not self.is_walkable(obstacle.x, obstacle.y):
            neighbour.parent = node
            node.force_neighbours.append(neighbour)
            print('------ force neighbour --------- ')
            print('node ({0}, {1})'.format(node.x, node.y))
            print('node force neighbour ({0}, {1})'.format(neighbour.x, neighbour.y))
            print('------ force neighbour --------- ')
            return True
        return False

    def check_diag_force_neighbour(self, node, dirction, sign):
        pre_node = Node(node.x - dirction[0], node.y - dirction[1])
        if sign == 1:
            obstacle_dir = dirction[0], 0
            neighbour_dir = dirction[0], 0
        else:
            obstacle_dir = 0, dirction[1]
            neighbour_dir = 0, dirction[1]
        obstacle = Node(pre_node.x + obstacle_dir[0], pre_node.y + obstacle_dir[1])
        neighbour = Node(obstacle.x + neighbour_dir[0], obstacle.y + neighbour_dir[1])
        if not self.is_walkable(neighbour.x, neighbour.y):
            return False
        if not self.is_walkable(obstacle.x, obstacle.y):
            neighbour.parent = node
            node.force_neighbours.append(neighbour)
            print('------ force neighbour --------- ')
            print('node ({0}, {1})'.format(node.x, node.y))
            print('node force neighbour ({0}, {1})'.format(neighbour.x, neighbour.y))
            print('------ force neighbour --------- ')
            return True
        return False

    def check_node(self, node, li):
        print('：：：open size:{}'.format(len(li)))
        v_h = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        if node.parent is None:
            # 搜索上下左右四个方向
            print('parent is None, 开始直线搜索。。。')
            for dirct in v_h:
                temp = Node(node.x + dirct[0], node.y + dirct[1])
                self.search_hv(node, temp, li)
            print('parent is None, 直线搜索完成。。。')
        else:
            # 水平、垂直
            print('has parent, 开始直线搜索。。。')
            hori_dir = self.dir(node.x, node.parent.x)
            vert_dir = self.dir(node.y, node.parent.y)
            if hori_dir != 0:
                temp = Node(node.x + hori_dir, node.y)
                self.search_hv(node, temp, li)
            if vert_dir != 0:
                temp = Node(node.x, node.y + vert_dir)
                self.search_hv(node, temp, li)
            print('has parent, 直线搜索完成。。。')

        if node.parent is None:
            # 搜索4个对角方向
            print('parent is None, 开始对角线搜索。。。。。。。。。。。')
            diag = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            for dirct in diag:
                print('    direction: ', dirct)
                temp = Node(node.x + dirct[0], node.y + dirct[1])
                self.search_diag(node, temp, li)
                print('    ----------------------          ')
            print('parent is None, 对角线搜索完成。。。。。。。。。。。。。。。')
        else:
            print('has parent, 开始对角线搜索。。。')
            print('HHHHHHHHH  open size: {}'.format(len(li)))
            hori_dir = self.dir(node.x, node.parent.x)
            vert_dir = self.dir(node.y, node.parent.y)
            if hori_dir != 0 and vert_dir != 0:
                temp = Node(node.x + hori_dir, node.y + vert_dir)
                self.search_diag(node, temp, li)
            print('has parent, 对角线搜索完成。。。')
            # 遍历 node的 强迫邻居
            print('node({0}, {1}) force neighbour is None? {2}'.format(node.x, node.y, False if len(
                node.force_neighbours) > 0 else True))
            print('has parent, 开始对角线搜索 node force neighbour 。。。')
            for force_neighbour in node.force_neighbours:
                print('node ({0},{1})'.format(node.x, node.y))
                print('node force neighbours ({0},{1})'.format(force_neighbour.x, force_neighbour.y))
                self.search_diag(node, force_neighbour, li)
            print('has parent, 对角线搜索 node force neighbour 完成。。。')
        print('：：：open size:{}'.format(len(li)))

    def search_path(self, li):
        if self.start.x == self.end.x and self.start.y == self.end.y:
            return
        # 计算开始节点的f值
        f = self.calc_euclidean(self.start, self.start) + self.calc_Manhattan(self.start, self.end)
        li.append((f, self.start))
        while len(li) > 0:
            # 取出当前F值最小的点
            li = sorted(li, key=lambda x: x[0])
            cur_node = li.pop(0)[1]
            if cur_node.x == self.end.x and cur_node.y == self.end.y:
                print('开始打印路径')
                while cur_node:
                    print((cur_node.x, cur_node.y), end='\t')
                    cur_node = cur_node.parent
                    if cur_node is None:
                        print()
                print('----------------------')
                return
            print('start search ({0}, {1})... '.format(cur_node.x, cur_node.y))
            self.check_node(cur_node, li)
            self.close.append(cur_node)


if __name__ == '__main__':
    # 小bug，不同路径搜索到相同的跳点，需要更新权重
    map_test = [[0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
                ]
    map_test2 = [[0]*16]*16
    import numpy as np
    map_test2 = np.array(map_test2)
    start = (9, 2)
    end = (9, 10)
    map_test2[0:3, 10] = 1

    map_test2[6, 8:13] = 1
    map_test2[6:12, 8] = 1
    map_test2[11, 8:13] = 1

    map_test2 = map_test2.tolist()
    # print(map_test2)

    # start = (0, 0)
    # end = (6, 6)
    l = []
    jps = JPS(map_test2, start, end)
    jps.search_path(l)


