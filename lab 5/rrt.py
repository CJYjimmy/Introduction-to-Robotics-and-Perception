# author: Jiayuan Chen, Michael Wang

import cozmo
import math
import sys
import time
import numpy as np
import random

from grid import *
from cmap import *
from gui import *
from utils import *
from particle_filter import *
from particle import *
from pf_gui import *
from utils import *

MAX_NODES = 20000


def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object
    d = get_dist(node0,node1)
    if (d < limit):
        return node1
    angle = np.arctan2((node1.y - node0.y),(node1.x - node0.x))
    newP = ((node0.x + np.cos(angle) * limit), (node0.y + np.sin(angle) * limit))
    return Node(newP, node0)
    ############################################################################


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    posibility = random.uniform(0, 1)
    if posibility < 0.05:
        goal = cmap._goals[random.randint(1,len(cmap._goals)) - 1]
        return Node((goal.x, goal.y))
    else:
        check = True
        w, h = cmap.get_size()
        while (check):
            x = random.randint(0,w)
            y = random.randint(0,h)
            rand_node = Node((x,y), None)
            if (cmap.is_inbound(rand_node)) and not cmap.is_inside_obstacles(rand_node):
                check = False
    ############################################################################
    return rand_node


def RRT(cmap, start):
    cmap.add_node(start)
    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        ########################################################################
        # TODO: please enter your code below.
        # 1. Use CozMap.get_random_valid_node() to get a random node. This
        #    function will internally call the node_generator above
        # 2. Get the nearest node to the random node from RRT
        # 3. Limit the distance RRT can move
        # 4. Add one path from nearest node to random node
        #
        rand_node = cmap.get_random_valid_node()
        nearest_node = None
        minD = sys.maxsize
        nodes = cmap.get_nodes()
        for n in nodes:
            if get_dist(n, rand_node) < minD:
                nearest_node = n
                minD = get_dist(rand_node, nearest_node)
        ########################################################################
        time.sleep(0.01)
        cmap.add_path(nearest_node, rand_node)
        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")

def getPosition(robot):
    sX = 6*25.4
    sY = 10*25.4
    curX = robot.pose.position.x
    curY = robot.pose.position.y
    return Node((sX+curX, sY+curY))

async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent

    ########################################################################
    # TODO: please enter your code below.
    # Description of function provided in instructions

    mapWidth, mapHeight = cmap.get_size()
    sX = 6*25.4
    sY = 10*25.4
    marked = {}
    check = 0
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    while True:
        cmap.set_start(getPosition(robot) )
        updated, goal = await detect_cube_and_update_cmap(robot, marked, getPosition(robot))
        if updated:
            cmap.reset()
        if not cmap.is_solved():
            if goal == None and len(cmap.get_goals()) == 0:
                check = check + 1
                if (check == 1):
                    nextPose = cozmo.util.Pose(mapWidth/2 - sX, mapHeight/2 - sY,0,angle_z = cozmo.util.Angle((i%12-6) * 30))
                    await robot.go_to_pose(nextPose).wait_for_completed()
                await robot.turn_in_place(cozmo.util.degrees(35)).wait_for_completed()
                continue
            if len(cmap.get_goals()) > 0:
                cmap.set_start(getPosition(robot) )
                RRT(cmap, cmap.get_start())
                if cmap.is_solved():
                    path = cmap.get_smooth_path()
                    nextNodeIndex = 1
        if cmap.is_solved():
            if nextNodeIndex == len(path):
                print("Arrived")
                continue
            preNode = path[nextNodeIndex - 1]
            nextNode = path[nextNodeIndex]
            angle = math.atan2(nextNode.y - preNode.y,nextNode.x - preNode.x)
            nodePose = cozmo.util.Pose(nextNode.x - sX,nextNode.y - sY,0,angle_z = cozmo.util.Angle(angle))
            await robot.go_to_pose(nodePose).wait_for_completed()
            nextNodeIndex += 1


def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object

        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.
    c = math.cos(local_angle)
    s = math.sin(local_angle)
    xO = local_origin.x
    yO = local_origin.y
    x = node.x
    y = node.y
    newX = xO + x * c + y * -s
    newY = yO + x * s + y * c

    return Node((newX,newY),None)


async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap

    # Padding of objects and the robot for C-Space
    cube_padding = 60.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # The goal cube is defined as robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id
        if robot.world.light_cubes[cozmo.objects.LightCube1Id].object_id == obj.object_id:

            # Calculate the approach position of the object
            local_goal_pos = Node((0, -cozmo_padding))
            goal_pos = get_global_node(object_angle, object_pos, local_goal_pos)

            # Check whether this goal location is valid
            if cmap.is_inside_obstacles(goal_pos) or (not cmap.is_inbound(goal_pos)):
                print("The goal position is not valid. Please remove the goal cube and place in another position.")
            else:
                cmap.clear_goals()
                cmap.add_goal(goal_pos)
                goal_center = object_pos

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center


class RobotThread(threading.Thread):
    """Thread to run cozmo code separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(CozmoPlanning,use_3d_viewer=False, use_viewer=False)
        stopevent.set()


class RRTThread(threading.Thread):
    """Thread to run RRT separate from main thread
    """

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        while not stopevent.is_set():
            RRT(cmap, cmap.get_start())
            time.sleep(100)
            cmap.reset()
        stopevent.set()


if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    robotFlag = False
    for i in range(0,len(sys.argv)):
        if (sys.argv[i] == "-robot"):
            robotFlag = True
    if (robotFlag):
        cmap = CozMap("maps/emptygrid.json", node_generator)
        robot_thread = RobotThread()
        robot_thread.start()
    else:
        cmap = CozMap("maps/map2.json", node_generator)
        sim = RRTThread()
        sim.start()
    visualizer = Visualizer(cmap)
    visualizer.start()
    stopevent.set()