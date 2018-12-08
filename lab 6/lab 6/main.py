
try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
import threading
import time
import sys
from PIL import Image
import PIL
import random
from markers import detect, annotator
from grid import *
from gui import *
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from imgclassification import ImageClassifier
from cmap import *
from mapGui import *
from pf_gui import *
from CoordinateTransformer import*
from Localize import *


MAX_NODES = 20000

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)



IDLE = "none"
DRONE = "drone"
PLANE = "plane"
INSPECTION = "inspection"
PLACE = "place"

# tmp cache
lastPose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))


# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)


# map
grid = CozGrid("map_arena.json")
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

# Constant
PI = 3.14159
POSITION_TOL = 1.5
HEADING_TOL = 12.0

##sign = lambda x: (1, -1)[x < 0]

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global lastPose
    last_x, last_y, last_h = lastPose.position.x, lastPose.position.y, \
        lastPose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))




async def marker_processing(robot, camera_settings, show_diagnostic_image=False, return_unwarped_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera.
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh)
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)

    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = PIL.Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image



    if return_unwarped_image == True:
      unwarped_image_list = [marker['unwarped_image'] for marker in markers]
      return marker_list, annotated_image, unwarped_image_list

    return marker_list, annotated_image


#add other methods
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

def get_current_pose_on_cmap(robot, star_p):
    start_x = 6 * 25.4
    start_y = 10 * 25.4
    current_x = robot.pose.position.x
    current_y = robot.pose.position.y

    return Node((current_x + start_x, current_y + start_y))

async def run(robot: cozmo.robot.Robot):

    global lastPose
    global grid, gui, pf
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(5)).wait_for_completed()
    await robot.set_lift_height(0.0).wait_for_completed()
    focalX, focalY = robot.camera.config.focal_length.x_y
    centerX, centerY = robot.camera.config.center.x_y
    camera_settings = np.array([
        [focalX,  0, centerX],
        [ 0, focalY, centerY],
        [ 0,  0,  1]
    ], dtype=np.float)
    markerLocation = {
        'drone': Localize(),
        'inspection': Localize(),
        'order': Localize(),
        'plane': Localize(),
        'truck': Localize(),
        'hands': Localize(),
        'place': Localize()
    }
    cube_target = {
        'A': 'drone',
        'B': 'plane',
        'C': 'inspection',
        'D': 'place'
    }

    img_clf = ImageClassifier()
    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)

    # train model
    img_clf.train_classifier(train_data, train_labels)

    ###################

    converged = False
    convergeScore = 0
    global cmap, stopevent
    while True:
        if not converged:
            await robot.drive_wheels(10.0, -10,0)
            currPose = robot.pose
            odom = compute_odometry(currPose, cvt_inch=True)
            lastPose = currPose
            marker_list, camera_image, unwarped_image_list = await marker_processing(robot, camera_settings, show_diagnostic_image=True, return_unwarped_image=True)
            if len(unwarped_image_list) > 0:
                unwarped = unwarped_image_list[0]
                relativePosition = marker_list[0]
                cm = matplotlib.cm.ScalarMappable()
                cm.set_cmap("viridis")
                img = cm.to_rgba(unwarped)
                img = (np.around(img*255))
                imageArray = np.array([img], dtype=np.int16)
                featureArray = imgClf.extract_image_features(imageArray)
                labels = imgClf.predict_labels(featureArray)
                print(labels)
                robotPose = (robot.pose.position.x, robot.pose.position.y, robot.pose.rotation.angle_z.radians)
                relativeToRobot = [i * INCHE_IN_MILLIMETERS for i in relativePosition[:2]]
                relativeToOrigin = (
                    robotPose[0] + relativeToRobot[0] * math.cos(robotPose[2]) - relativeToRobot[1] * math.sin(robotPose[2]), robotPose[1] + relativeToRobot[0] * math.sin(robotPose[2]) + relativeToRobot[1] * math.cos(robotPose[2]),
                )
                markerLocation[labels[0]].push(relativeToOrigin)
            (m_x, m_y, m_h, m_confident) = pf.update(odom, marker_list)
            if m_confident:
                convergeScore += 1
            else:
                convergeScore = 0
            if convergeScore > 10:
                converged = True
                originPose = (robot.pose.position.x, robot.pose.position.y, robot.pose.rotation.angle_z.degrees)
                mapPose = (m_x, m_y, m_h)
                ct = CoordinateTransformer(originPose, mapPose)
                startPose = ct.origin_to_map(originPose)
                obstaclePose = (WEIGHT / 2 -0.5, HEIGHT / 2 - 0.5, 0)
                obstaclePoseOrigin = ct.map_to_origin(obstaclePose)
                fixed_object = await robot.world.create_custom_fixed_object(cozmo.util.Pose(obstaclePoseOrigin[0], obstaclePoseOrigin[1], 0, angle_z=degrees(obstaclePoseOrigin[2])), INCHE_IN_MILLIMETERS*3, INCHE_IN_MILLIMETERS*3, INCHE_IN_MILLIMETERS*3,)
                cubes = await robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=30, include_existing = True)
                for i in cubes:
                    print(i)
        else:
            await robot.drive_wheels(0, 0)
            cmap.set_start(get_current_pose_on_cmap(robot, startPose))
            obstacleNode = []
            obstacleNode.append(Node((int((mapWidth / 2 - .5) * 25.4), int((mapHeight  / 2 - .5) * 25.4))))
            obstacleNode.append(Node((int((mapWidth / 2 - .5) * 25.4), int((mapHeight  / 2 + 0.5) * 25.4))))
            obstacleNode.append(Node((int((mapWidth / 2 + .5) * 25.4), int((mapHeight  / 2 - .5) * 25.4))))
            obstacleNode.append(Node((int((mapWidth / 2 + .5) * 25.4), int((mapHeight  / 2 + 0.5) * 25.4))))
            cmap.add_obstacle(obstacleNode)
            print(len(cubes))
            for i, cube in enumerate(cubes):
                mapPose = ct.origin_to_map((cube.pose.position.x, cube.pose.position.y, cube.pose.rotation.angle_z.degrees))
                print('Map pose:', mapPose)
                cube_label = recognize_cube((mapPose[0], mapPose[1]))
                print('Type of cube: ', cube_label)
                print("Picking up.")
                mapPoseCopy0 = mapPose[0];
                mapPoseCopy1 = mapPose[1];
                if mapPose[0] < 0:
                    mapPoseCopy0 = 2;
                if mapPose[1] < 0:
                    mapPoseCopy1 = 2;
                cmap.add_goal(Node((mapPoseCopy0*25.4 , mapPoseCopy1*25.4)))
                RRT(cmap, cmap.get_start())
                if cmap.is_solved():
                    path = cmap.get_smooth_path()
                cubePose = (cube.pose.position.x, cube.pose.position.y)
                startAngle = math.atan2(
                                    cubePose[1],
                                    cubePose[0]
                                ) / PI * 180.0
                await robot.go_to_pose(cozmo.util.Pose(0, 0, 0, angle_z=degrees(startAngle)))\
                           .wait_for_completed()
                await robot.pickup_object(cube, num_retries=100).wait_for_completed()
                # cmap.reset()
                # cmap.clear_goals()
                markerPose = markerLocation[cube_target[cube_label]].get()
                sq_avg = (markerPose[0] ** 2 + markerPose[1] ** 2) ** (1/2)
                markerPose[0] = markerPose[0] - (markerPose[0]/sq_avg) * 85
                markerPose[1] = markerPose[1] - (markerPose[1]/sq_avg) * 85
                print("Marker pose: ", markerPose)
                endAngle = math.atan2(
                                markerPose[1],
                                markerPose[0]
                            ) / PI * 180.0
                await robot.go_to_pose(cozmo.util.Pose(markerPose[0], markerPose[1], 0, angle_z=degrees(endAngle)))\
                           .wait_for_completed()
                print("Dropping the cube.")
                await robot.place_object_on_ground_here(cube).wait_for_completed()
                cmap.reset()
                cmap.clear_goals()
                cmap.set_start(Node((markerPose[0]*25.4 , markerPose[1]*25.4)))
            break


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


class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        # Please refrain from enabling use_viewer since it uses tk, which must be in main thread
        cozmo.run_program(run,use_3d_viewer=False, use_viewer=False)
        stopevent.set()

if __name__ == '__main__':
    global cmap
    global stopevent
    stopevent = threading.Event()

    robot_thread = CozmoThread()
    robot_thread.start()

    cmap = CozMap("maps/emptygrid.json", node_generator)

    visualizer = Visualizer(cmap)
    visualizer.start()

    gui.show_particles(delivery_robot.pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()

    print('Visualizer set')
    stopevent.set()


