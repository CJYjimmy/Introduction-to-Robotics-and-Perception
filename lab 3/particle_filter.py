# author: Jiayuan Chen, Michael Wang

from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np

# author: Jiayuan Chen, Michael Wang

def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []

    for particle in particles:
        dx, dy, dh = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        dx, dy = rotate_point(dx, dy, particle.h)
        particle.x += dx
        particle.y += dy
        particle.h += dh
        motion_particles.append(particle)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    remove_count = 0
    total = 1
    w = []

    if len(measured_marker_list) == 0:
        # if the measured_marker_list is empty, we can just make all particles's weight evenly
        for p in particles:
            w.append((p, 1/len(particles)))
    else:
        for p in particles:
            if not grid.is_in(p.x, p.y):
                w.append((p, 0))
            elif not grid.is_free(p.x, p.y):
                w.append((p, 0))
            else:
                # to match each closest markers btw the measured marker list and particle viewed marker list
                pairs = []
                p_see_markers = p.read_markers(grid)
                diff_btw_p_m = math.fabs(len(measured_marker_list) - len(p_see_markers))
                for m_marker in measured_marker_list:
                    if len(p_see_markers) != 0:
                        m_marker_x, m_marker_y, m_marker_h = add_marker_measurement_noise(m_marker, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)

                        closest_m = p_see_markers[0]
                        closest_dist = grid_distance(m_marker_x, m_marker_y, closest_m[0], closest_m[1])

                        for p_m in p_see_markers:
                            p_m_dist = grid_distance(m_marker_x, m_marker_y, p_m[0], p_m[1])
                            if closest_dist > p_m_dist:
                                closest_dist = p_m_dist
                                closest_m = p_m
                        pairs.append((closest_m, m_marker))
                        p_see_markers.remove(closest_m)
                    else:
                        break

                # calculate the weight according to the formula: P(x) = e ^ (- ((distance between markers)^2 / (2 sigma^2) + (angle difference between markers)^2 / (2 sigma^2)))
                prob = 1
                longest_dist_btw_markers = 0;
                for p_marker, m_marker in pairs:
                    p_m_diff_dist = grid_distance(p_marker[0], p_marker[1], m_marker[0], m_marker[1])
                    p_m_diff_angle = diff_heading_deg(p_marker[2], m_marker[2])
                    prob *= np.exp(-(p_m_diff_dist ** 2 / (2 * MARKER_TRANS_SIGMA ** 2) + p_m_diff_angle ** 2 / (2 * MARKER_ROT_SIGMA ** 2)))
                    if p_m_diff_dist > longest_dist_btw_markers:
                        longest_dist_btw_markers = p_m_diff_dist
                # to counter the effect of the different number of marker between two markers list
                # we can assume that the markers without forming pair is out of the other view field, and the senser only can detect 45 degree field, so we assume
                # the different angle btw them is 45 degrees and the different distance is the longest distance btw existing marker pairs.
                for i in range(int(diff_btw_p_m)):
                    prob *= np.exp(-((longest_dist_btw_markers ** 2) / (2 * (MARKER_TRANS_SIGMA ** 2)) + (45 ** 2) / (2 * (MARKER_ROT_SIGMA ** 2))))
                w.append((p, prob))
        # to sort the weight, and to remove some low weight particles
        w = sorted(w, key = lambda weight : weight[1])
        total = 0
        remove = int(PARTICLE_COUNT / 150)
        w = w[remove:]
        for i, j in w:
            if j == 0:
                remove_count += 1
            else:
                total += j
        w = w[remove_count:]
        remove_count += remove
    # to normalize the weight and to get the each particle's possibility
    particles_List = []
    new_w = []
    for i, j in w:
        particles_List.append(Particle(i.x, i.y, i.h))
        new_w.append(j / total)

    # to add some random particle into the measured_particles which the amount is the amount of particles we removing before
    random_particles = Particle.create_random(remove_count, grid)
    for random_p in random_particles:
        measured_particles.append(random_p)
    # to resample the particles_List and put into the measured_particles
    new_p_list = []
    if particles_List != []:
        new_p_list = np.random.choice(particles_List, size = len(particles_List), replace = True, p = new_w)
    for p in new_p_list:
        p_x, p_y, p_h = add_odometry_noise([p.x, p.y, p.h], ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        measured_particles.append(Particle(p_x, p_y, p_h))

    return measured_particles


