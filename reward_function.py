# -*- coding: utf-8 -*-

import math
# import traceback

"""
This is the source code you cut and paste into AWS console. It consists of RewardEvaluator class that is instantiated
by the code of the desired reward_function(). The  RewardEvaluator contains a set of elementary  "low level" functions 
 for example the distance calculation between waypoints, directions as well as higher-level functions (e.g. nearest turn 
direction and distance) allowing you to design more complex reward logic.
"""


class RewardEvaluator:

    # CALCULATION CONSTANTS - change for the performance fine tuning

    # Define minimum and maximum expected speed interval for the training. Both values should be corresponding to
    # parameters you are going to use for the Action space. Set MAX_SPEED equal to maximum speed defined there,
    # MIN_SPEED should be lower (just a bit) then expected minimum defined speed (e.g. Max speed set to 5 m/s,
    # speed granularity 3 => therefore, MIN_SPEED should be less than 1.66 m/s.
    MAX_SPEED = float(2.0)
    MIN_SPEED = float(0.6)

    # Define maximum steering angle according to the Action space settings. Smooth steering angle threshold is used to
    # set a steering angle still considered as "smooth". The value must be higher than minimum steering angle determined
    # by the steering Action space. E.g Max steering 30 degrees, granularity 3 => SMOOTH_STEERING_ANGLE_TRESHOLD should
    # be higher than 10 degrees.
    MAX_STEERING_ANGLE = 30
    SMOOTH_STEERING_ANGLE_TRESHOLD = 10  # Greater than minimum angle defined in action space

    # Constant value used to "ignore" turns in the corresponding distance (in meters). The car is supposed to drive
    # at MAX_SPEED (getting a higher reward). In case within the distance is a turn, the car is rewarded when slowing
    # down.
    SAFE_HORIZON_DISTANCE = 0.2  # meters, able to fully stop. See ANGLE_IS_CURVE.

    # Constant to define accepted distance of the car from the center line.
    CENTERLINE_FOLLOW_RATIO_TRESHOLD = 0.24

    # Constant to define a threshold (in degrees), representing max. angle within SAFE_HORIZON_DISTANCE. If the car is
    # supposed to start steering and the angle of the farthest waypoint is above the threshold, the car is supposed to
    # slow down
    ANGLE_IS_CURVE = 3

    # A range the reward value must fit in.
    PENALTY_MAX = 0.001
    REWARD_MAX = 89999  # 100000

    # params is a set of input values provided by the DeepRacer environment. For each calculation
    # this is provided
    params = None

    # Class properties - status values extracted from "params" input
    # reference: https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html

    all_wheels_on_track = None # Boolean,        # flag to indicate if the agent is on the track
    x = None # float,                            # agent's x-coordinate in meters
    y = None # float,                            # agent's y-coordinate in meters
    closest_objects = None # [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
    closest_waypoints = None # [int, int],       # indices of the two nearest waypoints.
    distance_from_center = None # float,         # distance in meters from the track center 
    is_crashed = None # Boolean,                 # Boolean flag to indicate whether the agent has crashed.
    is_left_of_center = None # Boolean,          # Flag to indicate if the agent is on the left side to the track center or not. 
    is_offtrack = None # Boolean,                # Boolean flag to indicate whether the agent has gone off track.
    is_reversed = None # Boolean,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    heading = None # float,                      # agent's yaw in degrees
    objects_distance = None # [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
    objects_heading = None # [float, ],          # list of the objects' headings in degrees between -180 and 180.
    objects_left_of_center = None # [Boolean, ], # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
    objects_location = None # [(float, float),], # list of object locations [(x,y), ...].
    objects_speed = None # [float, ],            # list of the objects' speeds in meters per second.
    progress = None # float,                     # percentage of track completed
    speed = None # float,                        # agent's speed in meters per second (m/s)
    steering_angle = None # float,               # agent's steering angle in degrees
    steps = None # int,                          # number steps completed
    track_length = None # float,                 # track length in meters.
    track_width = None # float,                  # width of the track
    waypoints = None # [(float, float), ]        # list of (x,y) as milestones along the track center

    # aditional parameters

    next_object_index = None
    nearest_previous_waypoint_ind = None
    nearest_next_waypoint_ind = None

    log_message = ""

    # method used to extract class properties (status values) from input "params"    
    def init_self(self, params):
        
        for key, value in params.items():
            setattr(self, key, value)

        _, self.next_object_index = params.get('closest_objects',[0,0])
        self.nearest_previous_waypoint_ind = params.get('closest_waypoints',[0,0])[0]
        self.nearest_next_waypoint_ind = params.get('closest_waypoints',[0,0])[1]

    # RewardEvaluator Class constructor
    def __init__(self, params):
        self.params = params
        self.init_self(params)

    # Method used to "print" status values and logged messages into AWS log. Be aware of additional cost Amazon will
    # charge you when logging is used heavily!!!
    def status_to_string(self):
        status = self.params
        if 'waypoints' in status: del status['waypoints']
        status['debug_log'] = self.log_message
        print(status)

    # Gets ind'th waypoint from the list of all waypoints retrieved in params['waypoints']. Waypoints are circuit track
    # specific (every time params is provided it is same list for particular circuit). If index is out of range (greater
    # than len(params['waypoints']) a waypoint from the beginning of the list ir returned.
    def get_way_point(self, index_way_point):
        if index_way_point > (len(self.waypoints) - 1):
            return self.waypoints[index_way_point - (len(self.waypoints))]
        elif index_way_point < 0:
            return self.waypoints[len(self.waypoints) + index_way_point]
        else:
            return self.waypoints[index_way_point]

    # Calculates distance [m] between two waypoints [x1,y1] and [x2,y2]
    @staticmethod
    def get_way_points_distance(previous_waypoint, next_waypoint):
        return math.sqrt(pow(next_waypoint[1] - previous_waypoint[1], 2) + pow(next_waypoint[0] - previous_waypoint[0], 2))

    # Calculates heading direction between two waypoints - angle in cartesian layout. Clockwise values
    # 0 to -180 degrees, anti clockwise 0 to +180 degrees
    @staticmethod
    def get_heading_between_waypoints(previous_waypoint, next_waypoint):
        track_direction = math.atan2(next_waypoint[1] - previous_waypoint[1], next_waypoint[0] - previous_waypoint[0])
        return math.degrees(track_direction)

    # Calculates the misalignment of the heading of the car () compared to center line of the track (defined by previous and
    # the next waypoint (the car is between them)
    def get_car_heading_error(self):  # track direction vs heading
        next_point = self.get_way_point(self.closest_waypoints[1])
        prev_point = self.get_way_point(self.closest_waypoints[0])
        track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        track_direction = math.degrees(track_direction)
        return track_direction - self.heading

    # Based on CarHeadingError (how much the car is misaligned with th direction of the track) and based on the "safe
    # horizon distance it is indicating the current speed (params['speed']) is/not optimal.
    def get_optimum_speed_ratio(self):
        if abs(self.get_car_heading_error()) >= self.MAX_STEERING_ANGLE:
            return float(0.34)
        if abs(self.get_car_heading_error()) >= (self.MAX_STEERING_ANGLE * 0.75):
            return float(0.67)
        current_position_xy = (self.x, self.y)
        current_wp_index = self.closest_waypoints[1]
        length = self.get_way_points_distance((self.x, self.y), self.get_way_point(current_wp_index))
        current_track_heading = self.get_heading_between_waypoints(self.get_way_point(current_wp_index),
                                                                   self.get_way_point(current_wp_index + 1))
        while True:
            from_point = self.get_way_point(current_wp_index)
            to_point = self.get_way_point(current_wp_index + 1)
            length = length + self.get_way_points_distance(from_point, to_point)
            if length >= self.SAFE_HORIZON_DISTANCE:
                heading_to_horizont_point = self.get_heading_between_waypoints(self.get_way_point(self.closest_waypoints[1]), to_point)
                if abs(current_track_heading - heading_to_horizont_point) > (self.MAX_STEERING_ANGLE * 0.5):
                    return float(0.33)
                elif abs(current_track_heading - heading_to_horizont_point) > (self.MAX_STEERING_ANGLE * 0.25):
                    return float(0.66)
                else:
                    return float(1.0)
            current_wp_index = current_wp_index + 1

    # Calculates angle of the turn the car is right now (degrees). It is angle between previous and next segment of the
    # track (previous_waypoint - closest_waypoint and closest_waypoint - next_waypoint)
    def get_turn_angle(self):
        current_waypoint = self.closest_waypoints[0]
        angle_ahead = self.get_heading_between_waypoints(self.get_way_point(current_waypoint),
                                                         self.get_way_point(current_waypoint + 1))
        angle_behind = self.get_heading_between_waypoints(self.get_way_point(current_waypoint - 1),
                                                          self.get_way_point(current_waypoint))
        result = angle_ahead - angle_behind
        if angle_ahead < -90 and angle_behind > 90:
            return 360 + result
        elif result > 180:
            return -180 + (result - 180)
        elif result < -180:
            return 180 - (result + 180)
        else:
            return result

    # Indicates the car is in turn
    def is_in_turn(self):
        if abs(self.get_turn_angle()) >= self.ANGLE_IS_CURVE:
            return True
        else:
            return False
        return False

    # Indicates the car has reached final waypoint of the circuit track
    def reached_target(self):
        max_waypoint_index = len(self.waypoints) - 1
        if self.closest_waypoints[1] == max_waypoint_index:
            return True
        else:
            return False
    
    def get_object_distances_reward(self):
        # Distance to the next object
        distance_closest_object = self.objects_distance[self.next_object_index]
        # Decide if the agent and the next object is on the same lane
        is_same_lane = self.objects_left_of_center[self.next_object_index] == self.is_left_of_center

        reward_avoid = 1.0

        if is_same_lane:
            if 0.5 <= distance_closest_object < 0.8: 
                reward_avoid *= 0.5
            elif 0.3 <= distance_closest_object < 0.5:
                reward_avoid *= 0.2
            elif distance_closest_object < 0.3:
                reward_avoid = 1e-3 # Likely crashed

        return reward_avoid

    # Provides direction of the next turn in order to let you reward right position to the center line (before the left
    # turn position of the car sligthly right can be rewarded (and vice versa) - see is_in_optimized_corridor()
    def get_expected_turn_direction(self):
        current_waypoint_index = self.closest_waypoints[1]
        length = self.get_way_points_distance((self.x, self.y), self.get_way_point(current_waypoint_index))
        while True:
            from_point = self.get_way_point(current_waypoint_index)
            to_point = self.get_way_point(current_waypoint_index + 1)
            length = length + self.get_way_points_distance(from_point, to_point)
            if length >= self.SAFE_HORIZON_DISTANCE * 4.5:
                result = self.get_heading_between_waypoints(self.get_way_point(self.closest_waypoints[1]), to_point)
                if result > 2:
                    return "LEFT"
                elif result < -2:
                    return "RIGHT"
                else:
                    return "STRAIGHT"
            current_waypoint_index = current_waypoint_index + 1

    # Based on the direction of the next turn it indicates the car is on the right side to the center line in order to
    # drive through smoothly - see get_expected_turn_direction().
    def is_in_optimized_corridor(self):
        if self.is_in_turn():
            turn_angle = self.get_turn_angle()
            if turn_angle > 0:  # Turning LEFT - better be by left side
                if (self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width) or
                        self.is_left_of_center == False and self.distance_from_center <= (
                                self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width)):
                    return True
                else:
                    return False
            else:  # Turning RIGHT - better be by right side
                if self.is_left_of_center == True and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False
        else:
            next_turn = self.get_expected_turn_direction()
            if next_turn == "LEFT":  # Be more righ side before turn
                if self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False
            elif next_turn == "RIGHT":  # Be more left side before turn:
                if self.is_left_of_center == True and self.distance_from_center <= (
                        self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width) or self.is_left_of_center == False and self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD / 2 * self.track_width):
                    return True
                else:
                    return False
            else:  # Be aligned with center line:
                if self.distance_from_center <= (self.CENTERLINE_FOLLOW_RATIO_TRESHOLD * 2 * self.track_width):
                    return True
                else:
                    return False

    def is_optimum_speed(self):
        if abs(self.speed - (self.get_optimum_speed_ratio() * self.MAX_SPEED)) < (self.MAX_SPEED * 0.15) and self.MIN_SPEED <= self.speed <= self.MAX_SPEED:
            return True
        else:
            return False

    # Accumulates all logging messages into one string which you may need to write to the log (uncomment line
    # self.status_to_string() in evaluate() if you want to log status and calculation outputs.
    def log_feature(self, message):
        if message is None:
            message = 'NULL'
        self.log_message = self.log_message + str(message) + '|'

    # Here you can implement your logic to calculate reward value based on input parameters (params) and use
    # implemented features (as methods above)
    def evaluate(self):
        self.init_self(self.params)
        result_reward = float(0.001)
        try:
            # No reward => Fatal behaviour, NOREWARD!  (out of track, reversed, sleeping)
            if self.all_wheels_on_track == False or self.is_reversed == True or (self.speed < (0.1 * self.MAX_SPEED)):
                self.log_feature("all_wheels_on_track or is_reversed issue")
                self.status_to_string()
                return float(self.PENALTY_MAX)

            # REWARD 50 - EARLY Basic learning => easy factors accelerate learning
            # Right heading, no crazy steering
            if abs(self.get_car_heading_error()) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("getCarHeadingOK")
                result_reward = result_reward + self.REWARD_MAX * 0.3

            if abs(self.steering_angle) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("getSteeringAngleOK")
                result_reward = result_reward + self.REWARD_MAX * 0.15

            # REWARD100 - LATER ADVANCED complex learning
            # Ideal path, speed wherever possible, carefully in corners
            if self.is_in_optimized_corridor():
                self.log_feature("is_in_optimized_corridor")
                result_reward = result_reward + float(self.REWARD_MAX * 0.45)

            if not (self.is_in_turn()) and (abs(self.speed - self.MAX_SPEED) < (0.1 * self.MAX_SPEED)) \
                    and abs(self.get_car_heading_error()) <= self.SMOOTH_STEERING_ANGLE_TRESHOLD:
                self.log_feature("isStraightOnMaxSpeed")
                result_reward = result_reward + float(self.REWARD_MAX * 1)

            if self.is_in_turn() and self.is_optimum_speed():
                self.log_feature("isOptimumSpeedinCurve")
                result_reward = result_reward + float(self.REWARD_MAX * 0.6)

            # REWAR - Progress bonus
            TOTAL_NUM_STEPS = 150
            if (self.steps % 100 == 0) and self.progress > (self.steps / TOTAL_NUM_STEPS):
                self.log_feature("progressingOk")
                result_reward = result_reward + self.REWARD_MAX * 0.4

            # Reach Max Waypoint - get extra reward
            if self.reached_target():
                self.log_feature("reached_target")
                result_reward = float(self.REWARD_MAX)

        except Exception as e:
            print("Error : " + str(e))
            # print(traceback.format_exc())


        result_reward = result_reward * self.get_object_distances_reward()

        # Finally - check reward value does not exceed maximum value
        if result_reward > 900000:
            result_reward = 900000

        self.log_feature(result_reward)
        # self.status_to_string()

        return float(result_reward)


"""
This is the core function called by the environment to calculate reward value for every point of time of the training. 
params: input values for the reward calculation (see above)

Usually, this function contains all reward calculations a logic implemented. Instead, this code example is instantiating 
RewardEvaluator which has implemented a set of features one can easily combine and use.
"""


def reward_function(params):
    re = RewardEvaluator(params)
    return float(re.evaluate())
