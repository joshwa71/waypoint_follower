#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from collections import deque


class WaypointFollower(Node):
    """
    A ROS2 node that uses a PD controller to follow a dynamically received
    waypoint (x, y, theta). Subscribes to /waypoint for the target and
    /odom for the robot's current state, and publishes velocity commands
    to /cmd_vel.
    """

    def __init__(self):
        super().__init__('waypoint_follower')

        # Target waypoint (x, y, theta)
        # Initially, set them to 0. They will be updated by /waypoint subscriber.
        self.x_target = None
        self.y_target = None
        self.orientation_target = None

        self.max_velo = 0.5

        # PD Controller Gains (tune as necessary)
        self.Kp_linear = 10.0
        self.Kd_linear = 0.1
        self.Kp_angular = 5.0
        self.Kd_angular = 0.1

        # Previous errors (for derivative term)
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_orientation = 0.0

        # Robot current state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0
        self.is_odom_received = False
        self.is_arrive_waypoint  = True

        # Collision detection parameters
        self.declare_parameter('min_height', 0.03)  # Minimum height to consider points (in meters)
        self.declare_parameter('max_height', 0.6)   # Maximum height to consider points (in meters)
        self.declare_parameter('collision_distance', 0.50)  # Distance threshold for collision detection (in meters)
        self.declare_parameter('collision_buffer_size', 10)  # Size of buffer for averaging collision detections
        self.declare_parameter('recovery_waypoints', 1)  # Number of waypoints to skip collision detection after a collision
        
        # Get parameters
        self.min_height = self.get_parameter('min_height').value
        self.max_height = self.get_parameter('max_height').value
        self.collision_distance = self.get_parameter('collision_distance').value
        self.collision_buffer_size = self.get_parameter('collision_buffer_size').value
        self.recovery_waypoints = self.get_parameter('recovery_waypoints').value
        
        # Collision state
        self.collision_detected = False
        self.collision_buffer = deque(maxlen=self.collision_buffer_size)
        self.collision_status_published = False
        self.recovery_mode = False
        self.recovery_count = 0

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_status_pub = self.create_publisher(String, '/path_planner/status', 10)

        # Subscriber to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/Odometry',
            self.odom_callback,
            10
        )

        # Subscriber to waypoint
        self.waypoint_sub = self.create_subscription(
            Pose2D,
            '/waypoint',
            self.waypoint_callback,
            10
        )
        
        # Subscriber to pointcloud for collision detection
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/Laser_map',
            self.pointcloud_callback,
            10
        )

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)

        self.get_logger().info("Waypoint Follower node started. Waiting for waypoints...")
        self.get_logger().info(f"Collision detection parameters: min_height={self.min_height}m, max_height={self.max_height}m, distance={self.collision_distance}m")
        self.get_logger().info(f"Recovery mode will disable collision detection for {self.recovery_waypoints} waypoints after a collision")

    def waypoint_callback(self, msg: Pose2D):
        """
        Updates the target waypoint when a new message is received.
        """
        if self.x_target==None or abs(self.x_target-msg.x) > 0.2 or abs(self.y_target-msg.y) > 0.2 or abs(self.orientation_target-msg.theta) > 0.2:
            self.is_arrive_waypoint = False

        self.x_target = msg.x
        self.y_target = msg.y
        self.orientation_target = msg.theta
        
        # Reset collision status when receiving a new waypoint
        self.collision_status_published = False
        
        # Log if a new waypoint is received while in recovery mode
        if self.recovery_mode:
            self.get_logger().info(f"Received new waypoint while in recovery mode. Remaining recovery waypoints: {self.recovery_count}")
                
        self.get_logger().info(
            f"Received new waypoint: x={self.x_target}, y={self.y_target}, orientation={self.orientation_target}. current state: x={self.current_x}, y={self.current_y}, orientation={self.current_orientation}"
        )

    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw (theta)
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)

        # keep stop when there is no waypoint published
        if self.x_target is None:
            self.x_target = self.current_x
            self.y_target = self.current_y
            self.orientation_target = self.current_orientation
            
    def pointcloud_callback(self, cloud_msg):
        """
        Process pointcloud data to detect potential collisions.
        """
        # Skip collision checking if in recovery mode
        if self.recovery_mode:
            return
            
        try:
            # Extract points from the pointcloud
            cloud_points = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
            
            if not cloud_points:
                return
                
            # Filter points by height to avoid ground or tall objects
            filtered_points = []
            for point in cloud_points:
                x, y, z = point
                if self.min_height <= z <= self.max_height:
                    filtered_points.append((x, y, z))
            
            if not filtered_points:
                return
                
            # Convert to numpy array for faster processing
            points_array = np.array(filtered_points)
            
            # Calculate distance from robot to each point in 2D (ignore z)
            robot_pos = np.array([self.current_x, self.current_y])
            points_2d = points_array[:, :2]  # Only x and y
            
            # Calculate Euclidean distances
            distances = np.sqrt(np.sum((points_2d - robot_pos) ** 2, axis=1))
            
            # Check if any point is within collision distance
            collision_count = np.sum(distances < self.collision_distance)
            
            # Add to buffer (1 for collision detected, 0 for no collision)
            self.collision_buffer.append(1 if collision_count > 0 else 0)
            
            # Check if majority of recent checks detect collision
            # Make detection more sensitive - require fewer points to trigger
            collision_threshold = max(1, len(self.collision_buffer) / 10)
            self.collision_detected = sum(self.collision_buffer) >= collision_threshold
            
            if self.collision_detected and not self.is_arrive_waypoint and not self.collision_status_published:
                self.get_logger().warn(f"Collision detected: {collision_count} points within {self.collision_distance}m")
                
        except Exception as e:
            self.get_logger().error(f"Error processing pointcloud: {str(e)}")

    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        if not self.is_odom_received:
            return
            
        # Check for collision and publish status if detected (skip if in recovery mode)
        if self.collision_detected and not self.is_arrive_waypoint and not self.collision_status_published and not self.recovery_mode:
            self.publish_status("COLLISION")
            self.collision_status_published = True
            self.is_arrive_waypoint = True  # Stop following the waypoint
            
            # Enable recovery mode for next waypoint
            self.recovery_mode = True
            self.recovery_count = self.recovery_waypoints
            self.get_logger().info(f"Entering recovery mode for {self.recovery_count} waypoints")
            
            # Stop the robot immediately
            twist_msg = Twist()
            self.cmd_vel_pub.publish(twist_msg)
            return
            
        # If we've already published the collision status for this waypoint, don't continue
        if self.collision_status_published and not self.recovery_mode:
            return
            
        # 1) Compute errors
        error_x = self.x_target - self.current_x
        error_y = self.y_target - self.current_y
        error_theta = self.normalize_angle(math.atan2(error_y, error_x)- self.current_orientation)
        error_orientation = self.normalize_angle(self.orientation_target - self.current_orientation)

        # 2) Compute derivative of errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        derivative_theta = error_theta - self.prev_error_theta
        derivative_orientation = error_orientation - self.prev_error_orientation

        # 3) PD control for linear velocities (x, y)
        vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y

        # 4) PD control for angular velocity
        vtheta = self.Kp_angular * error_theta + self.Kd_angular * derivative_theta
        vorientation = self.Kp_angular * error_orientation + self.Kd_angular * derivative_orientation

        # 5) Update previous error terms
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_error_theta = error_theta
        self.prev_error_orientation = error_orientation

        # 6) Publish velocity commands
        twist_msg = Twist()

        # Before arriving to the waypoint, decide whether to rotate in place or move forward: 
        # If the robot is not heading to the waypoint (with a margin 0.1 rad) and
        # the  waypoint is far away (more than 0.1 m), the robot needs to rotate in place
        # else, once the rotation is completed, enter the next phase, moving forward until
        # the distance between the robot and the waypoint is less than 0.1m
        if not self.is_arrive_waypoint:
            if abs(error_theta)>0.1 and math.hypot(error_x, error_y)>0.1:
                twist_msg.angular.z = min(vtheta, self.max_velo)
                self.get_logger().info( f"rotate before moving forward" )
            elif math.hypot(error_x, error_y)>0.1:
                twist_msg.linear.x = min(math.hypot(vx,vy), self.max_velo)
                twist_msg.angular.z = min(vtheta, self.max_velo)
                self.get_logger().info( f"moving forward" )                      
            else:
                self.is_arrive_waypoint = True
                # arrive the waypoint
                self.publish_status("WAYPOINT_REACHED")
                
                # Handle recovery mode exit *after* reaching the waypoint
                if self.recovery_mode:
                    self.recovery_count -= 1
                    if self.recovery_count <= 0:
                        self.recovery_mode = False
                        self.collision_detected = False
                        # Clear collision buffer
                        for _ in range(len(self.collision_buffer)):
                            self.collision_buffer.append(0)
                        self.get_logger().info("Recovery waypoint reached. Exiting recovery mode, re-enabling collision detection.")
                    else:
                        self.get_logger().info(f"Recovery waypoint reached. {self.recovery_count} recovery waypoints remaining.")
                        
        # After arriving to the waypoint, rotate in place to the target orientation: 
        else:
            if abs(error_orientation)>0.05:
                twist_msg.angular.z = min(vorientation, self.max_velo)
                self.get_logger().info( f"rotating to target orientation" )
            else:
                # arrive the target orientation
                self.publish_status("WAYPOINT_REACHED")

        self.cmd_vel_pub.publish(twist_msg)

    def publish_status(self, status):
        """Publish path planner status"""
        msg = String()
        msg.data = status
        self.path_status_pub.publish(msg)

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
