#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D


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

        # Publisher to cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

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

        # Timer to periodically publish velocity commands
        timer_period = 0.1  # [s] -> 10 Hz
        self.timer = self.create_timer(timer_period, self.control_loop_callback)

        self.get_logger().info("Waypoint Follower node started. Waiting for waypoints...")

    def waypoint_callback(self, msg: Pose2D):
        """
        Updates the target waypoint when a new message is received.
        """
        self.x_target = msg.x
        self.y_target = msg.y
        self.orientation_target = msg.theta
        self.get_logger().info(
            f"Received new waypoint: x={self.x_target}, y={self.y_target}, orientation={self.orientation_target}"
        )

    def odom_callback(self, msg: Odometry):
        """
        Extracts and stores the robot's current pose from the odometry.
        """
        self.is_odom_received = True
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw (theta)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_orientation = math.atan2(siny_cosp, cosy_cosp)
        self.get_logger().info(
            f"current state: x={self.current_x}, y={self.current_y}, orientation={self.current_orientation}"
        )

        # keep stop when there is no waypoint published
        if self.x_target is None:
            self.x_target = self.current_x
            self.y_target = self.current_y
            self.orientation_target = self.current_orientation

    def control_loop_callback(self):
        """
        Periodic control loop callback. Computes PD control and publishes velocity commands.
        """
        if not self.is_odom_received:
            return
        # 1) Compute x,y,theta errors
        error_x = self.x_target - self.current_x
        error_y = self.y_target - self.current_y
        # TBD: theta error?
        error_theta = 0.0

        # 2) Compute derivative of x,y,theta errors
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y
        # TBD: theta error derivative?

        # 3) PD control for linear velocities (x, y)
        vx = self.Kp_linear * error_x + self.Kd_linear * derivative_x
        vy = self.Kp_linear * error_y + self.Kd_linear * derivative_y

        # 4) PD control for angular velocity
        # TBD: PD for ang vel?
        vtheta = 0

        # 5) Update previous error terms
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        # TBD: error update for theta?

        # 6) Publish velocity commands
        twist_msg = Twist()

        if abs(error_theta)>0.1 and math.hypot(error_x, error_y)>0.3:
            twist_msg.angular.z = min(vtheta, self.max_velo)
            self.get_logger().info( f"rotate before moving forward" )
        elif math.hypot(error_x, error_y)>0.05:
            twist_msg.linear.x = min(math.hypot(vx,vy), self.max_velo)
            twist_msg.angular.z = min(vtheta, self.max_velo)
            self.get_logger().info( f"moving forward" )                      
        else:
            # arrive
            pass

        self.cmd_vel_pub.publish(twist_msg)

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
