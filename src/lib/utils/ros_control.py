import rospy
from geometry_msgs.msg import Twist
from math import pi


class ForwardAndSideway():
    def __init__(self):
        rospy.init_node('forward_and_sideway', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        # this "forward_and_back" node will publish Twist type msgs to /cmd_vel
        # topic, where this node act like a Publisher
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        # parameters
        rate = 50
        self.r = rospy.Rate(rate)

        self.forwardSpeed = 0.0
        self.sidewaySpeed = 0.0

    def cmd_init(self):
        self.forwardSpeed = 0.0
        self.sidewaySpeed = 0.0


    def ros_pub(self, gaitType = 1, forwardSpeed = 0.0, sidewaySpeed = 0.0,
                      rotateSpeed = 0.0, speedLevel = 0, bodyHeight = 0.0, footRaiseHeight = 0.0):
        self.forwardSpeed = forwardSpeed
        self.sidewaySpeed = sidewaySpeed

        linear_speed = self.forwardSpeed
        # goal_distance = 5
        # linear_duration = goal_distance / linear_speed

        angular_speed = self.sidewaySpeed
        # goal_angular = pi
        # angular_duration = goal_angular / angular_speed

        # forward->rotate->back->rotate
        for i in range(1):
            # this is the msgs variant, has Twist type, no data now
            move_cmd = Twist()

            move_cmd.angular_speed.z = angular_speed
            # ticks = int(angular_duration * rate)
            # for t in range(ticks):
            self.cmd_vel.publish(move_cmd)
            self.r.sleep()

            # stop 1 ms before ratate
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)

            move_cmd.linear.x = linear_speed  #
            # should keep publishing
            # ticks = int(linear_duration * rate)
            # for t in range(ticks):
                # one node can publish msgs to different topics, here only publish
                # to /cmd_vel
            self.cmd_vel.publish(move_cmd)
            self.r.sleep  # sleep according to the rate

        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def shutdown(self):
        rospy.loginfo("Stopping the robot")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        ForwardAndSideway()
    except:
        rospy.loginfo("forward_and_back node terminated by exception")