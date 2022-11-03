import rospy
import GraphMatcher


class GraphMatcherNode():
    def __init__(self):
        self.graph_matcher = GraphMatcher()
        self.defineROSInterface()
        rospy.init_node('graph_matcher_node')

    def defineROSInterface(self):
        rospy.Subscriber("bim_sgraph_pub", SgraphMsg, self.BIMSgraphCallback)
        rospy.Subscriber("real_sgraph_pub", SgraphMsg, self.RealSgraphCallback)
        self.match_pub = rospy.Publisher('match_pub', MatchMsg, queue_size=10)
        find_first_pose_server = rospy.Service('find_first_pose', FindFirstPoseRequest, self.findFirstPoseCallback)

    def findFirstPoseServiceCallback(self, request):
        response = responseMsg()
        response.success, response.matches = self.graph_matcher.findFirstPose()
        return response

    def BIMSgraphCallback(self):
        pass

    def RealSgraphCallback(self):
        pass

    


if __name__ == '__main__':
    try:
        GraphMatcherNode()
    except rospy.ROSInterruptException:
        pass