#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, time, actionlib
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveAction, MoveGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

def clamp(x, a, b): return a if x < a else b if x > b else x

def width_from_js(msg):
    if not msg.position: return None
    if msg.name:
        idx = [i for i,n in enumerate(msg.name) if 'finger' in n]
        if len(idx) == 2: return abs(msg.position[idx[0]]) + abs(msg.position[idx[1]])
        if len(idx) == 1: return 2.0 * abs(msg.position[idx[0]])
    if len(msg.position) == 2: return abs(msg.position[0]) + abs(msg.position[1])
    return 2.0 * abs(msg.position[0])

class FollowerClient:
    def __init__(self, ns_prefix, prefer='move', wait_s=2.0, speed=0.3, wmin=0.0, wmax=0.08):
        if not ns_prefix.startswith('/'): ns_prefix = '/' + ns_prefix
        self.mode = None
        self.speed, self.wmin, self.wmax = float(speed), float(wmin), float(wmax)
        self.last_cmd = None
        self.last_send = 0.0
        self._move_ns = ns_prefix + '/franka_gripper/move'
        self._gcmd_ns = ns_prefix + '/franka_gripper/gripper_action'
        self._move = actionlib.SimpleActionClient(self._move_ns, MoveAction)
        self._gcmd = actionlib.SimpleActionClient(self._gcmd_ns, GripperCommandAction)
        chosen = None
        if prefer in ('move','auto') and self._move.wait_for_server(rospy.Duration(wait_s)):
            chosen = 'move'
        if chosen is None and prefer in ('gripper','auto') and self._gcmd.wait_for_server(rospy.Duration(wait_s)):
            chosen = 'gripper'
        if chosen is None:
            ok1 = self._move.wait_for_server(rospy.Duration(wait_s))
            ok2 = self._gcmd.wait_for_server(rospy.Duration(wait_s))
            rospy.logfatal('No gripper action under %s (move=%s, gripper_action=%s)', ns_prefix, ok1, ok2)
            raise RuntimeError('No follower action server')
        self.mode = chosen
        rospy.loginfo('Follower uses %s', self._move_ns if self.mode=='move' else self._gcmd_ns)

    def send_width(self, w):
        w = clamp(float(w), self.wmin, self.wmax)
        if self.last_cmd is not None and abs(w - self.last_cmd) < 1e-5:  # only on real change
            return
        self.last_cmd, self.last_send = w, time.time()
        if self.mode == 'move':
            self._move.send_goal(MoveGoal(width=w, speed=self.speed))
        else:
            g = GripperCommandGoal()
            g.command.position = w
            g.command.max_effort = 0.0  # let controller decide; set >0 if you want a force cap
            self._gcmd.send_goal(g)

class WidthMirrorFast:
    def __init__(self):
        # Params you must point to EXACT topics from `rostopic list`
        self.master_js_topic   = rospy.get_param('~master_js_topic',   '/fr3_right/franka_gripper/joint_states')
        self.follower_ns       = rospy.get_param('~follower_ns',       'fr3_left')
        self.prefer_action     = rospy.get_param('~prefer_action',     'move')  # 'move'|'gripper'|'auto'
        self.speed             = float(rospy.get_param('~move_speed',  0.3))    # fast like you had before
        self.wmin              = float(rospy.get_param('~min_width',   0.0))
        self.wmax              = float(rospy.get_param('~max_width',   0.08))
        self.min_send_gap      = float(rospy.get_param('~min_send_gap',0.0))    # 0 = send every change
        self.eps               = float(rospy.get_param('~eps',         0.0001)) # re-send threshold

        # State
        self.master_w = None
        self.last_sent = None
        self.last_time = 0.0

        # IO
        self.follower = FollowerClient(self.follower_ns, prefer=self.prefer_action,
                                       wait_s=2.0, speed=self.speed, wmin=self.wmin, wmax=self.wmax)

        rospy.Subscriber(self.master_js_topic, JointState, self._on_master_js, queue_size=500)
        rospy.loginfo('Mirroring %s  ->  %s', self.master_js_topic,
                      self.follower_ns + ('/franka_gripper/move' if self.follower.mode=='move' else '/franka_gripper/gripper_action'))

    def _on_master_js(self, msg):
        w = width_from_js(msg)
        if w is None: return
        w = clamp(w, self.wmin, self.wmax)
        self.master_w = w

        now = time.time()
        if self.min_send_gap > 0.0 and (now - self.last_time) < self.min_send_gap:
            return
        if self.last_sent is None or abs(w - self.last_sent) >= self.eps:
            self.follower.send_width(w)
            self.last_sent, self.last_time = w, now

def main():
    rospy.init_node('gripper_width_mirror_fast')
    WidthMirrorFast()
    rospy.spin()

if __name__ == '__main__':
    main()




# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import rospy, time, actionlib
# from sensor_msgs.msg import JointState
# from franka_gripper.msg import MoveAction, MoveGoal
# from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# def clamp(x, a, b): return a if x < a else b if x > b else x

# def width_from_js(msg):
#     if not msg.position: return None
#     if msg.name:
#         idx = [i for i,n in enumerate(msg.name) if 'finger' in n]
#         if len(idx) == 2: return abs(msg.position[idx[0]]) + abs(msg.position[idx[1]])
#         if len(idx) == 1: return 2.0 * abs(msg.position[idx[0]])
#     if len(msg.position) == 2: return abs(msg.position[0]) + abs(msg.position[1])
#     return 2.0 * abs(msg.position[0])

# class FollowerClient:
#     def __init__(self, ns_prefix, prefer='move', wait_s=2.0, speed=0.3, wmin=0.0, wmax=0.08):
#         if not ns_prefix.startswith('/'): ns_prefix = '/' + ns_prefix
#         self.mode = None
#         self.speed, self.wmin, self.wmax = float(speed), float(wmin), float(wmax)
#         self.last_cmd = None
#         self.last_send = 0.0
#         self._move_ns = ns_prefix + '/franka_gripper/move'
#         self._gcmd_ns = ns_prefix + '/franka_gripper/gripper_action'
#         self._move = actionlib.SimpleActionClient(self._move_ns, MoveAction)
#         self._gcmd = actionlib.SimpleActionClient(self._gcmd_ns, GripperCommandAction)
#         chosen = None
#         if prefer in ('move','auto') and self._move.wait_for_server(rospy.Duration(wait_s)):
#             chosen = 'move'
#         if chosen is None and prefer in ('gripper','auto') and self._gcmd.wait_for_server(rospy.Duration(wait_s)):
#             chosen = 'gripper'
#         if chosen is None:
#             ok1 = self._move.wait_for_server(rospy.Duration(wait_s))
#             ok2 = self._gcmd.wait_for_server(rospy.Duration(wait_s))
#             rospy.logfatal('No gripper action under %s (move=%s, gripper_action=%s)', ns_prefix, ok1, ok2)
#             raise RuntimeError('No follower action server')
#         self.mode = chosen
#         rospy.loginfo('Follower uses %s', self._move_ns if self.mode=='move' else self._gcmd_ns)

#     def send_width(self, w):
#         w = clamp(float(w), self.wmin, self.wmax)
#         if self.last_cmd is not None and abs(w - self.last_cmd) < 1e-5:  # only on real change
#             return
#         self.last_cmd, self.last_send = w, time.time()
#         if self.mode == 'move':
#             self._move.send_goal(MoveGoal(width=w, speed=self.speed))
#         else:
#             g = GripperCommandGoal()
#             g.command.position = w
#             g.command.max_effort = 0.0  # let controller decide; set >0 if you want a force cap
#             self._gcmd.send_goal(g)

# class WidthMirrorFast:
#     def __init__(self):
#         # Params you must point to EXACT topics from `rostopic list`
#         self.master_js_topic   = rospy.get_param('~master_js_topic',   '/fr3_left/franka_gripper/joint_states')
#         self.follower_ns       = rospy.get_param('~follower_ns',       'fr3_right')
#         self.prefer_action     = rospy.get_param('~prefer_action',     'move')  # 'move'|'gripper'|'auto'
#         self.speed             = float(rospy.get_param('~move_speed',  0.3))    # fast like you had before
#         self.wmin              = float(rospy.get_param('~min_width',   0.0))
#         self.wmax              = float(rospy.get_param('~max_width',   0.08))
#         self.min_send_gap      = float(rospy.get_param('~min_send_gap',0.0))    # 0 = send every change
#         self.eps               = float(rospy.get_param('~eps',         0.0001)) # re-send threshold

#         # State
#         self.master_w = None
#         self.last_sent = None
#         self.last_time = 0.0

#         # IO
#         self.follower = FollowerClient(self.follower_ns, prefer=self.prefer_action,
#                                        wait_s=2.0, speed=self.speed, wmin=self.wmin, wmax=self.wmax)

#         rospy.Subscriber(self.master_js_topic, JointState, self._on_master_js, queue_size=500)
#         rospy.loginfo('Mirroring %s  ->  %s', self.master_js_topic,
#                       self.follower_ns + ('/franka_gripper/move' if self.follower.mode=='move' else '/franka_gripper/gripper_action'))

#     def _on_master_js(self, msg):
#         w = width_from_js(msg)
#         if w is None: return
#         w = clamp(w, self.wmin, self.wmax)
#         self.master_w = w

#         now = time.time()
#         if self.min_send_gap > 0.0 and (now - self.last_time) < self.min_send_gap:
#             return
#         if self.last_sent is None or abs(w - self.last_sent) >= self.eps:
#             self.follower.send_width(w)
#             self.last_sent, self.last_time = w, now

# def main():
#     rospy.init_node('gripper_width_mirror_fast')
#     WidthMirrorFast()
#     rospy.spin()

# if __name__ == '__main__':
#     main()



# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import rospy, actionlib, time
# from sensor_msgs.msg import JointState
# from franka_gripper.msg import MoveAction, MoveGoal

# def clamp(x, a, b): return a if x < a else b if x > b else x

# def width_from_js(msg):
#     if msg.name:
#         idx = [i for i, n in enumerate(msg.name) if 'finger' in n]
#         if len(idx) == 2: return abs(msg.position[idx[0]]) + abs(msg.position[idx[1]])
#         if len(idx) == 1: return 2.0 * abs(msg.position[idx[0]])
#     if len(msg.position) == 2: return abs(msg.position[0]) + abs(msg.position[1])
#     if len(msg.position) >= 1: return 2.0 * abs(msg.position[0])
#     return None

# class Follower:
#     def __init__(self, move_action, speed, wmin, wmax):
#         self.client = actionlib.SimpleActionClient(move_action, MoveAction)
#         rospy.loginfo('Waiting for move action: %s', move_action)
#         self.client.wait_for_server()
#         self.speed, self.wmin, self.wmax = speed, wmin, wmax
#         self.last_cmd = None
#         self.last_send = 0.0

#     def send_width(self, w, min_gap_s=0.0):
#         now = time.time()
#         if min_gap_s and (now - self.last_send) < min_gap_s:
#             return
#         w = clamp(w, self.wmin, self.wmax)
#         if self.last_cmd is not None and abs(w - self.last_cmd) < 1e-5:
#             return
#         self.last_cmd, self.last_send = w, now
#         self.client.send_goal(MoveGoal(width=float(w), speed=float(self.speed)))

# class WidthMirror:
#     def __init__(self):
#         self.master_js_topic   = rospy.get_param('~master_js_topic',   '/fr3_left/franka_gripper/joint_states')
#         self.follower_move_act = rospy.get_param('~follower_move_act', '/fr3_right/franka_gripper/move')

#         self.min_w   = float(rospy.get_param('~min_width', 0.0))
#         self.max_w   = float(rospy.get_param('~max_width', 0.08))
#         self.speed   = float(rospy.get_param('~move_speed', 0.03))

#         self.rate_hz = float(rospy.get_param('~rate_hz', 100.0))   
#         self.min_gap = float(rospy.get_param('~min_send_gap', 0.0)) 
#         self.eps     = float(rospy.get_param('~eps', 1e-4))         
#         self.keepalive_s = float(rospy.get_param('~keepalive_s', 0.5))  

#         self.master_w = None
#         self.last_sent_display = 0.0
#         self.last_keepalive = 0.0

#         self.follower = Follower(self.follower_move_act, self.speed, self.min_w, self.max_w)

#         rospy.Subscriber(self.master_js_topic, JointState, self._master_cb, queue_size=200)
#         rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)

#         rospy.loginfo('Mirroring "%s"  ->  "%s"', self.master_js_topic, self.follower_move_act)

#     def _master_cb(self, msg):
#         w = width_from_js(msg)
#         if w is not None:
#             self.master_w = clamp(w, self.min_w, self.max_w)

#     def _tick(self, _):
#         if self.master_w is None:
#             return
#         now = time.time()

#         # keepalive
#         if (now - self.last_keepalive) >= self.keepalive_s:
#             self.follower.send_width(self.master_w, self.min_gap)
#             self.last_keepalive = now
#             if (now - self.last_sent_display) > 0.5:
#                 rospy.loginfo_throttle(0.5, 'Master=%.4f -> Follower cmd=%.4f', self.master_w, self.follower.last_cmd or -1.0)
#             return

#         # send only on meaningful change
#         if self.follower.last_cmd is None or abs(self.master_w - self.follower.last_cmd) > self.eps:
#             self.follower.send_width(self.master_w, self.min_gap)

# def main():
#     rospy.init_node('gripper_width_relay')
#     WidthMirror()
#     rospy.spin()

# if __name__ == '__main__':
#     main()
