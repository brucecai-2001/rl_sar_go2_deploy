"""
    Thanks to rl_sar @ Ziqi Fan
    Deploy HimLoco on Unitree Go2 using Unitree_SDK2_Python
"""
import rospy
import sys
sys.path.append('/home/cxy/anaconda3/envs/unitree/lib/python3.8/site-packages')

import os
import torch
import threading
import time

# Add the scripts directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ 
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

from rl_sdk import *
from observation_buffer import ObservationBuffer
from unitree_wireless_controller import unitreeRemoteController

# global definiation
CSV_LOGGER = False
TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
PosStopF = 2.146e9
VelStopF = 16000.0




class Go2HimLocoReal(RL):
    """
        GO2 HimLoco Deployment Class based on RL class in rl_sdk.py
    """
    def __init__(self):
        super().__init__()

        ####### Config #######
        self.robot_name = "go2_isaacgym"
        self.ReadYaml(self.robot_name)
        for i in range(len(self.params.observations)):
            if self.params.observations[i] == "ang_vel":
                self.params.observations[i] = "ang_vel_world"



        ####### Robot #######
        # init robot state client
        self.running_state = STATE.STATE_WAITING

        # init channels
        ChannelFactoryInitialize(0)
        # cmd publisher 
        self.lowcmd_publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.lowcmd_publisher.Init()
        # state subscriber, callback function is LowStateMessageHandler
        self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # robot cmd and state
        self.unitree_low_cmd = unitree_go_msg_dds__LowCmd_()
        self.unitree_low_state = unitree_go_msg_dds__LowState_()
        self.unitree_wireless_controller = unitreeRemoteController()
        
        # init other things
        self.crc = CRC()
        self.InitLowCmd()
        self.InitObservations()
        self.InitOutputs()
        self.InitControl()



        ####### RL #######
        # init RL model
        model_path = os.path.join(os.path.dirname(__file__), f"../models/{self.robot_name}/{self.params.model_name}")
        self.model = torch.jit.load(model_path)

        # build history observation buffer
        if len(self.params.observations_history) != 0:
            self.history_obs_buf = ObservationBuffer(1, self.params.num_observations, len(self.params.observations_history))



        ####### System #######
        # init RL loop and Control loop
        self.thread_control = threading.Thread(target=self.ThreadControl)
        self.thread_rl = threading.Thread(target=self.ThreadRL)
        self.thread_control.start()
        self.thread_rl.start()

        # init keyboard loop
        self.listener_keyboard = keyboard.Listener(on_press=self.KeyboardInterface)
        self.listener_keyboard.start()


    def RobotControl(self):
        """
            Control robot based on running_state
        """
        # refresh state
        self.GetState(self.robot_state)
        # rl_sdk StateController
        self.StateController(self.robot_state, self.robot_command)
        # build low level command and write to robot
        self.SetCommand(self.robot_command)


    def RunModel(self):
        """
            observations to each joint's torque and position 
        """
        if self.running_state == STATE.STATE_RL_RUNNING:
            self.obs.ang_vel = torch.tensor(self.robot_state.imu.gyroscope).unsqueeze(0) # 角速度
            self.obs.commands = torch.tensor([[self.unitree_wireless_controller.Ly, -self.unitree_wireless_controller.Rx, -self.unitree_wireless_controller.Lx]]) # 指令
            self.obs.base_quat = torch.tensor(self.robot_state.imu.quaternion).unsqueeze(0) # 四元数
            self.obs.dof_pos = torch.tensor(self.robot_state.motor_state.q).narrow(0, 0, self.params.num_of_dofs).unsqueeze(0) # 关节位置
            self.obs.dof_vel = torch.tensor(self.robot_state.motor_state.dq).narrow(0, 0, self.params.num_of_dofs).unsqueeze(0) # 关节速度

            # 前向传播
            clamped_actions = self.Forward()

            # 缩放hip的动作
            for i in self.params.hip_scale_reduction_indices:
                clamped_actions[0][i] *= self.params.hip_scale_reduction

            self.obs.actions = clamped_actions

            # 使用actions计算力矩，pd控制
            origin_output_dof_tau = self.ComputeTorques(self.obs.actions)
            # 截断力矩
            self.output_dof_tau = torch.clamp(origin_output_dof_tau, -(self.params.torque_limits), self.params.torque_limits)

            self.output_dof_pos = self.ComputePosition(self.obs.actions)

            if CSV_LOGGER:
                tau_est = torch.zeros((1, self.params.num_of_dofs))
                for i in range(self.params.num_of_dofs):
                    tau_est[0, i] = self.joint_efforts[self.params.joint_controller_names[i]]
                self.CSVLogger(self.output_dof_tau, tau_est, self.obs.dof_pos, self.output_dof_pos, self.obs.dof_vel)
    

    def Forward(self):
        """
            RL model inference, input observation buffer, output actions
        """
        torch.set_grad_enabled(False)
        # 计算此刻observation
        clamped_obs = self.ComputeObservation()

        # 添加到buffer中
        if len(self.params.observations_history) != 0:
            self.history_obs_buf.insert(clamped_obs)
            history_obs = self.history_obs_buf.get_obs_vec(self.params.observations_history)
            actions = self.model.forward(history_obs)
        else:
            actions = self.model.forward(clamped_obs)
        if self.params.clip_actions_lower is not None and self.params.clip_actions_upper is not None:
            return torch.clamp(actions, self.params.clip_actions_lower, self.params.clip_actions_upper)
        else:
            return actions
        
    
    def InitLowCmd(self):
        """
            Init low level command
        """
        self.unitree_low_cmd.head[0] = 0xFE
        self.unitree_low_cmd.head[1] = 0xEF
        self.unitree_low_cmd.level_flag = 0xFF # low level
        self.unitree_low_cmd.gpio = 0

        for i in range(20):
            self.unitree_low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.unitree_low_cmd.motor_cmd[i].q= PosStopF
            self.unitree_low_cmd.motor_cmd[i].kp = 0
            self.unitree_low_cmd.motor_cmd[i].dq = VelStopF
            self.unitree_low_cmd.motor_cmd[i].kd = 0
            self.unitree_low_cmd.motor_cmd[i].tau = 0
    
    
    def GetState(self, state: RobotState):
        """
            Update robot state from the unitree low state, include
            i) quaternion, 
            ii) gyroscope, 
            iii) motor_state
        """
        state_mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        # 根据 controller 的组件状态设置 control_state
        if int(self.unitree_wireless_controller.R2) == 1:
            self.control.control_state = "STATE_POS_GETUP"
        elif int(self.unitree_wireless_controller.R1) == 1:
            self.control.control_state = "STATE_RL_INIT"
        elif int(self.unitree_wireless_controller.L2) == 1:
            self.control.control_state = "STATE_POS_GETDOWN"

        # 根据 params 的 framework 不同设置 state 的 imu 信息
        # isaacgym 
        state.imu.quaternion[3] = self.unitree_low_state.imu_state.quaternion[0]  # w
        state.imu.quaternion[0] = self.unitree_low_state.imu_state.quaternion[1]  # x
        state.imu.quaternion[1] = self.unitree_low_state.imu_state.quaternion[2]  # y
        state.imu.quaternion[2] = self.unitree_low_state.imu_state.quaternion[3]  # z

        # 设置 imu 的 gyroscope 信息
        for i in range(3):
            state.imu.gyroscope[i] = self.unitree_low_state.imu_state.gyroscope[i]

        # 设置 motor_state 信息
        for i in range(self.params.num_of_dofs):
            state.motor_state.q[i] = self.unitree_low_state.motor_state[state_mapping[i]].q
            state.motor_state.dq[i] = self.unitree_low_state.motor_state[state_mapping[i]].dq
            state.motor_state.tau_est[i] = self.unitree_low_state.motor_state[state_mapping[i]].tau_est



    def SetCommand(self, command: RobotCommand):
        """
            Build low command, calculate crc and publish
        """
        command_mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        for i in range(self.params.num_of_dofs):
            self.unitree_low_cmd.motor_cmd[i].mode =  0x01
            self.unitree_low_cmd.motor_cmd[i].q = command.motor_command.q[command_mapping[i]]
            self.unitree_low_cmd.motor_cmd[i].dq = command.motor_command.dq[command_mapping[i]]
            self.unitree_low_cmd.motor_cmd[i].kp = command.motor_command.kp[command_mapping[i]]
            self.unitree_low_cmd.motor_cmd[i].kd = command.motor_command.kd[command_mapping[i]]
            self.unitree_low_cmd.motor_cmd[i].tau = command.motor_command.tau[command_mapping[i]]

        # calculate crc
        self.unitree_low_cmd.crc = self.crc.Crc(self.unitree_low_cmd)

        # publish
        if self.lowcmd_publisher.Write(self.unitree_low_cmd):
            print("Publish success. msg:", self.unitree_low_cmd.crc)
        else:
            print("Waitting for subscriber.")

    def LowStateMessageHandler(self, msg: LowState_):
        """
            Refresh the robot state and wireless controller's data
        """
        self.unitree_low_state = msg
        wireless_controller_data = msg.wireless_remote
        self.unitree_wireless_controller.parse(wireless_controller_data)


    def ThreadControl(self):
        """
            Thread which controls the robot in a state machine
        """
        thread_period = self.params.dt
        thread_name = "thread_control"
        print(f"[Thread Start] named: {thread_name}, period: {thread_period * 1000:.0f}(ms), cpu unspecified")
        while not rospy.is_shutdown():
            self.RobotControl()
            time.sleep(thread_period)
        print("[Thread End] named: " + thread_name)


    def ThreadRL(self):
        """
            Thread that runs the rl model
        """
        thread_period = self.params.dt * self.params.decimation
        thread_name = "thread_rl"
        print(f"[Thread Start] named: {thread_name}, period: {thread_period * 1000:.0f}(ms), cpu unspecified")
        while not rospy.is_shutdown():
            self.RunModel()
            time.sleep(thread_period)
        print("[Thread End] named: " + thread_name)

if __name__ == '__main__':
    go2_rl_real_deploy = Go2HimLocoReal()