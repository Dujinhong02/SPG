import pybullet as p
import pybullet_data
import time
import math
import numpy as np

class FloatingGripperEnv:
    def __init__(self, gui=True):
        self.gui = gui
        self._client_id = p.connect(p.GUI if gui else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # 地面
        self.plane_id = p.loadURDF("plane.urdf")

        # 加载你自己的夹爪 URDF（路径自己改）
        self.gripper_id = p.loadURDF(
            "/home/ubuntu/task/more_than_grasp/assets/xljz_gripper/my_gripperv4.urdf",
            basePosition=[0, 0, 0.084336],
            baseOrientation=p.getQuaternionFromEuler([math.pi, 0, 0]),
            useFixedBase=False,
        )
        for jid in range(p.getNumJoints(self.gripper_id)):
            info = p.getJointInfo(self.gripper_id, jid)
            link_name = info[12].decode("utf-8")
            if link_name in ["finger1_link"]: 
                self.finger1_jid = info[0] # 按你 URDF 的 link 名来
                p.changeDynamics(
                    self.gripper_id, 
                    self.finger1_jid,
                    lateralFriction=1,
                    spinningFriction=0.1,
                    restitution=0.0,
                )
                
            elif link_name in ["finger2_link"]:
                self.finger2_jid = info[0]
                p.changeDynamics(
                    self.gripper_id, 
                    self.finger2_jid,
                    lateralFriction=1,
                    spinningFriction=0.1,
                    restitution=0.0,
                )
        self.init_gripper_pos = [0, 0, 0.08434]
        quat = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.gripper_cid = p.createConstraint(
                parentBodyUniqueId=self.gripper_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,        
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],            
                childFramePosition=self.init_gripper_pos,
                childFrameOrientation=quat  
            )
        c = p.createConstraint(
            self.gripper_id,
            self.finger1_jid,
            self.gripper_id,
            self.finger2_jid,
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)        
        print(f"finger1_jid={self.finger1_jid},finger2_jid={self.finger2_jid}")
        self.box_id = self._spawn_box([0.3, 0, 0.05], [0.05, 0.05, 0.05])

        # 当前是否用约束“抓住”物体
        self.grasp_constraint_id = None

    def _find_finger_joints(self):
        n_joints = p.getNumJoints(self.gripper_id)
        for jid in range(n_joints):
            info = p.getJointInfo(self.gripper_id, jid)
            name = info[1].decode("utf-8")
            if name == "finger1_joint":
                self.finger1_jid = info[0]
            elif name == "finger2_joint":
                self.finger2_jid = info[0]

        assert self.finger1_jid is not None and self.finger2_jid is not None, \
            "未找到 finger1_joint / finger2_joint，请检查 URDF 关节名字"
        print("finger joints:", self.finger1_jid, self.finger2_jid)

    def _spawn_box(self, pos, half_extents, mass=0.01):
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents)
        box_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=pos,
        )
        # 提高摩擦，方便靠摩擦力抓住
        p.changeDynamics(box_id, -1, lateralFriction=2, spinningFriction=0.5)
        return box_id
    def open_gripper(self, steps=80):
        """平滑张开夹爪（从当前关节位置插值到 0.0）"""
        # 当前关节位置
        cur1 = p.getJointState(self.gripper_id, self.finger1_jid)[0]
        cur2 = p.getJointState(self.gripper_id, self.finger2_jid)[0]

        start1, start2 = cur1, cur2
        end = 0.0

        for k in range(steps):
            alpha = (k + 1) / float(steps)

            target1 = (1 - alpha) * start1 + alpha * end
            target2 = (1 - alpha) * start2 + alpha * end

            for jid, tgt in [(self.finger1_jid, target1),
                            (self.finger2_jid, target2)]:
                p.setJointMotorControl2(
                    self.gripper_id, jid,
                    p.POSITION_CONTROL,
                    targetPosition=tgt,
                    force=40,              # 张开不需要太大力
                    positionGain=0.4,
                    velocityGain=1.0,
                    maxVelocity=0.2,       # 张开速度限制
                )

            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

    def close_gripper(self, close_pos=None, is_slow=True):
        # 既然是 Sim，直接用力挤压！
        force = 10 # 给足力气
        # 假设正方向是闭合（如果不动改负数）
        # 既然你之前的代码 targetVelocity=1 是闭合，那就用 1
        
        # 持续施加闭合指令
        p.setJointMotorControl2(self.gripper_id, self.finger1_jid, p.VELOCITY_CONTROL, targetVelocity=1, force=force)
        p.setJointMotorControl2(self.gripper_id, self.finger2_jid, p.VELOCITY_CONTROL, targetVelocity=1, force=force)
        
        # 给它时间去闭合
        for _ in range(100): # 约0.4秒
            p.stepSimulation()
            if self.gui: time.sleep(1./240.)

    def move_gripper_linear(self, target_pos, target_orn, duration=1.0, dt=1./240.):
        start_pos, start_orn = p.getBasePositionAndOrientation(self.gripper_id)
        start_pos = np.array(start_pos)
        target_pos = np.array(target_pos)
        # 设定坐标轴长度
        axis_length = 0.2  # 20cm
        line_width = 2     # 线条宽度
        life_time = 0
        # # 绘制 X 轴 (红色)
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],      # 起点 (相对于物体中心)
            lineToXYZ=[axis_length, 0, 0], # 终点 (X轴方向)
            lineColorRGB=[1, 0, 0],     # 红色
            lineWidth=line_width,
            lifeTime=0,                 # 0表示永久存在，直到手动移除
            parentObjectUniqueId=self.gripper_id, # 绑定到你的夹爪ID
            parentLinkIndex=-1          # -1 表示绑定到 Base Link，如果是某个关节链可以用具体的 link index
        )

        # # 绘制 Y 轴 (绿色)
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, axis_length, 0],
            lineColorRGB=[0, 1, 0],     # 绿色
            lineWidth=line_width,
            lifeTime=0,
            parentObjectUniqueId=self.gripper_id,
            parentLinkIndex=-1
        )

        # # 绘制 Z 轴 (蓝色)
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, axis_length],
            lineColorRGB=[0, 0, 1],     # 蓝色
            lineWidth=line_width,
            lifeTime=0,
            parentObjectUniqueId=self.gripper_id,
            parentLinkIndex=-1
        )

        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], 
            lineToXYZ=[axis_length, 0, 0], 
            lineColorRGB=[1, 0, 0], # R
            lineWidth=line_width, 
            lifeTime=life_time,
            parentObjectUniqueId=self.gripper_id, # 绑定该机器人
            parentLinkIndex=self.finger1_jid      # 绑定特定的 Link 索引
        )

        # 2. 画 Y 轴 (绿色)
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], 
            lineToXYZ=[0, axis_length, 0], 
            lineColorRGB=[0, 1, 0], # G
            lineWidth=line_width, 
            lifeTime=life_time,
            parentObjectUniqueId=self.gripper_id,
            parentLinkIndex=self.finger1_jid
        )

        # 3. 画 Z 轴 (蓝色)
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0], 
            lineToXYZ=[0, 0, axis_length], 
            lineColorRGB=[0, 0, 1], # B
            lineWidth=line_width, 
            lifeTime=life_time,
            parentObjectUniqueId=self.gripper_id,
            parentLinkIndex=self.finger1_jid
        )
        p.stepSimulation()

        n_steps = int(duration / dt)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            pos = (1 - alpha) * start_pos + alpha * target_pos

            p.changeConstraint(
                self.gripper_cid,
                jointChildPivot=pos.tolist(),
                jointChildFrameOrientation=target_orn,
                maxForce=1e10,   # 给足约束力，确保 gripper 紧跟
            )

            p.stepSimulation()
            if self.gui:
                time.sleep(dt)

    def grasp_and_place(self,
                        grasp_pos, grasp_orn,
                        approach_dir, pre_dist,
                        place_pos, place_orn,
                        use_constraint=False):
        """
        grasp_pos, grasp_orn: 抓取位姿 (世界坐标)
        approach_dir: 单位向量, 世界坐标下的 approach 方向, 比如 [0, 0, -1] 表示从上往下
        pre_dist: 预抓取距离, 比如 0.10 (m)
        place_pos, place_orn: 放置位姿
        """

        approach_dir = np.array(approach_dir, dtype=float)
        approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)

        grasp_pos = np.array(grasp_pos, dtype=float)

        # ---------- Step 1: 移到预抓取位姿 ----------
        pre_grasp_pos = grasp_pos - approach_dir * pre_dist
        self.open_gripper()
        self.move_gripper_linear(pre_grasp_pos, grasp_orn, duration=1.0)

        # ---------- Step 2: 沿 approach 方向到抓取位姿 ----------
        self.move_gripper_linear(grasp_pos, grasp_orn, duration=1.0)

        # ---------- Step 3: 闭合夹爪 ----------
        self.close_gripper()

        # （可选）检测是否有接触，再决定是否建立约束
        # if use_constraint:
        self._attach_object_if_contact(self.box_id)

        # ---------- Step 4: 搬运到放置位姿 ----------
        self.move_gripper_linear(place_pos, place_orn, duration=2.0)

        # 打开夹爪&解除约束
        if use_constraint and self.grasp_constraint_id is not None:
            p.removeConstraint(self.grasp_constraint_id)
            self.grasp_constraint_id = None

        self.open_gripper()
    def _attach_object_if_contact(self, obj_id):
        # 看看夹爪和物体是否有接触点
        cps = p.getContactPoints(bodyA=self.gripper_id, bodyB=obj_id)
        if len(cps) == 0:
            print("警告：闭合后夹爪与物体没有接触点，可能没有真正抓到。")
        else:
            print("finger touch with obj.")
            return


if __name__ == "__main__":
    env = FloatingGripperEnv(gui=True)
    # 假设我们要从上方抓住桌面上的箱子
    # 这里简单用 box 的当前姿态做目标 grasp pose
    box_pos, box_orn = p.getBasePositionAndOrientation(env.box_id)

    # 抓取时，让夹爪的 -Z 方向朝下（根据你自己的建模调整）
    grasp_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # approach 方向：从上往下
    approach_dir = [0, 0, -1]
    pre_dist = 0.2  # 预抓取高度 15cm

    # 放置位姿（随便给一个位置）
    place_pos = [box_pos[0], box_pos[1], 0.35]
    place_orn = grasp_orn

    env.grasp_and_place(
        grasp_pos=[box_pos[0], box_pos[1], box_pos[2] + 0.105],  # 比物体中心略高一点
        grasp_orn=grasp_orn,
        approach_dir=approach_dir,
        pre_dist=pre_dist,
        place_pos=place_pos,
        place_orn=place_orn,
        use_constraint=False
    )

    while True:
        p.stepSimulation()
        time.sleep(1./240.)