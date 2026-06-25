import time
import glob
import numpy as np
import random
import os
import pybullet as pb
import pybullet_data    
import env.cameras as cameras
from scipy.spatial.transform import Rotation as R
from env.constants import PIXEL_SIZE, WORKSPACE_LIMITS
import math
class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.
        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """
        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.w_open = 0.137          
        self.q_open = 0.00292 # q_open = pb.getJointState(gripper_id, finger1_jid)[0]
        self.close_sign = +1
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        # self.filtered_obj_list = []
        # Start PyBullet.
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)
        pb.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step,
            numSubSteps=1,
            numSolverIterations=150,
            deterministicOverlappingPairs=1,
        )
        pb.setGravity(0, 0, -9.8)
        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )
    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)
    
    # @property
    # def is_gripper_closed(self):
    #     gripper_angle = pb.getJointState(
    #         self.ee, self.gripper_main_joint, physicsClientId=self._client_id
    #     )[0]
    #     return gripper_angle < self.gripper_angle_close_threshold

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle 
    
    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_id in self.object_ids:
            
            pos, rot = pb.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._client_id
            )
            dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
            info[obj_id] = (pos, rot, dim)
        return info
    
    def _init_gripper_indices(self):
        """从 ee 上查找所有与夹爪相关的 joint / link index。"""
        self.joint_name_to_id = {}
        num_joints = pb.getNumJoints(self.ee, physicsClientId=self._client_id)
        for j in range(num_joints):
            info = pb.getJointInfo(self.ee, j, physicsClientId=self._client_id)
            name = info[1].decode("utf-8")
            self.joint_name_to_id[name] = j

        # 主关节
        self.gripper_main_joint = self.joint_name_to_id["finger_joint"]

        # mimic 关节
        self.gripper_mimic_joints = {
            "left_inner_knuckle_joint":  self.joint_name_to_id["left_inner_knuckle_joint"],
            "left_inner_finger_joint":   self.joint_name_to_id["left_inner_finger_joint"],
            "right_outer_knuckle_joint": self.joint_name_to_id["right_outer_knuckle_joint"],
            "right_inner_knuckle_joint": self.joint_name_to_id["right_inner_knuckle_joint"],
            "right_inner_finger_joint":  self.joint_name_to_id["right_inner_finger_joint"],
        }

        # 指尖 pad 对应的 link index（用来测两指间距）
        self.left_pad_link  = self.joint_name_to_id["left_inner_finger_pad_joint"]
        self.right_pad_link = self.joint_name_to_id["right_inner_finger_pad_joint"]

    def _reset_gripper_state_for_calib(self, q):
        """标定阶段：用 mimic 关系把 6 个关节都 reset 到角度 q。"""
        cid = self._client_id
        pb.resetJointState(self.ee, self.gripper_main_joint, q, physicsClientId=cid)

        pb.resetJointState(self.ee, self.gripper_mimic_joints["left_inner_knuckle_joint"],  q, physicsClientId=cid)
        pb.resetJointState(self.ee, self.gripper_mimic_joints["right_outer_knuckle_joint"], q, physicsClientId=cid)
        pb.resetJointState(self.ee, self.gripper_mimic_joints["right_inner_knuckle_joint"], q, physicsClientId=cid)

        pb.resetJointState(self.ee, self.gripper_mimic_joints["left_inner_finger_joint"],  -q, physicsClientId=cid)
        pb.resetJointState(self.ee, self.gripper_mimic_joints["right_inner_finger_joint"], -q, physicsClientId=cid)

    def _measure_width_for_calib(self, q):
        """给定主关节角 q，返回两指尖 pad 间距（米）。"""
        self._reset_gripper_state_for_calib(q)
        cid = self._client_id
        left_pos  = np.array(pb.getLinkState(self.ee, self.left_pad_link,  physicsClientId=cid)[0])
        right_pos = np.array(pb.getLinkState(self.ee, self.right_pad_link, physicsClientId=cid)[0])
        return float(np.linalg.norm(left_pos - right_pos))

    def _calibrate_gripper_width_lut(self, num_samples: int = 21):
        """初始化时调用一次，建立 angle ↔ width 的插值表。"""
        cid = self._client_id

        # 保存当前状态，标定后恢复
        all_joints = [self.gripper_main_joint] + list(self.gripper_mimic_joints.values())
        saved = {j: pb.getJointState(self.ee, j, physicsClientId=cid)[0] for j in all_joints}

        info = pb.getJointInfo(self.ee, self.gripper_main_joint, physicsClientId=cid)
        q_lower, q_upper = info[8], info[9]      # URDF 里 lower / upper
        qs = np.linspace(q_lower, q_upper, num_samples)

        widths = []
        for q in qs:
            w = self._measure_width_for_calib(q)
            widths.append(w)

        widths = np.asarray(widths)
        order = np.argsort(widths)   # 宽度单调，便于插值
        self._width_samples = widths[order]
        self._q_samples     = qs[order]

        self.min_width = float(self._width_samples.min())
        self.max_width = float(self._width_samples.max())

        # 恢复原始状态
        for j, q in saved.items():
            pb.resetJointState(self.ee, j, q, physicsClientId=cid)

        print(f"[Gripper calib] width range: "
            f"{self.min_width*1000:.1f}mm ~ {self.max_width*1000:.1f}mm")

    def _init_ee_frame_vis(self, axis_len=0.12, line_w=3):
        """在 reset() 里调用一次"""
        self._ee_axis_len = axis_len
        self._ee_line_w = line_w
        self._ee_axis_ids = [-1, -1, -1]  # x,y,z 三条线的 debug id

    def _draw_ee_frame(self):
        """在每个 sim step 后调用，用于实时更新 ee 坐标系"""
        # 1) 取末端位姿（世界系）
        pos = self.info[self.gripper_id][0] # world position of link frame
        orn = self.info[self.gripper_id][1] # world quaternion

        # 2) 四元数 -> 旋转矩阵（row-major 9 list）
        R = np.array(pb.getMatrixFromQuaternion(orn)).reshape(3, 3)

        L = self._ee_axis_len
        # PyBullet 的旋转矩阵列向量是 ee 局部 xyz 轴在世界系的方向
        x_end = pos + R[:, 0] * L
        y_end = pos + R[:, 1] * L
        z_end = pos + R[:, 2] * L

        # 3) 画/更新三条轴线（replaceItemUniqueId 实时覆盖）
        self._ee_axis_ids[0] = pb.addUserDebugLine(
            pos, x_end, [1, 0, 0],
            lineWidth=self._ee_line_w, lifeTime=0,
            replaceItemUniqueId=self._ee_axis_ids[0]
        )
        self._ee_axis_ids[1] = pb.addUserDebugLine(
            pos, y_end, [0, 1, 0],
            lineWidth=self._ee_line_w, lifeTime=0,
            replaceItemUniqueId=self._ee_axis_ids[1]
        )
        self._ee_axis_ids[2] = pb.addUserDebugLine(
            pos, z_end, [0, 0, 1],
            lineWidth=self._ee_line_w, lifeTime=0,
            replaceItemUniqueId=self._ee_axis_ids[2]
        )
        return pos

    def _init_grasp_frame_vis(self, axis_len=0.10, line_w=2):
        self._grasp_axis_len = axis_len
        self._grasp_line_w = line_w
        self._grasp_axis_ids = [-1, -1, -1]  # x,y,z 三轴线 id

    def _draw_grasp_frame(self, grasp_pose_world):
        """
        grasp_pose_world:
            - 形式1: (pos, quat) 其中 pos=(x,y,z), quat=(x,y,z,w)
            - 形式2: 4x4 齐次矩阵 T (world <- grasp)
        """
        # -------- 解析 pose --------
        if isinstance(grasp_pose_world, (list, tuple)) and len(grasp_pose_world) == 2:
            pos = np.asarray(grasp_pose_world[0], dtype=float)
            quat = grasp_pose_world[1]
            R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        else:
            T = np.asarray(grasp_pose_world, dtype=float)
            pos = T[:3, 3]
            R = T[:3, :3]

        L = self._grasp_axis_len
        x_end = pos + R[:, 0] * L
        y_end = pos + R[:, 1] * L
        z_end = pos + R[:, 2] * L

        # -------- 画/更新三轴 --------
        self._grasp_axis_ids[0] = pb.addUserDebugLine(
            pos, x_end, [1, 0, 0],
            lineWidth=self._grasp_line_w, lifeTime=0,
            replaceItemUniqueId=self._grasp_axis_ids[0],
        )
        self._grasp_axis_ids[1] = pb.addUserDebugLine(
            pos, y_end, [0, 1, 0],
            lineWidth=self._grasp_line_w, lifeTime=0,
            replaceItemUniqueId=self._grasp_axis_ids[1],
        )
        self._grasp_axis_ids[2] = pb.addUserDebugLine(
            pos, z_end, [0, 0, 1],
            lineWidth=self._grasp_line_w, lifeTime=0,
            replaceItemUniqueId=self._grasp_axis_ids[2],
        )
        for _ in range(5):
            pb.stepSimulation()

    def _draw_grasped_obj_frame(self, obj_id, axis_len=0.08, line_w=2):
        """
        grasp_pose_world:
            - 形式1: (pos, quat) 其中 pos=(x,y,z), quat=(x,y,z,w)
            - 形式2: 4x4 齐次矩阵 T (world <- grasp)
        """
        # -------- 懒初始化：为每个物体保存三轴线 id --------
        if not hasattr(self, "_grasped_obj_axis_ids"):
            self._grasped_obj_axis_ids = {}  # obj_id -> [xid,yid,zid]
        if not hasattr(self, "_grasped_obj_text_ids"):
            self._grasped_obj_text_ids = {}  # obj_id -> text id

        if obj_id not in self._grasped_obj_axis_ids:
            self._grasped_obj_axis_ids[obj_id] = [-1, -1, -1]
            self._grasped_obj_text_ids[obj_id] = -1

        # -------- 读取物体位姿 --------
        pos, rot, _ = self.obj_info(obj_id)  # 你已有的接口
        pos = np.asarray(pos, dtype=float)

        # rot 兼容：四元数 or 旋转矩阵
        rot_arr = np.asarray(rot)
        if rot_arr.shape == (4,):  # quat
            Rm = np.array(pb.getMatrixFromQuaternion(rot_arr.tolist())).reshape(3, 3)
        elif rot_arr.shape == (3, 3):
            Rm = rot_arr
        else:
            raise ValueError(f"Unsupported rot format: {rot_arr.shape}")

        L = axis_len
        x_end = pos + Rm[:, 0] * L
        y_end = pos + Rm[:, 1] * L
        z_end = pos + Rm[:, 2] * L

        ids = self._grasped_obj_axis_ids[obj_id]

        # -------- 画/更新三轴（我用偏亮的颜色区分物体）--------
        ids[0] = pb.addUserDebugLine(
            pos, x_end, [1, 0.2, 0.2],
            lineWidth=line_w, lifeTime=0,
            replaceItemUniqueId=ids[0]
        )
        ids[1] = pb.addUserDebugLine(
            pos, y_end, [0.2, 1, 0.2],
            lineWidth=line_w, lifeTime=0,
            replaceItemUniqueId=ids[1]
        )
        ids[2] = pb.addUserDebugLine(
            pos, z_end, [0.2, 0.2, 1],
            lineWidth=line_w, lifeTime=0,
            replaceItemUniqueId=ids[2]
        )
        return pos

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        fixed_rot = pb.getQuaternionFromEuler([np.pi, 0, 0])
        rot = fixed_rot
        info = (pos, rot, dim)
        return info


    def get_link_pose(self,body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]
    
    def go_home(self):
        return self.move_joints(self.home_joints)
    
    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def solve_ik(self, pose):
            """Calculate joint configuration with inverse kinematics."""
            joints = pb.calculateInverseKinematics(
                bodyUniqueId=self.ur5e,
                endEffectorLinkIndex=self.ur5e_ee_id,
                targetPosition=pose[0],
                targetOrientation=pose[1],
                lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
                upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
                jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
                restPoses=np.float32(self.ik_rest_joints).tolist(),
                # maxNumIterations=100,
                # residualThreshold=1e-5,
            )
            joints = np.array(joints, dtype=np.float32)
            # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
            return joints
    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform
    
    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    def add_objects(self, num_obj, workspace_limits):
        """Randomly dropped objects to the workspace"""
        # random.seed(seed)
        # np.random.seed(11111)
        # cam1 = (self.agent_cams[1]['position'],self.agent_cams[1]['rotation'])
        # self._init_grasp_frame_vis()
        # self._draw_grasp_frame(cam1)
        self.object_ids = []  # 保存创建的物体ID
        self.obj_folder = "/home/ubuntu/task/more_than_grasp/assets/train_dataset_center/"
        # self.obj_folder = "/home/ubuntu/task/more_than_grasp/assets/3dnet/"
        files = [f for f in os.listdir(self.obj_folder) if f.endswith(".obj")]
        k = min(15, len(files))
        rng = random.Random(time.time_ns() ^ os.getpid())
        picked = rng.sample(files, k)
        # print(picked)
        self.filtered_obj_list = [os.path.join(self.obj_folder, f) for f in picked]
        # with open('/home/ubuntu/task/more_than_grasp/obj_path_refine.txt', "r", encoding="utf-8", errors="ignore") as f:
        #     obj_list = [line.strip() for line in f if line.strip()]
        # self.filtered_obj_list = random.sample(obj_list, 6)

        pb.setGravity(0, 0, -9.8)
        for i in range(num_obj):
            # 随机生成物体的 (x, y) 坐标，z高度固定高一点让它掉下来
            x = np.random.uniform(workspace_limits[0][0]+0.1, workspace_limits[0][1]-0.1)
            y = np.random.uniform(workspace_limits[1][0]+0.1, workspace_limits[1][1]-0.1)
            z = workspace_limits[2][1]-0.1   # 高一点，从空中掉下来
            # 生成随机旋转角度（绕z轴旋转）
            angle = random.uniform(-np.pi, np.pi)
            orientation = pb.getQuaternionFromEuler([angle, 0, angle])
            obj_path = os.path.join(self.obj_folder, self.filtered_obj_list[i]) 
            # if i == 0:
            #     color = [1, 0, 0, 1]
            #     x = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
            #     y = np.random.uniform(workspace_limits[1][0]+0.3, workspace_limits[1][1]-0.3)
            #     # y = (workspace_limits[1][0] + workspace_limits[1][1]) / 2
            # else:
            color = [random.random(), random.random(), random.random(), 1]
            # 加载.obj的visual shape
            # if i != 0:
            #     obj_path = os.path.join(self.obj_folder, random.choice(self.obj_list))  # 随机选一个物体
            vis_id = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1],  # 根据需要缩放
                rgbaColor=color
            )
            # 简单地创建一个碰撞体，比如用球或盒子代表
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    collision_id = pb.createCollisionShape(
                        shapeType=pb.GEOM_MESH,
                        fileName=obj_path,
                        meshScale=[1, 1, 1]
                    )
                    break  # 成功了就退出循环
                except Exception as e:
                    print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                    time.sleep(0.1)  # 等一下再试
            else:
                raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

            # 组合成物体
            body_id = pb.createMultiBody(
                baseMass=0.1,  # 轻一点
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            pb.changeDynamics(
                body_id,
                -1,
                rollingFriction=0.001,
                spinningFriction=0.001,
            )
            self.object_ids.append(body_id)
            for _ in range(340):  
                pb.stepSimulation()
                time.sleep(1/500)
            self.wait_static()
        self.object_ids.append(self.gripper_id)
        return self.object_ids

    def add_one_objects(self, num_obj, workspace_limits, index):
        """Randomly dropped objects to the workspace"""
        random.seed(11111)
        np.random.seed(11111)
        self.obj_ids = {"fixed": [], "rigid": []}
        self.object_ids = []  # 保存创建的物体ID
        # self.obj_folder = "/home/ubuntu/task/more_than_grasp/assets/train_dataset_center"
        self.obj_folder = "/home/ubuntu/task/more_than_grasp/assets/train_dataset_center"
        if index == 0:
            obj_files = [f for f in os.listdir(self.obj_folder) if f.endswith(".obj")]
            def mixed_key(f):
                stem = os.path.splitext(f)[0]
                if stem.isdigit():
                    return (0, int(stem))   # 纯数字：按数值排，放前面
                else:
                    return (1, stem)        # 非数字：按字母排，放后面
            self.filtered_obj_list = sorted(obj_files, key=mixed_key)
        # index = int(index / 2)
        for i in range(num_obj):
            # 随机生成物体的 (x, y) 坐标，z高度固定高一点让它掉下来
            x = np.random.uniform(workspace_limits[0][0]+0.2, workspace_limits[0][1]-0.2)
            y = np.random.uniform(workspace_limits[1][0]+0.2, workspace_limits[1][1]-0.2)
            z = 0.2  # 高一点，从空中掉下来
            # 生成随机旋转角度（绕z轴旋转）
            angle = random.uniform(-np.pi, np.pi)
            orientation = pb.getQuaternionFromEuler([angle, angle, angle])
            color = [0.5, random.random(), random.random(), 1]
            # 加载.obj的visual shape
            obj_path = os.path.join(self.obj_folder, self.filtered_obj_list[index % 124])  
            # obj_path = '/home/ubuntu/task/more_than_grasp/assets/train_dataset_center/1ef68777bfdb7d6ba7a07ee616e34cd7.obj'
            print(f'\033[34m current obj path :{obj_path} \033[0m')
            vis_id = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1],  # 根据需要缩放
                rgbaColor=color
            )
            # 简单地创建一个碰撞体，比如用球或盒子代表
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    collision_id = pb.createCollisionShape(
                        shapeType=pb.GEOM_MESH,
                        fileName=obj_path,
                        meshScale=[1, 1, 1]
                    )
                    break  # 成功了就退出循环
                except Exception as e:
                    print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                    time.sleep(0.1)  # 等一下再试
            else:
                raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

            # 组合成物体
            body_id = pb.createMultiBody(
                baseMass=0.1,  # 轻一点
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            
            self.goal_obj_id = body_id
            pb.changeDynamics(
                body_id,
                -1,
                rollingFriction=0.001,
                spinningFriction=0.001,
            )
            self.object_ids.append(body_id)
            for _ in range(500):  
                pb.stepSimulation()
                time.sleep(1/500)
            self.wait_static()
        # center_obj = self.info[self.goal_obj_id]
        # self._draw_grasp_frame(center_obj[:2])
         # ========= 交互删除模块，从这里开始 =========
        try:
            user_input = input(
                f"\n当前物体: {obj_path}\n"
                f"输入 0 并回车：删除该 obj 文件\n"
                f"输入空格或直接回车：保留该 obj 文件\n"
                f"你的选择: "
            )
        except EOFError:
            # 非交互环境（比如被别的程序调用）时，直接跳过交互
            user_input = ""

        if user_input == "0":
            # 删除文件（可选：同时把场景里的物体也删掉）
            try:
                os.remove(obj_path)
                print(f"\033[31m已删除 obj 文件: {obj_path}\033[0m")
            except OSError as e:
                print(f"\033[31m删除 {obj_path} 失败: {e}\033[0m")

            # 如果你希望场景里也看不到这个物体，可以顺便移除：
            # pb.removeBody(body_id)
            # self.object_ids.remove(body_id)

        elif user_input == " " or user_input == "":
            # 空格 或 直接回车：不做任何操作
            with open('/home/ubuntu/task/more_than_grasp/non_symmetric.txt', 'a') as f:
                f.write(f"{obj_path}\n")
            print("\033[32m保留该 obj 文件。\033[0m")

        else:
            # 其他输入：默认视为“保留”
            print(f"\033[33m输入 '{user_input}' 无效，默认保留该 obj 文件。\033[0m")
        # ========= 交互删除模块到这里结束 =========
        self.object_ids.append(self.gripper_id)
        return self.object_ids, self.filtered_obj_list[index % 124]

    # def add_one_objects(self, num_obj, workspace_limits, index):
    #     """Randomly dropped objects to the workspace"""
    #     rng = self._random
    #     # random.seed(11111)
    #     # np.random.seed(11111)
    #     self.obj_ids = {"fixed": [], "rigid": []}
    #     self.object_ids = []  # 保存创建的物体ID
    #     self.obj_folder = "/home/ubuntu/task/more_than_grasp/assets/train_dataset_center"
    #     if index == 0:
    #         obj_files = [f for f in os.listdir(self.obj_folder) if f.endswith(".obj")]
    #         def mixed_key(f):
    #             stem = os.path.splitext(f)[0]
    #             if stem.isdigit():
    #                 return (0, int(stem))   # 纯数字：按数值排，放前面
    #             else:
    #                 return (1, stem)        # 非数字：按字母排，放后面
    #         self.filtered_obj_list = sorted(obj_files, key=mixed_key)
    #     # index = int(index / 2)
    #     for i in range(num_obj):
    #         x = float(rng.uniform(workspace_limits[0][0]+0.2, workspace_limits[0][1]-0.2))
    #         y = float(rng.uniform(workspace_limits[1][0]+0.2, workspace_limits[1][1]-0.2))
    #         z = 0.2

    #         angle = float(rng.uniform(-np.pi, np.pi))
    #         orientation = pb.getQuaternionFromEuler([angle, angle, angle])

    #         color = [0.5, float(rng.rand()), float(rng.rand()), 1.0]
    #         # 加载.obj的visual shape
    #         obj_path = os.path.join(self.obj_folder, self.filtered_obj_list[index % 124])  
    #         # obj_path = '/home/ubuntu/task/more_than_grasp/assets/train_dataset_center/1ef68777bfdb7d6ba7a07ee616e34cd7.obj'
    #         print(f'\033[34m current obj path :{obj_path} \033[0m')
    #         vis_id = pb.createVisualShape(
    #             shapeType=pb.GEOM_MESH,
    #             fileName=obj_path,
    #             meshScale=[1, 1, 1],  # 根据需要缩放
    #             rgbaColor=color
    #         )
    #         # 简单地创建一个碰撞体，比如用球或盒子代表
    #         max_retry = 3
    #         for attempt in range(max_retry):
    #             try:
    #                 collision_id = pb.createCollisionShape(
    #                     shapeType=pb.GEOM_MESH,
    #                     fileName=obj_path,
    #                     meshScale=[1, 1, 1]
    #                 )
    #                 break  # 成功了就退出循环
    #             except Exception as e:
    #                 print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
    #                 time.sleep(0.1)  # 等一下再试
    #         else:
    #             raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

    #         # 组合成物体
    #         body_id = pb.createMultiBody(
    #             baseMass=0.1,  # 轻一点
    #             baseCollisionShapeIndex=collision_id,
    #             baseVisualShapeIndex=vis_id,
    #             basePosition=[x, y, z],
    #             baseOrientation=orientation
    #         )
            
    #         self.goal_obj_id = body_id
    #         pb.changeDynamics(
    #             body_id,
    #             -1,
    #             rollingFriction=0.001,
    #             spinningFriction=0.001,
    #         )
    #         self.object_ids.append(body_id)
    #         for _ in range(500):  
    #             pb.stepSimulation()
    #             time.sleep(1/500)
    #         self.wait_static()
    #     # center_obj = self.info[self.goal_obj_id]
    #     # self._draw_grasp_frame(center_obj[:2])
    #      # ========= 交互删除模块，从这里开始 =========
    #     # try:
    #     #     user_input = input(
    #     #         f"\n当前物体: {obj_path}\n"
    #     #         f"输入 0 并回车：删除该 obj 文件\n"
    #     #         f"输入空格或直接回车：保留该 obj 文件\n"
    #     #         f"你的选择: "
    #     #     )
    #     # except EOFError:
    #     #     # 非交互环境（比如被别的程序调用）时，直接跳过交互
    #     #     user_input = ""

    #     # if user_input == "0":
    #     #     # 删除文件（可选：同时把场景里的物体也删掉）
    #     #     try:
    #     #         os.remove(obj_path)
    #     #         print(f"\033[31m已删除 obj 文件: {obj_path}\033[0m")
    #     #     except OSError as e:
    #     #         print(f"\033[31m删除 {obj_path} 失败: {e}\033[0m")

    #     #     # 如果你希望场景里也看不到这个物体，可以顺便移除：
    #     #     # pb.removeBody(body_id)
    #     #     # self.object_ids.remove(body_id)

    #     # elif user_input == " " or user_input == "":
    #     #     # 空格 或 直接回车：不做任何操作
    #     #     with open('/home/ubuntu/task/more_than_grasp/non_symmetric.txt', 'a') as f:
    #     #         f.write(f"{obj_path}\n")
    #     #     print("\033[32m保留该 obj 文件。\033[0m")

    #     # else:
    #     #     # 其他输入：默认视为“保留”
    #     #     print(f"\033[33m输入 '{user_input}' 无效，默认保留该 obj 文件。\033[0m")
    #     # ========= 交互删除模块到这里结束 =========
    #     self.object_ids.append(self.gripper_id)
    #     return self.object_ids, self.filtered_obj_list[index % 124]
    
    def add_one_objects_with_texture(self, num_obj, workspace_limits, index):
        """Randomly dropped objects to the workspace"""
        self.obj_ids = {"fixed": [], "rigid": []}
        self.object_ids = []

        # 根目录：每个子文件夹一个 YCB 物体
        self.obj_root = "/home/ubuntu/task/more_than_grasp/assets/YCB_obj"

        # 只保留子目录，如 ['000', '001', '002', ...]
        self.filtered_obj_list = sorted(
            d for d in os.listdir(self.obj_root)
            if os.path.isdir(os.path.join(self.obj_root, d))
        )

        # 选定一个物体目录
        obj_name = self.filtered_obj_list[index]
        print(f'current index is {index}')
        obj_dir = os.path.join(self.obj_root, obj_name)

        # mesh 和 texture 路径
        mesh_path = os.path.join(obj_dir, "textured.obj")      # 或 nontextured_simplified_*.obj
        # 预先加载贴图
        # texture_path = self.find_texture_path(obj_dir)
        # if texture_path != None:
        #     texture_id = pb.loadTexture(texture_path)
        for i in range(num_obj):
            # 位置和姿态
            x = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
            y = (workspace_limits[1][0] + workspace_limits[1][1]) / 2
            z = 0.1
            orientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

            # 可视形状：用 mesh，不再用随机 rgbaColor 去染色（避免把贴图颜色覆盖掉）
            vis_id = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName=mesh_path,
                meshScale=[1, 1, 1],
                rgbaColor=[1, 1, 1, 1]  # 保持白色，让贴图原色显示
            )

            # 碰撞形状：同一个 mesh
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    collision_id = pb.createCollisionShape(
                        shapeType=pb.GEOM_MESH,
                        fileName=mesh_path,
                        meshScale=[1, 1, 1]
                    )
                    break
                except Exception as e:
                    print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                    time.sleep(0.1)
            else:
                raise RuntimeError(f"Failed to create collision shape for {mesh_path} after {max_retry} attempts.")

            # 创建刚体
            body_id = pb.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )

            # 给刚体绑定贴图
            # if texture_path != None:
            #     pb.changeVisualShape(
            #         objectUniqueId=body_id,
            #         linkIndex=-1,              # base link
            #         textureUniqueId=texture_id
            #     )

            self.object_ids.append(body_id)
            self.obj_ids["rigid"].append(body_id)
            # 让物体掉落稳定
            for _ in range(240):
                pb.stepSimulation()
                time.sleep(1 / 500)

            self.wait_static()

        return self.object_ids, obj_name

    def find_texture_path(self, obj_dir: str):
        """在 obj_dir 中查找纹理文件，优先常见命名，其次任意 png/jpg"""
        # 1) 优先按常见文件名搜索
        cand_names = [
            "texture_map.png",
            "textured.png",
            "textured.jpg",
            "texture.png",
            "texture.jpg",
        ]
        for name in cand_names:
            path = os.path.join(obj_dir, name)
            if os.path.exists(path):
                return path

        # 2) 退而求其次：找任意 png/jpg/jpeg
        for f in os.listdir(obj_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(obj_dir, f)

        # 3) 找不到就返回 None
        return None

    def load_objects_from_txt(self, txt_path):
        """
        仅支持每行：obj  r g b  x y z  roll pitch yaw
        角度默认是弧度；若 angles_in_degrees=True 则自动转弧度。
        若 apply_z180=True，则对姿态做绕 z 轴 180°补偿，同时 x,y 取反。
        """
        self.obj_folder = "/home/ubuntu/task/MyProject_Grasp-Push/assets/obj"
        self.object_ids = []
        with open(txt_path, "r") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                parts = s.split()
                # if len(parts) != 10:
                #     raise ValueError(
                #         f"[line {ln}] 期望 10 列（obj + 9 数字），实际 {len(parts)} 列：{parts}"
                #     )

                obj_file = parts[0]
                try:
                    nums = list(map(float, parts[1:]))   # 9 个数字
                except Exception as e:
                    raise ValueError(f"[line {ln}] 数字解析失败：{parts[1:]} ({e})")
                """add challenge objects to the workspace"""
                pb.setGravity(0, 0, -10)
                xyz = nums[0:3]
                theta = nums[5]
                
                euler_angles = nums[3:]
                initial_orientation = pb.getQuaternionFromEuler(euler_angles)
                # orientation = pb.getQuaternionFromEuler(nums[3:])

                color = [random.random(), random.random(), random.random(), 1]
                # 加载.obj的visual shape
                
                obj_path = os.path.join(self.obj_folder, obj_file)  # 随机选一个物体
                vis_id = pb.createVisualShape(
                    shapeType=pb.GEOM_MESH,
                    fileName=obj_path,
                    meshScale=[1, 1, 1],  # 根据需要缩放
                    rgbaColor=color
                )
                # 简单地创建一个碰撞体，比如用球或盒子代表
                max_retry = 3
                for attempt in range(max_retry):
                    try:
                        collision_id = pb.createCollisionShape(
                            shapeType=pb.GEOM_MESH,
                            fileName=obj_path,
                            meshScale=[1, 1, 1],
                            # flags=pb.GEOM_FORCE_CONCAVE_TRIMESH 
                        )
                        break  # 成功了就退出循环
                    except Exception as e:
                        print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                        time.sleep(0.1)  # 等一下再试
                else:
                    raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

                # 组合成物体
                body_id = pb.createMultiBody(
                    baseMass=0.1,  
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=xyz,
                    baseOrientation=initial_orientation
                )
                self.object_ids.append(body_id)
                for _ in range(240):  
                    pb.stepSimulation()
                    time.sleep(1/500)
                self.wait_static()
        return self.object_ids

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # renderer=pb.ER_BULLET_HARDWARE_OPENGL,
            renderer=pb.ER_TINY_RENDERER,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        # segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def reset(self):  #加载ur5e机械臂，并设置工作空间
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            
        # pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = pb.loadURDF(
            "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )
        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = pb.loadURDF(
            "assets/ur5e/ur5e.urdf",
            basePosition=(0, 0, 0),
            useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        self._init_ee_frame_vis(axis_len=0.12, line_w=3)
        self._init_grasp_frame_vis()
        # if not getattr(self, "gripper_width_calibrated", False):
        #     self._init_gripper_indices()        # 找到指尖 link
        #     self._calibrate_gripper_width_lut() # 建立 angle<->width 映射
        #     self.gripper_width_calibrated = True

        # Move robot to home joint configuration.
        success = self.go_home()
        self.open_gripper_franka()
        self.close_gripper_franka()
        

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )
        return self.ur5e
    
    def reset_only_gripper(self):  #加载ur5e机械臂，并设置工作空间
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)
        self.home_pose = np.array([-0.2, 0, 0.2, 1, 0, 0, 0])
        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
            
        # pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = pb.loadURDF(
            "sim_exp/assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )
        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        self.setup_gripper_only()

        self._init_ee_frame_vis(axis_len=0.12, line_w=3)
        self._init_grasp_frame_vis()

        # Move robot to home joint configuration.
        # success = self.move_gripper_linear()
        self.gripper_open()
        self.gripper_close()
        # if not success:
        #     print("Simulation is wrong!")
        #     exit()
        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )
    def gripper_open(self, steps=80):
        """平滑张开夹爪（从当前关节位置插值到 0.0）"""
        # 当前关节位置
        cur1 = pb.getJointState(self.gripper_id, self.finger1_jid)[0]
        cur2 = pb.getJointState(self.gripper_id, self.finger2_jid)[0]

        start1, start2 = cur1, cur2
        end = 0.0

        for k in range(steps):
            alpha = (k + 1) / float(steps)

            target1 = (1 - alpha) * start1 + alpha * end
            target2 = (1 - alpha) * start2 + alpha * end

            for jid, tgt in [(self.finger1_jid, target1),
                            (self.finger2_jid, target2)]:
                pb.setJointMotorControl2(
                    self.gripper_id, jid,
                    pb.POSITION_CONTROL,
                    targetPosition=tgt,
                    force=40,              
                    positionGain=0.4,
                    velocityGain=1.0,
                    maxVelocity=0.2,       
                )
            pb.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)
            
    def gripper_close(self):
        force = 3.1 
        pb.setJointMotorControl2(self.gripper_id, self.finger1_jid, pb.VELOCITY_CONTROL, targetVelocity=1, force=force)
        pb.setJointMotorControl2(self.gripper_id, self.finger2_jid, pb.VELOCITY_CONTROL, targetVelocity=1, force=force)
        for _ in range(100): 
            pb.stepSimulation()
            if self.gui: time.sleep(1./240.)

    def set_gripper_width(self,
                      width_m: float,
                      force: float = 20.0,
                      max_vel: float = 1.0,
                      steps: int = 240,
                      tol: float = 1e-4,
                      positionGain: float = 0.3,
                      velocityGain: float = 1.0):
        """把夹爪收拢到目标开口宽度（预抓取宽度），并尽量停稳。"""
        q, _ = self._width_to_finger_q(width_m)

        pb.setJointMotorControl2(
            self.gripper_id, self.finger1_jid,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=q,
            force=force,
            maxVelocity=max_vel,
            positionGain=positionGain,
            velocityGain=velocityGain,
        )
        pb.setJointMotorControl2(
            self.gripper_id, self.finger2_jid,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=q,
            force=force,
            maxVelocity=max_vel,
            positionGain=positionGain,
            velocityGain=velocityGain,
        )

        # 让它跑到位（或接近到位）
        for _ in range(int(steps)):
            pb.stepSimulation()
            if self.gui:
                time.sleep(1.0/240.0)

            q1 = pb.getJointState(self.gripper_id, self.finger1_jid)[0]
            q2 = pb.getJointState(self.gripper_id, self.finger2_jid)[0]
            if abs(q1 - q) < tol and abs(q2 - q) < tol:
                break

    def _width_to_finger_q(self, w_target: float):
        # 宽度 clamp
        w = float(w_target)
        w = max(0.0, min(w, self.w_open))

        # 每个手指需要移动的量（对称两指：宽度变化的一半）
        dq = 0.5 * (self.w_open - w)

        # 方向：close_sign=+1 表示 q 增大 -> 宽度变小（闭合）
        q1 = self.q_open + self.close_sign * dq
        q2 = self.q_open + self.close_sign * dq

        # # 关节限位 clamp
        # q1 = max(self.f1_lower, min(q1, self.f1_upper))
        # q2 = max(self.f2_lower, min(q2, self.f2_upper))
        return q1, q2            

    def move_gripper_linear(self, target_pos, target_orn, duration=1.0, dt=1./240., force_detect=False):
        start_pos, start_orn = pb.getBasePositionAndOrientation(self.gripper_id)
        start_pos = np.array(start_pos)
        target_pos = np.array(target_pos)
        success = True
        n_steps = int(duration / dt)
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            pos = (1 - alpha) * start_pos + alpha * target_pos
            s = 0.5 - 0.5 * np.cos(np.pi * alpha)     # 0→1 且两端导数为 0
            orn = pb.getQuaternionSlerp(start_orn, target_orn, s)
            pb.changeConstraint(
                self.gripper_cid,
                jointChildPivot=pos.tolist(),
                jointChildFrameOrientation=orn,
                maxForce=1e10,   # 给足约束力，确保 gripper 紧跟
            )

            pb.stepSimulation()
            if force_detect:
                max_force = 100
                total = 0.0
                finger_links = [self.finger1_jid, self.finger2_jid]
                obj_ids = self.object_ids
                for oid in obj_ids:
                    for linkA in finger_links:
                        cps = pb.getContactPoints(bodyA=self.gripper_id, bodyB=oid, linkIndexA=linkA)
                        # contact tuple 的第 10 个字段 (index=9) 是 normalForce
                        for cp in cps:
                            total += float(cp[9])
                if total > max_force:
                    print(f"\033[33m Force is {total}, exceed the max force {max_force} \033[0m")
                    return False
            if self.gui:
                time.sleep(dt)
        hold_steps = 50 
        for _ in range(hold_steps):
            pb.changeConstraint(
                self.gripper_cid,
                jointChildPivot=target_pos, # 这里的参数见下文“原因三”的讨论
                jointChildFrameOrientation=target_orn,
                maxForce=1e10
            )
            pb.stepSimulation()
        end_pos, _ = pb.getBasePositionAndOrientation(self.gripper_id)
        if np.linalg.norm(target_pos - np.array(end_pos)) > 1e-3:
            success = False
        return success
    def setup_gripper(self):  
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9 # intial 0.9 change to 1.0
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def setup_gripper_only(self):
        self.gripper_id = pb.loadURDF(
            "/home/ubuntu/task/more_than_grasp/assets/xljz_gripper/my_gripperv3.urdf",
            basePosition=[0, 0, 0.2],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
        )
        for jid in range(pb.getNumJoints(self.gripper_id)):
            info = pb.getJointInfo(self.gripper_id, jid)
            link_name = info[12].decode("utf-8")
            if link_name in ["finger1_link"]: 
                self.finger1_jid = info[0] # 按你 URDF 的 link 名来
                pb.changeDynamics(
                    self.gripper_id, 
                    self.finger1_jid,
                    lateralFriction=0.9,
                    spinningFriction=0.1,
                    restitution=0.0,
                )
                
            elif link_name in ["finger2_link"]:
                self.finger2_jid = info[0]
                pb.changeDynamics(
                    self.gripper_id, 
                    self.finger2_jid,
                    lateralFriction=0.9,
                    spinningFriction=0.1,
                    restitution=0.0,
                )
        self.init_gripper_pos = [0, 0, 0.2]
        quat = pb.getQuaternionFromEuler([math.pi, 0, 0])
        self.gripper_cid = pb.createConstraint(
                parentBodyUniqueId=self.gripper_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,        
                childLinkIndex=-1,
                jointType=pb.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],            
                childFramePosition=self.init_gripper_pos,
                childFrameOrientation=quat,
            )
        pb.changeConstraint(self.gripper_cid, maxForce=10000)
        c = pb.createConstraint(
            self.gripper_id,
            self.finger1_jid,
            self.gripper_id,
            self.finger2_jid,
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000) 


    def move_joints(self, target_joints, speed=0.01, timeout=10):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                    self._draw_ee_frame()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
            # self._draw_ee_frame()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False
    
    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)
    
    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if success == False:
                return success
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
                self._draw_ee_frame()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
                    self._draw_ee_frame()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()
            self._draw_ee_frame()


    def step(self,pose=None, approach_is_down=False, target_obj=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        success, grasped_obj_id, valid_flag, distance_force = self.grasp(pose, approach_is_down, target_obj)
            # Grasping fails
        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()
            _ = self._draw_ee_frame()
        return success, grasped_obj_id, valid_flag, distance_force
    
    def grasp(self, pose, approach_is_down, target_obj, speed=0.001):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_tip_id, lateralFriction=0.9, spinningFriction=0.1
        )

        transform = pose
        valid_flag = True
        distance_force = None
        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform
        vis_ee_pose = np.copy(ee_transform)
        
        pos = (ee_transform[:3, 3]).T
        # for aligning with ee 
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        # approach direaction
        # over = np.array((pos[0], pos[1], pos[2] + 0.2))
        direction = ee_transform[:3, 0]
        approach_distance = 0.2
        if approach_is_down:
            over = np.array((pos[0], pos[1], pos[2] + 0.2))
        else:
            over = pos - direction * approach_distance
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        
        # Execute 6-dof grasping.
        grasped_obj_id = None
        # min_pos_dist = None  
        self.open_gripper()
        # self.set_gripper_width(width)
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot), speed)
            for _ in range(10):
                pb.stepSimulation()
                _ = self._draw_ee_frame()
            self._draw_grasp_frame(vis_ee_pose)
                
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
            if success==False:
                self.go_home()
                return success, grasped_obj_id, valid_flag, distance_force
            for _ in range(10):
                pb.stepSimulation()
                pose = self._draw_ee_frame()
            distance_error = np.linalg.norm(pose - pos)
            if distance_error > 0.022:
                valid_flag = False
                print(f'\033[33m gripper do not arrive grasp pose!!! invalid grasp action!!! {distance_error}\033[0m')

        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            for _ in range(5):
                pb.stepSimulation()
                _ = self._draw_ee_frame()
            
            # success &= (self.is_gripper_closed > 0.73)
            success = self.move_joints(self.ik_rest_joints)
            for _ in range(10):
                pb.stepSimulation()
                pose_ee = self._draw_ee_frame()
            pose_obj = self._draw_grasped_obj_frame(target_obj)
            # ensure the ee pose arrive goal position
            if not (self.is_gripper_closed > 0.795):
                pose1 = pose_ee[:2]
                pose2 = pose_obj[:2]
                distance_force = np.linalg.norm(pose2 - pose1)
            if success: # get grasped object id
                max_height = 0.2
                # grasped_obj_id = []
                for i in self.object_ids:
                    height = self.info[i][0][2]
                    
                    if height >= max_height:
                        grasped_obj_id = i
                        # break

        if success:
            success = self.move_joints(self.drop_joints1)
            # success &= self.is_gripper_closed
            self.open_gripper_franka(is_slow=True)
        self.go_home()

        # print(f"Grasp at {pose}, the grasp {success}")

        pb.changeDynamics(
            self.ee, self.ee_tip_id, lateralFriction=0.9
        )

        return success, grasped_obj_id, valid_flag, distance_force
    
    def step_only_gripper(self,pose=None, approach_is_down=False, goal_obj_id=None, width=None, is_refine=False):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        success, is_valid = self.grasp_only_gripper(pose, approach_is_down, goal_obj_id, width, is_refine)
            # Grasping fails
        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()
            # _ = self._draw_ee_frame()
        return success, is_valid

    def grasp_only_gripper(self, pose, approach_is_down, goal_obj_id, width, is_refine):
        """Execute grasping primitive.
        Args:
            pose: SE(3) grasping pose.
        Returns:
            success: robot movement success if True.
        """
        # Handle unexpected behavior
        is_valid = True
        transform = pose
        vis_ee_pose = np.copy(pose)
        center2pad_offset = 0.085326
        # transform from ee link to tip 
        if not is_refine:
            if approach_is_down:
                transform[2, 3] += center2pad_offset
            else:
                transform[:3,3] -= center2pad_offset * transform[:3, 2] 
        ee_transform = transform 
        
        pos = (ee_transform[:3, 3]).T
        # for aligning with ee 
        pos[2] = max(pos[2], self.bounds[2][0])
        # approach direaction
        if approach_is_down:
            over = np.array((pos[0], pos[1], pos[2] + 0.2))
        else:
            direction = ee_transform[:3, 2]
            approach_distance = 0.2
            over = pos - direction * approach_distance
        set_pos = np.copy(over)
        set_pos[0] = -0.2
        set_pos[2] += 0.1
        set_rot = pb.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()
        # Execute 6-dof grasping.
        self.gripper_open()
        # q_open = pb.getJointState(self.gripper_id, self.finger1_jid)[0]
        # print(f"q_open = {q_open}")
        self.set_gripper_width(width)
        self.move_gripper_linear(over, rot, duration=1.0)
        for _ in range(5):
            pb.stepSimulation()
            # pose = self._draw_ee_frame()
            # self._draw_grasp_frame(vis_ee_pose)
        # ---------- Step 2: along approach to goal pose ----------
        success = self.move_gripper_linear(pos, rot, duration=1.0, force_detect=True)
        for _ in range(5):
            pb.stepSimulation()
            # pose = self._draw_ee_frame()
        if not success:
            is_valid = False
        # ---------- Step 3: close gripper----------
        if success:
            self.gripper_close()
            for _ in range(5):
                pb.stepSimulation()
        # ---------- Step 4: compute force length ----------
        # obj_position = self.info[goal_obj_id][0]
        # gripper_pose =self.info[self.gripper_id]
        # approach_gripper = R.from_quat(gripper_pose[1]).as_matrix()
        # finger_position = gripper_pose[0] + center2pad_offset * approach_gripper[:3, 2]
        # force_length = np.linalg.norm(finger_position[:2] - obj_position[:2])

        # vis_finger_pose = np.eye(4)
        # vis_finger_pose[:3, :3] = approach_gripper
        # vis_finger_pose[:3, 3] = finger_position
        # self._draw_grasp_frame(vis_finger_pose)
        # self.move_gripper_linear(over, rot, duration=1.0)
        # success = self.move_gripper_linear(over, set_rot, duration=1.5)
        # ensure the ee pose arrive goal position           
        # if success:
        #     pose_obj = self.info[self.goal_obj_id][0]
        #     pose_ee = self.info[self.gripper_id][0]
        #     distance_force = np.linalg.norm(np.array(pose_ee[:2]) - np.array(pose_obj[:2]))
        self.move_gripper_linear(over, rot, duration=1)

        self.move_gripper_linear(set_pos, rot, duration=1)
        if success: # get grasped object id
            max_height = 0.08
            # grasped_obj_id = []
            pose_obj = self.info[goal_obj_id][0]
            if pose_obj[2] >= max_height:
                success = True
            else:
                success = False
        self.gripper_open()
        print(f"the grasp is {success}")

        return success, is_valid
    
    def is_in_workplace(self,pos):
        is_in_workplace = True
        if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
            or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
            is_in_workplace = False 
        return is_in_workplace

    def push(self, push_action, target_obj, speed=0.005,push_distance=0.125):
        """
        push_action = pose 4x4
        push_distance is fixed.
        """
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )
        transform = push_action
        # move_distance = 0.0
        # current_poses = []
        # move_poses = []
        # for id in obj_list:
        #     current_pose = self.obj_info(id)
        #     current_poses.append(current_pose)
        obj_is_moved = True
        current_poses = self.obj_info(target_obj)
        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

            # 方法 1：修正 ee_tip_transform（推荐）
    
        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.001, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.15))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        #execute push action
        self.close_gripper()

        # position = ee_transform[:3, 3]
        # displacement = np.array([push_distance,0,0])
        Rotation_grasp = ee_transform[:3,:3]
        # world_displacement = Rotation_grasp @ displacement
        # new_position = position + world_displacement
        # base_T_new = np.eye(4)
        # base_T_new[:3, 3] = new_position
        # base_T_new[:3, :3] = ee_transform[:3, :3]

        grasp_x = -Rotation_grasp[:3, 2]
        end_pos = pos + push_distance * grasp_x.T 
        end_pos[2] = ee_transform[2, 3]
        # T_relative = np.eye(4)
        # T_relative[0, 3] = push_distance  # 向 X 轴正方向平移

        # # 末端坐标在 base 中的目标位姿 = 当前 base->ee * ee->目标（相对）
        # T_target = ee_transform @ T_relative

        # 提取位置与姿态
        # end_pose = T_target[:3, 3]
        # end_rot = T_target[:3, :3]
        # grasp_x_proj = np.array([grasp_x[0], grasp_x[1]])
        # grasp_x_proj = grasp_x_proj / np.linalg.norm(grasp_x_proj)

        # push_angle = np.arctan2(grasp_x_proj[1], grasp_x_proj[0])
        # push_angle = np.degrees(push_angle)
        # if push_angle < 0:
        #     push_angle += 360.0

        # push_vec = np.array([np.cos(push_angle), np.sin(push_angle), 0.0])
        # end_pos = [pos[0] + push_distance * 4 ,
        #     pos[1] ,
        #     pos[2]]

        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot), speed)
            for _ in range(5):
                pb.stepSimulation()
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
            for _ in range(10):
                pb.stepSimulation()
        if success:
            success = self.straight_move(pos, end_pos, rot, speed=0.003, detect_force=False)
            for _ in range(10):
                pb.stepSimulation()
        
        # for id in obj_list:
        #     move_pose = self.obj_info(id)
        #     move_poses.append(move_pose)
        move_pose = self.obj_info(target_obj)
        success =True
        # for i in range(len(obj_list)):
        #     move_distance += np.linalg.norm(np.array(move_poses[i][0]) - np.array(current_poses[i][0]))
            # if not self.is_in_workplace(move_poses[i][0]):
            #     success = False
            #     break
        move_position = np.linalg.norm(np.array(move_pose[0]) - np.array(current_poses[0]))
        
        q1 = current_poses[1] / np.linalg.norm(current_poses[1])
        q2 = move_pose[1] / np.linalg.norm(move_pose[1])
        # 计算点积（四元数内积）
        dot_product = np.dot(q1, q2)
    
        # 由于四元数q和-q表示相同旋转，取绝对值确保得到最小角度
        dot_product = np.clip(np.abs(dot_product), 0.0, 1.0)
    
        # 计算角度差（弧度）
        move_angle = 2 * np.arccos(dot_product)

        if (move_position < 0.005) and (move_angle < 0.01):
            obj_is_moved = False
        self.go_home()
    
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success, obj_is_moved
    
    def save_state(self):
        self.state_id = pb.saveState()
        return self.state_id
    
    def load_state(self, state_id):
        pb.restoreState(stateId=state_id)

    # def set_gripper_width(self, width_m):
    #     """
    #     将 Robotiq 2F-85 夹爪张口设置为指定宽度（两指尖间距，单位：米）。

    #     width_m: 期望宽度，例如 0.04 = 40mm
    #     is_slow: 是否用慢速接近（一般 width 精确控制时用 False）
    #     """
    #     width_m = float(np.clip(width_m, self.min_width, self.max_width))

    #     target_angle = float(np.interp(width_m, self._width_samples, self._q_samples))

    #     self._move_gripper(target_angle, is_slow=True)

    def get_center_of_obj(self, num_id):
        """
        obj_position based on world base.
        """
        position, _ = pb.getBasePositionAndOrientation(num_id)
        return position