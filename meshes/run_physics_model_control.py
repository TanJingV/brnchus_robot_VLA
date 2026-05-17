import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import os
import math

# ================= 配置区域 =================
XML_PATH = "meshes\cable_robot_bronch_final_seg2.xml"  # 请确保文件名正确
CAMERA_NAME = "tip_camera"

# --- 物理引擎稳定性设置 ---
# 强制使用 implicitfast 以适应高刚度绳索
NEW_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST 
NEW_TIMESTEP = 0.001
STEPS_PER_RENDER = 100

# --- 物理参数 (用于UI显示理论参考) ---
# 镍钛合金 Seg (r=1.75mm) 计算出的 EI ≈ 0.55 Nm^2
THEO_EI_NITINOL = 0.55 
SEG_LENGTH = 0.027     # 27mm

# --- 控制参数 ---
MAX_TORQUE = 1.0       # 最大力矩 1.0 Nm
COMPASS_RADIUS = 130   # 罗盘半径
TORQUE_SMOOTH = 0.15   # 力矩平滑系数 (防止突变)
# ===========================================

def quat_diff_angle(q1, q2):
    """
    计算两个四元数之间的相对旋转角度 (输出: 弧度)
    用于计算 Seg1 和 Seg2 的真实弯曲
    """
    # MuJoCo quaternion is [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # 两个单位四元数的相对角度 theta = 2 * arccos( |q1 . q2| )
    dot = w1*w2 + x1*x2 + y1*y2 + z1*z2
    
    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(abs(dot)) 
    return angle

class BronchoBotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # # 1. [强制] 物理引擎稳定性覆盖
        self.model.opt.integrator = NEW_INTEGRATOR
        self.model.opt.timestep = NEW_TIMESTEP
        self.model.opt.impratio = 100 # 优先保证焊接约束不分离
        
        # 2. 关键刚体 ID (用于闭环几何测量)
        # 结构: Base -> Seg1 -> Seg2 -> Tip
        self.body_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_plate")
        self.body_mid_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seg2_body") # Seg1 End / Seg2 Start
        self.body_tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_6")     # Seg2 End
        
        self.slider_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_slid_M")
        
        # 3. 绳索与分段定义
        self.segments = [
            # Seg1 (近端): 由 s1, s3, s5 驱动
            {'name': 'Seg1 (Proximal)', 'sites': ['s1', 's3', 's5'], 
             'ctrl_vec': np.array([0., 0.]), 'torque_mag': 0.0, 'real_angle': 0.0},
            
            # Seg2 (远端): 由 s2, s4, s6 驱动
            {'name': 'Seg2 (Distal)',   'sites': ['s2', 's4', 's6'], 
             'ctrl_vec': np.array([0., 0.]), 'torque_mag': 0.0, 'real_angle': 0.0}
        ]
        
        self.cables = []
        print("[System] Mapping Cables & Sites...")
        for seg_idx, seg in enumerate(self.segments):
            for s_name in seg['sites']:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s_name)
                if sid != -1:
                    self.cables.append({
                        'id': sid,
                        'name': s_name,
                        'seg_idx': seg_idx,
                        'torque': np.zeros(3) # 当前施加的力矩向量
                    })
                    print(f"  - Mapped {s_name} to {seg['name']}")

        # 4. UI 初始化
        self.win_name = "Bronchoscope Control System"
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        
        # UI 布局坐标
        self.c1_center = (200, 480) 
        self.c2_center = (600, 480) 
        self.dragging = None 
        self.slider_val = 0.0

    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标拖拽罗盘"""
        if event == cv2.EVENT_LBUTTONDOWN:
            d1 = math.hypot(x - self.c1_center[0], y - self.c1_center[1])
            d2 = math.hypot(x - self.c2_center[0], y - self.c2_center[1])
            
            if d1 < COMPASS_RADIUS + 20: self.dragging = 0
            elif d2 < COMPASS_RADIUS + 20: self.dragging = 1
            elif y > 680: self.dragging = 2 # 底部滑条区域
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging == 0:
                dx = x - self.c1_center[0]
                dy = -(y - self.c1_center[1]) # Screen Y is down
                self.segments[0]['ctrl_vec'] = np.array([dx, dy]) / COMPASS_RADIUS
            elif self.dragging == 1:
                dx = x - self.c2_center[0]
                dy = -(y - self.c2_center[1])
                self.segments[1]['ctrl_vec'] = np.array([dx, dy]) / COMPASS_RADIUS
            elif self.dragging == 2:
                self.slider_val = np.clip((x - 50) / 700.0, 0.0, 1.0)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def measure_real_angles(self):
        """
        [闭环反馈] 读取四元数计算真实的物理弯曲角度
        """
        if self.body_base_id == -1 or self.body_mid_id == -1 or self.body_tip_id == -1: return

        q_base = self.data.xquat[self.body_base_id]
        q_mid  = self.data.xquat[self.body_mid_id]
        q_tip  = self.data.xquat[self.body_tip_id]
        
        # Seg1 角度 = Base 与 Mid 之间的相对旋转
        self.segments[0]['real_angle'] = np.degrees(quat_diff_angle(q_base, q_mid))
        
        # Seg2 角度 = Mid 与 Tip 之间的相对旋转 (局部解耦)
        self.segments[1]['real_angle'] = np.degrees(quat_diff_angle(q_mid, q_tip))

    def update_control(self):
        """计算力矩并进行平滑处理"""
        # 1. 进给电机控制
        self.data.ctrl[self.slider_act] = self.slider_val * 0.577
        
        # 2. 弯曲力矩计算
        for i, seg in enumerate(self.segments):
            ctrl = seg['ctrl_vec'] # 原始输入向量
            
            # --- [修改] 范围限制逻辑 ---
            # Seg1 范围 1.0, Seg2 范围 0.7
            limit = 0.7 if i == 1 else 1.0
            
            mag = np.linalg.norm(ctrl)
            
            # 创建生效的控制向量 (Clamped)
            ctrl_clamped = ctrl.copy()
            if mag > limit:
                # 保持方向，缩放幅度到 limit
                ctrl_clamped = ctrl * (limit / mag)
                mag = limit # 更新幅值用于显示
            # --------------------------
            
            # 计算总力矩标量
            total_moment = mag * MAX_TORQUE
            seg['torque_mag'] = total_moment
            
            # 映射方向: 
            # 使用经过限制的 ctrl_clamped 计算物理力矩
            torque_y = -ctrl_clamped[1] * MAX_TORQUE
            torque_z =  ctrl_clamped[0] * MAX_TORQUE
            
            target_torque_global = np.array([0.0, torque_y, torque_z])
            
            # 分配给该段的所有 Site (平均分配)
            site_torque = target_torque_global / 3.0
            
            for c in self.cables:
                if c['seg_idx'] == i:
                    # 低通滤波平滑，防止物理突变
                    c['torque'] = (1 - TORQUE_SMOOTH) * c['torque'] + TORQUE_SMOOTH * site_torque

    def apply_physics(self):
        """将计算好的力矩施加到仿真中"""
        for c in self.cables:
            t = c['torque']
            if np.linalg.norm(t) <= 1e-6: continue
            
            sid = c['id']
            # 直接在 Site 施加纯力矩 (Torque)
            mujoco.mj_applyFT(self.model, self.data, 
                              np.zeros(3),  # Force
                              t,            # Torque
                              self.data.site_xpos[sid], 
                              self.model.site_bodyid[sid], 
                              self.data.qfrc_applied)

    def draw_ui(self, cam_img=None):
        # 画布尺寸 750x800
        H, W = 750, 800
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # --- 1. 内窥镜画面 (顶部居中) ---
        if cam_img is not None:
            # 缩放画面
            disp_w, disp_h = 320, 240
            small_cam = cv2.resize(cam_img, (disp_w, disp_h))
            
            # 居中放置
            sx = (W - disp_w) // 2
            sy = 30
            img[sy:sy+disp_h, sx:sx+disp_w] = small_cam
            
            # 边框和标题
            cv2.rectangle(img, (sx, sy), (sx+disp_w, sy+disp_h), (100, 100, 100), 2)
            cv2.putText(img, "Bronchoscope View (Tip)", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- 2. 控制罗盘 ---
        for i, center in enumerate([self.c1_center, self.c2_center]):
            # 背景圆
            cv2.circle(img, center, COMPASS_RADIUS, (40, 40, 40), -1)
            cv2.circle(img, center, COMPASS_RADIUS, (120, 120, 120), 2)
            
            # --- [新增] Seg2 的限制范围指示圆 ---
            if i == 1: # Seg2
                limit_radius = int(COMPASS_RADIUS * 0.7)
                cv2.circle(img, center, limit_radius, (80, 80, 80), 1) # 灰色线圈
            # ----------------------------------

            # 名称
            cv2.putText(img, self.segments[i]['name'], (center[0]-70, center[1]-COMPASS_RADIUS-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # 获取数据
            M = self.segments[i]['torque_mag']
            real_ang = self.segments[i]['real_angle']
            
            # 理论估算
            theo_ang = 0.0
            if THEO_EI_NITINOL > 0:
                theo_ang = np.degrees(M * SEG_LENGTH / THEO_EI_NITINOL) * 5.0 
            
            # 显示文本
            cv2.putText(img, f"Torque: {M:.2f} Nm", (center[0]-90, center[1]+COMPASS_RADIUS+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            
            cv2.putText(img, f"Real Ang: {real_ang:.1f} deg", (center[0]-90, center[1]+COMPASS_RADIUS+70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(img, f"Ref(NiTi): ~{theo_ang:.1f}", (center[0]-90, center[1]+COMPASS_RADIUS+95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            # 绘制摇杆
            ctrl_vec = self.segments[i]['ctrl_vec']
            px = int(center[0] + ctrl_vec[0] * COMPASS_RADIUS)
            py = int(center[1] - ctrl_vec[1] * COMPASS_RADIUS)
            
            cv2.line(img, center, (px, py), (0, 255, 255), 2)
            cv2.circle(img, (px, py), 12, (0, 120, 255), -1)

        # --- 3. 底部进给滑条 ---
        bar_y = 700
        cv2.line(img, (50, bar_y), (750, bar_y), (80, 80, 80), 6)
        sx = int(50 + self.slider_val * 700)
        cv2.circle(img, (sx, bar_y), 15, (0, 255, 0), -1)
        cv2.putText(img, f"Insertion Depth: {self.slider_val*100:.0f}%", (320, bar_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow(self.win_name, img)

def main():
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file '{XML_PATH}' not found!")
        return

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load XML: {e}")
        return

    # 初始化渲染器
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    
    # 初始化控制器
    ctrl = BronchoBotController(model, data)
    
    print("=== 支气管镜机器人仿真系统启动 ===")
    print(f"Loading Model: {XML_PATH}")
    print(f"Seg2 Limit: 0.7 * MaxTorque")
    print("Controls: Drag Compasses for Bending, Slider for Insertion.")
    
    # 启动被动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ctrl.update_control()
            
            for _ in range(STEPS_PER_RENDER):
                data.qfrc_applied[:] = 0 
                ctrl.apply_physics()     
                mujoco.mj_step(model, data)
            
            ctrl.measure_real_angles()
            
            cam_img = None
            if cam_id != -1:
                renderer.update_scene(data, camera=CAMERA_NAME)
                rgb = renderer.render()
                cam_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            viewer.sync()
            ctrl.draw_ui(cam_img)
            
            if cv2.waitKey(1) == 27: 
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()