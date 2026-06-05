import sys
import os
import time
from scipy.spatial import cKDTree
from queue import Queue, Empty
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox,QTabWidget, QLabel, QPushButton,QSplitter,QWidget,QVBoxLayout,QComboBox,QHBoxLayout,QLineEdit,QSizePolicy
from pyvistaqt import QtInteractor
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5 import QtGui
import torch
from unet import Unet
from scipy.spatial.transform import Rotation as R
import pyvista as pv
import serial.tools.list_ports
from queue import Queue
from datetime import datetime
import numpy as np
from NDITrackerWrapper import NDITrackerWrapper
from compute_rigid_transform import compute_rigid_transform
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
import pyiges
from pathplan import pathplan
import vtk
import math
import Trio_UnifiedApi as TUA
import cv2
from torchvision import transforms
from PIL import Image
import pyqtgraph as pg
from collections import deque
def EventHandler(et, ival, sval):
    if et == TUA.EventType.Error or et == TUA.EventType.Warning:
        print("MC Error: (%x) %s" % (ival, sval))
    elif et == TUA.EventType.Message:
        print("MC Message:", sval)

def create_mc_connection(ip: str):
    key = ip.upper()
    if key == "PCMCAT":
        return TUA.TrioConnectionPCMCAT(EventHandler)
    elif key == "FLEX7":
        return TUA.TrioConnectionFlex7(EventHandler)
    else:
        return TUA.TrioConnectionTCP(EventHandler, ip)

class DebugImageWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("深度图 & Mask 调试")
        self.resize(800, 400)  # 变宽一点，准备显示两张图

        # 左边放深度图，右边放 Mask
        self.layout = QHBoxLayout(self)
        self.label_depth = QLabel("深度图")
        self.label_mask = QLabel("Mask")

        self.label_depth.setAlignment(Qt.AlignCenter)
        self.label_mask.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.label_depth)
        self.layout.addWidget(self.label_mask)

    def update_image(self, mask_arr, depth_arr=None):
        # 1. 显示 Mask (右边)
        if mask_arr is not None:
            h, w = mask_arr.shape
            img_mask = QImage(mask_arr.data, w, h, w, QImage.Format_Grayscale8)
            self.label_mask.setPixmap(QPixmap.fromImage(img_mask).scaled(
                self.label_mask.size(), Qt.KeepAspectRatio))

        # 2. 显示深度图 (左边) - 关键调试手段
        if depth_arr is not None:
            # 深度图是 float，范围可能是 0~60
            # 我们要把它归一化到 0~255 才能显示
            # 越深越亮，越浅越暗
            d_min = np.min(depth_arr)
            d_max = np.max(depth_arr)
            #print(f"热力图范围: 蓝(Min)={d_min:.2f} mm, 红(Max)={d_max:.2f} mm")
            if d_max - d_min < 1e-3: d_max = d_min + 1.0

            # 归一化
            depth_norm = ((depth_arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)

            # 为了看得更清楚，应用伪彩色 (热力图)
            # 蓝色=近(浅)，红色=远(深)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            h, w, ch = depth_color.shape
            img_depth = QImage(depth_color.data, w, h, ch * w, QImage.Format_BGR888)
            self.label_depth.setPixmap(QPixmap.fromImage(img_depth).scaled(
                self.label_depth.size(), Qt.KeepAspectRatio))
class CentroidOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None, color=QColor(255, 0, 0, 200)):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._centroids = []  # 存储 [(u1, v1), (u2, v2)...]
        self.color = color  # 保存颜色

    def set_centroids(self, points_list):
        """points_list: [(u, v), ...] 归一化坐标列表"""
        self._centroids = points_list if points_list else []
        self.update()

    def paintEvent(self, event):
        if not self._centroids:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.color)  # 红色

        w = self.width()
        h = self.height()
        r = max(4, min(w, h) * 0.015)  # 半径

        # 循环绘制所有点
        for (u, v) in self._centroids:
            x = u * w
            y = v * h
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)

            # 可选：画个编号
            # painter.setPen(Qt.white)
            # painter.drawText(QtCore.QPointF(x+5, y), "Target")
            # painter.setPen(Qt.NoPen)

        painter.end()


class NavigationOverlay(QtWidgets.QWidget):
    """
    【新增】专门用来画导航指引（方框、十字、文字）的层
    用途：只用在实际视角，显示虚实配准的结果
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)

        self.target_uv = None  # 目标坐标 (0~1)
        self.status = "LOST"  # LOCKED, GHOST, LOST

    def update_status(self, target_uv, status):
        self.target_uv = target_uv
        self.status = status
        self.update()

    def paintEvent(self, event):
        # 如果没有目标或者丢失状态，就不画或者画LOST
        if self.status == "LOST" or self.target_uv is None:
            # 可以选择画一个红色的 "LOST"
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        tx, ty = self.target_uv[0] * w, self.target_uv[1] * h

        if self.status == "LOCKED":
            # ====== 锁定状态：绿色实线框 + 十字 ======
            pen = QPen(QColor(0, 255, 0))  # 绿色
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            s = 40  # 框大小
            # 画框
            painter.drawRect(int(tx - s / 2), int(ty - s / 2), s, s)
            # 画十字
            painter.drawLine(int(tx), int(ty - s / 2), int(tx), int(ty + s / 2))
            painter.drawLine(int(tx - s / 2), int(ty), int(tx + s / 2), int(ty))

            # 文字
            painter.drawText(int(tx + s / 2) + 5, int(ty), "LOCKED")

        elif self.status == "GHOST":
            # ====== 幽灵状态：黄色虚线框 ======
            pen = QPen(QColor(255, 255, 0))  # 黄色
            pen.setWidth(3)
            pen.setStyle(Qt.DashLine)  # 虚线关键设置
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            s = 60  # 框大一点，表示不确定
            painter.drawRect(int(tx - s / 2), int(ty - s / 2), s, s)

            # 文字
            painter.setPen(QColor(255, 255, 0))  # 文字不需要虚线
            painter.drawText(int(tx + s / 2) + 5, int(ty), "GHOST")

        painter.end()

class SegmentationWorker(QtCore.QObject):
    # 修改信号：
    # mask: 完整的二值图（用于调试显示）
    # centroids: 一个列表，包含多个元组 [(u1, v1), (u2, v2), ...]
    resultReady = QtCore.pyqtSignal(object, list)
    finished = QtCore.pyqtSignal()

    def __init__(self, weights_path, frame_queue, img_size=400, parent=None):
        super().__init__(parent)
        self.weights_path = weights_path
        self.frame_queue = frame_queue
        self.img_size = img_size
        self._running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.model = None

    @QtCore.pyqtSlot()
    def process_loop(self):
        # ... (模型加载代码保持不变) ...
        if self.model is None:
            try:
                print(f"正在加载模型: {self.weights_path} ...")
                # 确保你导入了 Unet 类 (from unet import Unet)
                self.model = Unet(3, 1)
                # 加载权重
                state = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print("模型加载成功！开始推理循环...")
            except Exception as e:
                print(f"【严重错误】模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.finished.emit()
                return

        self._running = True

        while self._running:
            start_time = time.time()  # 记录开始时间

            try:
                # 1. 获取数据
                frame_bgr = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                # 2. 推理与分析 (保持你现在的代码不变)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                inp = self.transform(img_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(inp)
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

                mask = (prob > 0.05).astype(np.uint8) * 255

                # 连通域分析
                num_labels, labels, stats, centroids_data = cv2.connectedComponentsWithStats(mask, connectivity=8)
                found_centroids = []
                h_img, w_img = mask.shape
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 20: continue
                    cx, cy = centroids_data[i]
                    found_centroids.append((cx / w_img, cy / h_img))

                # 3. 发送结果
                self.resultReady.emit(mask, found_centroids)

            except Exception as e:
                print(f"Error: {e}")

            # 4. 【关键】智能休眠，限制帧率
            # 假设我们限制最大 FPS = 20 (即每帧至少间隔 0.05秒)
            elapsed = time.time() - start_time
            sleep_time = 0.05 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.finished.emit()

    def stop(self):
        self._running = False


class SquareInteractorContainer(QtWidgets.QWidget):
    """
    外壳控件：自己可随布局拉伸，
    内部 QtInteractor 始终正方形。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.interactor = QtInteractor(self)
        self.interactor.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                      QtWidgets.QSizePolicy.Fixed)

        # 外壳本身可拉伸
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        side = min(w, h)
        x = (w - side) // 2
        y = (h - side) // 2

        self.interactor.setGeometry(x, y, side, side)
        super().resizeEvent(event)


class SquareLabelContainer(QtWidgets.QWidget):
    """
    【自定义控件】
    功能：不管外部布局怎么拉伸，内部的 self.label 始终保持正方形，并居中显示。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # 让容器本身尽可能填满布局
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        # 内部真正的显示控件
        self.label = QLabel("摄像头未启动", self)
        self.label.setAlignment(Qt.AlignCenter)
        # 设置黑色背景，这样留白的地方就是黑色的（看起来像电影宽银幕）
        self.label.setStyleSheet("background-color: black;")

    def resizeEvent(self, event):
        """核心魔法：每次窗口大小改变，手动计算正方形的位置"""
        w = self.width()
        h = self.height()
        side = min(w, h)  # 取短边作为正方形边长

        # 计算居中坐标
        x = (w - side) // 2
        y = (h - side) // 2

        # 强制设置 Label 的大小和位置
        self.label.setGeometry(x, y, side, side)

        super().resizeEvent(event)
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,  data_queue: Queue):
        super().__init__()
        self._is_closing = False
        self.setWindowTitle("自动介入")
        self.data_queue = data_queue
        self.intrinsic_camera_matrix = np.array([[229.59784258, 0, 200.52304737],
                                                 [0, 227.66142868, 166.61526572],
                                                 [0, 0, 1]])
        self.distortion_coeffs = np.array([-0.31049457, 0.40897277, -0.00838951, 0.00158787, -0.38224321])
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(50)
        self.position = []
        self.direction = []
        self.R = []
        self.t = []
        self.scope_pos = None  # 镜子位置 (Port 1 / Handle 10)
        self.scope_dir = None  # 镜子姿态
        self.ref_pos = None  # 探针位置 (Port 2 / Handle 11)
        self.virtual_position = []
        self.all_points = []
        self.smoothpath = []
        self.endpoint = [50.776901, 144.23911, 39.526905]
        self.ready_control = False
        self.axis_history_len = 500
        # 创建 6 个队列，对应 6 个轴
        self.axis_histories = [deque(maxlen=self.axis_history_len) for _ in range(6)]
        # 初始化为 0，防止刚开始空图
        for h in self.axis_histories:
            for _ in range(self.axis_history_len):
                h.append(0.0)
        # ======================================

        self.cap = None
        self.cap = None  # 视频捕获对象
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)

        # ====== 分割相关 ======
        self.seg_thread = None
        self.seg_worker = None
        self.seg_running = False
        self.setup_ui()
        self.virt_opening_actor = None   # 第一次调用时创建
        self.virt_opening_poly = None    # 对应的 PolyData，用来更新点坐标
        # ====== 【新增】控制器连接状态标志 ======
        self.is_mc_connected = False
        self.has_reached_target = False
        # =======================================================
        # ====== 【新增】 轨迹与电机数据记录相关变量初始化 ======
        # =======================================================
        self.is_recording_trajectory = False  # 记录开关标志
        self.trajectory_data = []  # 存储所有数据 [(t, vx, vy, vz, ax0, ..., ax6), ...]
        self.recording_start_time = 0.0  # 记录开始的时间戳
        self.save_dir = os.path.join(os.getcwd(), "实际轨迹")  # 保存路径
        self.real_video_writer = None  # 真实 Mask 视频写入器
        self.virt_video_writer = None  # 虚拟 Mask + Depth 视频写入器
        # 缓存当前 7 个轴的位置，默认初始化为 0.0
        self.current_axis_values = [0.0] * 7

        # 预先创建文件夹
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                print(f"创建轨迹保存文件夹: {self.save_dir}")
            except Exception as e:
                print(f"无法创建文件夹: {e}")
    def setup_ui(self):
        # 创建一个水平分割器，左边显示 PyVista 窗口，右边放置按钮
        splitter =QSplitter(QtCore.Qt.Horizontal)

        # 左侧：PyVista 交互窗口
        self.plotter = QtInteractor(self)
        self.add_example_mesh()
        splitter.addWidget(self.plotter)



        # 右侧：按钮面板
        tab_widget = QTabWidget()
        # Tab 1：可以放置一些控件，比如按钮
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setAlignment(QtCore.Qt.AlignTop)
        tab_widget.addTab(tab1, "标定")
        #com口
        self.com_label = QtWidgets.QLabel("选择 COM 口:")
        self.comComboBox = QComboBox()
        self.refresh_ports()
        tab1_layout.addWidget(self.com_label)
        tab1_layout.addWidget(self.comComboBox)

        #connection
        self.ndi_connect_button = QPushButton("Connect")
        self.ndi_connect_button.clicked.connect(self.toggle_connection)
        tab1_layout.addWidget(self.ndi_connect_button)

        # 左侧：大按钮，用于触发点选择操作
        group_box = QtWidgets.QGroupBox("点选择")
        group_layout = QtWidgets.QHBoxLayout(group_box)
        self.select_points_button = QPushButton("选择点")
        # 这里设置一个较大的最小尺寸，你可以根据需要调整
        self.select_points_button.setMinimumSize(60, 100)
        self.select_points_button.clicked.connect(self.select_points)
        group_layout.addWidget(self.select_points_button)

        # 右侧：3×3 的网格，用来显示三个点的 X, Y, Z 坐标
        grid_widget = QWidget()
        grid_layout = QVBoxLayout(grid_widget)  # 或者使用 QGridLayout 更为灵活

        # 使用 QGridLayout 排列成 3 行 3 列
        coord_grid = QtWidgets.QGridLayout()
        self.coord_labels = []  # 用于保存标签引用，便于后续更新

        # 循环创建 3 行（3个点）× 3 列（X, Y, Z 坐标）
        for i in range(3):
            row_labels = []
            for j, coord in enumerate(["X", "Y", "Z"]):
                label = QLabel(f"P{i + 1}_{coord}: N/A")
                coord_grid.addWidget(label, i, j)
                row_labels.append(label)
            self.coord_labels.append(row_labels)

        # 将网格布局放入一个 QWidget 中
        grid_container = QWidget()
        grid_container.setLayout(coord_grid)
        group_layout.addWidget(grid_container)

        # 将 Group Box 添加到 Tab1 的布局中
        tab1_layout.addWidget(group_box)

        # 添加第二个 Group Box，结构类似，只不过左侧为三个垂直排列的按钮，右侧仍为 3×3 的网格显示数字
        group_box2 = QtWidgets.QGroupBox("电磁定位点")
        group_layout2 = QtWidgets.QHBoxLayout(group_box2)

        # 左侧：三个垂直排列的按钮
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        self.button1 = QPushButton("记录标记点1")
        self.button2 = QPushButton("记录标记点2")
        self.button3 = QPushButton("记录标记点3")
        self.button1.clicked.connect(self.record_points_1)
        self.button2.clicked.connect(self.record_points_2)
        self.button3.clicked.connect(self.record_points_3)
        button_layout.addWidget(self.button1)
        button_layout.addWidget(self.button2)
        button_layout.addWidget(self.button3)
        group_layout2.addWidget(button_container)

        # 右侧：3×3 网格，用来显示数字（或其它信息）
        coord_grid2 = QtWidgets.QGridLayout()
        self.coord_labels2 = []  # 保存新组标签引用
        # 这里以 A, B, C 表示列标题，你也可以根据需要修改
        for i in range(3):
            row_labels = []
            for j, label_text in enumerate(["X", "Y", "Z"]):
                label = QLabel(f"P{i + 1}_{label_text}: 0")
                coord_grid2.addWidget(label, i, j)
                row_labels.append(label)
            self.coord_labels2.append(row_labels)
        grid_container2 = QWidget()
        grid_container2.setLayout(coord_grid2)
        group_layout2.addWidget(grid_container2)

        # 将第二个 Group Box 添加到 Tab1 的布局中
        tab1_layout.addWidget(group_box2)
        #配准，计算矩阵
        self.comput_button = QPushButton("配准")
        self.comput_button.clicked.connect(self.Kabsch_computer)
        tab1_layout.addWidget(self.comput_button)
        coord_display_layout = QHBoxLayout()
        self.actual_point_label = QLabel("实际点:")
        self.actual_point_label.setAlignment(QtCore.Qt.AlignCenter)
        coord_display_layout.addWidget(self.actual_point_label)
        self.x_label = QLabel("X: 0.00")
        self.y_label = QLabel("Y: 0.00")
        self.z_label = QLabel("Z: 0.00")

        # 可选：设置标签对齐方式或最小宽度，保证显示效果
        self.x_label.setAlignment(QtCore.Qt.AlignCenter)
        self.y_label.setAlignment(QtCore.Qt.AlignCenter)
        self.z_label.setAlignment(QtCore.Qt.AlignCenter)

        coord_display_layout.addWidget(self.x_label)
        coord_display_layout.addWidget(self.y_label)
        coord_display_layout.addWidget(self.z_label)

        # 将这个水平布局添加到 tab1_layout 末尾
        tab1_layout.addLayout(coord_display_layout)
        coord_display_layout2 = QHBoxLayout()

        self.actual_point_label2 = QLabel("虚拟点:")
        self.actual_point_label2.setAlignment(QtCore.Qt.AlignCenter)
        coord_display_layout2.addWidget(self.actual_point_label2)
        self.xx_label = QLabel("X: 0.00")
        self.yy_label = QLabel("Y: 0.00")
        self.zz_label = QLabel("Z: 0.00")

        # 可选：设置标签对齐方式或最小宽度，保证显示效果
        self.xx_label.setAlignment(QtCore.Qt.AlignCenter)
        self.yy_label.setAlignment(QtCore.Qt.AlignCenter)
        self.zz_label.setAlignment(QtCore.Qt.AlignCenter)

        coord_display_layout2.addWidget(self.xx_label)
        coord_display_layout2.addWidget(self.yy_label)
        coord_display_layout2.addWidget(self.zz_label)

        # 将这个水平布局添加到 tab1_layout 末尾
        tab1_layout.addLayout(coord_display_layout2)
        # 终点选择
        self.endpoint_button = QPushButton("选择导航终点")  # 改个名字更贴切
        self.endpoint_button.clicked.connect(self.select_endpoint_mode)  # <--- 连到新函数
        tab1_layout.addWidget(self.endpoint_button)
        #路径规划
        self.pathplan_button = QPushButton("路径规划")
        self.pathplan_button.clicked.connect(self.route_plan)
        tab1_layout.addWidget(self.pathplan_button)

       # Tab 2：另一个 Tab 页，放置其他控件，比如标签
        # Tab 2：控制
        tab2 = QtWidgets.QWidget()
        tab_widget.addTab(tab2, "控制")
        tab2_layout = QtWidgets.QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(10, 10, 10, 10)

        # ========= 相机选择栏（在虚实视角上方） =========
        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("选择相机:"))

        self.camera_combo = QComboBox()
        self.cameras = QCameraInfo.availableCameras()
        for cam in self.cameras:
            self.camera_combo.addItem(cam.description())
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        cam_select_layout.addWidget(self.camera_combo)
        # ====== 新增：开始/停止分割按钮 ======
        self.seg_button = QPushButton("开始分割")
        self.seg_button.setCheckable(True)
        self.seg_button.clicked.connect(self.toggle_segmentation)
        cam_select_layout.addWidget(self.seg_button)
        from PyQt5.QtWidgets import QCheckBox
        self.debug_cb = QCheckBox("显示Mask调试")
        self.debug_cb.setChecked(False)
        self.debug_cb.toggled.connect(self.on_debug_toggled)
        cam_select_layout.addWidget(self.debug_cb)

        self.show_all_centroids_cb = QCheckBox("显示所有红点")
        self.show_all_centroids_cb.setChecked(False)
        cam_select_layout.addWidget(self.show_all_centroids_cb)

        self.virtual_debug_cb = QCheckBox("显示虚拟Mask")
        self.virtual_debug_cb.setChecked(False)
        self.virtual_debug_cb.toggled.connect(self.on_virtual_debug_toggled)
        cam_select_layout.addWidget(self.virtual_debug_cb)

        self.record_data_cb = QtWidgets.QCheckBox("记录数据")  # 使用 QCheckBox
        self.record_data_cb.setChecked(False)
        self.record_data_cb.toggled.connect(self.toggle_recording)  # 连接到新函数
        cam_select_layout.addWidget(self.record_data_cb)

        cam_select_layout.addStretch()
        tab2_layout.addLayout(cam_select_layout)
        self.debug_window = None
        self.virt_debug_window = None  # 虚拟 Mask 窗口 (新增)
        self.enable_virtual_seg = False  # 虚拟分割的总开关 (默认关闭)
        # ========= 虚拟视角 + 实际视角（左右各一半） =========
        top_layout = QtWidgets.QHBoxLayout()
        virtual_view_container = QtWidgets.QWidget()
        virtual_view_layout = QtWidgets.QVBoxLayout(virtual_view_container)
        virtual_view_layout.setContentsMargins(0, 0, 0, 0)
        virtual_view_layout.setSpacing(2)

        # 用刚才定义的“正方形外壳”
        self.virtual_square = SquareInteractorContainer(virtual_view_container)

        # 以后所有 camera 操作、add_mesh 等，仍然用 self.virtual_view 这个 QtInteractor
        self.virtual_view = self.virtual_square.interactor

        # 和原来一样：在虚拟视角里加模型
        self.virtual_view.add_mesh(
            self.mesh,
            color="#c06c6c",  # 1. 设置为肉粉色
            specular=0.3,  # 2. 高光强度 (0~1)，模拟黏膜的湿润反光
            specular_power=20,  # 3. 高光范围 (数值越大反光点越集中，越有"水"的感觉)
            diffuse=0.3,  # 4. 漫反射强度 (控制固有色亮度)
            ambient=0.6,  # 5. 环境光 (防止暗部死黑)
            smooth_shading=True,  # 6. 平滑着色 (去掉网格棱角)
            opacity=1.0
        )
        self.virt_green_points = vtk.vtkPoints()
        self.virt_green_cells = vtk.vtkCellArray()
        # 创建一个 PolyData 数据集
        self.virt_green_poly = vtk.vtkPolyData()
        self.virt_green_poly.SetPoints(self.virt_green_points)
        self.virt_green_poly.SetVerts(self.virt_green_cells)
        # 创建 2D 映射器 (专门画 2D 图形)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self.virt_green_poly)

        # 创建 2D Actor
        self.virt_green_actor = vtk.vtkActor2D()
        self.virt_green_actor.SetMapper(mapper)

        # 设置颜色 (绿色) 和 大小
        self.virt_green_actor.GetProperty().SetColor(0, 1, 0)  # R=0, G=1, B=0
        self.virt_green_actor.GetProperty().SetPointSize(10)  # 点的大小

        # 将这个 2D Actor 加到渲染器里
        # 注意：add_actor 是 PyVista 的底层接口
        self.virtual_view.renderer.AddActor2D(self.virt_green_actor)
        # 1. 数据结构
        self.virt_box_points = vtk.vtkPoints()
        self.virt_box_cells = vtk.vtkCellArray()
        self.virt_box_poly = vtk.vtkPolyData()
        self.virt_box_poly.SetPoints(self.virt_box_points)
        self.virt_box_poly.SetLines(self.virt_box_cells)  # 注意这里用 SetLines，因为是画线框

        # 2. 映射器
        box_mapper = vtk.vtkPolyDataMapper2D()
        box_mapper.SetInputData(self.virt_box_poly)

        # 3. Actor
        self.virt_box_actor = vtk.vtkActor2D()
        self.virt_box_actor.SetMapper(box_mapper)

        # 4. 样式设置 (红色，线宽稍微粗一点)
        self.virt_box_actor.GetProperty().SetColor(1, 0, 0)  # R=1, G=0, B=0 (红色)
        self.virt_box_actor.GetProperty().SetLineWidth(3)  # 线宽 3

        # 5. 加入渲染器
        self.virtual_view.renderer.AddActor2D(self.virt_box_actor)
        self.debug_guide_points = vtk.vtkPoints()
        self.debug_guide_cells = vtk.vtkCellArray()
        self.debug_guide_poly = vtk.vtkPolyData()
        self.debug_guide_poly.SetPoints(self.debug_guide_points)
        self.debug_guide_poly.SetVerts(self.debug_guide_cells)

        mapper_blue = vtk.vtkPolyDataMapper2D()
        mapper_blue.SetInputData(self.debug_guide_poly)

        self.debug_guide_actor = vtk.vtkActor2D()
        self.debug_guide_actor.SetMapper(mapper_blue)
        self.debug_guide_actor.GetProperty().SetColor(0, 0, 1)  # 蓝色
        self.debug_guide_actor.GetProperty().SetPointSize(15)  # 大一点

        self.virtual_view.renderer.AddActor2D(self.debug_guide_actor)

        self.sync_virtual_camera_with_intrinsics()

        virtual_label = QtWidgets.QLabel("虚拟视角")
        virtual_label.setAlignment(Qt.AlignCenter)

        # 把外壳加到布局里，标签在下面
        virtual_view_layout.addWidget(self.virtual_square)
        virtual_view_layout.addWidget(virtual_label)

        # ---- 实际视角 ----
        actual_view_container = QtWidgets.QWidget()
        actual_view_layout = QtWidgets.QVBoxLayout(actual_view_container)
        actual_view_layout.setContentsMargins(0, 0, 0, 0)
        actual_view_layout.setSpacing(0)

        # 【核心修改】实例化自定义容器
        self.actual_square_container = SquareLabelContainer()

        # 【关键】把 self.actual_view 指向容器内部的 label
        # 这样，你的 update_video_frame 往 actual_view 里塞图片时，其实是塞进了这个正方形 Label 里
        self.actual_view = self.actual_square_container.label

        # 把容器加入布局
        actual_view_layout.addWidget(self.actual_square_container)
        # ====== 新增：质心叠加层（透明） ======
        self.centroid_overlay = CentroidOverlay(self.actual_view)
        self.centroid_overlay.raise_()
        self.nav_overlay = NavigationOverlay(self.actual_view)
        self.nav_overlay.raise_()
        actual_label = QtWidgets.QLabel("实际视角")
        actual_label.setAlignment(Qt.AlignCenter)
        actual_view_layout.addWidget(actual_label)

        top_layout.addWidget(virtual_view_container)
        top_layout.addWidget(actual_view_container)
        top_layout.setStretchFactor(virtual_view_container, 1)
        top_layout.setStretchFactor(actual_view_container, 1)

        tab2_layout.addLayout(top_layout,4)

        # ========= 下面保持你原来的 MC / WDOG / 轴位置布局 =========
        mc_layout = QHBoxLayout()
        mc_layout.addWidget(QLabel("MC IP:"))
        self.mc_ip_edit = QLineEdit()
        self.mc_ip_edit.setText("192.168.0.250")
        mc_layout.addWidget(self.mc_ip_edit)

        self.mc_connect_btn = QPushButton("连接 MC")
        self.mc_connect_btn.clicked.connect(self.on_mc_connect)
        mc_layout.addWidget(self.mc_connect_btn)

        self.mc_disconnect_btn = QPushButton("断开 MC")
        self.mc_disconnect_btn.setEnabled(False)
        self.mc_disconnect_btn.clicked.connect(self.on_mc_disconnect)
        mc_layout.addWidget(self.mc_disconnect_btn)

        tab2_layout.addLayout(mc_layout)

        wdog_layout = QHBoxLayout()
        wdog_layout.addWidget(QLabel("WDOG:"))

        self.wdog_enable_btn = QPushButton("使能")
        self.wdog_enable_btn.clicked.connect(self.on_wdog_enable)
        wdog_layout.addWidget(self.wdog_enable_btn)

        self.wdog_disable_btn = QPushButton("关闭使能")
        self.wdog_disable_btn.clicked.connect(self.on_wdog_disable)
        wdog_layout.addWidget(self.wdog_disable_btn)

        tab2_layout.addLayout(wdog_layout)

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("轴位置:"))

        self.axis_pos_labels = []
        for i in range(7):
            lbl = QLabel(f"轴{i}: N/A")
            lbl.setAlignment(Qt.AlignCenter)
            self.axis_pos_labels.append(lbl)
            axis_layout.addWidget(lbl)

        tab2_layout.addLayout(axis_layout)
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w')  # 设置背景为白色 (默认是黑)
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)
        self.graph_widget.setTitle("轴位置实时曲线", color="k", size="12pt")
        self.graph_widget.setLabel('left', 'Position', color='k')
        self.graph_widget.setLabel('bottom', 'Time', color='k')

        # 添加图例 (Legend)
        # 1. 获取图例对象 (加一个变量 self.legend 接住它)
        self.legend = self.graph_widget.addLegend(offset=(10, 10))

        # 2. 【核心】设置行间距为 0 (默认比较大，改成0就会紧凑很多)
        self.legend.layout.setVerticalSpacing(0)

        # 可选：如果你觉得边框留白也太大了，可以把边距也改小 (左, 上, 右, 下)
        self.legend.layout.setContentsMargins(5, 5, 5, 5)

        # 2. 初始化 6 条曲线
        self.curves = []
        # 定义 6 种颜色 (红, 绿, 蓝, 橙, 紫, 青)
        colors = [
            (255, 0, 0),  # 轴0: 红
            (0, 200, 0),  # 轴1: 绿
            (0, 0, 255),  # 轴2: 蓝
            (255, 165, 0),  # 轴3: 橙
            (128, 0, 128),  # 轴4: 紫
            (0, 200, 200)  # 轴5: 青
        ]

        for i in range(6):
            # 创建曲线对象
            pen = pg.mkPen(color=colors[i], width=2)
            curve = self.graph_widget.plot(name=f"Axis {i}", pen=pen)
            name_str = f'<span style="font-size: 9pt">Axis {i}</span>'
            self.curves.append(curve)

        # 3. 将绘图控件添加到布局底部
        # 使用 setStretch 让图表占据剩余的所有空白区域
        tab2_layout.addWidget(self.graph_widget)
        tab2_layout.setStretchFactor(self.graph_widget, 1)

        # =======================================================
        # 将分割器设置为主窗口的中心部件
        splitter.addWidget(tab_widget)
        self.setCentralWidget(splitter)
        if len(self.cameras) > 0:
            self.on_camera_changed(0)
    def on_mc_connect(self):
        ip = self.mc_ip_edit.text().strip()
        try:
            # 创建并打开连接
            self.mc_connection = create_mc_connection(ip)
            self.mc_connection.OpenConnection()
            self.is_mc_connected = True
            QMessageBox.information(self, "MC 连接", f"已成功连接到 {ip}")
            # 按钮状态更新
            self.mc_connect_btn.setEnabled(False)
            self.mc_disconnect_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "MC 连接失败", str(e))
            self.is_mc_connected = False
            if hasattr(self, "mc_connection"):
                try: self.mc_connection.CloseConnection()
                except: pass

    def on_mc_disconnect(self):
        try:
            if hasattr(self, "mc_connection"):
                self.mc_connection.CloseConnection()
                QMessageBox.information(self, "MC 断开", "已断开 MC 连接")
                self.is_mc_connected = False
        except Exception as e:
            QMessageBox.warning(self, "MC 断开异常", str(e))
        finally:
            # 按钮状态恢复
            self.is_mc_connected = False
            self.mc_connect_btn.setEnabled(True)
            self.mc_disconnect_btn.setEnabled(False)

    def on_wdog_enable(self):
        """使能 WDOG"""
        try:
            self.mc_connection.SetSystemParameter_WDOG(True)
            QMessageBox.information(self, "WDOG", "已使能 WDOG")
            self.wdog_enable_btn.setEnabled(False)
            self.wdog_disable_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "WDOG 错误", str(e))

    def on_wdog_disable(self):
        """关闭 WDOG"""
        try:
            self.mc_connection.SetSystemParameter_WDOG(False)
            QMessageBox.information(self, "WDOG", "已关闭 WDOG")
        except Exception as e:
            QMessageBox.critical(self, "WDOG 错误", str(e))
        finally:
            self.wdog_enable_btn.setEnabled(True)
            self.wdog_disable_btn.setEnabled(False)

    def toggle_recording(self, checked: bool):
        """
        控制是否记录轨迹数据 + 视频数据
        """
        if checked:
            # --- 1. 基础数据记录初始化 ---
            self.is_recording_trajectory = True
            self.trajectory_data = []
            self.recording_start_time = datetime.now().timestamp()  # 使用时间戳
            self.current_axis_values = [0.0] * 7

            # --- 2. 【新增】视频记录初始化 ---
            try:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                real_video_name = f"RealMask_{timestamp_str}.avi"
                virt_video_name = f"VirtualMask_{timestamp_str}.avi"
                real_path = os.path.join(self.save_dir, real_video_name)
                virt_path = os.path.join(self.save_dir, virt_video_name)

                # 定义编码器 (XVID 兼容性较好)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                real_fps = 20.0  # 实际分割跑得快
                virt_fps = 4.0  # 虚拟视角被分频了

                self.real_video_writer = cv2.VideoWriter(real_path, fourcc, real_fps, (400, 400))
                # 虚拟: Mask(400) + Depth(400) = 800宽
                self.virt_video_writer = cv2.VideoWriter(virt_path, fourcc, virt_fps, (800, 400))
                print(f">>> [Record] 视频录制开始")

            except Exception as e:
                print(f"初始化视频录制失败: {e}")
                self.real_video_writer = None
                self.virt_video_writer = None
        else:
            # --- 停止记录 ---
            self.is_recording_trajectory = False
            print(">>> [Record] 停止记录，正在保存...")
            # 计算这一段录制的总时长
            total_time = time.time() - self.recording_start_time
            # 计算总共录了多少帧 (通过 trajectory_data 的长度估算)
            total_frames = len(self.trajectory_data)

            if total_time > 0:
                actual_fps = total_frames / total_time
                print(f"========================================")
                print(f"实际录制时长: {total_time:.2f} 秒")
                print(f"实际录制帧数: {total_frames} 帧")
                print(f"建议将 real_fps 修改为: {actual_fps:.2f}")
                print(f"========================================")
            self.save_trajectory_to_file()

            if self.real_video_writer is not None:
                self.real_video_writer.release()
                self.real_video_writer = None
            if self.virt_video_writer is not None:
                self.virt_video_writer.release()
                self.virt_video_writer = None

    # def save_trajectory_to_file(self):
    #     """将记录的数据保存为 CSV 文件"""
    #     if not self.trajectory_data:
    #         return
    #     try:
    #         timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"{timestamp_str}.csv"
    #         file_path = os.path.join(self.save_dir, filename)

    #         import csv
    #         with open(file_path, mode='w', newline='', encoding='utf-8') as f:
    #             writer = csv.writer(f)
    #             headers = ["Time(s)", "Proj_X", "Proj_Y", "Proj_Z", "Raw_X", "Raw_Y", "Raw_Z"]
    #             headers += [f"Axis_{i}" for i in range(7)]
    #             writer.writerow(headers)
    #             writer.writerows(self.trajectory_data)

    #         QMessageBox.information(self, "记录完成", f"数据已保存:\n{filename}")
    #     except Exception as e:
    #         print(f"保存失败: {e}")

    def save_trajectory_to_file(self):
        """将记录的数据保存为 CSV 文件"""
        if not self.trajectory_data:
            return
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp_str}.csv"
            file_path = os.path.join(self.save_dir, filename)

            import csv
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # --- 表头修改 ---
                # 增加了 "Q0", "QX", "QY", "QZ" 四列姿态数据 (四元数)
                headers = [
                    "Time(s)", 
                    "Proj_X", "Proj_Y", "Proj_Z",  # 投影后位置
                    "Raw_X", "Raw_Y", "Raw_Z",     # 原始位置
                    "Q0", "QX", "QY", "QZ"         # 姿态 (w, x, y, z)
                ]
                
                # 追加电机轴数据的列名 (Axis_0 到 Axis_6)
                headers += [f"Axis_{i}" for i in range(7)]
                
                writer.writerow(headers)
                writer.writerows(self.trajectory_data)

            QMessageBox.information(self, "记录完成", f"数据已保存:\n{filename}")
        except Exception as e:
            print(f"保存失败: {e}")

    '''def update_axis_positions(self):
        """从控制器获取轴位置并更新标签和曲线"""

        # 1. 连接检查
        if not getattr(self, "is_mc_connected", False):
            return
        if not hasattr(self, "mc_connection"):
            return

        # ====== 【核心修改】将循环范围从 6 改为 7 ======
        for i in range(7):
            try:
                # 1. 获取位置 (读取所有 7 个轴的数据)
                pos = self.mc_connection.GetAxisParameter_DPOS(i)

                # 2. 更新文字标签 (所有 7 个轴都要更新)
                if i < len(self.axis_pos_labels):
                    self.axis_pos_labels[i].setText(f"轴{i}: {pos:.2f}")
                if hasattr(self, "current_axis_values") and i < len(self.current_axis_values):
                    self.current_axis_values[i] = pos
                # 3. 更新曲线数据 (仅限前 6 个轴)
                # 因为 axis_histories 和 curves 列表只初始化了 6 个
                if i < 6:
                    self.axis_histories[i].append(pos)
                    self.curves[i].setData(list(self.axis_histories[i]))

            except Exception:
                if i < len(self.axis_pos_labels):
                    self.axis_pos_labels[i].setText(f"轴{i}: ERR")'''

    def update_axis_positions(self):
        """
        从控制器获取轴位置并更新标签和曲线
        【修改版】实现轴 1,2,3 (虚拟) 跟随 轴 4,5,6 (真实)
        """

        # 1. 连接检查
        if not getattr(self, "is_mc_connected", False):
            return
        if not hasattr(self, "mc_connection"):
            return

        # ==========================================================
        # ====== 第一步：读取真实的电机位置 (轴 3, 4, 5) ======
        # ==========================================================
        # 对应关系 (假设 Python list index 从 0 开始):
        # 轴 3 -> Motor 4 (60度)
        # 轴 4 -> Motor 5 (180度)
        # 轴 5 -> Motor 6 (300度)

        real_indices = [3, 4, 5]

        # 先把真实的读出来存好
        for i in real_indices:
            try:
                pos = self.mc_connection.GetAxisParameter_DPOS(i)
                # 更新缓存
                if i < len(self.current_axis_values):
                    self.current_axis_values[i] = pos
                # 更新UI
                if i < len(self.axis_pos_labels):
                    self.axis_pos_labels[i].setText(f"轴{i}: {pos:.2f}")
                # 更新曲线
                if i < 6:
                    self.axis_histories[i].append(pos)
                    self.curves[i].setData(list(self.axis_histories[i]))
            except Exception:
                pass

        # ==========================================================
        # ====== 第二步：计算虚拟的电机位置 (轴 0, 1, 2) ======
        # ==========================================================
        # 对应关系:
        # 轴 0 -> Motor 1 (0度)   = M4 + M6
        # 轴 1 -> Motor 2 (120度) = M4 + M5
        # 轴 2 -> Motor 3 (240度) = M5 + M6

        # 从缓存中获取刚刚读到的值
        val_4 = self.current_axis_values[3]  # Axis 3 (60 deg)
        val_5 = self.current_axis_values[4]  # Axis 4 (180 deg)
        val_6 = self.current_axis_values[5]  # Axis 5 (300 deg)

        # 计算虚拟值
        sim_val_1 = val_4 + val_6  # Axis 0
        sim_val_2 = val_4 + val_5  # Axis 1
        sim_val_3 = val_5 + val_6  # Axis 2

        # 封装成列表方便循环更新
        sim_results = {0: sim_val_1, 1: sim_val_2, 2: sim_val_3}

        for i, sim_pos in sim_results.items():
            # 1. 更新缓存 (这很重要，因为数据记录CSV是读这个列表的)
            if i < len(self.current_axis_values):
                self.current_axis_values[i] = sim_pos

            # 2. 更新UI标签
            if i < len(self.axis_pos_labels):
                # 标记一下这是虚拟计算值 (Sim)
                self.axis_pos_labels[i].setText(f"轴{i}: {sim_pos:.2f}")

            # 3. 更新曲线
            if i < 6:
                self.axis_histories[i].append(sim_pos)
                self.curves[i].setData(list(self.axis_histories[i]))

        # 轴 6 (如果有第7个轴) 保持原样读取
        try:
            pos_6 = self.mc_connection.GetAxisParameter_DPOS(6)
            if 6 < len(self.current_axis_values):
                self.current_axis_values[6] = pos_6
            if 6 < len(self.axis_pos_labels):
                self.axis_pos_labels[6].setText(f"轴6: {pos_6:.2f}")
        except:
            pass
    def on_virtual_debug_toggled(self, checked):
        """控制虚拟 Mask 调试窗口的开关"""
        if checked:
            if self.virt_debug_window is None:
                self.virt_debug_window = DebugImageWindow()
                self.virt_debug_window.setWindowTitle("虚拟 Mask 调试")
            self.virt_debug_window.show()
        else:
            if self.virt_debug_window:
                self.virt_debug_window.hide()
    def on_debug_toggled(self, checked):
        if checked:
            if self.debug_window is None:
                self.debug_window = DebugImageWindow()
            self.debug_window.show()
        else:
            if self.debug_window:
                self.debug_window.hide()

    # [修改] 接收信号的槽函数
    @QtCore.pyqtSlot(object, list)
    def on_segmentation_result(self, mask, centroids):
        if not self.seg_running:
            return

        # 1. 准备数据：获取虚拟指引点 (蓝点)
        virtual_target_uv = self.get_path_lookahead_uv()

        # 2. 执行匹配算法 (获取 匹配索引、状态、目标位置)
        matched_idx = -1
        status = "LOST"
        target_display_uv = None

        if virtual_target_uv is not None:
            # perform_robust_matching 返回: best_idx, status, target_point_uv
            matched_idx, status, target_display_uv = self.perform_robust_matching(centroids, virtual_target_uv)
        else:
            status = "LOST"

        # 3. ====== 【核心修改】红点显示逻辑 ======
        points_to_draw = []

        if self.show_all_centroids_cb.isChecked():
            # A. 如果勾选了"显示所有"，那就把识别到的全画出来 (乱一点但信息全)
            points_to_draw = centroids
        else:
            # B. 如果没勾选，只显示被锁定的那个 (干净)
            if status == "LOCKED" and matched_idx != -1:
                # 只取那个匹配成功的红点
                points_to_draw = [centroids[matched_idx]]
            else:
                # 如果是 GHOST 或 LOST，说明没锁住任何点，那就一个红点也不画
                points_to_draw = []
        if getattr(self, "is_recording_trajectory", False) and self.real_video_writer is not None:
            if mask is not None:
                self.real_video_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        # 更新红点层
        self.centroid_overlay.set_centroids(points_to_draw)

        # 4. 更新导航框层 (绿框/黄框)
        # 这一层始终要更新，告诉医生现在是锁定还是丢失
        if hasattr(self, "nav_overlay"):
            self.nav_overlay.update_status(target_display_uv, status)

        # 5. 更新调试窗口 (保持不变)
        if self.debug_cb.isChecked() and self.debug_window and self.debug_window.isVisible():
            self.debug_window.update_image(mask)

    def on_camera_changed(self, index: int):
        """切换相机（OpenCV模式）"""
        # 1. 关闭旧设备
        self.video_timer.stop()
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # 2. 尝试打开新相机
        # 注意：Qt的index和OpenCV的index通常是一致的，如果不一致可能需要尝试 0, 1
        # cv2.CAP_DSHOW 是 Windows 下推荐的后端，启动更快
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if self.cap.isOpened():
            # 设置分辨率（可选，根据需要调整，越高越卡）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
            self.video_timer.start(30)  # 30ms 刷新一次，约 33 FPS
            print(f"相机 {index} 已通过 OpenCV 打开")
        else:
            self.actual_view.setText(f"无法打开相机 {index}")
            print(f"相机 {index} 打开失败")

    def add_example_mesh(self):
        self.mesh = pv.read("支气管.stl")
        self.plotter.add_mesh(self.mesh, color="lightgray", opacity=0.5)
        self.plotter.add_axes()  # 添加坐标轴显示
        self.plotter.reset_camera()

    def toggle_connection(self):
        """Handle connect/disconnect button click"""
        if self.ndi_connect_button.text() == "Connect":
            selected_port = self.comComboBox.currentData()
            if selected_port:
                try:
                    self.tracker_wrapper = NDITrackerWrapper()
                    self.tracker_wrapper.COM_PORT = selected_port
                    self.data_queue = self.tracker_wrapper.visualization_queue

                    if self.tracker_wrapper.initialize_tracker():
                        self.ndi_connect_button.setText("Disconnect")
                        self.comComboBox.setEnabled(False)
                        self.tracker_wrapper.start_tracking_thread()
                    else:
                        QMessageBox.critical(self, "Error", "Failed to initialize tracker")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Connection failed: {str(e)}")
        else:
            # Disconnect
            if self.tracker_wrapper:
                self.tracker_wrapper.stop_tracking()
                self.tracker_wrapper = None
            self.ndi_connect_button.setText("Connect")
            self.comComboBox.setEnabled(True)

    def refresh_ports(self):
        """Refresh available COM ports"""
        self.comComboBox.clear()
        ports = list(serial.tools.list_ports.comports())
        selected_index = 0
        for i, port in enumerate(ports):
            self.comComboBox.addItem(f"{port.device} - {port.description}", port.device)
            if port.device == "COM3":  # 如果存在 COM5,则默认选中它
                selected_index = i
        self.comComboBox.setCurrentIndex(selected_index)

    def select_points(self):
        """
        点击“选择点”按钮时，启用点选择，并保存返回的 observer ID
        """
        self.picked_points = []
        # 保存 observer ID
        self.point_picker_observer = self.plotter.enable_point_picking(
            callback=self.point_picked_callback,
            show_message=True,
            use_picker=True
        )
        print("点选择已启用，请在 PyVista 窗口中点击选择点...")

    def point_picked_callback(self, point, *args):
        self.picked_points.append(point)
        print(f"选取的点: {point}")

        # 如果没有保存球体 actor 的列表，则初始化
        if not hasattr(self, "picked_point_actors"):
            self.picked_point_actors = []

        # 为当前选取的点创建一个黄色球体（半径可根据需要调整）
        sphere = pv.Sphere(radius=2, center=point)
        sphere_actor = self.plotter.add_mesh(sphere, color="yellow")
        self.picked_point_actors.append(sphere_actor)

        if len(self.picked_points) == 3:
            # 当选取了3个点后，关闭点选择模式
            self.plotter.disable_picking()
            # 同时更新界面上的标签显示选取的点坐标
            self.update_labels_with_points()

    def update_labels_with_points(self):
        """将选取的三个点的坐标显示到界面上相应的标签中"""
        for i, point in enumerate(self.picked_points):
            self.coord_labels[i][0].setText(f"P{i+1}_X: {point[0]:.3f}")
            self.coord_labels[i][1].setText(f"P{i+1}_Y: {point[1]:.3f}")
            self.coord_labels[i][2].setText(f"P{i+1}_Z: {point[2]:.3f}")

    def record_points_1(self):
        if self.ref_pos is None:
            QMessageBox.warning(self, "警告", "未检测到配准探针 (Port 2)！")
            return
        # 使用 self.ref_pos (Port 2)
        self.coord_labels2[0][0].setText(f"P1_X: {self.ref_pos[0]:.3f}")
        self.coord_labels2[0][1].setText(f"P1_Y: {self.ref_pos[1]:.3f}")
        self.coord_labels2[0][2].setText(f"P1_Z: {self.ref_pos[2]:.3f}")

    def record_points_2(self):
        if self.ref_pos is None:
            QMessageBox.warning(self, "警告", "未检测到配准探针 (Port 2)！")
            return
        # 使用 self.ref_pos (Port 2)
        self.coord_labels2[1][0].setText(f"P2_X: {self.ref_pos[0]:.3f}")
        self.coord_labels2[1][1].setText(f"P2_Y: {self.ref_pos[1]:.3f}")
        self.coord_labels2[1][2].setText(f"P2_Z: {self.ref_pos[2]:.3f}")

    def record_points_3(self):
        if self.ref_pos is None:
            QMessageBox.warning(self, "警告", "未检测到配准探针 (Port 2)！")
            return
        # 使用 self.ref_pos (Port 2)
        self.coord_labels2[2][0].setText(f"P3_X: {self.ref_pos[0]:.3f}")
        self.coord_labels2[2][1].setText(f"P3_Y: {self.ref_pos[1]:.3f}")
        self.coord_labels2[2][2].setText(f"P3_Z: {self.ref_pos[2]:.3f}")

    def select_endpoint_mode(self):
        """
        [修正版] 激活终点选择模式
        修复了 TypeError: "message" is an invalid keyword argument
        """
        # 1. 清除旧的终点标记 (如果有)
        if hasattr(self, "endpoint_actor") and self.endpoint_actor:
            self.plotter.remove_actor(self.endpoint_actor)
            self.endpoint_actor = None

        # 2. 显示提示文字 (替代原来的 message 参数)
        # name='msg' 保证下次调用 add_text 时会覆盖这条，不会重叠
        self.plotter.add_text("请在模型上【右键】点击选择导航终点...", position='upper_left', font_size=12,
                              color='yellow', name='msg')

        # 3. 启用点选
        # 注意：去掉了不支持的 'message' 参数
        self.plotter.enable_point_picking(
            callback=self.on_endpoint_picked,
            show_message=True,  # 选完后显示坐标信息
            font_size=12,
            use_picker=True,
            show_point=False  # 不显示默认的红色选点，我们自己画球
        )
        print(">>> 进入终点选择模式，请在模型上【右键】点击...")

    def on_endpoint_picked(self, point, *args):
        """
        [修正版] 终点被选中后的回调函数
        """
        if point is None: return

        print(f"选中终点坐标: {point}")
        self.endpoint = list(point)  # 更新终点坐标

        # 1. 清除之前的提示文字，换成成功提示
        self.plotter.add_text("导航终点已设定!", position='upper_left', font_size=12, color='green', name='msg')
        self.has_reached_target = False
        # 2. 画一个蓝色小球标记终点
        sphere = pv.Sphere(radius=3, center=point)
        self.endpoint_actor = self.plotter.add_mesh(sphere, color="blue", label="Goal")

        # 3. 选完一个点后，立即关闭选择模式
        self.plotter.disable_picking()

        # 4. 标记系统已就绪
        self.ready_control = True

        # 5. 刷新界面
        self.plotter.render()

    def update_visualization(self):
        """Update visualization with latest data"""
        if self._is_closing:
            return
        try:
            self.update_axis_positions()

            # ====== 1. 获取并分流 NDI 数据 ======
            if hasattr(self, 'data_queue') and not self.data_queue.empty():

                # --- 【核心修改】清空队列，只取最新的一帧 ---
                tools_dict = self.data_queue.get()
                while not self.data_queue.empty():
                    tools_dict = self.data_queue.get()  # 只要队列还有，就继续取，覆盖掉旧的

                # 定义端口对应关系 (根据您的测试结果)
                PORT_SCOPE = 10  # 端口1 (镜子) -> Handle 10
                PORT_REF = 11  # 端口2 (探针) -> Handle 11

                # --- A. 处理配准探针 (Port 2) ---
                if PORT_REF in tools_dict:
                    mat_ref = tools_dict[PORT_REF]
                    self.ref_pos = mat_ref[:3, 3]  # 存下来给 record_points 用

                    # 【UI反馈】在配准之前，界面上的 XYZ 显示探针的位置，方便您确认是否接触到了点
                    if len(self.R) == 0:
                        self.x_label.setText("Ref X: {:.2f}".format(self.ref_pos[0]))
                        self.y_label.setText("Ref Y: {:.2f}".format(self.ref_pos[1]))
                        self.z_label.setText("Ref Z: {:.2f}".format(self.ref_pos[2]))

                # --- B. 处理支气管镜 (Port 1) ---
                if PORT_SCOPE in tools_dict:
                    mat_scope = tools_dict[PORT_SCOPE]
                    fix_mat = np.array([
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]
                    ])
                    mat_scope = mat_scope @ fix_mat
                    self.scope_pos = mat_scope[:3, 3]  # 存下来给导航用
                    self.scope_dir = mat_scope[:3, :3] # 存下来给姿态记录用

                    # 【UI反馈】配准成功后，界面 XYZ 改为显示镜子的位置
                    if len(self.R) > 0:
                        self.x_label.setText("Scope X: {:.2f}".format(self.scope_pos[0]))
                        self.y_label.setText("Scope Y: {:.2f}".format(self.scope_pos[1]))
                        self.z_label.setText("Scope Z: {:.2f}".format(self.scope_pos[2]))

            # ====== 2. 计算虚拟位置 (仅使用 Port 1 + 配准矩阵) ======
            # 只有当 (配准已完成) 且 (镜子数据存在) 时才更新
            if len(self.R) > 0 and len(self.t) > 0 and self.scope_pos is not None:

                # 【核心修改】这里使用 self.scope_pos (端口1)，而不是通用的 self.position
                raw_virtual = np.dot(self.R, self.scope_pos) + self.t

                self.virtual_position_raw = raw_virtual.copy()

                # 优先投影到中心线
                if hasattr(self, "centerline_points"):
                    self.virtual_position = self.project_onto_centerline(raw_virtual)
                else:
                    self.virtual_position = raw_virtual

                # ====== 数据记录部分 (含新增姿态) ======
                if getattr(self, "is_recording_trajectory", False) and len(self.virtual_position) == 3:
                    current_t = time.time() - self.recording_start_time
                    vx, vy, vz = self.virtual_position  # 投影后
                    rx, ry, rz = raw_virtual  # 投影前
                    axes_pos = list(self.current_axis_values)

                    # [新增] 计算四元数姿态 (x, y, z, w)
                    # self.scope_dir 是 3x3 旋转矩阵
                    if self.scope_dir is not None:
                         # scipy 的 as_quat() 返回 [x, y, z, w]
                         quat = R.from_matrix(self.scope_dir).as_quat()
                         qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]
                    else:
                         qx, qy, qz, qw = 0, 0, 0, 1
                    
                    # 组合数据: 时间, 投影XYZ, 原始XYZ, 姿态Quat(4), 电机轴(7)
                    row_data = [current_t, vx, vy, vz, rx, ry, rz, qx, qy, qz, qw] + axes_pos
                    self.trajectory_data.append(row_data)

                # 计算旋转 (同样使用 scope_dir)
                R_calib = np.dot(self.R, self.scope_dir)
                dir_z = R_calib[:, 2]
                norm_z = np.linalg.norm(dir_z)
                if norm_z > 1e-6:
                    dir_z = dir_z / norm_z
                else:
                    dir_z = np.array([0, 0, 1])

                # 更新虚拟点坐标标签
                self.xx_label.setText("X: {:.2f}".format(self.virtual_position[0]))
                self.yy_label.setText("Y: {:.2f}".format(self.virtual_position[1]))
                self.zz_label.setText("Z: {:.2f}".format(self.virtual_position[2]))

                # 更新红色圆柱体 (Actor)
                base_z = np.array([0.0, 0.0, 1.0])
                axis = np.cross(base_z, dir_z)
                if np.linalg.norm(axis) < 1e-6:
                    axis = np.array([1.0, 0.0, 0.0])
                else:
                    axis = axis / np.linalg.norm(axis)
                angle = math.degrees(math.acos(np.clip(np.dot(base_z, dir_z), -1.0, 1.0)))

                if not hasattr(self, 'cyl_transform'):
                    self.cyl_transform = vtk.vtkTransform()
                    self.cyl_transform.PostMultiply()
                self.cyl_transform.Identity()
                self.cyl_transform.RotateWXYZ(angle, axis.tolist())
                self.cyl_transform.Translate(self.virtual_position.tolist())

                if hasattr(self, "virtual_point_actor") and self.virtual_point_actor is not None:
                    self.virtual_point_actor.SetUserTransform(self.cyl_transform)
                else:
                    # 如果还没有 actor，创建它
                    cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1, height=5)
                    self.virtual_point_actor = self.plotter.add_mesh(cyl, color="black")
                    self.virtual_point_actor.SetUserTransform(self.cyl_transform)

            if hasattr(self, "endpoint") and self.endpoint is not None and len(self.virtual_position) > 0:
                # 1. 获取当前点和目标点
                curr_pos = np.array(self.virtual_position)
                goal_pos = np.array(self.endpoint)

                # 2. 计算欧氏距离
                dist_to_goal = np.linalg.norm(curr_pos - goal_pos)

                # 3. 判断阈值 (这里设为 2mm，您可以根据精度需求调整)
                ARRIVAL_THRESHOLD = 2.0

                # 4. 如果距离小于阈值，且之前没弹过窗
                if dist_to_goal < ARRIVAL_THRESHOLD and not self.has_reached_target:
                    self.has_reached_target = True  # 锁住，防止重复弹窗

                    # 弹出提示
                    QMessageBox.information(self, "导航提示", "已到达目标位置！\nTarget Reached.")
                    print(f">>> 提示: 到达终点 (距离 {dist_to_goal:.2f} mm)")

            # 3. 渲染主窗口
            if hasattr(self, 'plotter') and not self.plotter._closed:
                self.plotter.render()

            # 4. 强制刷新虚拟视角 (使用新的 self.scope_pos 计算的结果)
            should_run_virtual = False
            if self.ready_control: should_run_virtual = True
            if self.virtual_debug_cb.isChecked(): should_run_virtual = True

            if should_run_virtual and hasattr(self, 'virtual_view'):
                if not hasattr(self, 'cyl_transform'):
                    self.cyl_transform = vtk.vtkTransform()
                    self.cyl_transform.PostMultiply()
                    self.cyl_transform.Identity()
                    if len(self.virtual_position) == 0:
                        self.virtual_position = np.array([10.0, 10.0, 10.0])
                    self.cyl_transform.Translate(self.virtual_position.tolist())
                self.update_virtual_camera()

        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()


    # def update_visualization(self):
    #     """Update visualization with latest data"""
    #     if self._is_closing:
    #         return
    #     try:
    #         self.update_axis_positions()

    #         # ====== 1. 获取并分流 NDI 数据 ======
    #         if hasattr(self, 'data_queue') and not self.data_queue.empty():

    #             # --- 【核心修改】清空队列，只取最新的一帧 ---
    #             tools_dict = self.data_queue.get()
    #             while not self.data_queue.empty():
    #                 tools_dict = self.data_queue.get()  # 只要队列还有，就继续取，覆盖掉旧的

    #             # 定义端口对应关系 (根据您的测试结果)
    #             PORT_SCOPE = 10  # 端口1 (镜子) -> Handle 10
    #             PORT_REF = 11  # 端口2 (探针) -> Handle 11

    #             # --- A. 处理配准探针 (Port 2) ---
    #             if PORT_REF in tools_dict:
    #                 mat_ref = tools_dict[PORT_REF]
    #                 self.ref_pos = mat_ref[:3, 3]  # 存下来给 record_points 用

    #                 # 【UI反馈】在配准之前，界面上的 XYZ 显示探针的位置，方便您确认是否接触到了点
    #                 if len(self.R) == 0:
    #                     self.x_label.setText("Ref X: {:.2f}".format(self.ref_pos[0]))
    #                     self.y_label.setText("Ref Y: {:.2f}".format(self.ref_pos[1]))
    #                     self.z_label.setText("Ref Z: {:.2f}".format(self.ref_pos[2]))

    #             # --- B. 处理支气管镜 (Port 1) ---
    #             if PORT_SCOPE in tools_dict:
    #                 mat_scope = tools_dict[PORT_SCOPE]
    #                 fix_mat = np.array([
    #                     [-1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, -1, 0],
    #                     [0, 0, 0, 1]
    #                 ])
    #                 mat_scope = mat_scope @ fix_mat
    #                 self.scope_pos = mat_scope[:3, 3]  # 存下来给导航用
    #                 self.scope_dir = mat_scope[:3, :3]

    #                 # 【UI反馈】配准成功后，界面 XYZ 改为显示镜子的位置
    #                 if len(self.R) > 0:
    #                     self.x_label.setText("Scope X: {:.2f}".format(self.scope_pos[0]))
    #                     self.y_label.setText("Scope Y: {:.2f}".format(self.scope_pos[1]))
    #                     self.z_label.setText("Scope Z: {:.2f}".format(self.scope_pos[2]))

    #         # ====== 2. 计算虚拟位置 (仅使用 Port 1 + 配准矩阵) ======
    #         # 只有当 (配准已完成) 且 (镜子数据存在) 时才更新
    #         if len(self.R) > 0 and len(self.t) > 0 and self.scope_pos is not None:

    #             # 【核心修改】这里使用 self.scope_pos (端口1)，而不是通用的 self.position
    #             raw_virtual = np.dot(self.R, self.scope_pos) + self.t

    #             self.virtual_position_raw = raw_virtual.copy()

    #             # 优先投影到中心线
    #             if hasattr(self, "centerline_points"):
    #                 self.virtual_position = self.project_onto_centerline(raw_virtual)
    #             else:
    #                 self.virtual_position = raw_virtual
    #             if getattr(self, "is_recording_trajectory", False) and len(self.virtual_position) == 3:
    #                 current_t = time.time() - self.recording_start_time
    #                 vx, vy, vz = self.virtual_position  # 投影后
    #                 rx, ry, rz = raw_virtual  # 投影前
    #                 axes_pos = list(self.current_axis_values)

    #                 row_data = [current_t, vx, vy, vz, rx, ry, rz] + axes_pos
    #                 self.trajectory_data.append(row_data)
    #             # 计算旋转 (同样使用 scope_dir)
    #             R_calib = np.dot(self.R, self.scope_dir)
    #             dir_z = R_calib[:, 2]
    #             norm_z = np.linalg.norm(dir_z)
    #             if norm_z > 1e-6:
    #                 dir_z = dir_z / norm_z
    #             else:
    #                 dir_z = np.array([0, 0, 1])

    #             # 更新虚拟点坐标标签
    #             self.xx_label.setText("X: {:.2f}".format(self.virtual_position[0]))
    #             self.yy_label.setText("Y: {:.2f}".format(self.virtual_position[1]))
    #             self.zz_label.setText("Z: {:.2f}".format(self.virtual_position[2]))

    #             # 更新红色圆柱体 (Actor)
    #             base_z = np.array([0.0, 0.0, 1.0])
    #             axis = np.cross(base_z, dir_z)
    #             if np.linalg.norm(axis) < 1e-6:
    #                 axis = np.array([1.0, 0.0, 0.0])
    #             else:
    #                 axis = axis / np.linalg.norm(axis)
    #             angle = math.degrees(math.acos(np.clip(np.dot(base_z, dir_z), -1.0, 1.0)))

    #             if not hasattr(self, 'cyl_transform'):
    #                 self.cyl_transform = vtk.vtkTransform()
    #                 self.cyl_transform.PostMultiply()
    #             self.cyl_transform.Identity()
    #             self.cyl_transform.RotateWXYZ(angle, axis.tolist())
    #             self.cyl_transform.Translate(self.virtual_position.tolist())

    #             if hasattr(self, "virtual_point_actor") and self.virtual_point_actor is not None:
    #                 self.virtual_point_actor.SetUserTransform(self.cyl_transform)
    #             else:
    #                 # 如果还没有 actor，创建它
    #                 cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=1, height=5)
    #                 self.virtual_point_actor = self.plotter.add_mesh(cyl, color="black")
    #                 self.virtual_point_actor.SetUserTransform(self.cyl_transform)
    #         if hasattr(self, "endpoint") and self.endpoint is not None and len(self.virtual_position) > 0:
    #             # 1. 获取当前点和目标点
    #             curr_pos = np.array(self.virtual_position)
    #             goal_pos = np.array(self.endpoint)

    #             # 2. 计算欧氏距离
    #             dist_to_goal = np.linalg.norm(curr_pos - goal_pos)

    #             # 3. 判断阈值 (这里设为 2mm，您可以根据精度需求调整)
    #             ARRIVAL_THRESHOLD = 2.0

    #             # 4. 如果距离小于阈值，且之前没弹过窗
    #             if dist_to_goal < ARRIVAL_THRESHOLD and not self.has_reached_target:
    #                 self.has_reached_target = True  # 锁住，防止重复弹窗

    #                 # 弹出提示
    #                 QMessageBox.information(self, "导航提示", "已到达目标位置！\nTarget Reached.")
    #                 print(f">>> 提示: 到达终点 (距离 {dist_to_goal:.2f} mm)")
    #         # 3. 渲染主窗口
    #         if hasattr(self, 'plotter') and not self.plotter._closed:
    #             self.plotter.render()

    #         # 4. 强制刷新虚拟视角 (使用新的 self.scope_pos 计算的结果)
    #         should_run_virtual = False
    #         if self.ready_control: should_run_virtual = True
    #         if self.virtual_debug_cb.isChecked(): should_run_virtual = True

    #         if should_run_virtual and hasattr(self, 'virtual_view'):
    #             if not hasattr(self, 'cyl_transform'):
    #                 self.cyl_transform = vtk.vtkTransform()
    #                 self.cyl_transform.PostMultiply()
    #                 self.cyl_transform.Identity()
    #                 if len(self.virtual_position) == 0:
    #                     self.virtual_position = np.array([10.0, 10.0, 10.0])
    #                 self.cyl_transform.Translate(self.virtual_position.tolist())
    #             self.update_virtual_camera()

    #     except Exception as e:
    #         print(f"Error updating visualization: {e}")
    #         import traceback
    #         traceback.print_exc()

    def get_path_lookahead_uv(self):
        """
        【修正版】计算路径前瞻点投影
        修正点：使用动态计算的焦距 + 原始内参的主点偏移 (cx, cy)。
        """
        if not hasattr(self, "smoothpath") or len(self.smoothpath) == 0:
            return None

        # 1. 找到路径上离当前虚拟相机最近的点
        path_arr = np.array(self.smoothpath)
        dists = np.linalg.norm(path_arr - self.virtual_position, axis=1)
        curr_idx = np.argmin(dists)

        # 2. 找前瞻点
        look_ahead_steps = 12
        target_idx = min(curr_idx + look_ahead_steps, len(path_arr) - 1)
        target_pt = path_arr[target_idx]

        # 3. 坐标系转换 (世界 -> 相机)
        if not hasattr(self, "cam_R_cw") or not hasattr(self, "cam_C"):
            return None

        P_w_minus_C = target_pt - self.cam_C
        P_c = self.cam_R_cw @ P_w_minus_C
        x, y, z = P_c

        # 如果点在相机后面，丢弃
        if z < 1e-3:
            return None

        # ====== 【核心修改：动态计算投影参数】 ======

        # A. 获取当前渲染用的真实 FOV
        cam = self.virtual_view.camera
        fov_deg = cam.GetViewAngle()

        # B. 传感器尺寸
        h_sensor = 400.0
        w_sensor = 400.0

        # C. 根据 FOV 反推焦距
        # f = (h / 2) / tan(fov / 2)
        f_pixel_new = (h_sensor / 2.0) / math.tan(math.radians(fov_deg / 2.0))

        # D. 【修正】使用新的焦距 + 原始的主点偏移
        fx = f_pixel_new
        fy = f_pixel_new

        # 从原始矩阵读取 cx, cy，确保与 SetWindowCenter 逻辑一致
        K = self.intrinsic_camera_matrix
        cx = w_sensor - K[0, 2]
        cy = h_sensor - K[1, 2]

        # E. 投影公式
        u = fx * x / z + cx
        v = fy * y / z + cy

        # 4. 归一化 (0.0 ~ 1.0)
        u_norm = u / w_sensor
        v_norm = v / h_sensor

        return (u_norm, v_norm)
    def get_virtual_visual_center(self):
        """
        【核心算法】基于深度图提取虚拟视角的腔道中心
        修改点：使用 SwapBuffers(0) 禁止后台渲染上屏，彻底消除闪烁。
        """
        if not hasattr(self, 'virtual_view') or self.virtual_view is None:
            return [], None

        # 获取渲染窗口对象
        ren_win = self.virtual_view.GetRenderWindow()

        # ====== 【核心修改 1：禁止画面刷新上屏】 ======
        # 0 = 禁止交换缓冲区 (意味着接下来的 render 只在内存里画，屏幕不更新)
        # 这样用户就看不到"线消失"的那一帧了
        ren_win.SetSwapBuffers(0)

        actors_to_hide = []

        # 1. 收集要隐藏的物体 (逻辑不变)
        if hasattr(self, "path_actor_virtual") and self.path_actor_virtual:
            if self.path_actor_virtual.GetVisibility():
                actors_to_hide.append(self.path_actor_virtual)

        if hasattr(self, "virt_opening_actor") and self.virt_opening_actor:
            if self.virt_opening_actor.GetVisibility():
                actors_to_hide.append(self.virt_opening_actor)

        if hasattr(self, "virt_green_actor") and self.virt_green_actor:
            if self.virt_green_actor.GetVisibility():
                actors_to_hide.append(self.virt_green_actor)

        if hasattr(self, "virtual_point_actor") and self.virtual_point_actor:
            if self.virtual_point_actor.GetVisibility():
                actors_to_hide.append(self.virtual_point_actor)

        if hasattr(self, "picked_point_actors") and self.picked_point_actors:
            for actor in self.picked_point_actors:
                if actor and actor.GetVisibility():
                    actors_to_hide.append(actor)
        if hasattr(self, "virt_box_actor") and self.virt_box_actor:
            actors_to_hide.append(self.virt_box_actor)
        if hasattr(self, "debug_guide_actor") and self.debug_guide_actor:
            actors_to_hide.append(self.debug_guide_actor)
        # 2. 执行隐藏
        for actor in actors_to_hide:
            actor.SetVisibility(False)

        # 3. 后台渲染 (这一步屏幕不会闪烁，因为SwapBuffers关了)
        if actors_to_hide:
            self.virtual_view.render()

        try:
            # 4. 抓取深度图 (此时是干净的)
            depth = self.virtual_view.get_image_depth(fill_value=60.0)
        except AttributeError:
            # 出错恢复：别忘了把屏幕刷新打开
            for actor in actors_to_hide:
                actor.SetVisibility(True)
            ren_win.SetSwapBuffers(1)  # <--- 恢复
            return [], -1, None, None

        # 5. 恢复显示
        for actor in actors_to_hide:
            actor.SetVisibility(True)

        # ====== 【核心修改 2：允许画面刷新上屏】 ======
        # 1 = 允许交换缓冲区 (恢复正常显示)
        ren_win.SetSwapBuffers(1)

        # 6. 最后渲染一次 (把带线的画面推给用户)
        if actors_to_hide:
            self.virtual_view.render()
            self.virtual_view.render()

        # ... (以下图像处理逻辑完全保持不变) ...

        if depth is None or depth.size == 0:
            return [], -1, None, None
        # 2. 生成 Mask
        dist = np.abs(depth)
        valid_mask = (dist > 1e-3) & (~np.isnan(dist))
        if np.any(valid_mask):
            max_valid = np.max(dist[valid_mask])
        else:
            max_valid = 10.0  # 兜底默认值

        # 2. 设定一个"无穷远"的值 (比当前看到的最远还要远)
        infinity_val = max_valid * 2.0
        # 防止过大导致后续计算溢出，也可以直接给个固定大值比如 100.0
        if infinity_val < 100.0: infinity_val = 100.0

        # 3. 替换坏值
        # 将 NaN 替换为无穷远
        dist = np.nan_to_num(dist, nan=infinity_val, posinf=infinity_val, neginf=infinity_val)
        # 将 0.0 (背景) 也替换为无穷远 <--- 关键就在这！
        dist[dist < 1e-3] = infinity_val
        # 【调试关键】: 深层时，视野很窄，不需要看 60mm 那么远
        # 动态调整 Clip，增加对比度
        current_max = np.percentile(dist, 98)
        # 如果当前最大深度只有 10mm (贴墙了)，那就只看 10mm 范围
        effective_clip = min(60.0, current_max * 1.2)
        dist = np.clip(dist, 0, effective_clip)

        max_d = np.percentile(dist, 98)

        if max_d < 1.0:  # 门槛也调低点
            return [], -1, None, dist  # 把原始 dist 返回去调试

        # ====== 【核心修改 A：调低阈值】 ======
        # 0.5 甚至可以试 0.4，让它更容易把浅坑当成路
        ratio = 0.8
        thresh_val = max_d * ratio
        mask = (dist > thresh_val).astype(np.uint8) * 255

        # ====== 【核心修改 B：禁用或减小腐蚀】 ======
        # 之前的 (5,5) 太大了，深部支气管只有 3-4 个像素宽，会被直接抹掉！
        # 建议直接注释掉，或者改用 (1,1) / (2,2)
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        centers_list = []
        h, w = mask.shape

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # ====== 【核心修改 C：允许小洞】 ======
            # 深处的小洞可能只有 5-10 个像素
            if area < 5: continue

            cx, cy = centroids[i]
            u_norm = cx / w
            v_norm = cy / h
            centers_list.append((u_norm, v_norm))
        if not centers_list:
            return [], -1, mask, dist  # 返回空列表和无效索引 -1
            # 1. 获取路径指引点
        guide_uv = self.get_path_lookahead_uv()
        best_idx = -1  # 默认没有最佳
        if guide_uv is not None:
            g_u, g_v = guide_uv
            min_dist = float('inf')

            # 2. 遍历找最近
            for i, (c_u, c_v) in enumerate(centers_list):
                # ====== 【核心修改】换个变量名！别覆盖 dist 数组！ ======
                match_dist = (c_u - g_u) ** 2 + (c_v - g_v) ** 2

                if match_dist < min_dist:
                    min_dist = match_dist
                    best_idx = i
        else:
            best_idx = -1

        return centers_list, best_idx, mask, dist

    def update_virtual_camera(self):
        """
        更新虚拟相机视角，并计算虚拟视觉中心
        """
        if self._is_closing: return  # 防爆锁
        if not hasattr(self, 'virtual_view') or self.virtual_view is None:
            return
        if hasattr(self.virtual_view, '_closed') and self.virtual_view._closed:
            return

        # 仍然沿用你原来的保护逻辑
        if not hasattr(self, 'cyl_transform') or not hasattr(self, 'virtual_position'):
            return

        # ====== 第一部分：完全保持你原来的相机设置 ======
        # 从 transform 矩阵里取出旋转部分第 3 列，作为本地 z 轴在世界坐标系下的方向
        mat = self.cyl_transform.GetMatrix()  # vtkMatrix4x4
        dir_z = [mat.GetElement(i, 2) for i in range(3)]
        # 单位化
        norm = math.sqrt(dir_z[0] ** 2 + dir_z[1] ** 2 + dir_z[2] ** 2)
        if norm > 1e-6:
            dir_z = [v / norm for v in dir_z]
        else:
            # 兜底：如果没法单位化，就用默认朝向
            dir_z = [0, 1, 0]

        # 设置相机位置到圆柱中心（和你原来一样）
        cam = self.virtual_view.camera
        cam.SetPosition(self.virtual_position.tolist())

        # 上向始终为世界坐标 Z 轴（和你原来一样）
        cam.SetViewUp([0, 0, 1])

        # 焦点沿着 dir_z 方向偏移一点，让相机看向圆柱长轴（和你原来一样）
        focal = [self.virtual_position[i] + dir_z[i] for i in range(3)]
        cam.SetFocalPoint(focal)

        # 【渲染】必须先渲染，才能获取最新的深度图
        self.virtual_view.render()
        # ====== 第二部分：在不再改动相机的前提下，求出相机坐标系 ======
        # 这里仅仅是为了后续数学投影，不会再影响显示效果

        z_cam = np.array(dir_z, dtype=float)
        z_norm = np.linalg.norm(z_cam)
        if z_norm > 1e-6:
            z_cam /= z_norm
        else:
            z_cam = np.array([0.0, 1.0, 0.0])

        cam_pos = np.array(self.virtual_position, dtype=float)

        # 用世界 Z 轴当“参考上向”，与 z_cam 叉乘得到 x_cam
        # 用世界 Z 轴当“参考上向”
        up_world = np.array([0.0, 0.0, 1.0])

        # ====== 【核心修正】构建 CV 坐标系 (Right-Down-Forward) ======

        # 1. 计算 X 轴 (Right)：Forward × Up = Right
        # 之前是 Up × Forward = Left，所以我们交换一下叉乘顺序
        x_cam = np.cross(z_cam, up_world)

        if np.linalg.norm(x_cam) < 1e-6:
            # 万一死锁（相机垂直朝上/朝下），给个默认右方向
            x_cam = np.array([1.0, 0.0, 0.0])
        else:
            x_cam /= np.linalg.norm(x_cam)

        # 2. 计算 Y 轴 (Down)：Forward × Right = Down
        # 这一行不用改公式，但因为 x_cam 变了，y_cam 会自动变成向下的
        y_cam = np.cross(z_cam, x_cam)

        # world -> camera 旋转矩阵：Pc = R_cw * (Pw - C)
        self.cam_R_cw = np.vstack([x_cam, y_cam, z_cam])  # 3×3
        self.cam_C = cam_pos  # 3×1
        self.cam_z_dir = z_cam  # 记一下相机前向方向，画圆盘会用到

        # 分频计算
        if not hasattr(self, "opening_update_counter"):
            self.opening_update_counter = 0
        self.opening_update_counter += 1

        if self.opening_update_counter >= 5:
            self.opening_update_counter = 0

            is_debug = self.virtual_debug_cb.isChecked()
            need_calc = getattr(self, "enable_virtual_seg", False) or is_debug

            if need_calc:
                if hasattr(self, "get_virtual_visual_center"):
                    # 【注意】接收 3 个返回值
                    v_centers_list, best_idx, v_mask, v_dist = self.get_virtual_visual_center()
                    if getattr(self, "is_recording_trajectory", False) and getattr(self, "virt_video_writer",
                                                                                   None) is not None:
                        try:
                            # 1. 处理 Mask (转 BGR 以便拼接)
                            if v_mask is None:
                                # 如果 Mask 为空，给全黑
                                mask_bgr = np.zeros((400, 400, 3), dtype=np.uint8)
                            else:
                                mask_bgr = cv2.cvtColor(v_mask, cv2.COLOR_GRAY2BGR)

                            # 2. 处理深度图 (归一化 + 转伪彩色热力图)
                            if v_dist is None:
                                depth_color = np.zeros((400, 400, 3), dtype=np.uint8)
                            else:
                                # 深度值是 float，可能包含 inf，替换为最远距离 60.0
                                safe_dist = np.nan_to_num(v_dist, nan=60.0, posinf=60.0, neginf=60.0)
                                safe_dist = np.clip(safe_dist, 0, 60.0)

                                # 归一化到 0-255
                                depth_norm = (safe_dist / 60.0 * 255).astype(np.uint8)
                                # 应用伪彩色 (蓝色近，红色远)
                                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                                # ====== 【核心修复】强制统一尺寸 ======
                                # 防止因为 PyVista 窗口大小变动导致拼接失败或写入失败

                                # A. 先分别把两个图强行缩放到 400x400
                            if mask_bgr.shape[:2] != (400, 400):
                                mask_bgr = cv2.resize(mask_bgr, (400, 400))

                            if depth_color.shape[:2] != (400, 400):
                                depth_color = cv2.resize(depth_color, (400, 400))
                            # 3. 左右拼接 (左 Mask，右 Depth)，总宽 800
                            combined_frame = np.hstack([mask_bgr, depth_color])

                            # 4. 再次确保最终尺寸是 800x400 (双重保险)
                            if combined_frame.shape[:2] != (400, 800):  # 注意 numpy shape 是 (H, W)
                                combined_frame = cv2.resize(combined_frame, (800, 400))

                            # 5. 写入
                            self.virt_video_writer.write(combined_frame)

                        except Exception as e:
                            print(f"写入虚拟视频失败: {e}")
                    # ====== 调试：画出指引点位置 ======
                    guide_uv = self.get_path_lookahead_uv()
                    self.debug_guide_points.Reset()
                    self.debug_guide_cells.Reset()

                    if guide_uv is not None:
                        gu, gv = guide_uv
                        w = self.virtual_view.width()
                        h = self.virtual_view.height()

                        gx_px = gu * w
                        gy_vtk = h - (gv * h)

                        pid = self.debug_guide_points.InsertNextPoint(gx_px, gy_vtk, 0)
                        self.debug_guide_cells.InsertNextCell(1)
                        self.debug_guide_cells.InsertCellPoint(pid)

                    self.debug_guide_points.Modified()
                    self.debug_guide_cells.Modified()
                    self.debug_guide_poly.Modified()
                    # 1. 重置数据
                    self.virt_green_points.Reset()
                    self.virt_green_cells.Reset()
                    self.virt_box_points.Reset()
                    self.virt_box_cells.Reset()

                    if v_centers_list:
                        w = self.virtual_view.width()
                        h = self.virtual_view.height()

                        # ====== 绘制所有绿点 ======
                        for i, (u_norm, v_norm) in enumerate(v_centers_list):
                            # 坐标转换
                            x_px = u_norm * w
                            y_vtk = h - (v_norm * h)  # Y轴翻转

                            # 插入绿点
                            pid = self.virt_green_points.InsertNextPoint(x_px, y_vtk, 0)
                            self.virt_green_cells.InsertNextCell(1)
                            self.virt_green_cells.InsertCellPoint(pid)

                            # ====== 如果是最佳点，额外画个红框 ======
                            if i == best_idx:
                                # 定义框的大小 (比如 半径 20 像素)
                                r = 20

                                # 四个角的坐标
                                p0 = (x_px - r, y_vtk - r)
                                p1 = (x_px + r, y_vtk - r)
                                p2 = (x_px + r, y_vtk + r)
                                p3 = (x_px - r, y_vtk + r)

                                # 插入4个角点
                                base_id = self.virt_box_points.GetNumberOfPoints()
                                self.virt_box_points.InsertNextPoint(p0[0], p0[1], 0)
                                self.virt_box_points.InsertNextPoint(p1[0], p1[1], 0)
                                self.virt_box_points.InsertNextPoint(p2[0], p2[1], 0)
                                self.virt_box_points.InsertNextPoint(p3[0], p3[1], 0)

                                # 插入4条线 (组成口字型)
                                # 0-1
                                line1 = vtk.vtkLine()
                                line1.GetPointIds().SetId(0, base_id + 0)
                                line1.GetPointIds().SetId(1, base_id + 1)
                                self.virt_box_cells.InsertNextCell(line1)

                                # 1-2
                                line2 = vtk.vtkLine()
                                line2.GetPointIds().SetId(0, base_id + 1)
                                line2.GetPointIds().SetId(1, base_id + 2)
                                self.virt_box_cells.InsertNextCell(line2)

                                # 2-3
                                line3 = vtk.vtkLine()
                                line3.GetPointIds().SetId(0, base_id + 2)
                                line3.GetPointIds().SetId(1, base_id + 3)
                                self.virt_box_cells.InsertNextCell(line3)

                                # 3-0
                                line4 = vtk.vtkLine()
                                line4.GetPointIds().SetId(0, base_id + 3)
                                line4.GetPointIds().SetId(1, base_id + 0)
                                self.virt_box_cells.InsertNextCell(line4)

                                # 缓存这个最佳点用于控制
                                self.virtual_center_cache = (u_norm, v_norm)

                    # 标记修改
                    self.virt_green_points.Modified()
                    self.virt_green_cells.Modified()
                    self.virt_green_poly.Modified()

                    self.virt_box_points.Modified()
                    self.virt_box_cells.Modified()
                    self.virt_box_poly.Modified()

                    if is_debug and self.virt_debug_window:
                        self.virt_debug_window.update_image(v_mask, v_dist)
    def extract_points_from_labels(self,label_grid):
        points = []
        for row in label_grid:
            point = []
            for label in row:
                text = label.text()  # 获取标签文本，例如 "P1_X: 0.00"
                try:
                    # 分割字符串，取冒号后的部分，再转换为浮点数
                    number_str = text.split(":")[-1].strip()
                    value = float(number_str)
                except Exception as e:
                    value = 0.0
                point.append(value)
            points.append(point)
        return np.array(points)
    def Kabsch_computer(self):
        points_A = self.extract_points_from_labels(self.coord_labels2)
        points_B = self.extract_points_from_labels(self.coord_labels)
        self.R, self.t = compute_rigid_transform(points_A, points_B)
        print(points_A, points_B)
        print(self.R, self.t)
    def route_plan(self):
        # 1) 读取 IGES 中的所有中心线
        self.all_points = self.read_iges('老模型中心线.igs')

        # 2) 把所有中心线点拼成一个大数组，后面找最近点就用它
        self.centerline_points = np.vstack(self.all_points)

        print(f"正在构建 KDTree，共有 {len(self.centerline_points)} 个点...")
        self.kdtree = cKDTree(self.centerline_points)

        # 3) 用终点在中心线上找最近点
        nearest_point_goal, d, i = self.find_nearest_point(self.centerline_points, self.endpoint)

        path = []
        path, self.smoothpath, graph = pathplan(self.all_points,[9.1551647 ,-94.828995 ,43.316994 ],nearest_point_goal)
        # 1. 用规划好的路径点覆盖原来的全肺点
        self.centerline_points = np.array(self.smoothpath)

        # 2. 重新构建 KDTree，现在的树里只有这条红线上的点了
        print(f"正在切换导航树，锁定路径点数: {len(self.centerline_points)}")
        self.kdtree = cKDTree(self.centerline_points)
        line = pv.lines_from_points(self.smoothpath)
        if hasattr(self, "path_actor"):
            self.plotter.remove_actor(self.path_actor)

            # 将生成的线添加到 PyVista 的 plotter 中，并设置为红色、宽度为4
        self.path_actor = self.plotter.add_mesh(line, color="red", line_width=4)

        # 渲染更新界面
        self.plotter.render()
        if hasattr(self, "path_actor_virtual"):
            self.virtual_view.remove_actor(self.path_actor_virtual)
        self.path_actor_virtual = self.virtual_view.add_mesh(line, color="red", line_width=4)
        self.virtual_view.render()
        self.enable_virtual_seg = True
        print("虚拟视觉分割已就绪。")
    def read_iges(self,file_path):
        iges = pyiges.read(file_path)  # 读取 IGES 文件
        all_points = []
        for entity in iges:
            parameters = entity.parameters
            num_points = int(parameters[0][2])  # 提取点的数量
            coords_strings = parameters[0][3:]  # 提取坐标字符串
            coords_floats = [float(value.strip()) for value in coords_strings]  # 转换为浮点数

            # 将坐标转换为 (X, Y, Z) 格式
            points = []
            for i in range(num_points):
                x = coords_floats[i * 3]
                y = coords_floats[i * 3 + 1]
                z = coords_floats[i * 3 + 2]
                points.append([x, y, z])

            # 转换为 NumPy 数组并缩放
            points = np.array(points)
            all_points.append(points)
        return all_points

    def find_nearest_point(self, points, point):
        """
        查找 point 在 points 中最近的点。
        如果构建了 self.kdtree，则使用 KDTree 加速查询。
        """
        point = np.array(point)

        # 1. 优先使用 KDTree 查询 (极快)
        if hasattr(self, 'kdtree') and self.kdtree is not None:
            # query 返回 (距离, 索引)
            # k=1 表示只找最近的 1 个点
            distance, index = self.kdtree.query(point, k=1)

            nearest_point = self.centerline_points[index]
            return tuple(nearest_point), distance, index

        # 2. 如果没有 Tree，回退到暴力计算 (旧逻辑)
        # (通常只在刚启动还没点“路径规划”时会用到这里)
        points = np.array(points)
        distances = np.linalg.norm(points - point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = points[nearest_index]
        nearest_distance = distances[nearest_index]

        return tuple(nearest_point), nearest_distance, nearest_index

    def get_closest_point_on_segment(self, p, a, b):
        """
        计算点 p 在线段 ab 上的投影点
        """
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)

        ab = b - a
        length_sq = np.sum(ab ** 2)

        if length_sq < 1e-6:
            return a  # a 和 b 重合

        # 投影公式: t = (ap . ab) / (ab . ab)
        ap = p - a
        t = np.dot(ap, ab) / length_sq

        # 限制 t 在 [0, 1] 之间 (保证在很多线段内)
        t = np.clip(t, 0.0, 1.0)

        # 计算投影点坐标
        closest = a + t * ab
        return closest
    def project_onto_centerline(self, pt):
        if not hasattr(self, "centerline_points") or self.centerline_points is None:
            # 还没做路径规划 / 没有中心线数据，就暂时不处理
            return pt

        # 1. 先用 KDTree 找到最近的那个"顶点" (粗定位)
        # distance: 到顶点的距离, idx: 顶点的索引
        _, distance, idx = self.find_nearest_point(self.centerline_points, pt)

        # 如果只有一个点，没法连线，直接返回
        N = len(self.centerline_points)
        if N < 2:
            return self.centerline_points[idx]

        # 2. 获取前后相邻的点，组成线段
        # 我们要检查两个线段：(prev, curr) 和 (curr, next)
        # 看看投影到哪个线段上距离更近

        candidates = []

        # 当前最近点 P_i
        P_i = self.centerline_points[idx]

        # 尝试线段 1: P_i-1 -> P_i
        if idx > 0:
            P_prev = self.centerline_points[idx - 1]
            proj_prev = self.get_closest_point_on_segment(pt, P_prev, P_i)
            dist_prev = np.linalg.norm(proj_prev - pt)
            candidates.append((dist_prev, proj_prev))

        # 尝试线段 2: P_i -> P_i+1
        if idx < N - 1:
            P_next = self.centerline_points[idx + 1]
            proj_next = self.get_closest_point_on_segment(pt, P_i, P_next)
            dist_next = np.linalg.norm(proj_next - pt)
            candidates.append((dist_next, proj_next))

        # 3. 比较哪个线段更近
        if not candidates:
            return P_i  # 孤立点

        # 按距离排序，取最小的
        candidates.sort(key=lambda x: x[0])
        best_point = candidates[0][1]

        return np.array(best_point, dtype=float)

    def compute_virtual_branch_openings(self, plane_z=15.0):

        if not hasattr(self, "all_points") or self.all_points is None:
            return []
        if not hasattr(self, "cam_R_cw") or not hasattr(self, "cam_C"):
            return []

        K = self.intrinsic_camera_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        w = max(self.virtual_view.width(), 1)
        h = max(self.virtual_view.height(), 1)

        openings = []

        for bid, poly in enumerate(self.all_points):
            if poly is None:
                continue
            poly = np.asarray(poly)
            if poly.shape[0] < 2:
                continue

            Pw = poly
            Pw_minus_C = Pw - self.cam_C[None, :]
            Pc_all = (self.cam_R_cw @ Pw_minus_C.T).T  # (N,3)

            open_point_cam = None
            for i in range(len(Pc_all) - 1):
                p1 = Pc_all[i]
                p2 = Pc_all[i + 1]
                z1, z2 = p1[2], p2[2]

                if z1 <= 1e-6 and z2 <= 1e-6:
                    continue
                val1 = z1 - plane_z
                val2 = z2 - plane_z
                if val1 * val2 > 0:
                    continue
                denom = (z2 - z1)
                if abs(denom) < 1e-6:
                    continue
                t = (plane_z - z1) / denom
                if t < -0.1 or t > 1.1:
                    continue

                open_point_cam = p1 + t * (p2 - p1)
                break

            if open_point_cam is None:
                continue

            x, y, z = open_point_cam
            if z <= 1e-6:
                continue

            u = fx * x / z + cx
            v = fy * y / z + cy
            if not (0 <= u < w and 0 <= v < h):
                continue

            Pw_open = self.cam_R_cw.T @ open_point_cam + self.cam_C

            openings.append({
                "branch_id": bid,
                "Pw": Pw_open,
                "Pc": open_point_cam,
                "uv": (float(u), float(v)),
            })

        return openings

    def update_virtual_opening_glyphs(self, plane_z=15.0):

        openings = self.compute_virtual_branch_openings(plane_z=plane_z)

        # 如果当前没有任何开口点
        if not openings:
            # 如果之前创建过 actor，就先隐藏它
            if self.virt_opening_actor is not None:
                try:
                    self.virt_opening_actor.SetVisibility(False)
                except Exception:
                    pass
            self.virtual_view.render()
            return

        # 有开口点：取出所有 3D 世界坐标
        pts = np.array([o["Pw"] for o in openings], dtype=float)  # (N,3)

        # 第一次调用：创建 PolyData 和 actor
        if self.virt_opening_actor is None or self.virt_opening_poly is None:
            # 建一个 PolyData 来保存点
            self.virt_opening_poly = pv.PolyData(pts)

            # 用 add_mesh（内部和 add_points 类似），只创建一次 actor
            self.virt_opening_actor = self.virtual_view.add_mesh(
                self.virt_opening_poly,
                color="green",
                point_size=10,
                render_points_as_spheres=True
            )
        else:
            # 后续调用：只更新 PolyData 的 points
            try:
                # 更新点坐标
                self.virt_opening_poly.points = pts
                # 确保 actor 是可见的
                self.virt_opening_actor.SetVisibility(True)
            except Exception as e:
                print("更新虚拟开口点时出错：", e)

        # 最后渲染一次
        self.virtual_view.render()
    def sync_virtual_camera_with_intrinsics(self):
        K = self.intrinsic_camera_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 用手册中的真实参数
        pixel_size_mm = 1.008e-3  # 1.008 µm
        h_pixels = 400
        sensor_height = pixel_size_mm * h_pixels

        f_mm = fy * pixel_size_mm
        fovy = 100

        cam = self.virtual_view.camera
        cam.SetViewAngle(fovy)

        w, h = 400, 400
        wx = (cx - w / 2) / (w / 2)
        wy = (h / 2 - cy) / (h / 2)
        cam.SetWindowCenter(wx, wy)
        cam.SetClippingRange(0.1, 30)
        self.virtual_view.render()

    def update_video_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret: return

        if not self._is_closing:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.actual_view.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self.actual_view.setPixmap(scaled_pixmap)

            # Overlay 直接铺满 Label (因为 Label 就是有效画面)
            self.centroid_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())
            self.nav_overlay.setGeometry(0, 0, self.actual_view.width(), self.actual_view.height())
        # ... (发送给线程的代码不变，建议发原始 frame) ...
        if self.seg_running and hasattr(self, 'seg_queue'):
            try:
                if self.seg_queue.full():
                    try:
                        self.seg_queue.get_nowait()
                    except:
                        pass
                self.seg_queue.put(frame, block=False)
            except Exception:
                pass
    def toggle_segmentation(self, checked: bool):

        if checked:
            # ====== 启动分割 ======
            self.seg_button.setText("停止分割")
            self.start_segmentation()
        else:
            # ====== 停止分割 ======
            self.seg_button.setText("开始分割")
            self.stop_segmentation()

    def start_segmentation(self):
        if self.seg_running:
            return

        self.seg_queue = Queue(maxsize=1)

        self.seg_thread = QtCore.QThread(self)

        self.seg_worker = SegmentationWorker(
            weights_path="1205weights_49.pth",
            frame_queue=self.seg_queue,  # <--- 传入队列
            img_size=400
        )
        self.seg_worker.moveToThread(self.seg_thread)

        self.seg_thread.started.connect(self.seg_worker.process_loop)
        self.seg_worker.finished.connect(self.seg_thread.quit)
        self.seg_worker.finished.connect(self.seg_worker.deleteLater)
        self.seg_thread.finished.connect(self.seg_thread.deleteLater)
        self.seg_worker.resultReady.connect(self.on_segmentation_result)

        self.seg_running = True
        self.seg_thread.start()

    def stop_segmentation(self):
        if not self.seg_running:
            return
        self.seg_running = False
        if self.seg_worker is not None:
            self.seg_worker.stop()
        self.centroid_overlay.set_centroids([])

    def perform_robust_matching(self, real_centroids, virtual_target_uv):
        """
        鲁棒虚实匹配算法
        【修改版】针对偏移较大的情况，增加了宽松匹配逻辑
        """

        # 1. 基础检查
        if virtual_target_uv is None:
            return -1, "LOST", None

        g_u, g_v = virtual_target_uv

        if not real_centroids:
            return -1, "GHOST", (g_u, g_v)

        # ==========================================================
        # ====== 策略 A: 单对单强行锁定 (之前的逻辑) ======
        # ==========================================================
        if len(real_centroids) == 1:
            return 0, "LOCKED", real_centroids[0]

        # ==========================================================
        # ====== 策略 B: 寻找最近邻 (Nearest Neighbor) ======
        # ==========================================================
        best_idx = -1
        min_dist = float('inf')

        for i, (r_u, r_v) in enumerate(real_centroids):
            # 计算欧氏距离平方
            dist = (r_u - g_u) ** 2 + (r_v - g_v) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        # ==========================================================
        # ====== 策略 C: 宽松匹配 (针对您图片中的情况) ======
        # ==========================================================
        # 现象：画面只有2个洞，虽然偏移了，但"最近的那个"肯定是对的。
        # 逻辑：如果洞的数量很少(<=3)，我们将阈值放宽到 0.4 (40%屏幕宽度)
        # 这样即使有 15%-20% 的电磁漂移，也能锁住。

        LOOSE_THRESHOLD = 0.3  # 宽松阈值 (用于少于3个点的简单场景)
        STRICT_THRESHOLD = 0.2  # 严格阈值 (用于复杂场景)

        current_threshold = STRICT_THRESHOLD
        if len(real_centroids) <= 3:
            current_threshold = LOOSE_THRESHOLD

        # 判定
        if min_dist < current_threshold ** 2:
            return best_idx, "LOCKED", real_centroids[best_idx]
        else:
            # 只有真的离谱得太远了 (比如指东打西)，才显示 Ghost
            return -1, "GHOST", (g_u, g_v)
    def closeEvent(self, event):
        print("[Exit] 正在强制退出...")

        # 1. 停止定时器 (防止 UI 刷新)
        if hasattr(self, 'timer'): self.timer.stop()
        if hasattr(self, 'video_timer'): self.video_timer.stop()

        # 2. 停止 NDI Tracker
        if hasattr(self, "tracker_wrapper") and self.tracker_wrapper:
            try:
                self.tracker_wrapper.stop_tracking()
            except:
                pass

        # 3. 【新增】停止分割线程 (防止 GPU 在最后一刻还在运算)
        if hasattr(self, "seg_worker") and self.seg_worker:
            self.seg_worker.stop()  # 把 _running 设为 False

        # 4. 【新增】关闭可能打开的调试窗口
        if hasattr(self, "debug_window") and self.debug_window:
            self.debug_window.close()
        if hasattr(self, "virt_debug_window") and self.virt_debug_window:
            self.virt_debug_window.close()
        self.virt_green_actor = None
        print("[Exit] Bye!")
        event.accept()
        os._exit(0)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(Queue())
    window.show()
    sys.exit(app.exec_())