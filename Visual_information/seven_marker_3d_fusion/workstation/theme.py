"""Visual tokens and application stylesheet."""

PRODUCT_QSS = r"""
* {
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
    color: #1D1D1F;
}
QMainWindow, QWidget#appRoot { background: #F5F5F7; }
QFrame#sidebar { background: #15161A; border: none; }
QLabel#brandMark {
    min-width: 38px; min-height: 38px; max-width: 38px; max-height: 38px;
    border-radius: 11px; background: #0A84FF; color: white;
    font-size: 18px; font-weight: 700;
}
QLabel#brandTitle { color: #FFFFFF; font-size: 15px; font-weight: 650; }
QLabel#brandCaption { color: #8E8E93; font-size: 11px; }
QPushButton#navButton {
    background: transparent; color: #A9A9AF; border: none; border-radius: 10px;
    text-align: left; padding: 0 14px; min-height: 42px; font-weight: 550;
}
QPushButton#navButton:hover { background: #23252B; color: #FFFFFF; }
QPushButton#navButton:checked { background: #2B2E36; color: #FFFFFF; }
QLabel#sidebarStatus { color: #8E8E93; font-size: 11px; }
QFrame#topbar { background: rgba(255,255,255,245); border-bottom: 1px solid #E1E1E6; }
QLabel#pageTitle { font-size: 22px; font-weight: 680; color: #151517; }
QLabel#pageSubtitle { color: #6E6E73; font-size: 12px; }
QFrame#card { background: #FFFFFF; border: 1px solid #E2E2E7; border-radius: 16px; }
QFrame#heroCard { background: #FFFFFF; border: 1px solid #DDE5F0; border-radius: 18px; }
QLabel#cardTitle { font-size: 14px; font-weight: 650; color: #1D1D1F; }
QLabel#sectionTitle { font-size: 18px; font-weight: 670; color: #151517; }
QLabel#muted { color: #73737A; font-size: 12px; }
QLabel#micro { color: #8E8E93; font-size: 10px; }
QLabel#successBadge { color: #117A45; background: #E7F7ED; border-radius: 9px; padding: 4px 9px; font-weight: 620; }
QLabel#warningBadge { color: #8A5B00; background: #FFF3CF; border-radius: 9px; padding: 4px 9px; font-weight: 620; }
QLabel#errorBadge { color: #A22A25; background: #FDEAE8; border-radius: 9px; padding: 4px 9px; font-weight: 620; }
QLabel#stepBadge { color: #0067CE; background: #E9F3FF; border-radius: 11px; padding: 4px 10px; font-weight: 650; }
QPushButton {
    min-height: 34px; padding: 0 14px; border: 1px solid #D2D2D7;
    border-radius: 9px; background: #FFFFFF; font-weight: 550;
}
QPushButton:hover { background: #F2F2F4; }
QPushButton:pressed { background: #E7E7EA; }
QPushButton:disabled { color: #AEAEB2; background: #ECECEF; border-color: #E2E2E5; }
QPushButton#primary {
    background: #007AFF; color: white; border: none; font-weight: 650;
}
QPushButton#primary:hover { background: #0873DD; }
QPushButton#secondaryBlue { color: #006FD6; border-color: #B8D8FA; background: #F6FAFF; }
QPushButton#danger { color: #C9342C; border-color: #F0C5C1; background: #FFF8F7; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #FFFFFF; border: 1px solid #D2D2D7; border-radius: 8px;
    min-height: 32px; padding: 0 8px; selection-background-color: #007AFF;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #007AFF; }
QComboBox::drop-down { border: none; width: 26px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 17px; height: 17px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #C7C7CC; border-radius: 4px; min-height: 28px; }
QSlider::groove:horizontal { height: 4px; background: #D8D8DD; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #007AFF; }
QProgressBar { border: none; background: #E5E5EA; border-radius: 4px; height: 8px; text-align: center; }
QProgressBar::chunk { background: #0A84FF; border-radius: 4px; }
QTableWidget { background: #FFFFFF; border: 1px solid #E2E2E7; border-radius: 11px; gridline-color: #EFEFF2; }
QHeaderView::section { background: #F7F7F9; color: #6E6E73; border: none; border-bottom: 1px solid #E2E2E7; padding: 7px; font-weight: 620; }
QTabWidget::pane { border: 1px solid #E2E2E7; background: #FFFFFF; border-radius: 12px; }
QTabBar::tab { background: transparent; color: #6E6E73; padding: 9px 15px; }
QTabBar::tab:selected { color: #007AFF; font-weight: 650; }
QGroupBox { background: #FFFFFF; border: 1px solid #E2E2E7; border-radius: 12px; margin-top: 12px; padding-top: 12px; font-weight: 650; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QToolTip { background: #1D1D1F; color: #FFFFFF; border: none; padding: 6px; }
"""


PAGE_TITLES = (
    ("数据", "选择并检查采集会话"),
    ("初始化", "一次点击 K0–K6，建立跟踪身份"),
    ("重建", "执行唯一识别主链路并查看进度"),
    ("结果", "检查三维形状、坐标和质量"),
    ("设置", "管理可选先验、配准和输出"),
)
