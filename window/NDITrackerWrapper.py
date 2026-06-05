import time
import keyboard
import logging
from typing import Tuple, List, Optional
from sksurgerynditracker.nditracker import NDITracker
import socket
import numpy as np
import sys
import serial
import serial.tools.list_ports
import os
from datetime import datetime
from queue import Queue
from threading import Thread, Event
import json


class NDITrackerWrapper:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('tracker.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.load_config()

        # Initialize variables
        self.tracker = None
        self.udp_socket = None
        self.visualization_queue = Queue()
        self.stop_event = Event()

        # Check system
        self.check_rom_file()
        self.initialize_udp()

    def load_config(self):
        """Load settings from configuration file"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    self.DEST_ADDR = config.get('dest_addr', "127.0.0.1")
                    self.DEST_PORT = config.get('dest_port', 8192)
                    self.SEND_INTERVAL = config.get('send_interval', 0.1)
                    self.ROM_FILE = config.get('rom_file', "8700339.rom")
                    # self.COM_PORT = config.get('com_port', "COM5")
                    # 强制使用 COM3，或者您可以改回从 config 读取
                    self.COM_PORT = "COM3"
                    self.BAUD_RATE = config.get('baud_rate', 9600)
            else:
                # Use default values
                self.DEST_ADDR = "127.0.0.1"
                self.DEST_PORT = 8192
                self.SEND_INTERVAL = 0.1
                self.ROM_FILE = "8700339.rom"
                self.COM_PORT = "COM3"  # 默认 COM3
                self.BAUD_RATE = 9600
        except Exception as e:
            self.logger.error(f"Failed to load config file: {e}")
            self.DEST_ADDR = "127.0.0.1"
            self.DEST_PORT = 8192
            self.SEND_INTERVAL = 0.1
            self.ROM_FILE = "8700339.rom"
            self.COM_PORT = "COM3"
            self.BAUD_RATE = 9600

    def check_rom_file(self):
        """Check if ROM file exists and is accessible"""
        self.logger.info(f"Current working directory: {os.getcwd()}")
        if os.path.exists(self.ROM_FILE):
            self.logger.info(f"ROM file exists: {self.ROM_FILE}")
            file_stats = os.stat(self.ROM_FILE)
            self.logger.info(f"ROM file size: {file_stats.st_size} bytes")
            self.logger.info(f"ROM file permissions: {oct(file_stats.st_mode)[-3:]}")
        else:
            raise FileNotFoundError(f"ROM file not found: {self.ROM_FILE}")

    def verify_com_port(self):
        """Verify COM port availability"""
        ports = list(serial.tools.list_ports.comports())
        port_found = False

        for p in ports:
            if p.device == self.COM_PORT:
                port_found = True
                self.logger.info(f"Target port found: {p.device}")
                self.logger.info(f"  Description: {p.description}")
                self.logger.info(f"  Hardware ID: {p.hwid}")

                try:
                    with serial.Serial(p.device, self.BAUD_RATE, timeout=1) as ser:
                        self.logger.info(f"  Port test: Success")
                except Exception as e:
                    raise ConnectionError(f"Port test failed: {e}")

        if not port_found:
            raise ConnectionError(f"Target port not found: {self.COM_PORT}")

    def initialize_udp(self):
        """Initialize UDP connection"""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.logger.info("UDP socket initialized successfully")
        except Exception as e:
            raise ConnectionError(f"UDP socket initialization failed: {e}")

    def initialize_tracker(self) -> bool:
        """Initialize NDI tracker"""
        try:
            self.logger.info("Starting tracker initialization...")
            self.verify_com_port()

            settings = {
                "tracker type": "aurora",
                "serial port": self.COM_PORT,
                "baud rate": self.BAUD_RATE,
                "tool ports": None,
                "romfiles": [self.ROM_FILE],
            }

            self.logger.info(f"Initialization settings: {settings}")

            with serial.Serial(settings["serial port"], settings["baud rate"], timeout=1) as ser:
                ser.write(b'\x00')
                time.sleep(0.1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                self.logger.info("Serial communication test successful")

            self.tracker = NDITracker(settings)

            if hasattr(self.tracker, '_initialize_hardware'):
                self.logger.info("Initializing hardware...")
                self.tracker._initialize_hardware()

            if hasattr(self.tracker, '_configure_tools'):
                self.logger.info("Configuring tools...")
                self.tracker._configure_tools()

            if not all(hasattr(self.tracker, attr) for attr in ['start_tracking', 'get_frame']):
                raise AttributeError("Tracker missing required methods")

            self.logger.info("Tracker initialization successful")
            return True

        except Exception as e:
            self.logger.error(f"Tracker initialization failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def validate_tracking_data(self, tracking_data):
        """Validate tracking data"""
        try:
            if tracking_data is None:
                return False

            if isinstance(tracking_data, np.ndarray):
                if tracking_data.size == 0:
                    return False
                if np.isnan(tracking_data).all():
                    return False
                if tracking_data.shape == (4, 4):
                    return True
                return not np.isnan(tracking_data).all()

            self.logger.warning(f"Unknown data type: {type(tracking_data)}")
            return False

        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False

    def format_tracking_data(self, port_handles: List[int],
                             tracking_data: List[Optional[Tuple]],
                             frame_numbers: List[int]) -> str:
        """Format tracking data as CSV string"""
        output = []

        for i, (port_handle, data, frame_no) in enumerate(zip(port_handles, tracking_data, frame_numbers)):
            self.logger.debug(f"Processing port {port_handle} data: {data}")

            try:
                if not self.validate_tracking_data(data):
                    self.logger.debug(f"Port {port_handle} data invalid")
                    continue

                flat_data = data.flatten().tolist()
                position = flat_data[0:3]
                quaternion = flat_data[3:7]

                line = (f"{len(port_handles)},Tool{port_handle},Frame{frame_no},"
                        f"{port_handle},0,OK,"
                        f"{quaternion[0]:.6f},{quaternion[1]:.6f},"
                        f"{quaternion[2]:.6f},{quaternion[3]:.6f},"
                        f"{position[0]:.6f},{position[1]:.6f},{position[2]:.6f},"
                        f"0.0,0")
                output.append(line)

            except Exception as e:
                self.logger.error(f"Error formatting port {port_handle} data: {e}")
                continue

        return "\n".join(output)

    def start_tracking_thread(self):
        """Start tracking in a separate thread"""
        self.stop_event = Event()
        self.tracking_thread = Thread(target=self.tracking_thread)
        self.tracking_thread.start()

    def stop_tracking(self):
        """Stop tracking and clean up"""
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join()
        self.cleanup()

    def tracking_thread(self):
        """Thread for processing tracking data - 修正为支持多传感器字典格式"""
        try:
            self.tracker.start_tracking()
            self.logger.info("Tracking started")

            while not self.stop_event.is_set():
                try:
                    # 获取原始数据列表
                    port_handles, timestamps, frame_numbers, tracking, quality = self.tracker.get_frame()

                    # 准备一个字典来存放这一帧的所有有效数据
                    frame_data_dict = {}

                    if port_handles:
                        for i, handle in enumerate(port_handles):
                            data = tracking[i]

                            # 校验数据有效性
                            if data is None: continue
                            if isinstance(data, np.ndarray) and data.shape == (4, 4):
                                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                    continue

                                # 将该端口的数据存入字典: { Handle : Matrix }
                                frame_data_dict[handle] = data

                        # 如果字典不为空，发送给主界面
                        if frame_data_dict:
                            self.visualization_queue.put(frame_data_dict)

                            # (可选) 如果还需要UDP发送，保持原有逻辑
                            data_str = self.format_tracking_data(port_handles, tracking, frame_numbers)
                            if data_str:
                                self.udp_socket.sendto(data_str.encode(), (self.DEST_ADDR, self.DEST_PORT))
                                self.logger.debug("Data sent via UDP")

                    time.sleep(self.SEND_INTERVAL)

                except Exception as e:
                    self.logger.error(f"Error in tracking loop: {e}")
                    time.sleep(1)

        except Exception as e:
            self.logger.error(f"Tracking thread error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.tracker:
                if hasattr(self.tracker, 'stop_tracking'):
                    self.logger.info("Stopping tracking...")
                    self.tracker.stop_tracking()

                if hasattr(self.tracker, 'close'):
                    self.logger.info("Closing device...")
                    self.tracker.close()

            if self.udp_socket:
                self.logger.info("Closing UDP connection...")
                self.udp_socket.close()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            import traceback
            self.logger.error(traceback.format_exc())