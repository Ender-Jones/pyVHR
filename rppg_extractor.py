import json
from pathlib import Path
import cv2
import numpy as np
import tqdm

from pyVHR.extraction.sig_processing import SignalProcessing
from pyVHR.extraction.skin_extraction_methods import SkinExtractionFaceParsing
from pyVHR.BVP.methods import cpu_OMIT

"""
它的核心功能是:
1. 读取一个视频文件。
2. 使用 pyVHR 的 `SkinExtractionFaceParsing` 和 `Holistic` 方法提取整个视频的原始RGB信号。
3. 对RGB信号应用滑动窗口（例如，60秒窗口，5秒重叠）。
4. 对每个窗口的信号，使用 `cpu_OMIT` 方法将其转换为 BVP(rPPG) 信号。
5. 将所有窗口的BVP信号以及元数据保存到一个JSON文件中。

这个脚本的输出将作为您主代码库处理流程的输入。
"""

class RppgExtractor:
    def __init__(self, window_length_sec=60, step_length_sec=5, device='CPU'):
        """
        初始化 rPPG 提取器。

        Args:
            window_length_sec (int): 滑动窗口的长度（秒）。
            step_length_sec (int): 窗口滑动的步长（秒）。
            device (str): 运行设备 ('CPU' or 'GPU')。
        """
        self.window_length_sec = window_length_sec
        self.step_length_sec = step_length_sec

        # --- 规范化 device，避免大小写导致误入 GPU 分支 ---
        device_norm = str(device).upper()
        if device_norm not in ('CPU', 'GPU'):
            print(f"[WARN] 未知的 device='{device}', 已回退到 'CPU'")
            device_norm = 'CPU'

        # 传给 torch.load 的设备标识（小写）
        torch_device = 'cuda' if device_norm == 'GPU' else 'cpu'

        # --- 初始化 pyVHR 核心组件 ---
        print("正在初始化 pyVHR 组件...")
        # 构造时使用小写，避免 torch.device('CPU') 报错
        skin_extractor = SkinExtractionFaceParsing(device=torch_device)
        # 立刻覆盖成大写，匹配库内部 self.device == 'CPU' / 'GPU' 判断
        skin_extractor.device = device_norm
        self.signal_processor = SignalProcessing()
        self.signal_processor.set_skin_extractor(skin_extractor)
        print("pyVHR 组件初始化完成。")

    @staticmethod
    def _get_video_metadata(video_path):
        """辅助函数: 获取视频的FPS和总帧数"""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps == 0 or total_frames == 0:
            raise ValueError(f"视频文件可能已损坏或元数据无效: {video_path}")
        return fps, total_frames

    def extract_and_save_signals_from_video(self, video_path, output_dir):
        """
        处理单个视频,提取所有窗口的rPPG信号并保存到文件。

        Args:
            video_path (Path): 输入视频文件的路径。
            output_dir (Path): 保存输出JSON文件的目录。
        """
        try:
            fps, total_frames = self._get_video_metadata(video_path)
        except (IOError, ValueError) as e:
            print(f"错误: {e}")?
            return

        # 定义输出文件路径，并检查是否已存在
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file_path = output_dir / f"{video_path.stem}_rppg.json"
        if json_file_path.exists():
            print(f"文件已存在，跳过: {json_file_path}")
            return

        # --- 步骤 1: 一次性提取整个视频的原始RGB信号 ---
        # 这是一个耗时操作，但每个视频只执行一次
        print(f"正在从视频 '{video_path.name}' 提取完整的RGB信号...")
        try:
            # full_rgb_signal 的形状是 (num_frames, 1, 3)
            full_rgb_signal = self.signal_processor.extract_holistic(str(video_path))
            # 移除中间多余的维度 -> (num_frames, 3), 
            # 因为extract_holistic情况下, 选择器数量永远都是1,所以其返回值的
            # [num_frames, 1, rgb_channels]中, 1是多余的.
            full_rgb_signal = np.squeeze(full_rgb_signal, axis=1) #squeeze第二个数
        except Exception as e:
            print(f"!! 在处理视频 '{video_path.name}' 时提取RGB信号失败: {e}")
            return
        
        print(f"RGB信号提取完成, 形状为: {full_rgb_signal.shape}")

        # --- 步骤 2: 应用滑动窗口并转换信号 ---
        window_frames = int(self.window_length_sec * fps)
        # 注意：步长是基于非重叠部分计算的
        step_frames = int(self.step_length_sec * fps)

        all_windows_data = []
        
        print("开始应用滑动窗口并转换为 BVP 信号...")
        for window_id, start_frame in enumerate(range(0, total_frames, step_frames)):
            end_frame = start_frame + window_frames
            # 如果窗口超出视频总长度，则忽略这个窗口
            if end_frame > total_frames:
                continue

            # 2.1 切分窗口
            rgb_window = full_rgb_signal[start_frame:end_frame, :]

            # 2.2 调整信号形状以匹配 cpu_OMIT 的输入要求
            # cpu_OMIT 需要 [num_estimators, rgb_channels, num_frames]
            # 当前是 [num_frames, rgb_channels] -> (window_frames, 3)
            # 转置 -> (3, window_frames)
            # 增加维度 -> (1, 3, window_frames)
            rgb_window_transposed = rgb_window.T
            rgb_window_final = np.expand_dims(rgb_window_transposed, axis=0)

            # 2.3 调用 OMIT 进行转换
            # bvp_window 的形状是 (1, window_frames)
            bvp_window = cpu_OMIT(rgb_window_final)
            # 展平为一维数组
            bvp_signal_list = bvp_window.flatten().tolist()

            # 2.4 收集数据
            window_data = {
                'video_name': video_path.name,
                'window_id': window_id,
                'start_frame': start_frame,
                'end_frame': end_frame - 1,
                'fps': fps,
                'bvp_signal': bvp_signal_list
            }
            all_windows_data.append(window_data)
        
        # --- 步骤 3: 保存到 JSON 文件 ---
        with open(json_file_path, 'w') as f:
            json.dump(all_windows_data, f, indent=4)
            
        print(f"rPPG信号已成功保存到: {json_file_path}")


def process_dataset():
    """
    这是一个专用于论文复现的处理函数。
    它会根据 UBFC-Phys 数据集的特定结构查找视频并提取rPPG信号。
    """
    # --- 1. 配置区域 ---
    # !! 重要: 请在这里设置您的数据集根目录 !!
    # 该目录应包含 s1, s2, ... 等子目录
    DATASET_ROOT_PATH = Path("/mnt/f/DataSet/UBFC-Phys/Data")  # <--- 根据您的截图设置

    # -- 论文复现的固定参数 --
    WINDOW_LENGTH_SEC = 60
    STEP_LENGTH_SEC = 5
    DEVICE = "CPU"  

    # --- 2. 逻辑：根据数据集结构生成视频文件列表 ---
    if not DATASET_ROOT_PATH.exists():
        print(f"错误: 数据集根目录不存在: {DATASET_ROOT_PATH}")
        return

    print("正在根据 UBFC-Phys 结构查找视频文件...")
    all_video_paths = []
    levels = ['T1', 'T2', 'T3']
    
    # 获取所有 subject 目录 (s1, s2, ...) 并进行自然排序
    subject_dirs = sorted(
        [d for d in DATASET_ROOT_PATH.iterdir() if d.is_dir() and d.name.startswith('s')],
        key=lambda p: int(p.name[1:])
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        for level in levels:
            video_path = subject_dir / f"vid_{subject_id}_{level}.avi"
            if video_path.is_file():
                all_video_paths.append(video_path)
            else:
                print(f"警告: 视频文件不存在，已跳过: {video_path}")
    
    if not all_video_paths:
        print(f"在 '{DATASET_ROOT_PATH}' 中没有根据指定规则找到任何视频文件。")
        return

    # --- 3. 执行提取 ---
    extractor = RppgExtractor(
        window_length_sec=WINDOW_LENGTH_SEC,
        step_length_sec=STEP_LENGTH_SEC,
        device=DEVICE
    )

    print(f"共找到 {len(all_video_paths)} 个视频文件。开始处理...")
    for video_path in tqdm.tqdm(all_video_paths, desc="整体进度"):
        # 输出目录将位于每个 subject 文件夹内，名为 'rppg_signals'
        # e.g., /.../s1/rppg_signals/
        subject_dir = video_path.parent
        output_dir = subject_dir / "rppg_signals"
        
        print(f"\n正在处理: {video_path}")
        extractor.extract_and_save_signals_from_video(video_path, output_dir)

    print("\n--- 所有视频处理完成 ---")


if __name__ == '__main__':
    process_dataset()