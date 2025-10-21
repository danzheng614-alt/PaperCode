import pandas as pd
import os
import logging
import numpy as np
import re
from tqdm import tqdm
from scipy.io import loadmat

# --------------------------
# 1. 配置参数
# --------------------------
PARSED_ROOT = "/home/sxzheng/code/sjyck/Time-Series-Library-main/capsule-5421243-code/data/Phase1And2_Paesed_four"  # 需根据实际路径调整
OUTPUT_ROOT = "./final_clean_Phase1And2_fourData"  
LOG_PATH = "final_process_Phase1And2_Fourlog.txt"  # 日志名同步调整
MIN_CONTINUOUS_LENGTH = 400  # 最小连续段长度阈值（400个时间步，对应2秒数据）

# 确保输出目录存在
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# --------------------------
# 2. 受试者特殊规则
# --------------------------
UNSPLIT_TASKS = {
    "BT01": ["curb_up_1_on"],
    "BT11": ["curb_down_1_1", "curb_down_1_2"]
}

INVALID_TASKS = {
    # 补充：BT01（Phase1）的无效力矩任务
    "BT01": ["incline_walk_3_down5_on", "incline_walk_4_down5_off", "incline_walk_5_down10_on", "incline_walk_6_down10_off",
             "dynamic_walk_1_butt-kicks_on", "dynamic_walk_1_high-knees_on", "dynamic_walk_2_butt-kicks_off", 
             "dynamic_walk_2_heel-walk_off", "dynamic_walk_2_high-knees_off", "dynamic_walk_2_toe-walk_off",
             "incline_walk_3_up5_on", "incline_walk_4_up5_off", "incline_walk_5_up10_on", "incline_walk_6_up10_off",
             "normal_walk_1_0-6_on", "normal_walk_1_1-2_on", "normal_walk_1_1-8_on", "normal_walk_3_0-6_off",
             "normal_walk_3_1-2_off", "normal_walk_3_1-8_off", "normal_walk_3_shuffle_off", "normal_walk_3_skip_off",
             "obstacle_walk_1_off", "walk_backward_1_0-6_off", "walk_backward_1_0-6_on", "walk_backward_1_0-8_off",
             "walk_backward_1_0-8_on", "walk_backward_1_1-0_off", "walk_backward_1_1-0_on", "weighted_walk_1_25lbs_off",
             "weighted_walk_1_25lbs_on"],
    "BT09": ["lift_weight_1_2"],
    "BT13": ["curb_down_1_1_left_on", "curb_down_1_1_right_on",
             "curb_up_1_1_right_on", "dynamic_walk_1_1_high-knees_on",
             "stairs_1_1_4"],
    "BT14": ["stairs_1_1_8"],
    "BT17": ["side_shuffle_1_1"],
    "BT20": ["walk_backward_1_1_1-0_on"],
    "BT23": ["stairs_2", "walk_backward_1_1_1-0_on"]
}

EXTRA_TASKS = {
    "BT02": ["lunges_1_1"],
    "BT03": ["lunges_1_1"],
    "BT07": ["stairs_2_11_up_off", "stairs_2_12_down_on", "walk_backward_2_1"],
    "BT10": ["stairs_2_11_up_off", "stairs_2_12_down_on"],
    "BT12": ["normal_walk_1_1_0-6_on", "normal_walk_1_1_1-2_on"],
    "BT17": ["lunges_1_1_set6_on"]
}

HILO_IDENTIFIER = "hilo"  # 启发式控制器标识

CONTROL_MODES = {
    "BT13": "闭环（第一天：净力矩；第二天：生物力矩）",
    "BT14": "闭环（生物力矩）",
    "BT15": "闭环（生物力矩）",
    "BT16": "闭环（生物力矩）",
    "BT17": "闭环（生物力矩）"
}

SENSOR_ISSUES = {
    "BT06": {
        "type": "grf_ml_noise",
        "fields": ["fp_left_force_x", "fp_left_force_z"],
        "threshold": 1e-6
    }
}

STATIC_TASKS = ["poses"]
DYNAMIC_TASKS = ["normal_walk", "incline_walk", "dynamic_walk",
                 "curb_up", "curb_down", "lunges", "stairs",
                 "walk_backward", "side_shuffle", "tire_run",
                 "lift_weight", "jump", "cutting", "step_ups"]

INVALID_SUBJECT_SUFFIX = "_Slice"

# 用于存储各任务的最大连续段长度，用于跨个体统一长度
TASK_SEGMENT_LENGTHS = {}

# --------------------------
# 3. 辅助函数
# --------------------------
def get_valid_legs(moment_df):
    """
    筛选有效的左右腿力矩字段，并检查是否全为NaN
    返回: 
        - 若双腿均非全NaN: (True, left_cols, right_cols)
        - 若任一条腿全NaN: (False, None, None)
    """
    # 定义正则模式（精确匹配腿标识，不区分大小写）
    left_pattern = re.compile(r'_l_|^l_|_l$|_left_', re.IGNORECASE)
    right_pattern = re.compile(r'_r_|^r_|_r$|_right_', re.IGNORECASE)
    exclude_combo = ["lr", "rl", "lrl", "rlr"]  # 排除左右腿组合字段
    
    # 左腿力矩字段筛选
    left_moment_cols = []
    for col in moment_df.columns:
        col_lower = col.lower()
        if ("moment" in col_lower and
            left_pattern.search(col_lower) and
            not right_pattern.search(col_lower) and
            not any(excl in col_lower for excl in exclude_combo)):
            left_moment_cols.append(col)
        elif "moment" in col_lower and left_pattern.search(col_lower):
            if right_pattern.search(col_lower):
                logging.debug(f"排除字段[{col}]：含左腿标识但同时含右腿标识")
            if any(excl in col_lower for excl in exclude_combo):
                logging.debug(f"排除字段[{col}]：含左腿标识但含组合词")
    
    logging.debug(f"筛选到的左腿力矩字段：{left_moment_cols}")
    if not left_moment_cols:
        logging.error("未筛选到任何左腿力矩字段")
        return (False, None, None)
    
    # 右腿力矩字段筛选
    right_moment_cols = []
    for col in moment_df.columns:
        col_lower = col.lower()
        if ("moment" in col_lower and
            right_pattern.search(col_lower) and
            not left_pattern.search(col_lower) and
            not any(excl in col_lower for excl in exclude_combo)):
            right_moment_cols.append(col)
        elif "moment" in col_lower and right_pattern.search(col_lower):
            if left_pattern.search(col_lower):
                logging.debug(f"排除字段[{col}]：含右腿标识但同时含左腿标识")
            if any(excl in col_lower for excl in exclude_combo):
                logging.debug(f"排除字段[{col}]：含右腿标识但含组合词")
    
    logging.debug(f"筛选到的右腿力矩字段：{right_moment_cols}")
    if not right_moment_cols:
        logging.error("未筛选到任何右腿力矩字段")
        return (False, None, None)
    
    # 检查是否全为NaN
    left_all_nan = moment_df[left_moment_cols].isna().all().all()
    right_all_nan = moment_df[right_moment_cols].isna().all().all()
    
    if left_all_nan:
        logging.error("左腿力矩字段全为NaN")
        return (False, None, None)
    
    if right_all_nan:
        logging.error("右腿力矩字段全为NaN")
        return (False, None, None)
    
    return (True, left_moment_cols, right_moment_cols)

def find_longest_continuous_segment(moment_df, left_cols, right_cols):
    """
    寻找双腿力矩均有效的最长连续段
    返回: (start_idx, end_idx, length)
    """
    # 标记有效行（双腿均无NaN）
    valid_rows = ~moment_df[left_cols].isna().any(axis=1) & ~moment_df[right_cols].isna().any(axis=1)
    
    max_length = 0
    current_length = 0
    start_idx = 0
    current_start = 0
    end_idx = 0
    
    for i, is_valid in enumerate(valid_rows):
        if is_valid:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_idx = current_start
                end_idx = i
        else:
            current_length = 0
    
    return (start_idx, end_idx, max_length)

def is_task_split(subject, task_name):
    """
    判断任务是否拆分左右腿：
    - 仅 step_ups（上台阶）任务的 _left/_right 表示“肢体拆分”
    - 其他任务（如 lunges、ball_toss、turn_and_step）的 _left/_right 是“方向”，不拆分
    """
    # 1. 先检查是否在“特殊不拆分任务”列表中（优先排除）
    if subject in UNSPLIT_TASKS and task_name in UNSPLIT_TASKS[subject]:
        return False
    
    # 2. 仅 step_ups 任务的 _left/_right 表示“拆分左右腿”
    if "step_ups" in task_name.lower():  # 判断是否是 step_ups 任务
        return "_left" in task_name.lower() or "_right" in task_name.lower()
    
    # 3. 其他任务的 _left/_right 是“方向”，不拆分
    return False

def get_exo_state(task_name):
    """判断外骨骼状态"""
    if "_on" in task_name.lower():
        return "on"
    elif "_off" in task_name.lower():
        return "off"
    else:
        return "unknown"

def get_task_type(task_name):
    """判断任务类型"""
    if any(static in task_name.lower() for static in STATIC_TASKS):
        return "static"
    elif any(dynamic in task_name.lower() for dynamic in DYNAMIC_TASKS):
        return "dynamic"
    else:
        return "other"

def read_csv_auto_delim(path):
    """自动识别分隔符读取CSV"""
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:  # 若仅1列，尝试分号分隔
            df = pd.read_csv(path, sep=";")
        return df
    except Exception as e:
        logging.error(f"读取 CSV 失败 {path}：{str(e)}")
        raise e

def clean_task_name(task_name):
    """清理任务名称"""
    cleaned_name = task_name.replace(HILO_IDENTIFIER, "")
    return cleaned_name

# --------------------------
# 4. 单个任务处理函数（第一阶段：统计最大连续段长度）
# --------------------------
def process_single_task_statistics(exo_path, moment_path, task_dir, subject, task_name):
    try:
        # 任务跳过逻辑
        if subject in INVALID_TASKS and task_name in INVALID_TASKS[subject]:
            logging.info(f"[{subject}] {task_name}：跳过，已知无效任务")
            return None

        # 加载数据
        exo_df = read_csv_auto_delim(exo_path)
        moment_df = read_csv_auto_delim(moment_path)
        if "time" not in exo_df.columns or "time" not in moment_df.columns:
            logging.error(f"[{subject}] {task_name} 失败：时间戳字段不是 'time'")
            return None

        # 验证腿力矩字段
        legs_valid, left_cols, right_cols = get_valid_legs(moment_df)
        if not legs_valid:
            logging.error(f"[{subject}] {task_name} 失败：单腿或双腿全为NaN")
            return None

        # activity_flag筛选（先应用活动筛选）
        activity_files = [f for f in os.listdir(task_dir) if "activity_flag.csv" in f]
        if activity_files:
            activity_path = os.path.join(task_dir, activity_files[0])
            activity_df = read_csv_auto_delim(activity_path)
            if "left" in activity_df.columns and "right" in activity_df.columns:
                activity_valid = activity_df[(activity_df["left"] == 1) | (activity_df["right"] == 1)]
                valid_times = set(activity_valid["time"])
                exo_valid = exo_df[exo_df["time"].isin(valid_times)]
                moment_valid = moment_df[moment_df["time"].isin(valid_times)]
                logging.info(f"[{subject}] {task_name}：activity_flag 筛选，保留 {len(exo_valid)}/{len(exo_df)} 行数据")
            else:
                exo_valid = exo_df.copy()
                moment_valid = moment_df.copy()
        else:
            exo_valid = exo_df.copy()
            moment_valid = moment_df.copy()

        # 数据同步
        moment_times = set(moment_valid["time"])
        exo_sync = exo_valid[exo_valid["time"].isin(moment_times)].sort_values("time").reset_index(drop=True)
        moment_sync = moment_valid[moment_valid["time"].isin(set(exo_sync["time"]))].sort_values("time").reset_index(drop=True)
        if len(exo_sync) == 0:
            logging.error(f"[{subject}] {task_name} 失败：同步后无数据")
            return None

        # 寻找最长连续段
        start_idx, end_idx, max_length = find_longest_continuous_segment(moment_sync, left_cols, right_cols)
        logging.info(f"[{subject}] {task_name}：最长连续段长度：{max_length}，是否满足阈值：{max_length >= MIN_CONTINUOUS_LENGTH}")
        
        return max_length

    except Exception as e:
        logging.error(f"[{subject}] {task_name} 统计失败：{str(e)}")
        return None

# --------------------------
# 5. 单个任务处理函数（第二阶段：根据统一长度截取）
# --------------------------
def process_single_task_crop(exo_path, moment_path, task_dir, subject, task_name, target_length):
    try:
        # 任务跳过逻辑
        if subject in INVALID_TASKS and task_name in INVALID_TASKS[subject]:
            logging.info(f"[{subject}] {task_name}：跳过，已知无效任务")
            return True

        # 加载数据
        exo_df = read_csv_auto_delim(exo_path)
        moment_df = read_csv_auto_delim(moment_path)
        if "time" not in exo_df.columns or "time" not in moment_df.columns:
            logging.error(f"[{subject}] {task_name} 失败：时间戳字段不是 'time'")
            return False

        # 验证腿力矩字段
        legs_valid, left_cols, right_cols = get_valid_legs(moment_df)
        if not legs_valid:
            logging.error(f"[{subject}] {task_name} 失败：单腿或双腿全为NaN")
            return False

        # 任务属性
        task_split = is_task_split(subject, task_name)
        exo_state = get_exo_state(task_name)
        task_type = get_task_type(task_name)
        cleaned_task_name = clean_task_name(task_name)
        
        if not task_split:
            logging.info(f"[{subject}] {task_name}：未拆分左右腿")
        else:
            logging.info(f"[{subject}] {task_name}：已拆分左右腿")

        # 保留exo字段（排除torque_estimated）
        exo_valid = exo_df.copy()
        if "torque_estimated" in exo_valid.columns:
            exo_valid = exo_valid.drop(columns=["torque_estimated"])
            logging.info(f"[{subject}] {task_name}：排除 torque_estimated 字段")

        # 外骨骼关闭时删除控制字段
        if exo_state == "off":
            control_fields = [col for col in exo_valid.columns 
                            if "torque_desired" in col or "torque_measured" in col]
            if control_fields:
                exo_valid = exo_valid.drop(columns=control_fields)
                logging.info(f"[{subject}] {task_name}：外骨骼关闭，删除控制字段：{', '.join(control_fields)}")

        # 筛选有效腿力矩数据
        moment_cols = ["time"] + left_cols + right_cols
        moment_valid = moment_df[moment_cols].copy()

        # activity_flag筛选（先应用活动筛选）
        activity_files = [f for f in os.listdir(task_dir) if "activity_flag.csv" in f]
        if activity_files:
            activity_path = os.path.join(task_dir, activity_files[0])
            activity_df = read_csv_auto_delim(activity_path)
            if "left" in activity_df.columns and "right" in activity_df.columns:
                activity_valid = activity_df[(activity_df["left"] == 1) | (activity_df["right"] == 1)]
                valid_times = set(activity_valid["time"])
                exo_valid = exo_valid[exo_valid["time"].isin(valid_times)]
                moment_valid = moment_valid[moment_valid["time"].isin(valid_times)]
                logging.info(f"[{subject}] {task_name}：activity_flag 筛选，保留 {len(exo_valid)}/{len(exo_df)} 行数据")

        # 数据同步
        moment_times = set(moment_valid["time"])
        exo_sync = exo_valid[exo_valid["time"].isin(moment_times)].sort_values("time").reset_index(drop=True)
        moment_sync = moment_valid[moment_valid["time"].isin(set(exo_sync["time"]))].sort_values("time").reset_index(drop=True)
        if len(exo_sync) == 0:
            logging.error(f"[{subject}] {task_name} 失败：同步后无数据")
            return False

        # 寻找最长连续段并截取
        start_idx, end_idx, max_length = find_longest_continuous_segment(moment_sync, left_cols, right_cols)
        
        if max_length < target_length:
            logging.error(f"[{subject}] {task_name} 失败：最长连续段长度 {max_length} < 目标长度 {target_length}")
            return False

        # 截取目标长度的连续段（从最长段的起始位置开始）
        segment_end_idx = start_idx + target_length - 1
        if segment_end_idx > len(moment_sync) - 1:
            logging.error(f"[{subject}] {task_name} 失败：无法截取目标长度的连续段")
            return False

        # 获取连续段的时间范围
        start_time = moment_sync.iloc[start_idx]["time"]
        end_time = moment_sync.iloc[segment_end_idx]["time"]

        # 截取exo和moment数据
        exo_segment = exo_sync[(exo_sync["time"] >= start_time) & (exo_sync["time"] <= end_time)].reset_index(drop=True)
        moment_segment = moment_sync[(moment_sync["time"] >= start_time) & (moment_sync["time"] <= end_time)].reset_index(drop=True)

        # 验证截取后的数据长度
        if len(exo_segment) != target_length or len(moment_segment) != target_length:
            logging.error(f"[{subject}] {task_name} 失败：截取后长度不匹配，exo: {len(exo_segment)}, moment: {len(moment_segment)}")
            return False

        # 传感器噪声修正
        if subject in SENSOR_ISSUES:
            issue = SENSOR_ISSUES[subject]
            if issue["type"] == "grf_ml_noise":
                grf_files = [f for f in os.listdir(task_dir) if f.endswith("_grf.csv")]
                if grf_files:
                    grf_path = os.path.join(task_dir, grf_files[0])
                    grf_df = read_csv_auto_delim(grf_path)
                    # 仅保留连续段时间范围内的数据
                    grf_segment = grf_df[(grf_df["time"] >= start_time) & (grf_df["time"] <= end_time)].reset_index(drop=True)
                    
                    for field in issue["fields"]:
                        if field in grf_segment.columns:
                            grf_segment[field] = grf_segment[field].apply(lambda x: pd.NA if abs(x) < issue["threshold"] else x)
                    
                    output_dir = os.path.join(OUTPUT_ROOT, subject, task_name)
                    os.makedirs(output_dir, exist_ok=True)
                    grf_save = os.path.join(output_dir, f"{os.path.basename(grf_path).replace('.csv', '_clean.csv')}")
                    grf_segment.to_csv(grf_save, index=False)
                    logging.info(f"[{subject}] {task_name}：修正 GRF 噪声字段")

        # 动态任务周期统计（基于截取的连续段）
        if task_type == "dynamic":
            # 尝试基于力矩数据识别周期
            # 这里简单实现：计算力矩峰值间隔来识别周期
            cycle_counts = {}
            for leg, cols in [("left", left_cols), ("right", right_cols)]:
                # 取第一个力矩字段计算峰值
                if cols:
                    moment_data = moment_segment[cols[0]].values
                    # 简单峰值检测（大于前后相邻值）
                    peaks = np.where((moment_data[1:-1] > moment_data[:-2]) & (moment_data[1:-1] > moment_data[2:]))[0] + 1
                    cycle_counts[leg] = len(peaks) - 1  # 峰值间的间隔为周期数
            
            logging.info(f"[{subject}] {task_name}：动态任务，连续段内周期统计 - 左腿: {cycle_counts.get('left', 0)}, 右腿: {cycle_counts.get('right', 0)}")

        # 保存数据
        output_task_dir = os.path.join(OUTPUT_ROOT, subject, task_name)
        os.makedirs(output_task_dir, exist_ok=True)
        length_suffix = f"_{target_length}len"

        # 调整：含HILO标识的任务添加_hilo后缀，区分启发式控制器数据
        if HILO_IDENTIFIER in task_name.lower():
            exo_filename = os.path.basename(exo_path).replace(".csv", f"_{HILO_IDENTIFIER}{length_suffix}_clean.csv")
            moment_filename = os.path.basename(moment_path).replace(".csv", f"_{HILO_IDENTIFIER}{length_suffix}_clean.csv")
        else:
            exo_filename = os.path.basename(exo_path).replace(".csv", f"{length_suffix}_clean.csv")
            moment_filename = os.path.basename(moment_path).replace(".csv", f"{length_suffix}_clean.csv")

        # 保存exo数据
        exo_save = os.path.join(output_task_dir, exo_filename)
        exo_segment.to_csv(exo_save, index=False)

        # 保存力矩数据
        moment_save = os.path.join(output_task_dir, moment_filename)
        moment_segment.to_csv(moment_save, index=False)

        logging.info(f"[{subject}] {task_name} 成功：保存至 {output_task_dir}，长度: {target_length}")
        return True

    except Exception as e:
        logging.error(f"[{subject}] {task_name} 失败：{str(e)}")
        return False

# --------------------------
# 6. 批量处理主函数
# --------------------------
def batch_process():
    global TASK_SEGMENT_LENGTHS
    
    # 第一步：统计各任务的最大连续段长度
    logging.info("===== 开始第一阶段：统计各任务的最大连续段长度 =====")
    for subject in tqdm(os.listdir(PARSED_ROOT), desc="第一阶段：处理受试者"):
        subject_dir = os.path.join(PARSED_ROOT, subject)
        if INVALID_SUBJECT_SUFFIX in subject:
            logging.info(f"跳过无效受试者文件夹：{subject}")
            continue
        if not os.path.isdir(subject_dir):
            logging.info(f"跳过非文件夹：{subject}")
            continue

        task_list = os.listdir(subject_dir)
        if not task_list:
            logging.info(f"[{subject}] 无任务文件夹，跳过")
            continue

        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            if not os.path.isdir(task_dir):
                logging.info(f"[{subject}] 跳过非任务文件夹：{task}")
                continue
            if subject in INVALID_TASKS and task in INVALID_TASKS[subject]:
                logging.info(f"[{subject}] {task}：跳过无效任务")
                continue

            # 查找exo和moment文件
            exo_files = [f for f in os.listdir(task_dir)
                        if f.endswith("exo.csv") and "power" not in f and not f.startswith(".")]
            moment_files = [f for f in os.listdir(task_dir)
                          if f.endswith("moment_filt.csv") and not f.startswith(".")]

            if len(exo_files) != 1 or len(moment_files) != 1:
                logging.warning(f"[{subject}] {task}：文件数量异常，跳过统计")
                continue

            exo_path = os.path.join(task_dir, exo_files[0])
            moment_path = os.path.join(task_dir, moment_files[0])
            max_length = process_single_task_statistics(exo_path, moment_path, task_dir, subject, task)
            
            if max_length and max_length >= MIN_CONTINUOUS_LENGTH:
                # 使用清理后的任务名作为键，确保同一任务的不同变体被归为一类
                cleaned_task = clean_task_name(task)
                if cleaned_task not in TASK_SEGMENT_LENGTHS:
                    TASK_SEGMENT_LENGTHS[cleaned_task] = []
                TASK_SEGMENT_LENGTHS[cleaned_task].append(max_length)

    # 计算每个任务的目标长度（取所有受试者该任务最大连续段长度的最小值）
    TASK_TARGET_LENGTHS = {}
    for task, lengths in TASK_SEGMENT_LENGTHS.items():
        if lengths:
            min_length = min(lengths)
            # 确保目标长度不小于最小连续段阈值
            TASK_TARGET_LENGTHS[task] = max(min_length, MIN_CONTINUOUS_LENGTH)
            logging.info(f"任务 {task} 目标长度：{TASK_TARGET_LENGTHS[task]}（基于 {len(lengths)} 个有效受试者）")
        else:
            logging.warning(f"任务 {task} 没有有效长度数据，将被排除")

    # 第二步：根据目标长度处理并截取各任务
    logging.info("\n===== 开始第二阶段：根据目标长度处理任务 =====")
    success_count = 0
    fail_count = 0
    skip_count = 0
    invalid_subject_count = 0

    for subject in tqdm(os.listdir(PARSED_ROOT), desc="第二阶段：处理受试者"):
        subject_dir = os.path.join(PARSED_ROOT, subject)
        if INVALID_SUBJECT_SUFFIX in subject:
            logging.info(f"跳过无效受试者文件夹：{subject}")
            invalid_subject_count += 1
            continue
        if not os.path.isdir(subject_dir):
            logging.info(f"跳过非文件夹：{subject}")
            skip_count += 1
            continue

        task_list = os.listdir(subject_dir)
        if not task_list:
            logging.info(f"[{subject}] 无任务文件夹，跳过")
            skip_count += 1
            continue

        for task in task_list:
            task_dir = os.path.join(subject_dir, task)
            if not os.path.isdir(task_dir):
                logging.info(f"[{subject}] 跳过非任务文件夹：{task}")
                skip_count += 1
                continue
            if subject in INVALID_TASKS and task in INVALID_TASKS[subject]:
                logging.info(f"[{subject}] {task}：跳过无效任务")
                skip_count += 1
                continue

            # 查找exo和moment文件
            exo_files = [f for f in os.listdir(task_dir)
                        if f.endswith("exo.csv") and "power" not in f and not f.startswith(".")]
            moment_files = [f for f in os.listdir(task_dir)
                          if f.endswith("moment_filt.csv") and not f.startswith(".")]

            if len(exo_files) == 0:
                logging.error(f"[{subject}] {task} 失败：未找到 xxx_exo.csv")
                fail_count += 1
                continue
            elif len(moment_files) == 0:
                logging.error(f"[{subject}] {task} 失败：未找到 xxx_moment_filt.csv")
                fail_count += 1
                continue
            elif len(exo_files) != 1 or len(moment_files) != 1:
                logging.error(f"[{subject}] {task} 失败：文件数量异常")
                fail_count += 1
                continue

            # 检查任务是否有目标长度
            cleaned_task = clean_task_name(task)
            if cleaned_task not in TASK_TARGET_LENGTHS:
                logging.info(f"[{subject}] {task}：无有效目标长度，跳过")
                skip_count += 1
                continue

            target_length = TASK_TARGET_LENGTHS[cleaned_task]
            exo_path = os.path.join(task_dir, exo_files[0])
            moment_path = os.path.join(task_dir, moment_files[0])
            
            result = process_single_task_crop(exo_path, moment_path, task_dir, subject, task, target_length)
            if result:
                success_count += 1
            else:
                fail_count += 1

    # 输出统计结果
    total = success_count + fail_count + skip_count
    print(f"\n 批量处理完成：")
    print(f"总受试者数：{len(os.listdir(PARSED_ROOT))}（跳过无效受试者 {invalid_subject_count} 个）")
    print(f"总任务数：{total}")
    print(f"成功：{success_count} 个")
    print(f"失败：{fail_count} 个")
    print(f"其他跳过：{skip_count} 个")
    print(f"详细日志：{LOG_PATH}")
    logging.info(f"批量处理完成：成功 {success_count}/{total}，失败 {fail_count}/{total}，其他跳过 {skip_count}/{total}，跳过无效受试者 {invalid_subject_count} 个")

# --------------------------
# 7. 运行入口
# --------------------------
if __name__ == "__main__":
    if not os.path.exists(PARSED_ROOT):
        print(f"错误：Parsed 根目录不存在 - {PARSED_ROOT}")
    else:
        batch_process()