import pandas as pd
import os
import logging
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

# --------------------------
# 1. 配置参数（请确认路径正确）
# --------------------------
PARSED_ROOT = "/home/sxzheng/code/sjyck/Time-Series-Library-main/capsule-5421243-code/data/test"
OUTPUT_ROOT = "./final_clean_data"
LOG_PATH = "final_process_log.txt"

# 确保输出目录存在
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 配置日志（增加控制台输出，方便实时查看）
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# --------------------------
# 2. 受试者特殊规则（保留原配置）
# --------------------------
UNSPLIT_TASKS = {
    "BT01": ["curb_up_1_on"],
    "BT11": ["curb_down_1_1", "curb_down_1_2"]
}
INVALID_TASKS = {
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
HILO_IDENTIFIER = "_hilo"
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
                 "lift_weight", "jump", "cutting"]
SIGN_CORRECTION_FIELDS = ["hip_angle_", "knee_angle_"]
SIGN_CORRECTION_SUFFIXES = ["deg", "velocity"]

# 无效受试者文件夹标识（需跳过）
INVALID_SUBJECT_SUFFIX = "_Slice"


# --------------------------
# 3. 辅助函数（删除插值填充相关函数）
# --------------------------
def get_valid_legs(moment_df, nan_ratio_threshold=0.05):
    """用NaN比例阈值判断有效腿（默认允许5%以内NaN）"""
    left_keywords = ["_l_", "left", "l_", "_l"]
    right_keywords = ["_r_", "right", "r_", "_r"]
    
    left_valid = False
    left_moment_cols = [col for col in moment_df.columns if (any(kw in col.lower() for kw in left_keywords)) and "moment" in col.lower()]
    if left_moment_cols:
        total_left_values = len(moment_df) * len(left_moment_cols)
        left_nan_count = moment_df[left_moment_cols].isna().sum().sum()
        left_nan_ratio = left_nan_count / total_left_values if total_left_values > 0 else 1.0
        left_valid = (left_nan_ratio <= nan_ratio_threshold)
    
    right_valid = False
    right_moment_cols = [col for col in moment_df.columns if (any(kw in col.lower() for kw in right_keywords)) and "moment" in col.lower()]
    if right_moment_cols:
        total_right_values = len(moment_df) * len(right_moment_cols)
        right_nan_count = moment_df[right_moment_cols].isna().sum().sum()
        right_nan_ratio = right_nan_count / total_right_values if total_right_values > 0 else 1.0
        right_valid = (right_nan_ratio <= nan_ratio_threshold)
    
    valid_legs = []
    if left_valid:
        valid_legs.append("left")
    if right_valid:
        valid_legs.append("right")
    return valid_legs

def is_task_split(subject, task_name):
    if subject in UNSPLIT_TASKS and task_name in UNSPLIT_TASKS[subject]:
        return False
    return "_left" in task_name.lower() or "_right" in task_name.lower()

def get_exo_state(task_name):
    if "_on" in task_name.lower():
        return "on"
    elif "_off" in task_name.lower():
        return "off"
    else:
        return "unknown"

def get_task_type(task_name):
    if any(static in task_name.lower() for static in STATIC_TASKS):
        return "static"
    elif any(dynamic in task_name.lower() for dynamic in DYNAMIC_TASKS):
        return "dynamic"
    else:
        return "other"

def correct_exo_sign(exo_df):
    correct_cols = [col for col in exo_df.columns if (any(field in col.lower() for field in SIGN_CORRECTION_FIELDS) and any(suffix in col.lower() for suffix in SIGN_CORRECTION_SUFFIXES))]
    for col in correct_cols:
        exo_df[col] = exo_df[col] * -1
    return exo_df, correct_cols

def read_csv_auto_delim(path):
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception as e:
        logging.error(f"读取CSV失败 {path}：{str(e)}")
        raise e

def clean_task_name(task_name):
    cleaned_name = task_name.replace(HILO_IDENTIFIER, "")
    return cleaned_name


# --------------------------
# 4. 单个任务处理函数（删除插值填充调用）
# --------------------------
def process_single_task(exo_path, moment_path, task_dir, subject, task_name):
    try:
        if subject in INVALID_TASKS and task_name in INVALID_TASKS[subject]:
            logging.info(f"[{subject}] {task_name}：跳过，已知无效任务")
            return True
        
        if subject in EXTRA_TASKS and task_name in EXTRA_TASKS[subject]:
            logging.info(f"[{subject}] {task_name}：检测到额外测试数据")
        
        if subject in CONTROL_MODES:
            logging.info(f"[{subject}] {task_name}：外骨骼控制模式 - {CONTROL_MODES[subject]}")
        
        if HILO_IDENTIFIER in task_name:
            logging.info(f"[{subject}] {task_name}：含高低启发式控制器数据")
        
        exo_df = read_csv_auto_delim(exo_path)
        moment_df = read_csv_auto_delim(moment_path)
        
        if "time" not in exo_df.columns or "time" not in moment_df.columns:
            logging.error(f"[{subject}] {task_name} 失败：时间戳字段不是'time'")
            return False
        
        task_split = is_task_split(subject, task_name)
        exo_state = get_exo_state(task_name)
        task_type = get_task_type(task_name)
        cleaned_task_name = clean_task_name(task_name)
        leg_keywords_map = {"left": ["_l_", "left", "l_", "_l"], "right": ["_r_", "right", "r_", "_r"]}
        
        # 未拆分任务优化：初始有效腿为空则跳过
        if not task_split:
            valid_legs_initial = get_valid_legs(moment_df)
            if not valid_legs_initial:
                logging.info(f"[{subject}] {task_name}：未拆分左右腿，但初始有效腿为空，跳过")
                return True
            valid_legs = ["left", "right"]
            logging.info(f"[{subject}] {task_name}：未拆分左右腿，强制保留双腿字段（初始有效腿：{valid_legs_initial}）")
        else:
            valid_legs = get_valid_legs(moment_df)
            if not valid_legs:
                logging.error(f"[{subject}] {task_name} 失败：双腿力矩均有NaN")
                return False
            logging.info(f"[{subject}] {task_name} 有效腿：{valid_legs}")
        
        # 筛选外骨骼数据
        exo_cols = ["time"]
        for leg in valid_legs:
            leg_keywords = leg_keywords_map[leg]
            opposite_leg = "right" if leg == "left" else "left"
            opposite_keywords = leg_keywords_map[opposite_leg]
            leg_exo_cols = [col for col in exo_df.columns if (any(kw in col.lower() for kw in leg_keywords)) 
                           and not (any(kw in col.lower() for kw in opposite_keywords)) 
                           and "torque_estimated" not in col.lower()]
            exo_cols.extend(leg_exo_cols)
        
        exo_cols = list(set(exo_cols))
        exo_cols.sort(key=lambda x: 0 if x == "time" else 1)
        if len(exo_cols) <= 1:
            logging.error(f"[{subject}] {task_name} 失败：未找到有效腿的输入特征")
            return False
        exo_valid = exo_df[exo_cols].copy()
        
        # 外骨骼关闭时删除控制字段
        if exo_state == "off":
            control_fields = [col for col in exo_valid.columns if "torque_desired" in col or "torque_measured" in col]
            exo_valid = exo_valid.drop(columns=control_fields)
            logging.info(f"[{subject}] {task_name}：外骨骼关闭，删除 {len(control_fields)} 个控制字段")
        
        # 修正角度字段符号
        exo_valid, corrected_cols = correct_exo_sign(exo_valid)
        if corrected_cols:
            logging.info(f"[{subject}] {task_name}：修正 {len(corrected_cols)} 个字段的符号")
        
        # 筛选力矩数据
        moment_cols = ["time"]
        for leg in valid_legs:
            leg_keywords = leg_keywords_map[leg]
            leg_moment_cols = [col for col in moment_df.columns if (any(kw in col.lower() for kw in leg_keywords)) and "moment" in col.lower()]
            moment_cols.extend(leg_moment_cols)
        moment_valid = moment_df[moment_cols].copy()
        
        # activity_flag筛选
        activity_files = [f for f in os.listdir(task_dir) if "activity_flag.csv" in f]
        if activity_files:
            activity_path = os.path.join(task_dir, activity_files[0])
            activity_df = read_csv_auto_delim(activity_path)
            if "left" in activity_df.columns and "right" in activity_df.columns:
                activity_valid = activity_df[(activity_df["left"] == 1) | (activity_df["right"] == 1)]
                valid_times = set(activity_valid["time"])
                original_exo_rows = len(exo_valid)
                original_moment_rows = len(moment_valid)
                exo_valid = exo_valid[exo_valid["time"].isin(valid_times)]
                moment_valid = moment_valid[moment_valid["time"].isin(valid_times)]
                logging.info(f"[{subject}] {task_name}：通过activity_flag筛选，保留 {len(exo_valid)}/{original_exo_rows} 行数据")
        
        # （已删除：插值填充相关代码）
        
        # 数据同步
        moment_times = set(moment_valid["time"])
        exo_sync = exo_valid[exo_valid["time"].isin(moment_times)].sort_values("time").reset_index(drop=True)
        moment_sync = moment_valid[moment_valid["time"].isin(set(exo_sync["time"]))].sort_values("time").reset_index(drop=True)
        if len(exo_sync) == 0:
            logging.error(f"[{subject}] {task_name} 失败：同步后无数据")
            return False
        
        # 传感器噪声修正
        if subject in SENSOR_ISSUES:
            issue = SENSOR_ISSUES[subject]
            if issue["type"] == "grf_ml_noise":
                grf_files = [f for f in os.listdir(task_dir) if f.endswith("_grf.csv")]
                if grf_files:
                    grf_path = os.path.join(task_dir, grf_files[0])
                    grf_df = read_csv_auto_delim(grf_path)
                    for field in issue["fields"]:
                        if field in grf_df.columns:
                            grf_df[field] = grf_df[field].apply(lambda x: pd.NA if abs(x) < issue["threshold"] else x)
                    output_task_dir = os.path.join(OUTPUT_ROOT, subject, task_name)
                    os.makedirs(output_task_dir, exist_ok=True)
                    output_grf_path = os.path.join(output_task_dir, f"{os.path.basename(grf_path).replace('.csv', '_clean.csv')}")
                    grf_df.to_csv(output_grf_path, index=False)
                    logging.info(f"[{subject}] {task_name}：修正GRF噪声字段")
        
        # 动态任务周期统计
        if task_type == "dynamic":
            parsed_parent = os.path.dirname(os.path.dirname(task_dir))
            segmented_dir = os.path.join(parsed_parent, "Segmented", subject, task_name)
            if os.path.exists(segmented_dir):
                parsing_files = [f for f in os.listdir(segmented_dir) if "_parsing.mat" in f]
                if parsing_files:
                    parsing_mat = loadmat(os.path.join(segmented_dir, parsing_files[0]))
                    left_cycles = len(parsing_mat.get("left_cycles", []))
                    right_cycles = len(parsing_mat.get("right_cycles", []))
                    logging.info(f"[{subject}] {task_name}：动态任务，左腿周期 {left_cycles} 个，右腿周期 {right_cycles} 个")
        
        # 同步后NaN校验（按任务是否拆分校验）
        if not task_split:
            # 未拆分任务：仅校验初始有效腿的字段
            initial_valid_moment_cols = ["time"]
            for leg in valid_legs_initial:
                leg_keywords = leg_keywords_map[leg]
                initial_valid_moment_cols.extend([col for col in moment_sync.columns 
                                               if any(kw in col.lower() for kw in leg_keywords) 
                                               and "moment" in col.lower()])
            exo_nan = exo_sync.isna().sum().sum() > 0
            moment_nan = moment_sync[initial_valid_moment_cols].isna().sum().sum() > 0
            if exo_nan or moment_nan:
                logging.error(f"[{subject}] {task_name} 失败：同步后有效腿字段存在NaN")
                return False
        else:
            # 拆分任务：校验所有有效腿的字段
            valid_moment_cols = ["time"]
            for leg in valid_legs:
                leg_keywords = leg_keywords_map[leg]
                valid_moment_cols.extend([col for col in moment_sync.columns 
                                       if any(kw in col.lower() for kw in leg_keywords) 
                                       and "moment" in col.lower()])
            if exo_sync.isna().sum().sum() > 0 or moment_sync[valid_moment_cols].isna().sum().sum() > 0:
                logging.error(f"[{subject}] {task_name} 失败：同步后有效腿字段存在NaN")
                return False
        
        # 保存清洗后的数据
        output_task_dir = os.path.join(OUTPUT_ROOT, subject, task_name)
        os.makedirs(output_task_dir, exist_ok=True)
        leg_suffix = "mixed_legs" if not task_split else "_".join(valid_legs)
        
        exo_filename = os.path.basename(exo_path).replace(".csv", f"_{leg_suffix}_clean.csv")
        exo_save_path = os.path.join(output_task_dir, exo_filename)
        exo_sync.to_csv(exo_save_path, index=False)
        
        moment_filename = os.path.basename(moment_path).replace(".csv", f"_{leg_suffix}_clean.csv")
        moment_save_path = os.path.join(output_task_dir, moment_filename)
        moment_sync.to_csv(moment_save_path, index=False)
        
        logging.info(f"[{subject}] {task_name} 成功：保存至 {output_task_dir}")
        return True
    
    except Exception as e:
        logging.error(f"[{subject}] {task_name} 失败：{str(e)}")
        return False


# --------------------------
# 5. 批量处理主函数
# --------------------------
def batch_process():
    success_count = 0
    fail_count = 0
    skip_count = 0
    invalid_subject_count = 0
    
    for subject in tqdm(os.listdir(PARSED_ROOT), desc="处理受试者"):
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
            
            exo_files = [
                f for f in os.listdir(task_dir) 
                if f.endswith("_exo.csv") and "power" not in f and not f.startswith("._")
            ]
            moment_files = [
                f for f in os.listdir(task_dir) 
                if f.endswith("_moment_filt.csv") and not f.startswith("._")
            ]
            
            logging.info(f"[{subject}] {task} 找到的exo文件：{exo_files}")
            logging.info(f"[{subject}] {task} 找到的moment_filt文件：{moment_files}")
            
            if len(exo_files) == 0:
                logging.error(f"[{subject}] {task} 失败：未找到xxx_exo.csv")
                fail_count += 1
                continue
            elif len(moment_files) == 0:
                logging.error(f"[{subject}] {task} 失败：未找到xxx_moment_filt.csv")
                fail_count += 1
                continue
            elif len(exo_files) != 1 or len(moment_files) != 1:
                logging.error(f"[{subject}] {task} 失败：文件数量异常")
                fail_count += 1
                continue
            
            exo_path = os.path.join(task_dir, exo_files[0])
            moment_path = os.path.join(task_dir, moment_files[0])
            result = process_single_task(exo_path, moment_path, task_dir, subject, task)
            if result:
                success_count += 1
            else:
                fail_count += 1
    
    total = success_count + fail_count + skip_count
    print(f"\n批量处理完成：")
    print(f"总受试者数：{len(os.listdir(PARSED_ROOT))}（跳过无效受试者{invalid_subject_count}个）")
    print(f"总任务数：{total}")
    print(f"成功/跳过：{success_count} 个")
    print(f"失败：{fail_count} 个")
    print(f"其他跳过：{skip_count} 个")
    print(f"详细日志：{LOG_PATH}")
    logging.info(f"批量处理完成：成功/跳过 {success_count}/{total}，失败 {fail_count}/{total}，其他跳过 {skip_count}/{total}，跳过无效受试者{invalid_subject_count}个")


# --------------------------
# 6. 运行入口
# --------------------------
if __name__ == "__main__":
    if not os.path.exists(PARSED_ROOT):
        print(f"错误：Parsed根目录不存在 - {PARSED_ROOT}")
    else:
        batch_process()
