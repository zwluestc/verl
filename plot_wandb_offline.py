import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

# 导入 wandb 内部的 protobuf 和 datastore 用于读取二进制 .wandb 文件
try:
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
except ImportError:
    print("Error: 请确保安装了 wandb (pip install wandb)")
    sys.exit(1)

def plot_from_binary_wandb(run_dir):
    wandb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wandb_files:
        print(f"Error: 在 {run_dir} 找不到 .wandb 数据文件！")
        return
        
    wandb_file = wandb_files[0]
    print(f"正在读取二进制日志: {wandb_file} ...")
    
    # 1. 解析 Protobuf 数据提取字典
    data = []
    ds = DataStore()
    ds.open_for_scan(wandb_file)
    
    record_count = 0
    history_count = 0
    for record_raw in ds.scan_data():
        record_count += 1
        
        # 兼容不同版本的 wandb
        if hasattr(record_raw, "history"):
            # 有的版本 scan_data 直接解包出了 pb 对象 (Record 对象)
            pb = record_raw
        else:
            record_bytes = None
            if isinstance(record_raw, tuple):
                for item in record_raw:
                    if isinstance(item, bytes):
                        record_bytes = item
                        break
                if record_bytes is None and len(record_raw) > 2:
                    # 某些版本里可能是 (offset, dict, ...)，我们尝试把原样打出来看
                    if record_count <= 2:
                        print(f"Debug Tuple: {[type(x) for x in record_raw]}")
            elif isinstance(record_raw, bytes):
                record_bytes = record_raw
            elif isinstance(record_raw, str):
                record_bytes = record_raw.encode("utf-8")
                
            pb = wandb_internal_pb2.Record()
            try:
                if record_bytes is not None:
                    pb.ParseFromString(record_bytes)
            except Exception as e:
                if record_count <= 5:
                    print(f"解析出错 record {record_count}: {e}")
                continue
        
        # 提取 history
        if pb.HasField('history'):
            history_count += 1
            row_dict = {}
            for item in pb.history.item:
                try:
                    row_dict[item.key] = json.loads(item.value_json)
                except Exception:
                    pass
            if row_dict:
                if pb.history.HasField('step'):
                    row_dict['_step'] = pb.history.step.num
                data.append(row_dict)
                
    if not data:
        print("记录文件中没有解析出任何图像指标历史！")
        return
        
    # 2. 数据处理与绘图
    df = pd.DataFrame(data)
    
    metric_cols = [col for col in df.columns if col != '_step' and not col.startswith('system/') and not col.startswith('_')]
    
    save_dir = os.path.join(run_dir, 'saved_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    x_axis = '_step' if '_step' in df.columns else df.index
    
    print(f"已提取到 {len(df)} 步记录，下面开始画图...")
    for metric in metric_cols:
        plot_df = df[[x_axis, metric]].dropna()
        if len(plot_df) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(plot_df[x_axis], plot_df[metric], marker='o', markersize=2, linestyle='-', label=metric)
        plt.title(f'{metric}')
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        safe_name = metric.replace('/', '_')
        save_path = os.path.join(save_dir, f"{safe_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"✅ 成功! 共有 {len(metric_cols)} 张图已被保存至目录: {save_dir}")

if __name__ == "__main__":
    run_dir = "/mnt/data/zwl/verl/wandb/offline-run-20260418_001121-wqd46r5l"
    # 忽略 Jupyter/交互式窗口自动传入的 -f 参数
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        run_dir = sys.argv[1]
    plot_from_binary_wandb(run_dir)
