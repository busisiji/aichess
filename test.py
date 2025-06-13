import pickle
import os
import logging
from logging.handlers import RotatingFileHandler
import argparse


# 创建日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志系统
LOG_FILE = os.path.join(LOG_DIR, "test_output.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024 * 5, backupCount=5, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PickleViewer")


def view_pickle_file(filepath):
    """
    查看 .pkl 文件内容，适配 collect_multi_thread.py 的格式。
    支持：
      - 单个样本（tuple）
      - 字典结构 {'data_buffer': list/deque, 'iters': int}
    """
    if not os.path.exists(filepath):
        logger.error(f"❌ 文件 {filepath} 不存在")
        return

    logger.info(f"\n📄 正在查看文件: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            idx = 0
            while True:
                try:
                    obj = pickle.load(f)

                    logger.info(f"\n📦 对象 #{idx}:")
                    logger.info(f"Type: {type(obj)}")

                    if isinstance(obj, dict):
                        keys = list(obj.keys())
                        logger.info(f"Keys: {keys}")

                        if 'data_buffer' in obj:
                            data_len = len(obj['data_buffer'])
                            logger.info(f"DataBuffer length: {data_len}")
                        if 'iters' in obj:
                            logger.info(f"Iterations: {obj['iters']}")

                        if 'data_buffer' in obj and len(obj['data_buffer']) > 0:
                            sample = obj['data_buffer'][0]
                            logger.info(f"First Sample:\n{str(sample)[:500]}...")

                    else:
                        logger.info(f"Raw Data (length {len(obj) if hasattr(obj, '__len__') else '?'}):\n{str(obj)[:500]}...")

                    idx += 1
                except EOFError as e:
                    break
        logger.info(f"\n✅ 共读取到 {idx} 个对象。")
    except Exception as e:
        logger.error(f"❌ 读取文件失败: {e}")


def scan_and_view_all_pkl_files(directory="data"):
    """
    扫描指定目录下所有的 .pkl 文件并调用 view_pickle_file 查看
    """
    if not os.path.exists(directory):
        logger.error(f"❌ 目录 {directory} 不存在")
        return

    logger.info(f"\n🔍 开始扫描目录: {directory}")
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    if not pkl_files:
        logger.warning(f"⚠️ 在 {directory} 中未找到任何 .pkl 文件")
        return

    logger.info(f"📂 找到 {len(pkl_files)} 个 .pkl 文件:")
    for filename in pkl_files:
        filepath = os.path.join(directory, filename)
        logger.info(f"🔎 正在处理文件: {filename}")
        view_pickle_file(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 .pkl 文件内容")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", type=str,default="data/train_data_buffer.pkl", help="指定要查看的 .pkl 文件路径")
    # group.add_argument("--dir", type=str, default="data", help="指定要扫描的目录，默认为 data/")
    args = parser.parse_args()

    if args.file:
        view_pickle_file(args.file)
    # if args.dir:
    #     scan_and_view_all_pkl_files(args.dir)
    else:
        scan_and_view_all_pkl_files("data")  # 默认扫描 data/
