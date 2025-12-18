import json
import os
import signal
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import hydrus_api
from croniter import croniter
from PIL import Image
from tqdm import tqdm

import app

# 全局变量用于优雅关闭
shutdown_flag = False
stats = {
    "total_processed": 0,
    "success": 0,
    "failed": 0,
    "last_run": None,
    "next_run": None
}


def load_config(config_path="config.json"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def setup_logging(config):
    """设置日志系统"""
    import logging
    from logging.handlers import RotatingFileHandler
    
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    log_level = config.get("logging", {}).get("level", "INFO")
    
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 日志文件名：hydrus_tagger_YYYY-MM-DD.log
    log_file = os.path.join(log_dir, f"hydrus_tagger_{datetime.now().strftime('%Y-%m-%d')}.log")
    
    # 配置日志格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建 logger
    logger = logging.getLogger("hydrus_tagger")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)
    
    return logger


def format_tags(response, ai_tag_name):
    """格式化标签"""
    tags = []
    tags.append(f'rating:{max(response[1], key=response[1].get)}')
    tags.extend(map(lambda x: f'character:{x}', response[2].keys()))
    tags.extend(response[3].keys())
    tags.append(ai_tag_name)
    return tags


def hydrus_add_tags(predictor: app.Predictor, client: hydrus_api.Client, file_id: int, 
                    service_key: str, config: dict, logger):
    """为 Hydrus 文件添加标签"""
    try:
        image_bytes = BytesIO(client.get_file(file_id=file_id).content)
        image = Image.open(image_bytes)
        image = image.convert("RGBA")
        
        model_config = config["model"]
        response = predictor.predict(
            image=image,
            model_repo=model_config["repo"],
            general_thresh=model_config["general_thresh"],
            general_mcut_enabled=model_config["general_mcut_enabled"],
            character_thresh=model_config["character_thresh"],
            character_mcut_enabled=model_config["character_mcut_enabled"],
        )
        
        ai_tag_name = f"{model_config['repo']} ai tags"
        tags = format_tags(response, ai_tag_name)
        client.add_tags(file_ids=[file_id], service_keys_to_tags={
            service_key: tags
        })
        return True
    except Exception as e:
        logger.error(f"处理文件 {file_id} 时出错: {str(e)}", exc_info=True)
        return False


def run_task(config, logger):
    """执行一次标签任务"""
    global stats, shutdown_flag
    
    logger.info("=" * 60)
    logger.info("开始执行标签任务")
    stats["last_run"] = datetime.now().isoformat()
    
    try:
        # 初始化
        predictor = app.Predictor()
        hydrus_config = config["hydrus"]
        client = hydrus_api.Client(hydrus_config["api_key"], hydrus_config["host"])
        
        # 构建搜索标签（排除已标记的）
        model_repo = config["model"]["repo"]
        ai_tag_name = f"{model_repo} ai tags"
        search_tags = config["search_tags"].copy()
        search_tags.append(f"-{ai_tag_name}")
        
        logger.info(f"搜索标签: {search_tags}")
        
        # 搜索文件
        try:
            # https://hydrusnetwork.github.io/hydrus/developer_api.html#get_files_search_files
            file_ids = client.search_files(search_tags)['file_ids']
        except Exception as e:
            logger.error(f"搜索文件失败: {str(e)}", exc_info=True)
            return
        
        total_files = len(file_ids)
        logger.info(f"找到 {total_files} 个未标记的文件")
        
        if total_files == 0:
            logger.info("没有需要处理的文件")
            return
        
        # 获取服务密钥
        try:
            service_key = client.get_service(hydrus_config["tag_service"])['service']['service_key']
        except Exception as e:
            logger.error(f"获取服务密钥失败: {str(e)}", exc_info=True)
            return
        
        # 处理文件
        success_count = 0
        failed_count = 0
        
        for file_id in tqdm(file_ids, desc="处理文件", disable=not sys.stdout.isatty()):
            if shutdown_flag:
                logger.warning("收到关闭信号，停止处理")
                break
            
            if hydrus_add_tags(predictor, client, file_id, service_key, config, logger):
                success_count += 1
            else:
                failed_count += 1
        
        # 更新统计
        stats["total_processed"] += total_files
        stats["success"] += success_count
        stats["failed"] += failed_count
        
        logger.info(f"任务完成: 成功 {success_count}, 失败 {failed_count}, 总计 {total_files}")
        logger.info(f"累计统计: 总处理 {stats['total_processed']}, 成功 {stats['success']}, 失败 {stats['failed']}")
        
    except Exception as e:
        logger.error(f"执行任务时发生错误: {str(e)}", exc_info=True)
    finally:
        logger.info("=" * 60)


def signal_handler(signum, frame):
    """信号处理器，用于优雅关闭"""
    global shutdown_flag
    import logging
    logger = logging.getLogger("hydrus_tagger")
    if logger.handlers:  # 只有在 logger 已初始化时才记录
        logger.info(f"收到信号 {signum}，准备优雅关闭...")
    else:
        print(f"收到信号 {signum}，准备优雅关闭...")
    shutdown_flag = True


def main():
    """主函数，实现定时循环运行"""
    global shutdown_flag, stats
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 加载配置
    try:
        config = load_config()
    except Exception as e:
        print(f"加载配置失败: {e}")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("Hydrus Tagger 启动")
    logger.info(f"配置文件: config.json")
    logger.info(f"调度表达式: {config['schedule']}")
    
    # 解析 crontab 表达式
    try:
        cron = croniter(config["schedule"], datetime.now())
    except Exception as e:
        logger.error(f"无效的 crontab 表达式: {config['schedule']}, 错误: {e}")
        sys.exit(1)
    
    # 计算下次运行时间
    next_run_time = cron.get_next(datetime)
    stats["next_run"] = next_run_time.isoformat()
    logger.info(f"下次运行时间: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 主循环
    logger.info("进入主循环，等待执行时间...")
    
    while not shutdown_flag:
        current_time = datetime.now()
        
        # 检查是否到达执行时间
        if current_time >= next_run_time:
            # 执行任务
            run_task(config, logger)
            
            # 计算下次运行时间
            if not shutdown_flag:
                next_run_time = cron.get_next(datetime)
                stats["next_run"] = next_run_time.isoformat()
                logger.info(f"下次运行时间: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 等待一段时间（避免 CPU 占用过高）
        time.sleep(10)
    
    logger.info("程序退出")


if __name__ == "__main__":
    import logging
    main()
