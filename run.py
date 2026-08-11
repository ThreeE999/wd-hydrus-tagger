"""Hydrus 定时打标入口：配置/日志 → job 编排 → worker → 调度循环。

支持独立启用的 tagger / classification，各自拥有 search_tags。
"""

import gc
import json
import logging
import multiprocessing
import os
import queue
import signal
import sys
import time
from datetime import datetime
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path

import hydrus_api
from croniter import croniter
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

CONFIG_PATH = "config.json"
JOB_ORDER = ("tagger", "classification")

shutdown_flag = False
current_worker_process = None
stats = {
    "total_processed": 0,
    "success": 0,
    "failed": 0,
    "last_run": None,
    "next_run": None,
}


# ---------------------------------------------------------------------------
# 配置 / 日志
# ---------------------------------------------------------------------------

def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config_mtime(config_path=CONFIG_PATH):
    if os.path.exists(config_path):
        return os.path.getmtime(config_path)
    return None


def setup_logging(config):
    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    log_level = log_cfg.get("level", "INFO")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"hydrus_tagger_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    logger = logging.getLogger("hydrus_tagger")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def logging_config_changed(old_config, new_config):
    old_log = old_config.get("logging", {})
    new_log = new_config.get("logging", {})
    return (
        old_log.get("level", "INFO") != new_log.get("level", "INFO")
        or old_log.get("log_dir", "logs") != new_log.get("log_dir", "logs")
    )


# ---------------------------------------------------------------------------
# Job 解析 / 标签工具
# ---------------------------------------------------------------------------

def make_tagger_done_tag(model_repo):
    return f"{model_repo} ai tags"


def make_classification_done_tag(repo, model_name):
    return f"{repo}/{model_name} ai tags"


def job_done_tag(job):
    if job["type"] == "tagger":
        return make_tagger_done_tag(job["model"]["repo"])
    if job["type"] == "classification":
        return make_classification_done_tag(job["repo"], job["model_name"])
    raise ValueError(f"未知 job 类型: {job.get('type')}")


def build_search_tags(job):
    search_tags = list(job["search_tags"])
    search_tags.append(f"-{job_done_tag(job)}")
    return search_tags


def format_tagger_tags(response, done_tag):
    _, rating, characters, general = response
    tags = [f"rating:{max(rating, key=rating.get)}"]
    tags.extend(f"character:{name}" for name in characters)
    tags.extend(general)
    tags.append(done_tag)
    return tags


def format_classification_tags(scores, done_tag):
    best = max(scores, key=scores.get)
    return [f"type:{best}", done_tag]


def resolve_enabled_jobs(config):
    """解析启用的 job 列表，兼容旧版顶层 model + search_tags。"""
    jobs = []

    if "tagger" in config:
        section = config["tagger"]
        if section.get("enabled", True):
            jobs.append(
                {
                    "type": "tagger",
                    "search_tags": section["search_tags"],
                    "model": section["model"],
                }
            )
    elif "model" in config and "search_tags" in config:
        # 旧配置兼容
        jobs.append(
            {
                "type": "tagger",
                "search_tags": config["search_tags"],
                "model": config["model"],
            }
        )

    classification = config.get("classification")
    if classification and classification.get("enabled", False):
        jobs.append(
            {
                "type": "classification",
                "search_tags": classification["search_tags"],
                "repo": classification["repo"],
                "model_name": classification["model_name"],
                "imgsize": classification.get("imgsize", 384),
            }
        )

    # 稳定顺序：tagger → classification
    order = {name: idx for idx, name in enumerate(JOB_ORDER)}
    jobs.sort(key=lambda j: order.get(j["type"], 99))
    return jobs


def next_run_from_now(schedule, base_time=None):
    """从当前时间起计算下一次运行，跳过已错过的槽位。"""
    base = base_time or datetime.now()
    return croniter(schedule, base).get_next(datetime)


def terminate_worker(process, logger=None, join_timeout=10):
    """优雅终止子进程：terminate → join → kill。"""
    if process is None or not process.is_alive():
        return
    if logger:
        logger.info("终止正在运行的子进程...")
    process.terminate()
    process.join(timeout=join_timeout)
    if process.is_alive():
        if logger:
            logger.warning("子进程未响应，强制终止")
        process.kill()
        process.join()


# ---------------------------------------------------------------------------
# 子进程 worker（按 job.type 加载对应模型）
# ---------------------------------------------------------------------------

def _predict_tags_for_file(file_id, client, job, model, done_tag):
    image = Image.open(BytesIO(client.get_file(file_id=file_id).content))
    image = image.convert("RGBA")

    if job["type"] == "tagger":
        model_cfg = job["model"]
        response = model.predict(
            image=image,
            model_repo=model_cfg["repo"],
            general_thresh=model_cfg["general_thresh"],
            general_mcut_enabled=model_cfg["general_mcut_enabled"],
            character_thresh=model_cfg["character_thresh"],
            character_mcut_enabled=model_cfg["character_mcut_enabled"],
        )
        return format_tagger_tags(response, done_tag)

    if job["type"] == "classification":
        scores = model.predict(
            image=image,
            repo=job["repo"],
            model_name=job["model_name"],
            imgsize=job.get("imgsize", 384),
        )
        return format_classification_tags(scores, done_tag)

    raise ValueError(f"未知 job 类型: {job.get('type')}")


def worker_process(task_queue, result_queue, hydrus_config, job):
    """在独立进程中加载单个 job 的模型并处理预测。"""
    logger = logging.getLogger("hydrus_tagger_worker")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    model = None
    client = None
    success_count = 0
    failed_count = 0

    try:
        if job["type"] == "tagger":
            import service.tagger.app as tagger_app

            model = tagger_app.Predictor()
        elif job["type"] == "classification":
            import service.classification.app as classification_app

            model = classification_app.ImageTypeClassifier()
        else:
            raise ValueError(f"未知 job 类型: {job.get('type')}")

        logger.info(f"子进程：{job['type']} 模型加载完成")

        client = hydrus_api.Client(hydrus_config["api_key"], hydrus_config["host"])
        service_key = client.get_service(hydrus_config["tag_service"])["service"][
            "service_key"
        ]
        done_tag = job_done_tag(job)

        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if task is None:
                break

            file_id = task
            try:
                tags = _predict_tags_for_file(file_id, client, job, model, done_tag)
                client.add_tags(
                    file_ids=[file_id],
                    service_keys_to_tags={service_key: tags},
                )
                result_queue.put(("success", file_id))
                success_count += 1
            except Exception as e:
                logger.error(f"处理文件 {file_id} 时出错: {e}", exc_info=True)
                result_queue.put(("failed", file_id))
                failed_count += 1

        logger.info(f"子进程：任务完成，成功 {success_count}，失败 {failed_count}")
        result_queue.put(("done", {"success": success_count, "failed": failed_count}))

    except Exception as e:
        logger.error(f"子进程发生错误: {e}", exc_info=True)
        result_queue.put(("error", str(e)))
    finally:
        if model is not None:
            del model
        if client is not None:
            del client
        gc.collect()
        logger.info("子进程退出，内存已释放")


# ---------------------------------------------------------------------------
# 主进程任务编排
# ---------------------------------------------------------------------------

def search_untagged_files(hydrus_config, job, logger):
    client = hydrus_api.Client(hydrus_config["api_key"], hydrus_config["host"])
    search_tags = build_search_tags(job)
    logger.info(f"[{job['type']}] 搜索标签: {search_tags}")
    # https://hydrusnetwork.github.io/hydrus/developer_api.html#get_files_search_files
    return client.search_files(search_tags)["file_ids"]


def enqueue_tasks(task_queue, file_ids, logger):
    """将 file_id 入队，返回实际入队数量；收到关闭信号时提前停止。"""
    queued = 0
    for file_id in file_ids:
        if shutdown_flag:
            logger.warning("收到关闭信号，停止添加任务")
            break
        task_queue.put(file_id)
        queued += 1
    task_queue.put(None)
    return queued


def _apply_result(result_type, result_data, counts, pbar, logger):
    """处理单条结果。返回 status 字符串，或 None 表示继续收集。"""
    success_count, failed_count, processed_count = counts
    if result_type == "success":
        success_count += 1
        processed_count += 1
        pbar.update(1)
        return (success_count, failed_count, processed_count), None
    if result_type == "failed":
        failed_count += 1
        processed_count += 1
        pbar.update(1)
        return (success_count, failed_count, processed_count), None
    if result_type == "done":
        return (success_count, failed_count, processed_count), "completed"
    if result_type == "error":
        logger.error(f"子进程报告错误: {result_data}")
        return (success_count, failed_count, processed_count), "worker_error"
    return (success_count, failed_count, processed_count), None


def collect_worker_results(result_queue, worker, queued_count, logger, desc="处理文件"):
    """收集子进程结果。返回 (success, failed, status)。"""
    counts = (0, 0, 0)
    status = "completed"

    pbar = tqdm(
        total=queued_count,
        desc=desc,
        disable=not sys.stdout.isatty() or queued_count == 0,
    )

    try:
        while counts[2] < queued_count:
            if shutdown_flag:
                logger.warning("收到关闭信号，终止子进程")
                terminate_worker(worker, logger)
                status = "interrupted"
                break

            try:
                result_type, result_data = result_queue.get(timeout=1)
            except queue.Empty:
                if worker.is_alive():
                    continue
                terminal = None
                while True:
                    try:
                        result_type, result_data = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    counts, terminal = _apply_result(
                        result_type, result_data, counts, pbar, logger
                    )
                    if terminal:
                        break
                if terminal:
                    status = terminal
                elif counts[2] < queued_count:
                    logger.warning("子进程异常退出")
                    status = "worker_died"
                break

            counts, terminal = _apply_result(
                result_type, result_data, counts, pbar, logger
            )
            if terminal:
                status = terminal
                break
    finally:
        pbar.close()

    return counts[0], counts[1], status


def wait_worker_exit(worker, logger, join_timeout=300):
    if worker is None:
        return
    if worker.is_alive():
        logger.info("等待子进程完成...")
        worker.join(timeout=join_timeout)
        if worker.is_alive():
            logger.warning("子进程未在预期时间内完成，强制终止")
            terminate_worker(worker, logger)


def update_stats(success_count, failed_count):
    processed = success_count + failed_count
    stats["total_processed"] += processed
    stats["success"] += success_count
    stats["failed"] += failed_count
    return processed


def run_job(hydrus_config, job, logger):
    """执行单个 job（独立搜索 + 独立子进程）。"""
    global current_worker_process

    job_type = job["type"]
    logger.info("-" * 60)
    logger.info(f"开始执行 job: {job_type}")

    if current_worker_process is not None and current_worker_process.is_alive():
        logger.warning("检测到正在运行的子进程，跳过本 job")
        return

    worker = None
    try:
        try:
            file_ids = search_untagged_files(hydrus_config, job, logger)
        except Exception as e:
            logger.error(f"[{job_type}] 搜索文件失败: {e}", exc_info=True)
            return

        total_found = len(file_ids)
        logger.info(f"[{job_type}] 找到 {total_found} 个未标记的文件")
        if total_found == 0:
            logger.info(f"[{job_type}] 没有需要处理的文件")
            return

        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        queued_count = enqueue_tasks(task_queue, file_ids, logger)

        if queued_count == 0:
            logger.warning(f"[{job_type}] 未入队任何任务，跳过处理")
            return

        if queued_count < total_found:
            logger.info(f"[{job_type}] 实际入队 {queued_count}/{total_found} 个文件")

        logger.info(f"[{job_type}] 启动子进程（模型将在子进程中加载）")
        worker = multiprocessing.Process(
            target=worker_process,
            args=(task_queue, result_queue, hydrus_config, job),
        )
        current_worker_process = worker
        worker.start()

        success_count, failed_count, status = collect_worker_results(
            result_queue,
            worker,
            queued_count,
            logger,
            desc=f"{job_type}",
        )
        wait_worker_exit(worker, logger)
        current_worker_process = None

        processed = update_stats(success_count, failed_count)
        status_label = {
            "completed": "完成",
            "interrupted": "中断",
            "worker_error": "异常（子进程报错）",
            "worker_died": "异常（子进程退出）",
        }.get(status, "结束")

        logger.info(
            f"[{job_type}] {status_label}: 成功 {success_count}, 失败 {failed_count}, "
            f"已处理 {processed}/{queued_count}（搜索到 {total_found}）"
        )
        if status == "completed":
            logger.info(f"[{job_type}] 子进程已退出，模型内存已释放")

    except Exception as e:
        logger.error(f"[{job_type}] 执行时发生错误: {e}", exc_info=True)
        terminate_worker(worker or current_worker_process, logger)
        current_worker_process = None


def run_task(config, logger):
    """执行一轮调度：依次跑所有 enabled job。"""
    logger.info("=" * 60)
    logger.info("开始执行标签任务")
    stats["last_run"] = datetime.now().isoformat()

    jobs = resolve_enabled_jobs(config)
    if not jobs:
        logger.warning("没有启用的 job（tagger / classification），跳过本轮")
        logger.info("=" * 60)
        return

    logger.info(f"本轮启用 job: {[j['type'] for j in jobs]}")
    hydrus_config = config["hydrus"]

    for job in jobs:
        if shutdown_flag:
            logger.warning("收到关闭信号，停止后续 job")
            break
        run_job(hydrus_config, job, logger)

    logger.info(
        f"累计统计: 总处理 {stats['total_processed']}, "
        f"成功 {stats['success']}, 失败 {stats['failed']}"
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# 调度循环 / 热重载 / 信号
# ---------------------------------------------------------------------------

def signal_handler(signum, frame):
    global shutdown_flag
    logger = logging.getLogger("hydrus_tagger")
    msg = f"收到信号 {signum}，准备优雅关闭..."
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)
    shutdown_flag = True
    terminate_worker(current_worker_process, logger if logger.handlers else None)


def apply_config_reload(config, new_config, logger):
    """应用热重载。返回 (config, logger, next_run_time_or_None)。"""
    next_run_time = None
    merged = dict(new_config)

    if new_config.get("schedule") != config.get("schedule"):
        try:
            next_run_time = next_run_from_now(new_config["schedule"])
            logger.info(f"调度表达式已更新: {new_config['schedule']}")
            logger.info(
                f"下次运行时间: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except Exception as e:
            logger.error(
                f"无效的 crontab 表达式: {new_config['schedule']}, 错误: {e}"
            )
            logger.warning("保持使用旧的调度表达式")
            merged["schedule"] = config["schedule"]

    if logging_config_changed(config, merged):
        logger.info("日志配置已更新，正在重建 logger")
        logger = setup_logging(merged)

    return merged, logger, next_run_time


def main():
    global shutdown_flag, stats

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        config = load_config()
    except Exception as e:
        print(f"加载配置失败: {e}")
        sys.exit(1)

    logger = setup_logging(config)
    logger.info("Hydrus Tagger 启动")
    logger.info(f"配置文件: {CONFIG_PATH}")
    logger.info(f"调度表达式: {config['schedule']}")

    enabled = [j["type"] for j in resolve_enabled_jobs(config)]
    logger.info(f"启用 job: {enabled or ['(无)']}")

    last_config_mtime = get_config_mtime(CONFIG_PATH)

    try:
        next_run_time = next_run_from_now(config["schedule"])
    except Exception as e:
        logger.error(f"无效的 crontab 表达式: {config['schedule']}, 错误: {e}")
        sys.exit(1)

    stats["next_run"] = next_run_time.isoformat()
    logger.info(f"下次运行时间: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("进入主循环，等待执行时间...")
    logger.info("配置文件监控已启用，修改 config.json 后会自动重载")

    while not shutdown_flag:
        current_time = datetime.now()

        current_mtime = get_config_mtime(CONFIG_PATH)
        if current_mtime and current_mtime != last_config_mtime:
            logger.info("检测到配置文件更新，正在重新加载...")
            try:
                new_config = load_config(CONFIG_PATH)
            except Exception as e:
                logger.error(f"重新加载配置失败: {e}", exc_info=True)
            else:
                config, logger, reloaded_next = apply_config_reload(
                    config, new_config, logger
                )
                if reloaded_next is not None:
                    next_run_time = reloaded_next
                    stats["next_run"] = next_run_time.isoformat()
                enabled = [j["type"] for j in resolve_enabled_jobs(config)]
                logger.info(f"启用 job: {enabled or ['(无)']}")
                last_config_mtime = current_mtime
                logger.info("配置重载完成")

        if current_time >= next_run_time:
            run_task(config, logger)
            if not shutdown_flag:
                next_run_time = next_run_from_now(config["schedule"])
                stats["next_run"] = next_run_time.isoformat()
                logger.info(
                    f"下次运行时间: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        time.sleep(10)

    logger.info("程序退出")


if __name__ == "__main__":
    main()
