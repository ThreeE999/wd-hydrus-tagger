# Hydrus Tagger

一个基于 WD14 Tagger / anime classification 的 Hydrus 图片自动标签工具，支持定时持续运行。

## 功能特性

- 使用 WD14 Tagger 为 Hydrus 图片添加标签
- 可选启用 anime classification，写入 `type:<label>`
- tagger / classification 可独立开关，并各自配置 `search_tags`
- 支持 crontab 风格定时调度与配置热重载
- 适合容器化部署；GitHub Actions 自动构建 Docker 镜像

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/threee999/wd-hydrus-tagger.git
cd wd-hydrus-tagger
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

复制并编辑配置文件：

```bash
cp config.json.example config.json
# 或直接编辑 config.json
```

编辑 `config.json` 文件，配置以下内容：

- **schedule**: crontab 表达式，定义运行频率
  - `*/5 * * * *` - 每 5 分钟
  - `0 * * * *` - 每小时
  - `0 */2 * * *` - 每 2 小时
  - `0 0 * * *` - 每天午夜
- **hydrus**: Hydrus API 配置（`host` / `api_key` / `tag_service`）
- **tagger**: WD14 打标任务
  - `enabled`: 是否启用
  - `search_tags`: 仅作用于本任务的搜索范围
  - `model`: `repo` / 阈值 / MCut 开关
- **classification**: 图片类型分类任务（可选）
  - `enabled`: 是否启用
  - `search_tags`: 仅作用于本任务的搜索范围
  - `repo` / `model_name` / `pre_long_side`（预缩小长边，先缩小再拉到模型输入）
  - 写入 `type:<最高分类>`，完成标记为 `{repo}/{model_name} ai tags`
- **logging**: `level` / `log_dir`

### 4. 运行

```bash
python run.py
```

程序会持续运行，按照配置的 crontab 表达式定时执行标签任务。

### 配置文件热重载

程序支持配置文件热重载功能。修改 `config.json` 后，程序会自动检测并重新加载配置，无需重启。支持以下配置的动态更新：

- 调度表达式（schedule）
- 日志级别（logging.level）
- 其他所有配置项

修改配置文件后，程序会在 10 秒内检测到变化并自动重载，相关日志会记录在日志文件中。

## Crontab 表达式说明

使用标准的 crontab 格式：`分 时 日 月 周`

```
* * * * *
│ │ │ │ │
│ │ │ │ └── 星期几 (0-7, 0 和 7 都表示周日)
│ │ │ └──── 月份 (1-12)
│ │ └────── 日期 (1-31)
│ └──────── 小时 (0-23)
└────────── 分钟 (0-59)
```

示例：
- `*/5 * * * *` - 每 5 分钟
- `0 * * * *` - 每小时整点
- `0 9 * * *` - 每天上午 9 点
- `0 0 * * 0` - 每周日午夜

## 日志

日志文件保存在 `logs/` 目录中，按日期命名：
- `hydrus_tagger_2024-01-01.log`
- `hydrus_tagger_2024-01-02.log`
- ...

日志会自动轮转，单个文件最大 10MB，保留 5 个备份。

## 容器化运行

### 使用 GitHub Container Registry 镜像

项目已配置 GitHub Actions 自动构建 Docker 镜像并推送到 GitHub Container Registry (ghcr.io)。

```bash
# 拉取最新镜像
docker pull ghcr.io/threee999/wd-hydrus-tagger:latest

# 运行容器
docker run -d \
  --name hydrus-tagger \
  -v /path/to/config.json:/app/config.json \
  -v /path/to/logs:/app/logs \
  -v /path/to/models:/app/models \
  --restart unless-stopped \
  ghcr.io/threee999/wd-hydrus-tagger:latest
```

### docker-compose.yml 示例

```yaml
version: '3.8'

services:
  hydrus-tagger:
    image: ghcr.io/threee999/wd-hydrus-tagger:latest
    # 或使用本地构建
    # build: .
    volumes:
      - ./config.json:/app/config.json
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
```


## 统计信息

程序会记录以下统计信息：
- 总处理文件数
- 成功数量
- 失败数量
- 最后运行时间
- 下次运行时间

## 模型

- `SmilingWolf/wd-eva02-large-tagger-v3`
- `SmilingWolf/wd-vit-large-tagger-v3`
- `SmilingWolf/wd-swinv2-tagger-v3`
- `SmilingWolf/wd-convnext-tagger-v3`
- 更多模型请参考 [WD Tagger](https://huggingface.co/SmilingWolf)

## 许可证

MIT License


