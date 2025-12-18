# Hydrus Tagger

一个基于 WD14 Tagger 的 Hydrus 图片自动标签工具，支持定时持续运行。

## 功能特性

- 🤖 使用 WD14 Tagger 模型自动为 Hydrus 中的图片添加标签
- ⏰ 支持 crontab 风格的定时调度（如 `*/5 * * * *` 每 5 分钟）
- 🔄 持续运行模式，适合容器化部署
- 🔥 配置文件热重载，修改 `config.json` 后自动生效，无需重启
- 🐳 GitHub Actions 自动构建 Docker 镜像

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
- **hydrus**: Hydrus API 配置
  - `host`: Hydrus 服务器地址
  - `api_key`: API 密钥
  - `tag_service`: 标签服务名称
- **model**: 模型配置
  - `repo`: 模型仓库名称
  - `general_thresh`: 通用标签阈值
  - `character_thresh`: 角色标签阈值
  - `general_mcut_enabled`: 是否启用 MCut 阈值
  - `character_mcut_enabled`: 是否启用角色 MCut 阈值
- **search_tags**: 搜索标签列表
- **logging**: 日志配置
  - `level`: 日志级别 (DEBUG, INFO, WARNING, ERROR)
  - `log_dir`: 日志目录

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


