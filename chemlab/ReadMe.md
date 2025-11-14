README：How to Add a New Script to chemlab

（完全避免 Markdown 代码块，适合直接复制到 GitHub）

====================================================

1. 目录结构

chemlab 使用以下结构管理脚本：

chemlab/
scripts/
ml_data/
export_numpy.py
prepare_tddft_inp.py
qchem/
convert_out_to_inp.py
cli/
config/
config.toml

任何放在 scripts/<group>/<script>.py 的文件
都会自动变成命令：

chemlab <group> <script>

例如：
文件：chemlab/scripts/ml_data/export_numpy.py
命令：chemlab ml_data export_numpy

====================================================

2. Script 类的格式

每个脚本必须包含一个继承自 Script 的类：

示例（注意：以下为伪代码，不会被渲染成代码块）

from chemlab.scripts.base import Script
from chemlab.config import ConfigBase

class ExportNumpyConfig(ConfigBase):
section_name = "export_numpy"

class ExportNumpy(Script):
name = "export_numpy"
config = ExportNumpyConfig

def run(self, cfg):
    ...执行任务...


字段说明：

name: CLI 中显示的子命令名
config: 对应的配置类（可选）
run(cfg): 脚本执行时运行的函数

====================================================

3. 配置文件 config.toml

每个脚本的配置默认写在：

chemlab/config/config.toml

例如：

[export_numpy]
data = "./dataset/"
out = "./npy/"
prefix = "phbdi_"
state_idx = 1

所有字段会自动生成命令行参数：

--data
--out
--prefix
--state_idx

====================================================

4. 自动加载机制

chemlab 会自动扫描：

chemlab/scripts/**

查找每一个 Script 子类，步骤如下：

根据脚本路径推导命令组 (group)
scripts/ml_data/export_numpy.py → group = ml_data

根据类名或文件名生成子命令 (command)
ExportNumpy → export_numpy

在 CLI 中注册命令
chemlab ml_data export_numpy

你不需要手写任何 CLI 代码。

====================================================

5. 添加一个新脚本的完整流程

示例：添加一个打印数据的脚本

步骤 1：创建文件
chemlab/scripts/ml_data/print_data.py

步骤 2：写 Script 类
class PrintData(Script):
name = "print_data"
config = PrintDataConfig
def run(self, cfg):
print("Dataset path:", cfg.data)

步骤 3：添加 config.toml 片段
[print_data]
data = "./dataset/"

步骤 4：使用
chemlab ml_data print_data --data mydata/

====================================================

6. 完整示例：零配置脚本

文件：chemlab/scripts/tools/hello.py

内容（纯文本示例）：
class Hello(Script):
name = "hello"
config = None
def run(self, cfg):
print("Hello from chemlab!")

执行：
chemlab tools hello

====================================================

7. 总结

要添加一个新脚本：

在 chemlab/scripts/<group>/ 下创建 .py 文件

定义一个 Script 子类（带 name / run）
3.（可选）定义 config class + 写入 config.toml

完成，脚本会自动出现在 CLI 中！

====================================================