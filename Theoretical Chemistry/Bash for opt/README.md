# Bash for opt

------------------------------
Caution! This markdown file is mainly written in Chinese.

------------------------------
主要是一些自己写来方便计算用的脚本，包含以下内容
1. sub.sh **主文件**，请在linux客户端下使用nohup bash sub.sh &运行该脚本,在大多数情况下，你也只需要调用该脚本中的函数即可。
2. modify.py 用于产生gaussian输入文件的python脚本，如果你想要更改脚本生成的输入文件关键词，可以修改它。
3. judge.py **请勿修改**用于判断输出/输入文件的任务类型.
4. extract.py **请勿修改**,用于从gaussian输出文件提取内容的脚本，通常提取热力学修正值和电子能量。
5. Conversion_Script.sh **请勿修改**,用于转化，chk，fchk文件等，made by Jianyong Yuan  

------------------------------

## sub.sh
sub.sh文件中包含以下部分
### 任务内容定义模块
#!/bin/bash\
source ~/.bashrc

optsolvent="dichloroethane"\
enesolvent="dichloroethane"\
chkpath=/data/zxwei\
mem=12\
nproc=18\
OptBasisSet="def2svp"\
EneBasisSet="def2tzvp"\
theory="UM062x"

在这里修改计算的一些必须内容，包括:
1. optsolvent= "" 这是opt类型任务的溶剂环境，如果只需要优化气相分子，请将溶剂改为"gas"
2. enesolvent ="" 这是单点任务的溶剂环境，同上。
3. chkpath ="" 在这里填写chk文件的存储位置
4. mem= "" 在这里填写最大可用内存
5. nproc = "" 在这里填写核数
6. OptBasisSet = "" 在这里填写opt类型任务的基组
7. EneBasisSet = "" 在这里填写单点类型任务的基组
8. theory = "" 在这里填写理论级别


### 函数模块