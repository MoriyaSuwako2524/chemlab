# Bash for opt

------------------------------
Caution! This markdown file is mainly written in Chinese.

------------------------------
主要是一些自己写来方便计算用的脚本，包含以下内容
1. sub.sh **主文件**，请在linux客户端下使用nohup bash sub.sh &运行该脚本,在大多数情况下，你也只需要调用该脚本中的函数即可。
2. tem文件夹 **在其中放入tem1.gjf 与tem2.gjf等输入文件作为输入模板**
3. modify.py 用于产生gaussian输入文件的python脚本，如果你想要更改脚本生成的输入文件关键词，可以修改它。
4. judge.py **请勿修改**用于判断输出/输入文件的任务类型.
5. extract.py **请勿修改**,用于从gaussian输出文件提取内容的脚本，通常提取热力学修正值和电子能量。
6. Conversion_Script.sh **请勿修改**,用于转化，chk，fchk文件等，made by Jianyong Yuan  

------------------------------




### 函数模块