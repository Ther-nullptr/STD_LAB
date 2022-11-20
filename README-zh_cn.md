# 《视听信息系统导论》课程大作业：噪声干扰下的音视频匹配

## records

实验报告地址：https://www.overleaf.com/3393918221kzwhkjdscmfh

实验数据记录地址：https://1drv.ms/x/s!Agcs68x5s4XGhngeyctknqAYWlr5?e=swsuEa

## milestone

- [x] 仓库配置、数据下载 
- [x] 任务一：基于滤波器的图像噪声处理
- [x] 任务二：基于谱减法的音频噪声处理
- [ ] 任务三：音视频匹配

## setup

```bash
conda config --append channels conda-forge  
conda install --yes --file requirements.txt    
```

## notice

### 文件说明
   
本仓库为所有开发工作共用仓库，请勿上传不必要的文件。本仓库采用 git 作为版本控制系统，主目录内的 .gitignore 非必要尽量不要修改；为了防止行尾不一致的问题，主目录内已经配置了 .gitattributes 以进行行尾标准化，非必要也尽量不要修改，如果有必要可以在子目录内自定义 .gitattributes。

### 开发流程

使用 Git 与 GitHub 进行协作开发的过程中，各个成员应当遵守下面的流程：

1. 将主仓库 fork 到个人 GitHub 仓库中
2. 创建自己的分支
3. 进行开发
4. 开发取得一定成果时需要commit一次，格式如下：`git commit -m '<messages>'`，需要在message中简要说明开发内容
5. 将主仓库 dev 分支的最新进度 pull 到自己的仓库中
6. 向主仓库的 dev 分支提出 pull request
