# 操作手记

## 虚拟机
+ `VMware` 使用 `Ctrl + G` 使得鼠标键盘进入虚拟机并用 `Ctrl + Alt` 退出
+ `Ubunto` 使用 `Ctrl + Alt + T` 打开命令行
+ `sudo apt-get install` 远程安装

**文本和命令行的复制黏贴**
完成虚拟机和客户机之间的复制黏贴的方法是使用 `ssh` 在客户机中访问虚拟机。
+ 保证虚拟机中安装了 `ssh`，`sudo apt-get install openssh-server`
+ 在 `ubunto` 中的网络连接查看本机 **IP** 地址
+ 在客户机中用 `ssh` 访问
+ `putty` 中使用 **鼠标右键** 进行黏贴

**vi上下左右键显示为ABCD的问题**
依次执行以下两个命令即可完美解决Ubuntu下vi编辑器方向键变字母的问题。
+ 执行命令 `sudo apt-get remove vim-common`
+ 执行命令 `sudo apt-get install vim`

**虚拟机与客户机共享的挂载**
+ 在 `VMware` 中的 `虚拟机-设置-选项` 中设置共享文件夹
+ 共享的文件夹在 `/mnt/hgfs` 中

**设置环境变量**
+ **需在root权限下**
+ `vi  ~/bashrc` 打开环境变量配置文件
+ **添加** `export PATH=$PATH:/bin:/usr/bin:/sbin:xxx`
+ `source  ~/bashrc`
