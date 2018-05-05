# 编程手记

## Java
+ 在函数的声明后面添加 `throws Exception` 表示这个函数允许有未处理的异常，并将该异常上抛，不加这一句，则会出现未处理的异常的情况
+ `f(String... s)` 可以用 `...` 表示多个参数或一个字符串数组
+ 写入文件的写操作要记得 `flush` 这样才会输出到文件中
+ `byte` 转 `String` 利用 `String` 的 **构造方法** 即可

**正则表达式与CSV文件**
+ 对于一个 `.csv` 文件，对于元素内部含有 `,` 的情况会使用 `""` 包围该元素，这时直接按照 `,` 分隔文件会出错
+ 正确做法是按照正则表达式 `,(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)` 分隔，
+ 正则表达式中 `^` 表示行头， `$` 表示行尾
  > `^[a-z] `只能匹配以小写字母为行首的行: `"a..."`
`[a-z]$`只能匹配以小写字母为行尾的行: `"...a"`
`^[a-z]$` 应该只能匹配只有一个小写字母的行: `"a"`
+  `(?=中国)人` 表示 **只匹配** `中国人` 中的 `人`，而不能是其他部位的人

**javacsv**
```java
        try{
            // create csv
            CsvReader csvReader = new CsvReader(filename);
            // read header
            csvReader.readHeaders();
            String[] headers = csvReader.getHeaders();

            int rowKey = 0;
            while(csvReader.readRecord()){
                hbd.putData(headers, csvReader.getValues(), rowKey);
                rowKey++;
            }

        }catch (IOException ex){
            System.err.println(ex.getMessage());
        }
```

## Ubuntu虚拟机
+ `VMware` 使用 `Ctrl + G` 使得鼠标键盘进入虚拟机并用 `Ctrl + Alt` 退出
+ `Ubunto` 使用 `Ctrl + Alt + T` 打开命令行
+ `sudo apt-get install` 远程安装
+ `Ctrl + h` 查看隐藏文件夹
+ `unzip` 解压 `.zip` 文件

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

**切换用户**
+ `su lei` 切换用户，不添加参数则直接切换到 `root` 用户

**“按照步骤”和找替代品**
+ 本以为按照步骤来，实际上找了替代品，导致多花费了太多时间
+ 在 `Mysql` 的安装过程中需要给出初始密码，如果不严格按照这个方式来，而是在后来设置密码，会导致 `Hive` 启动报错
+ 即便是 `root` 权限也不是万能的，`Hive` 需要在 `hadoop` 系统用户（可能因为只有这个用户对 `hadoop` 文件夹有修改权限）下启动，在 `root` 用户下启动报错
+ 对于已经详细给出过程的教程，报错的原因只有可能是 **没有按照步骤来** 或 **以为自己按照步骤来了**， 后者带来的时间浪费更为严重

**权限问题**
+ 修改、查看文件时需要注意权限，比如 `vi` 命令，在用户没有读权限时是打不开文件的，类似的命令还有 `gedit`
+ 对于一些 `ubuntu`中只有读权限的文件夹，可以更改权限
  ```
  sudo chmod -R 755 /usr/local/hadoop/hadoop-2.7.5
  sudo chown -R hadoop:hadoop /usr/local/hadoop/hadoop-2.7.5
  ```
+ `-R` 表示 **递归**
+ 这样 `sudo` 的作用就体现了，在桌面端中，有时用户没有权限创建文件夹，就可以利用命令行的 `sudo` 命令创建文件夹
+ 有时程序需要写入，但是若用户没有权限，那么无法写入，从而运行不成功，这时要使得需要写入的文件夹拥有权限

**hostname hosts**
+ 这两个文件都在 `/etc` 目录下，前者给出了主机名称，后者给出了 IP地址
+ `localhost` 一般指 `127.0.0.1`

**进程**
+ 察看进程 `jps`
+ 关闭进程 `kill 进程号`

## hadoop
**没有datanode**
在每次执行 `hadoop namenode -format` 时，都会为 `NameNode` 生成 `namespaceID`，但是在 `hadoop.tmp.dir` 目录下的 `DataNode` 还是保留上次的 `namespaceID`，因为 `namespaceID` 的不一致，而导致 `DataNode` 无法启动

所以 **只要在每次执行 `hadoop namenode -format` 之前，先删除 `hadoop.tmp.dir`（路径为 `/usr/local/hadoop/` 下的）`tmp` 目录就可以启动成功**

以后在 `hadoop format` 过程中 要注意不要频繁地 `reformat  namnode`（格式化命令为  `./bin/hadoop namenode -format`）的ID信息。`format` 过程中选择 **N（否)** 就是了

**关闭安全模式**
+ 开启安全模式时，`hbase` 无法正常工作，`hdfs dfsadmin -safemode leave`
可以关闭安全模式

**设置Hadoop和hbase的IP即端口**
+ 要让用户在可能写入的地址拥有 **写入的权限**
+ `hbase` 在 `hbase-site.xml` 文件中 `master:9000`
+ `hadoop` 在 `core-site.xml` 文件中 `master:9000`
+ **保证他们的端口一致** , 这样 `Hmaster` 进程才不会消失
+ **master info** 的端口需要配置
```py
  <property>
        <name>hbase.master.info.port</name>
        <value>60010</value>
  </property>
```
+ **启动顺序**
```shell
# 重启时
rm -r /tmp/dfs # 删除临时文件
./bin/hadoop namenode -format # 格式化namenode
# 以上出故障再使用

# 正常关闭正常启动即可
cd $HADOOP_HOME
./sbin/start-dfs.sh # 需要查看进程
hdfs dfsadmin -safemode leave # 全部重开始再设置，关闭安全模式

cd $HBASE_HOME
./bin/start-hbase.sh # 需要查看进程

./bin/hbase shell # 进入shell
```

+ 附配置文件总览

修改 `hostname`：`master`
修改 `host`文件 ： `192.168.238.128 master`

*hbase-site.xml*
```xml
    <property>  
        <name>hbase.rootdir</name>  
        <value>hdfs://master:9000/hbase</value>  
    </property>  
    <property>  
        <name>hbase.master</name>  
        <value>hdfs://master:60000</value>  
    </property>
    <property>  
        <name>hbase.cluster.distributed</name>  
        <value>true</value>  
    </property>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>  
        <name>hbase.zookeeper.quorum</name>  
        <value>192.168.238.128</value>  
    </property>  
    <property>
    	<name>hbase.zookeeper.property.dataDir</name>
    <value>/usr/local/hbase/zk_data</value>
    </property>
    <property>
        <name>hbase.master.info.port</name>
        <value>60010</value>
    </property>
```

*core-site.xml*
```xml
    <property>
             <name>hadoop.tmp.dir</name>
             <value>file:/usr/local/hadoop/hadoop-2.7.5/tmp</value>
             <description>Abase for other temporary directories.</description>
        </property>
        <property>
             <name>fs.defaultFS</name>
             <value>hdfs://master:9000</value>
        </property>
```

**hive**
+ 需要导入 `hbase` 的库
+ 需要选择正确 `mysql connector`， 正确版本为 `mysql-connector-java.5.1.30`
+ 允许远程连接
  ```shell
  hive --service metastore
  hive --service hiveserver2
  ```
+ `mysql-connector-java`
  ```java
  stmt.excute(sql) // 没有 resultset
  stmt.excuteQuery(sql) // 有 resultset
  ```
  执行建表、删除表操作时没有 `resultset` 而执行查询操作时有，要区分开来
