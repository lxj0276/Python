import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellScanner;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.util.Bytes;

public class HBaseDemo {
    public Connection connection;
    //用hbaseconfiguration初始化配置信息时会自动加载当前应用classpath下的hbase-site.xml
    public static Configuration configuration = HBaseConfiguration.create();
    public Table table;
    public Admin admin;

    public HBaseDemo() throws Exception{
        //对connection初始化
        connection = ConnectionFactory.createConnection(configuration);
        admin = connection.getAdmin();
    }

    public void connnectTable(String name) throws Exception{
        TableName tableName = TableName.valueOf(name);
        table = connection.getTable(tableName);
    }

    //创建表
    public void createTable(String tablename,String... cf1) throws Exception{
        //获取admin对象
        admin = connection.getAdmin();
        //创建tablename对象描述表的名称信息
        TableName tname = TableName.valueOf(tablename);//bd17:mytable
        //创建HTableDescriptor对象，描述表信息
        HTableDescriptor tDescriptor = new HTableDescriptor(tname);
        //判断是否表已存在
        if(admin.tableExists(tname)){
            System.out.println("表"+tablename+"已存在");
            return;
        }
        //添加表列簇信息
        for(String cf:cf1){
            HColumnDescriptor famliy = new HColumnDescriptor(cf);
            tDescriptor.addFamily(famliy);
        }
        //调用admin的createtable方法创建表
        admin.createTable(tDescriptor);
        System.out.println("表"+tablename+"创建成功");
    }

    //新增数据到表里面Put
    public void putData(String[] head, String[] line, int rowKey) throws Exception{
        Put put = new Put(Bytes.toBytes("rowkey_" + rowKey));
        for(int i=0; i<head.length; ++i){
            put.addColumn(Bytes.toBytes(head[i]), Bytes.toBytes(head[i]), Bytes.toBytes(line[i]));
        }

        table.put(put);
    }

    public void get(String tableName, int rowKey, String attr) throws Exception{
        connnectTable(tableName);

        Get get = new Get(Bytes.toBytes("rowkey_" + rowKey));
        Result rs = table.get(get);
        System.out.println(new String(rs.getValue(Bytes.toBytes(attr), Bytes.toBytes(attr))));
    }
}
