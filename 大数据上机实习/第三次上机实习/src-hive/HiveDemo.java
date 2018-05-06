import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import org.apache.log4j.Logger;

public class HiveDemo {

    private  String driverName = "org.apache.hive.jdbc.HiveDriver";
    private  String url = "jdbc:hive2://192.168.238.128:10000";
    private  String user = "hadoop";
    private  String password = "hadoop";
    private  String sql = "";
    private  ResultSet res;
    private  final Logger log = Logger.getLogger(HiveDemo.class);

    public void tableHead(Statement stmt, String tableName, int len)
            throws SQLException {
        sql = "select * from " + tableName + " limit 5";
        System.out.println("Running:" + sql);
        res = stmt.executeQuery(sql);
        while (res.next()) {
            for(int i=0; i<len; ++i)
                System.out.print(res.getString(i+1) + "\t");
                System.out.print('\n');
        }
    }

    public void selectData(Statement stmt, String tableName, int len, String key, String value)
            throws SQLException {
        sql = "select * from " + tableName + " where "+ key + " = " + value;
        System.out.println("Running:" + sql);
        res = stmt.executeQuery(sql);
        while (res.next()) {
            for(int i=0; i<len; ++i)
                System.out.print(res.getString(i+1) + "\t");
            System.out.print('\n');
        }
    }

    public void loadData(Statement stmt, String tableName, String filepath)
            throws SQLException {
        StringBuilder sb = new StringBuilder();
        sb.append("load data local inpath '" + filepath);
        sb.append("' into table " + tableName);

        sql = sb.toString();
        System.out.println("Running:" + sql);
        stmt.execute(sql);
        System.out.println("Success!");
    }

    public void showTables(Statement stmt, String tableName)
            throws SQLException {
        sql = "show tables '" + tableName + "'";
        System.out.println("Running:" + sql);
        res = stmt.executeQuery(sql);
        if (res.next()) {
            System.out.println(res.getString(1));
        }
    }

    public void createTable(Statement stmt, String tableName, String[] headers, String[] firstLine)
            throws SQLException {

        StringBuilder sb = new StringBuilder("(");
        for(int i=0; i<headers.length; ++i){
            sb.append(headers[i] + "1");
            if(firstLine[i].matches("^[0-9]+$"))
                sb.append(" int, ");
            else
                sb.append(" string, ");
        }

        String state = sb.toString();
        state = state.replaceAll(",\\s$", ")");


        sql = "create table "
                + tableName
                + " " + state
                + " ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'\n";
               // + "WITH SERDEPROPERTIES (\n"
               // + "   \"quoteChar\"     = \"\\\"\"\n"
               // + ")  ";
        System.out.println("Running:" + sql);
        stmt.execute(sql);
        System.out.println("Table Created: " + tableName);
    }

    public void dropTable(Statement stmt, String tableName) throws SQLException {
        // 创建的表名
        sql = "drop table " + tableName;
        stmt.execute(sql);
        System.out.println("Table dropped: " + tableName);
    }

    public Connection getConn() throws ClassNotFoundException,
            SQLException {
        Class.forName(driverName);
        Connection conn = DriverManager.getConnection(url, user, password);
        return conn;
    }
}
