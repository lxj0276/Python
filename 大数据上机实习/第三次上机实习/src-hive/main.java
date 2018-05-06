import com.csvreader.CsvReader;
import java.io.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;


public class main {
    public static void main(String[] args) throws Exception{
        Connection conn = null;
        Statement stmt = null;
        String fileNameMovie = "tmdb_5000_movies.csv";
        String fileNameCredits = "tmdb_5000_credits.csv";
        String fileTest = "test.csv";
        String file = fileNameMovie;
        HiveDemo hd = new HiveDemo();
        try{
            conn = hd.getConn();
            stmt = conn.createStatement();
            // connect mysql

            CsvReader csvReader = new CsvReader(file);
            String tableName = file.split("\\.")[0];
            // read file
            csvReader.readHeaders();
            String[] headers = csvReader.getHeaders();
            csvReader.readRecord();
            String[] firstLine = csvReader.getValues();

            // jdbc
            // drop table
            hd.dropTable(stmt, tableName);
            // create table
            hd.createTable(stmt, tableName, headers, firstLine);
            // show tables
            hd.showTables(stmt, tableName);
            String filepath = "/home/hadoop/Desktop/HiveDemo/nohead_" + file;
            // load csv
            hd.loadData(stmt, tableName, filepath);
            // table head
            hd.tableHead(stmt, tableName, headers.length);
            // simple query
            hd.selectData(stmt,tableName, headers.length, "budget1", "300000000");
        }catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        } catch (SQLException e) {
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                if (stmt != null) {
                    stmt.close();
                }
                if (conn != null) {
                    conn.close();
                }
            } catch (SQLException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
    }
}