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

        HiveDemo hd = new HiveDemo();
        try{
            conn = hd.getConn();
            stmt = conn.createStatement();
            // connect mysql

            CsvReader csvReader = new CsvReader(fileNameMovie);
            String tableName = fileNameMovie.split("\\.")[0];
            // read file
            csvReader.readHeaders();
            String[] headers = csvReader.getHeaders();

            // jdbc
            hd.dropTable(stmt, tableName);
            hd.createTable(stmt, tableName, headers);

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
    /*
    public static HBaseDemo hbd = null;

    public static void readAndInsert(String filename) throws Exception{
        try{
            // create csv
            CsvReader csvReader = new CsvReader(filename);
            // read header
            csvReader.readHeaders();
            String[] headers = csvReader.getHeaders();

            // create table and connect
            hbd.createTable(filename.split("\\.")[0]);
            hbd.connnectTable(filename.split("\\.")[0]);

            int rowKey = 0;
            while(csvReader.readRecord()){
                hbd.putData(headers, csvReader.getValues(), rowKey);
                rowKey++;
            }

        }catch (IOException ex){
            System.err.println(ex.getMessage());
        }

        System.out.println("Insert Compelte!");
    }

    public static void main(String[] args) throws Exception{
        hbd = new HBaseDemo();
        String fileNameMovies = "tmdb_5000_movies.csv";
        String fileNameCredits = "tmdb_5000_credits.csv";

        // readAndInsert(fileNameMovies);
        // readAndInsert(fileNameCredits);

        hbd.get("tmdb_5000_movies", 1, "budget");
    }*/