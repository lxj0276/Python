import com.csvreader.CsvReader;
import java.io.*;

public class main {
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
    }
}
