import com.csvreader.CsvReader;
import java.io.*;

public class test {
    public static void main(String[] args) throws Exception{
        CsvReader csvReader = new CsvReader("tmdb_5000_credits.csv");

        csvReader.readHeaders();
        String[] headers = csvReader.getHeaders();

        csvReader.readRecord();
        String[] records = csvReader.getValues();

        for(int i=0; i<headers.length; ++i){
            System.out.println(String.format("%s : %s", headers[i], records[i]));
        }

        StringBuilder sb = new StringBuilder("(");
        for(String s : headers){
            sb.append(s);
            sb.append(":chararray, ");
        }
        String state = sb.toString();
        state = state.replaceAll(",\\s$", ")");
        System.out.println(state);

        StringBuilder sb1 = new StringBuilder();
        for(String s : headers){
            sb1.append(s + ", ");
        }
        String state1 = sb1.toString();
        state1 = state1.replaceAll(",\\s$", "");
        System.out.println(state1);

        String filename = "pigScriptCredits.pig";
        PrintWriter pw = new PrintWriter(filename);
        pw.write("A = load '/pig_data/tmdb_data/tmdb_5000_credits.csv'");
        pw.write(" using org.apache.pig.piggybank.storage.CSVExcelStorage()");
        pw.write(" as " + state + ";\n");
        pw.write("B = foreach A generate");
        pw.write(" " + state1 + ";\n");
        pw.write("dump B;\n");
        pw.write("store B into '/pig_data/tmdb_result2.txt';\n");
        pw.flush();
        pw.close();
    }
}
