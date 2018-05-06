import com.csvreader.CsvReader;

public class test {
    public static void main(String[] args) throws Exception{
        CsvReader csvReader = new CsvReader("tmdb_5000_movies.csv");

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
            sb.append(" string, ");
        }
        String state = sb.toString();
        state = state.replaceAll(",\\s$", ")");
        System.out.println(state);

        String test = "12345";
        System.out.println(test.matches("^[0-9]+$"));
    }
}
