import java.io.BufferedReader;
import java.io.FileReader;
import java.math.BigInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class test {
    public static void main(String[] args) throws Exception{
        BufferedReader file = new BufferedReader(new FileReader("tmdb_5000_movies.csv"));
        file.readLine();
        String line = file.readLine();
        String[] splits = line.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)");
        for(String s : splits){
            System.out.println(s);
        }
        System.out.println(splits.length);
        System.out.println("Double:" + Double.parseDouble(splits[18]));
        System.out.println(Double.parseDouble(splits[18]) < 6.5);
        BigInteger bi = new BigInteger(splits[12]);
        System.out.println("BigInteger:" + bi);
        bi = bi.add(bi);
        System.out.println("BigInteger:" + bi);
        String companies = splits[9];
        Pattern p = Pattern.compile("(?<=name\"\":\\s\\\"\\\")[a-zA-Z\\s]*");
        Matcher m = p.matcher(companies);
        System.out.println(companies);
        while(m.find()){
            System.out.println(m.group());
        }
    }
}
