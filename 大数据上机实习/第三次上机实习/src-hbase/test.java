public class test {
    public static void main(String[] args){
        String testStr = "hello.java";
        String[] result = testStr.split("\\.");
        for(String s:result)
            System.out.println(s);
    }
}
