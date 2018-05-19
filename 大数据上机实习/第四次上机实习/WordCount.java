package org.apache.hadoop.examples;
import java.io.IOException;
import java.lang.NumberFormatException;
import java.math.BigInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
public class WordCount {
  public static class TokenizerMapper 
        extends Mapper<Object, Text, Text, Text>{
    private Text company = new Text();
    private Text revenue = new Text();
    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      String line = value.toString();
      String[] splits = line.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)");
      if(splits.length != 20)
    	  return;
      try {
    	  if(splits[18]=="" || Double.parseDouble(splits[18]) < 6.5)
    	  return;
      } catch(NumberFormatException nfe) {
    	  return;
      }
      
      revenue.set(splits[12]);
      String companies = splits[9];
      Pattern p = Pattern.compile("(?<=name\"\":\\s\\\"\\\")[a-zA-Z\\s]*");
      Matcher m = p.matcher(companies);

      while(m.find()){
          company.set(m.group());
          context.write(company, revenue);
      }
    }
  }
  public static class SumReducer 
        extends Reducer<Text,Text,Text,Text> {
    private Text result = new Text();
    public void reduce(Text key, Iterable<Text> values, 
                        Context context
                        ) throws IOException, InterruptedException {
      BigInteger sum = new BigInteger("0");
      for (Text val : values) {
    	BigInteger bi = new BigInteger(val.toString());
        sum = sum.add(bi);
      }
      result.set(sum.toString());
      context.write(key, result);
    }
  }
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
      System.err.println("Usage: wordcount <in> <out>");
      System.exit(2);
    }
    Job job = new Job(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(SumReducer.class);
    job.setReducerClass(SumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}