package org.apache.hadoop.examples;
import java.net.URI;
import java.io.IOException;
import java.io.File;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class PageRank {
	private static final float d = (float) 0.85, N = 10;

	public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			System.out.println(value.toString());
			String[] tokens = value.toString().split("\t");
			int firstComma = tokens[1].indexOf(',');
			if (firstComma <= 0) {
				return;
			}
			String rankStr = tokens[1].substring(0, firstComma);
			String linksStr = tokens[1].substring(firstComma + 1);
			String[] linksto = linksStr.split(",");
			String url = tokens[0].trim();
			float rank;
			try {
				rank = Float.parseFloat(rankStr);
			} catch (NumberFormatException e_) {
				System.out.println("Float cast error" + rankStr);
				return;
			}
			rank /= linksto.length;
			for (String u:linksto) {
				context.write(new Text(u), new Text("" + rank));
			}
			context.write(new Text(url), new Text("," + linksStr));
		}
	}

	public static class MmSumReducer extends Reducer<Text, Text, Text, Text> {

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			String linksStr = "";
			float sum = 0;
			for (Text val : values) {
				String valStr = val.toString();
				if (valStr.charAt(0) == ',') {
					linksStr = valStr;
				} else {
					float valFloat;
					try {
						valFloat = Float.parseFloat(valStr);
					} catch (NumberFormatException e_) {
						System.out.println("Float cast error" + valStr);
						continue;
					}
					sum += valFloat;
				}
			}
			float newRank = sum * d + (1 - d) / N;
			context.write(key, new Text(newRank + linksStr));
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		Path output = new Path("/user/hadoop/output");
		Path input = new Path("/user/hadoop/input");
		Path ps = new Path("/user/hadoop/output/_SUCCESS");
		Path pi = new Path("/user/hadoop/input/part-r-00000");
		Path po = new Path("/user/hadoop/output/part-r-00000");
		
		int exit_status=0;	
		fs.delete(output, true);
		
		for(int i=0;i<2;++i) {
			System.out.println("task:" + Integer.toString(i));
			Job job = new Job(conf, "word count");
			job.setJarByClass(PageRank.class);
			job.setMapperClass(TokenizerMapper.class);
			job.setReducerClass(MmSumReducer.class);
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			
			// input, output
			FileInputFormat.addInputPath(job, input);
			FileOutputFormat.setOutputPath(job, output);
			
			job.submit();
			while(!job.isComplete()) continue;
			job.killJob();
			
			// move file		
			FileUtil.copy(fs, po, fs, pi, true,true,conf);
			fs.delete(output, true);	
		}
		
		System.exit(exit_status);
	}
}
