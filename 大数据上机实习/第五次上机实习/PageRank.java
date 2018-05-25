package org.apache.hadoop.examples;
import java.net.URI;
import java.io.IOException;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

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

	public static ArrayList<Float> readList(FileSystem fs, Path p) throws Exception{
		BufferedReader file = new BufferedReader(new InputStreamReader(fs.open(p)));
		ArrayList<Float> l = new ArrayList<>();
		String line = file.readLine();
		while(line != null) {
			line = line.split("\t")[1];
			Float f = Float.parseFloat(line.split(",")[0]);
			l.add(f);
			line = file.readLine();
		}
		return l;
	}
	
	public static float distance(ArrayList<Float> l1, ArrayList<Float> l2) {
		float f = 0;
		for(int i=0; i<l1.size();++i) {
			f += (l1.get(i) - l2.get(i)) * (l1.get(i) - l2.get(i));
		}
		return f;
	}
	public static boolean compare(FileSystem fs, Path p1, Path p2, float precision) throws Exception{
		ArrayList<Float> l1 = readList(fs, p1);
		ArrayList<Float> l2 = readList(fs, p2);
		float f = distance(l1, l2);
		return f < precision;
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
		
		for(int i=0; i<7; i++) {
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
			
			boolean b = compare(fs, pi, po, 0.01f);
			System.out.println(b);
			if(b)
				break;
			
			// move file		
			FileUtil.copy(fs, po, fs, pi, true,true,conf);
			fs.delete(output, true);	
		}
		
		System.exit(exit_status);
	}
}
