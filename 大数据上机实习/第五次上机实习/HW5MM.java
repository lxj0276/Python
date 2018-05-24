package org.apache.hadoop.examples;

//files: A B
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class HW5MM {
	//shape of A:m*n shape of B:n*p
	private static final int m = 10, p = 10;

	public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
			String[] tokens = value.toString().split(",");
			if (tokens.length != 3) {
				System.out.println("len error");
				return;
			}
			int i, j, e;
			try {
				i = Integer.parseInt(tokens[0]);
				j = Integer.parseInt(tokens[1]);
				e = Integer.parseInt(tokens[2]);
			} catch (NumberFormatException e_) {
				System.out.println("Integer cast error");
				return;
			}
			if (e == 0)
				return;
			if (fileName.equals("A")) {
				String va = "a," + j + "," + e;
				for (int k = 0; k < p; k++) {
					String ke = i + "," + k;
					context.write(new Text(ke), new Text(va));
				}
			} else if (fileName.equals("B")) {
				String va = "b," + i + "," + e;
				for (int k = 0; k < m; k++) {
					String ke = k + "," + j;
					context.write(new Text(ke), new Text(va));
				}
			} else {
				System.out.println("fileName error");
			}
		}
	}

	public static class MmSumReducer extends Reducer<Text, Text, Text, Text> {

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Map<Integer,Integer> map = new HashMap<Integer, Integer>();
			int sum = 0;
			for (Text val : values) {
				String[] tokens = val.toString().split(",");
				if (tokens.length != 3) {
					System.out.println("len error");
					break;
				}
				int ij, e;
				try {
					ij = Integer.parseInt(tokens[1]);
					e = Integer.parseInt(tokens[2]);
				} catch (NumberFormatException e_) {
					System.out.println("Integer cast error");
					return;
				}
				if(map.containsKey(ij)) {
					sum+=e*map.get(ij);
					map.remove(ij);
				}else {
					map.put(ij, e);
				}
			}
			
			context.write(key, new Text(" " + sum));
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
		job.setJarByClass(HW5MM.class);
		job.setMapperClass(TokenizerMapper.class);
		//job.setCombinerClass(MmSumReducer.class);
		job.setReducerClass(MmSumReducer.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		long startTime = System.currentTimeMillis();
		System.out.println("before job");
		int exit_status = job.waitForCompletion(true) ? 0 : 1;
		long endTime = System.currentTimeMillis();
		System.out.println("after job, spend " + (endTime - startTime) / 1000);
		System.exit(exit_status);
	}
}
