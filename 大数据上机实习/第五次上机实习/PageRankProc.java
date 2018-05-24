package org.apache.hadoop.examples;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class PageRankProc {

	public static class TokenizerMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] tokens = value.toString().split(",");
			if (tokens.length != 3) {
				System.out.println("len error");
				return;
			}
			int url, linkto, e;
			try {
				url = Integer.parseInt(tokens[0]);
				linkto = Integer.parseInt(tokens[1]);
				e = Integer.parseInt(tokens[2]);
			} catch (NumberFormatException e_) {
				System.out.println("Integer cast error");
				return;
			}
			if (e == 0)
				return;
			context.write(new IntWritable(url), new IntWritable(linkto));

		}
	}

	public static class MmSumReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

		public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			String s = "1";
			for (IntWritable val : values) {
				s +=","+ val;
			}
			context.write(key, new Text(s));
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
		job.setJarByClass(PageRank.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(MmSumReducer.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
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
