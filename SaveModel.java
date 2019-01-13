/*
 * ����Ȩ�ؾ����Ԥ����
 */
import java.io.*;

public class SaveModel {
	public static void savemodel(String filename, double [] W) throws IOException{
		File f = new File(filename);
		// ����FileOutputStream����
		FileOutputStream fip = new FileOutputStream(f);
		// ����OutputStreamWriter����
		OutputStreamWriter writer = new OutputStreamWriter(fip,"UTF-8");
		//����ģ�;����Ԫ�ظ���
		int n = W.length;
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < n-1; i ++) {
			sb.append(String.valueOf(W[i]));
			sb.append("\t");
		}
		sb.append(String.valueOf(W[n-1]));
		String sb1 = sb.toString();
		writer.write(sb1);
		writer.close();
		fip.close();
	}
	
	public static void saveresults(String filename, double [] pre_results) throws IOException{
		File f = new File(filename);
		// ����FileOutputStream����
		FileOutputStream fip = new FileOutputStream(f);
		// ����OutputStreamWriter����
		OutputStreamWriter writer = new OutputStreamWriter(fip,"UTF-8");
		//����Ԥ�����ĸ���
		int n = pre_results.length;
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < n-1; i ++) {
			sb.append(String.valueOf(pre_results[i]));
			sb.append("\n");
		}
		sb.append(String.valueOf(pre_results[n-1]));
		String sb1 = sb.toString();
		writer.write(sb1);
		writer.close();
		fip.close();
	}

}
