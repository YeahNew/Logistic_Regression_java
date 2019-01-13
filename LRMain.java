/*
����
 */
import java.io.*;

public class LRMain {
	public static void main(String[] args) throws IOException{
		// filename 
		String filename = "data.txt";
		// �������������ͱ�ǩ
		double [][] feature = LoadData.Loadfeature(filename);
		double [] Label = LoadData.LoadLabel(filename); 
		// ��������
		int samNum = feature.length;
		int paraNum = feature[0].length;
		double rate = 0.01;
		int maxCycle = 1000;
		// LRģ��ѵ��
		LRtrainGradientDescent LR = new LRtrainGradientDescent(feature,Label,paraNum,rate,samNum,maxCycle);
		double [] W = LR.Updata(feature, Label, maxCycle, rate);
		//����ģ��
		String model_path = "wrights.txt";
		SaveModel.savemodel(model_path, W);
		//ģ�Ͳ���
		double [] pre_results = LRTest.lrtest(paraNum, samNum, feature, W);
		//������Խ��
		String results_path = "pre_results.txt";
		SaveModel.saveresults(results_path, pre_results);
	}

}
