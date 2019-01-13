/*
 * ʹ���ݶ��½��㷨ѵ��LRģ��
 */
public class LRtrainGradientDescent {
	int paraNum; //Ȩ�ز����ĸ���
    double rate; //ѧϰ��
    int samNum; //��������
    double [][] feature; //������������
    double [] Label;//������ǩ
    int maxCycle; //����������
    
    public LRtrainGradientDescent(double [][] feature, double [] Label, int paraNum,double rate, int samNum,int maxCycle) {
    	this.feature = feature;
    	this.Label = Label;
    	this.maxCycle = maxCycle;
    	this.paraNum = paraNum;
    	this.rate = rate;
    	this.samNum = samNum;    	
    }
    // Ȩֵ�����ʼ��
    public double [] ParaInitialize(int paraNum) {
    	double [] W = new double[paraNum];
    	for (int i = 0; i < paraNum; i ++) {
    		W[i] =  1.0;
    	}
    	return W;   	
    }
    //����ÿ�ε������Ԥ�����
    public double [] PreVal(int samNum,int paraNum, double [][] feature,double [] W) {
    	double [] Preval = new double[samNum];
    	for (int i = 0; i< samNum; i ++) {
    		double tmp = 0;
    		for(int j = 0; j < paraNum; j ++) {
    			tmp += feature[i][j] * W[j];
    		}
    		Preval[i] = Sigmoid.sigmoid(tmp);
    	}
    	return Preval;
    }
    //���������
    public double error_rate(int samNum, double [] Label, double [] Preval) {
    	double sum_err = 0.0;
    	for(int i = 0; i < samNum; i ++) {
    		sum_err += Math.pow(Label[i] - Preval[i], 2);  		
    	}
    	return sum_err;
    }
    //LRģ��ѵ��
    public double[] Updata(double [][] feature, double[] Label, int maxCycle, double rate) {
    	// �ȼ���������������������
    	int samNum = feature.length;
    	int paraNum = feature[0].length;
    	//��ʼ��Ȩ�ؾ���
    	double [] W = ParaInitialize(paraNum);
    	// ѭ�������Ż�Ȩ�ؾ���
    	for (int i = 0; i < maxCycle; i ++) {
    		// ÿ�ε���������Ԥ��ֵ
    		double [] Preval = PreVal(samNum,paraNum,feature,W);
    		double sum_err = error_rate(samNum,Label,Preval);
    		if (i % 10 == 0) {
    			System.out.println("��" + i + "�ε�����Ԥ�����Ϊ:" + sum_err);
    		}
    		//Ԥ��ֵ���ǩ�����
    		double [] err = new double[samNum];
    		for(int j = 0; j < samNum; j ++) {
    			err[j] = Label[j] - Preval[j];
    		}
    		// ����Ȩ�ؾ�����ݶȷ���
    		double [] Delt_W = new double[paraNum];
    		for (int n = 0 ; n < paraNum; n ++) {
    			double tmp = 0;
    			for(int m = 0; m < samNum; m ++) {
    				tmp += feature[m][n] * err[m];
    			}
    			Delt_W[n] = tmp / samNum;
    		} 
    		
    		for(int m = 0; m < paraNum; m ++) {
    			W[m] = W[m] + rate * Delt_W[m];
    		}
    	}
    return W;
    }

}
