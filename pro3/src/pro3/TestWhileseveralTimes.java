package pro3;
/**
 * Ƕ��ѭ�������Ƕ��ѭ��������ʹ��
 * @author hanyi_gm
 *
 */
public class TestWhileseveralTimes {

	
	public static void main(String[] args) {
		
		/**for(int i = 0; i <=5 ; i++) {
			for(int j = 1; j<=5; j++) {
				System.out.println(j);
				System.out.print(i +"\t");
			//System.out.println();
			
		//} */
		// ���Դ�ӡһ�žžų˷���
		
		for(int h =1; h <= 19; h++ ) {
			for(int y = 1; y <=h; y++) {
				System.out.print( h +"*"+ y + "=" +  h * y  + "\t");
			}
			System.out.println();
			
			System.out.println("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
			
		//����100���������Լ�ż���ĺ�	
			int sum1  = 0;
			int sum2  = 0;
			
			for (int t =1; t <=100; t++) {
				if(t%2 == 0) {
					sum1  +=  t;
				}else {
					sum2   += t;
				}
				
			}
			System.out.println("�������ǣ�" + sum1);
			System.out.println("ż�����ǣ�" + sum2);
			
		}   
		
		
		System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");

	//���1-1000֮����Ա�5������������ÿһ�з�5��
		int p = 0;
		for (int k = 1; k<=1000; k++ ) {
		
			if(k%5 == 0) {
			System.out.print(k +"\t");
			p++;
		}if(p == 5) {
			System.out.println();
			p = 0;
		}
		}
	
	
	
	
	}
	
	
	
	
	
	
	
	
	
	}



