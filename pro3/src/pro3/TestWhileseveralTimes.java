package pro3;
/**
 * 嵌套循环，就是多个循环套起来使用
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
		// 尝试打印一张九九乘法表
		
		for(int h =1; h <= 19; h++ ) {
			for(int y = 1; y <=h; y++) {
				System.out.print( h +"*"+ y + "=" +  h * y  + "\t");
			}
			System.out.println();
			
			System.out.println("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
			
		//计算100以内奇数以及偶数的和	
			int sum1  = 0;
			int sum2  = 0;
			
			for (int t =1; t <=100; t++) {
				if(t%2 == 0) {
					sum1  +=  t;
				}else {
					sum2   += t;
				}
				
			}
			System.out.println("奇数和是：" + sum1);
			System.out.println("偶数和是：" + sum2);
			
		}   
		
		
		System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");

	//输出1-1000之间可以被5整除的数并且每一行放5个
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



