package pro2;
/**
 * test opreator
 * @author hanyi_gm
 *
 */
public class Testopreator1 {

	
	public static void main(String [] args) {
		
/*		byte a  = 1;
		int b = 2;
		long b2 = 3;
		
		//byte c = a+b; nonononono
		//int d = b + b2; // again you shouldn't put the b into the wrong palce
		
		float p = 3.14F;
		double m = b + b2;
		float m2 = b + b2;
		//float m3 = p + 6.14;//������һ����double���͵�����
			out.println(-9%5);
			*/
			//�����������Լ�
			int a = 3;
					int b = a++ ;//�ȸ�ֵ��b��������������
					System.out.println("a =" + a +"\nb=" + b);// 
			a = 3;
			b = ++a ;
			System.out.println("a =" + a +"\nb=" + b);
		
		
			
			int r = 3;
			int y = 4;
			r+= y;// r = r+y
			System.out.println("r ="+  r + "\ny = " + y);
			r*= y+63;
			System.out.println("r ="+  r + "\ny = " + y);
			
		
	}
}
