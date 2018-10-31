package pro2;
/**
 * 测试的是移位运算
 * @author hanyi_gm
 *
 */
public class TestOperator4 {

	
	public static void main(String[] args) {
		
		int a = 3;
		int b = 4;
		System.out.println(a&b);
		System.out.println(a|b);
		System.out.println(a^b);
		System.out.println(~a);
		
		//移位运算
		int c= 3<<2;
		 System.out.println(c);
	}
}
