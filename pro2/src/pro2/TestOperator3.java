package pro2;


/**
 * 逻辑运算符牛逼
 * 
 * @author hanyi_gm
 *
 */
public class TestOperator3 {

	public static void main(String[] args) {
		boolean b1  = true;
		boolean b2  = false;
		System.out.println(b1&b2);
		System.out.println(b1|b2);
		System.out.println(b1^b2);
		System.out.println(!b2);
		
		//短路
		boolean b3 = 1>2&&2<(3/0);//第一个输入的值为false则后面的东西就不再计算了
		
		System.out.println(b3);
		
	}
}
