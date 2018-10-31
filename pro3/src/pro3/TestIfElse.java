package pro3;

/**
 * 测试if else双选择句式
 * @author hanyi_gm
 *
 */
public class TestIfElse {

	
	public static void main(String [] args) {
		double d = 4* Math.random();
		double area =  Math.PI * Math.pow(d, 2);
		double circle = 2 * Math.PI * d;
		System.out.println("半径为 ：" + d);
		System.out.println("面积为：" + area);
		System.out.println("周长为: " + circle);
		
		if(area > circle) {
			System.out.println("面积大于周长");
			
		}else {
			System.out.println("面积小于周长");
			
		}
		
	}
}
