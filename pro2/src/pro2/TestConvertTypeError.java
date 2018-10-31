package pro2;
/**
 * 测试转化类型错误处理
 * @author hanyi_gm
 *
 */
public class TestConvertTypeError {

	
	public static void main(String[] args) {
		int a  = 1000000000;
		int b = 20;
		System.out.println("babajiushiwo=" + a*b);//这里出现了一个错误表示的就是乘积已经超出了int范围
		
		long total = a*b;
		System.out.println(total);//乘的时候依然是作为两个int类型的数据做乘积所以不行
		
		long total1 = (long)a * b;
		
		System.out.println(total1);
		
				
				
	}
}
