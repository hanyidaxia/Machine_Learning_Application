package pro2;
/**
 * 
 * 测试语法 条件运算符（三元条件运算符）
 * @author hanyi_gm
 *
 */
public class TestOperator6 {

	
	public static void mian(String[] args) {
		int score = 80;
		String type =score<60?"不及格":"及格";
		System .out.println(type);
		
		if (score<60) {
			System.out.println("不及格");
		}else {
			System.out.println("及格");
		}
		
	}
}
// 这节课讲了条件语句的简化用法，就是说明了先加个判断符，然后后面跟上？然后后面前边是满足的输出，之后是不满足的的输出
