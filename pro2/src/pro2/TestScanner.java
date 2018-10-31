package pro2;

import java.util.Scanner;


/**
 * 
 * 测试一个基本的输入功能以及加上了收集数据的功能
 * 
 * @author hanyi_gm
 *
 */
public class TestScanner {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		System.out.println("请输入你的名字:");
		String name = scanner.nextLine();
		System.out.println("请输入你的爱好:");
		String favor = scanner.nextLine();
		System.out.println("请输入你的年龄：");
		int age = scanner.nextInt();
		
		
		System.out.println("########################################################################");
		System.out.println(name);
		System.out.println(favor);
		System.out.println(age);
		System.out.println("离开没人搭理这个状态的天数 ：" + age * 365);
		
	}
}
