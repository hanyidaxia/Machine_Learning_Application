package pro2;

import java.util.Scanner;


/**
 * 
 * ����һ�����������빦���Լ��������ռ����ݵĹ���
 * 
 * @author hanyi_gm
 *
 */
public class TestScanner {

	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		System.out.println("�������������:");
		String name = scanner.nextLine();
		System.out.println("��������İ���:");
		String favor = scanner.nextLine();
		System.out.println("������������䣺");
		int age = scanner.nextInt();
		
		
		System.out.println("########################################################################");
		System.out.println(name);
		System.out.println(favor);
		System.out.println(age);
		System.out.println("�뿪û�˴������״̬������ ��" + age * 365);
		
	}
}
