package pro3;

/**
 * ����if else˫ѡ���ʽ
 * @author hanyi_gm
 *
 */
public class TestIfElse {

	
	public static void main(String [] args) {
		double d = 4* Math.random();
		double area =  Math.PI * Math.pow(d, 2);
		double circle = 2 * Math.PI * d;
		System.out.println("�뾶Ϊ ��" + d);
		System.out.println("���Ϊ��" + area);
		System.out.println("�ܳ�Ϊ: " + circle);
		
		if(area > circle) {
			System.out.println("��������ܳ�");
			
		}else {
			System.out.println("���С���ܳ�");
			
		}
		
	}
}
