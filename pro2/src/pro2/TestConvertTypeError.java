package pro2;
/**
 * ����ת�����ʹ�����
 * @author hanyi_gm
 *
 */
public class TestConvertTypeError {

	
	public static void main(String[] args) {
		int a  = 1000000000;
		int b = 20;
		System.out.println("babajiushiwo=" + a*b);//���������һ�������ʾ�ľ��ǳ˻��Ѿ�������int��Χ
		
		long total = a*b;
		System.out.println(total);//�˵�ʱ����Ȼ����Ϊ����int���͵��������˻����Բ���
		
		long total1 = (long)a * b;
		
		System.out.println(total1);
		
				
				
	}
}
