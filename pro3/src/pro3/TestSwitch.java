package pro3;
/**
 * ����switch���
 * @author hanyi_gm
 *
 */
public class TestSwitch {

	public static void main(String[] args) {
		
		int month = 1 +(int)(12*Math.random());
		System.out.println("�·�"+ month);
		
		
		switch(month) {
		case 1:
			System.out.println("һ�·��ˣ���������");
			break;
		case 2:
			System.out.println("���·��ˣ����¶���̧ͷ");
			break;
		default:
			System.out.println("���������·�");
			
			
			System.out.println("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
	//������ֵ�жϵ�ʱ��ϣ��ʹ��switach		
			
			char c = 'a';
			int rand =  (int)(26 * (Math.random()));
			char c1 = (char)(c + rand);
			System.out.println(c1 + ":");
			switch(c1) {
			case 'a':
			case 'e':
			case 'i':
			case 'o':
			case 'u':
				System.out.println("this is the support section");
				break;
			case 'y':
			case 'w':
				System.out.println("half support section");
				break;
			default:
				System.out.println("those are the support section");
				
			
			
			}
			
			
		}
	}
}
