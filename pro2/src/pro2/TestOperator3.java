package pro2;


/**
 * �߼������ţ��
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
		
		//��·
		boolean b3 = 1>2&&2<(3/0);//��һ�������ֵΪfalse�����Ķ����Ͳ��ټ�����
		
		System.out.println(b3);
		
	}
}
