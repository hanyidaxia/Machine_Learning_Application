package pro3;
/**
 * 
 * ����continue���
 * @author hanyi_gm
 *continue ���ڽ�������ѭ������ʼ��һ����yeah
 */
public class TestContinue {

	public static void main(String[] args) {
		// 100��150֮�䲻�ܱ�3���������ҳ���������ÿ��5����
		int count = 0;
		for(int i = 100; i<=150; i++) {
			if(i%3 == 0) {
				continue;
			}
			System.out.print("��������Ա������������" + i);
			count++;
			if(count%5 == 0) {
				System.out.println();
			}
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	}
}
