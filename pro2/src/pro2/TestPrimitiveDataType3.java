package pro2;


/**
 * test the �ַ����ͺͲ�������
 * @author hanyi_gm
 *
 */
public class TestPrimitiveDataType3 {

	
	public static void main(String[] args) {
		
		char a  = 'T';
		char b = '��';
		char c = '\u0061';
		System.out.println(c);
			
			System.out.println("" + 'a' +'\n'+ 'b');
			System.out.println("" + 'a' +'\t'+ 'b');
			System.out.println("" + 'a' + '\''  +  'b');
			
			//char d  = "asd"; // it's wrong
			// String ���� a list of numbers
			String d  = "asdfaf";
			
			
			//test boolean
			boolean man = true;
			
			
			if(man ) { //���䲻�Ƽ��� man == true�� ��Ϊ������д�������ж�����Ϊ��ֵ���
				System.out.println("����");
			}else {
				System.out.println("Ů��");
			}
	}
}
