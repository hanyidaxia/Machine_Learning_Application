package pro3;

/**
 * ���Է������ǳ�����function
 * @author hanyi_gm
 *
 */
public class TestMethod {

	public static void main(String[] args) {
		//ͨ�����������ͨ����
		TestMethod tm = new TestMethod();
		tm.printsxt();
		tm.add("����", 5464, 455);
	}
 void printsxt() {
	 System.out.println("������������˧");
 }
 		void add(String a, int b, int c) {
 			int sum = b+c;
 			System.out.println(a + "������" +sum +"������");
 			//return ;
 		}
 		
 		
 		int add(int m, int l, int j) {
 			int sum = l+j;
 			System.out.println(m +sum);
 			return sum;//1.return ���ص�ֵ��2.��������������
 		}

}
