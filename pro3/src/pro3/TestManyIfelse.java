package pro3;
/**
 * ����else�ṹ
 * @author hanyi_gm
 *
 */
public class TestManyIfelse {

	public static void main(String[] args) {
		int age = (int)(100 * Math.random());
		System.out.println("������" + age + "Ӧ��");
		if(age < 15) {
			System.out.println("Сƨ���ӣ��త��");
		}else if(age < 25) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 35) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 45) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 55) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 65) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 75) {
			System.out.println("�������ˣ��ཻ��");
		}else if(age < 85) {
			System.out.println("�������ˣ��ཻ��");
		}else  {
			System.out.println("���˰ɣ��ܽ����þͽ����ã�������׬����");
		}
	}
}
