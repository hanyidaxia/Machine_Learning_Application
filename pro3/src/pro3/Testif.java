package pro3;
/**
 * ����ifѡ�����
 * @author hanyi_gm
 *
 */
public class Testif {

	public static void main(String[] args) {
		double d = Math.random();//���ص���[0��1��
		
		System.out.println(d);
		
		
		int h = (1+ (int)(6 * (Math.random())));
		System.out.println(h);
		if(h <= 3) {
			System.out.println("��������С����");
			
		}else {
			System.out.println("�������Ǵ��������������̽��Ұְ�");
		}
			
		
		System.out.println("###################################################################################");
	
		int i = 1+(int)(6 * (Math.random()));
		int j = 1+(int)(6 * (Math.random()));
		int k = 1+(int)(6 * (int)(Math.random()));
	   int count  = i + j+ k;
	   System.out.println(count);
	   if(count >15);{
	   System.out.println("�����������ֱ��ը");
	   }
	   
	   if(count <=15 && count >=10);{
	   System.out.println("������������е���Ҳ��������");
	   }
	   
	   if(count <10);{
	   System.out.println("������������ը��");
	   }
	
	
	
	}
	
}
