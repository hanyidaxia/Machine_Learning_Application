package pro3;
/**
 * ����break��continue
 * @author hanyi_gm
 *
 */
public class TestBreak {

	public static void main(String[] args) {
		int total = 0;
		System.out.println("begin");
	    
		while(true) {
		total ++;
		int i =(int) Math.round(100 * Math.random());
		System.out.println(i);
	    if (i == 56) {
	    	break;
	    }
		}
	System.out.println("Ϊ���ҵ�������������ܹ�����"+ total +"��ѭ��");
	
	
	
	
	
	
	
	
	
	
	
	}
}
