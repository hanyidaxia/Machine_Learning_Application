package pro3;
/**
 * 测试break和continue
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
	System.out.println("为了找到这个几把数，总共用了"+ total +"次循环");
	
	
	
	
	
	
	
	
	
	
	
	}
}
