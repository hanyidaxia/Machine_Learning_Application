package pro3;
/**
 * 
 * 测试continue语句
 * @author hanyi_gm
 *continue 用于结束本次循环，开始下一步，yeah
 */
public class TestContinue {

	public static void main(String[] args) {
		// 100到150之间不能被3整除的数找出来，并且每行5个数
		int count = 0;
		for(int i = 100; i<=150; i++) {
			if(i%3 == 0) {
				continue;
			}
			System.out.print("这个数可以被他妈的三整除" + i);
			count++;
			if(count%5 == 0) {
				System.out.println();
			}
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	}
}
