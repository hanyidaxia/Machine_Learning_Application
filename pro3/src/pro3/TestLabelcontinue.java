package pro3;
/**
 * 测试带标签的break和goto
 * @author hanyi_gm
 *
 */
public class TestLabelcontinue {
	public static void main(String[] args) {
		outer: for(int i = 100; i<=150; i++) {
			for(int j= 2; j < i/2; j++) {
				if(i %j ==0) {
					continue outer;
				}
				System.out.println(i + "	");
				
			}
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
	}
}
