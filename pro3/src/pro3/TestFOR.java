package pro3;
/**
 * FOR SENTECE AND COMPARISION WITH WHILE 
 * @author hanyi_gm
 *
 */
public class TestFOR {

	public static void main(String[] args) {
		
		int sum  = 0;
		for (int i = 1; i <=100; i ++) {
			sum  = sum+ i;
		}
		 System.out.println(sum);

		
		/**
 * 1. excute the first condition that is i = 1
 * 2. decide if the second condition 
 * 3. excute the last condition that is i++
 * 4.excute the loop
 * 5. go to tghe first senctence
 */
		// for 循环中定义的变量只有在for内部才可以使用，现在外面就不可以使用i这个变量了
		for(;;) {
			
			System.out.println("印度人全是傻逼，偷我的水杯，还他妈踩我的电脑线");
			
		}
	 
	  
	}
}
