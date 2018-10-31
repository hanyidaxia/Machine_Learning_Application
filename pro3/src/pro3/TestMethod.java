package pro3;

/**
 * 测试方法，非常棒的function
 * @author hanyi_gm
 *
 */
public class TestMethod {

	public static void main(String[] args) {
		//通过对象调用普通方法
		TestMethod tm = new TestMethod();
		tm.printsxt();
		tm.add("韩溢", 5464, 455);
	}
 void printsxt() {
	 System.out.println("韩溢可真他娘的帅");
 }
 		void add(String a, int b, int c) {
 			int sum = b+c;
 			System.out.println(a + "创造了" +sum +"个文明");
 			//return ;
 		}
 		
 		
 		int add(int m, int l, int j) {
 			int sum = l+j;
 			System.out.println(m +sum);
 			return sum;//1.return 返回的值，2.结束方法的运行
 		}

}
