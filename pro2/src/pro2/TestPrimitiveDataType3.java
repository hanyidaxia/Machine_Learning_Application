package pro2;


/**
 * test the 字符类型和布尔类型
 * @author hanyi_gm
 *
 */
public class TestPrimitiveDataType3 {

	
	public static void main(String[] args) {
		
		char a  = 'T';
		char b = '韩';
		char c = '\u0061';
		System.out.println(c);
			
			System.out.println("" + 'a' +'\n'+ 'b');
			System.out.println("" + 'a' +'\t'+ 'b');
			System.out.println("" + 'a' + '\''  +  'b');
			
			//char d  = "asd"; // it's wrong
			// String 就是 a list of numbers
			String d  = "asdfaf";
			
			
			//test boolean
			boolean man = true;
			
			
			if(man ) { //及其不推荐： man == true， 因为经常会写错，导致判断语句成为赋值语句
				System.out.println("男性");
			}else {
				System.out.println("女性");
			}
	}
}
