package pro3;
/**
 * 测试switch语句
 * @author hanyi_gm
 *
 */
public class TestSwitch {

	public static void main(String[] args) {
		
		int month = 1 +(int)(12*Math.random());
		System.out.println("月份"+ month);
		
		
		switch(month) {
		case 1:
			System.out.println("一月份了，过新年了");
			break;
		case 2:
			System.out.println("二月份了，二月二龙抬头");
			break;
		default:
			System.out.println("我是其他月份");
			
			
			System.out.println("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
	//遇到多值判断的时候希望使用switach		
			
			char c = 'a';
			int rand =  (int)(26 * (Math.random()));
			char c1 = (char)(c + rand);
			System.out.println(c1 + ":");
			switch(c1) {
			case 'a':
			case 'e':
			case 'i':
			case 'o':
			case 'u':
				System.out.println("this is the support section");
				break;
			case 'y':
			case 'w':
				System.out.println("half support section");
				break;
			default:
				System.out.println("those are the support section");
				
			
			
			}
			
			
		}
	}
}
