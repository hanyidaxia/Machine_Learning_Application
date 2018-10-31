package pro3;
/**
 * 多重else结构
 * @author hanyi_gm
 *
 */
public class TestManyIfelse {

	public static void main(String[] args) {
		int age = (int)(100 * Math.random());
		System.out.println("年龄是" + age + "应该");
		if(age < 15) {
			System.out.println("小屁孩子，多挨揍");
		}else if(age < 25) {
			System.out.println("成年人了，多交配");
		}else if(age < 35) {
			System.out.println("成年人了，多交配");
		}else if(age < 45) {
			System.out.println("成年人了，多交配");
		}else if(age < 55) {
			System.out.println("成年人了，多交配");
		}else if(age < 65) {
			System.out.println("成年人了，多交配");
		}else if(age < 75) {
			System.out.println("成年人了，多交配");
		}else if(age < 85) {
			System.out.println("老年人了，多交配");
		}else  {
			System.out.println("得了吧，能交配多久就交配多久，反正是赚到了");
		}
	}
}
