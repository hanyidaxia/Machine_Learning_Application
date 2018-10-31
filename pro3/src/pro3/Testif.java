package pro3;
/**
 * 测试if选择语句
 * @author hanyi_gm
 *
 */
public class Testif {

	public static void main(String[] args) {
		double d = Math.random();//返回的是[0，1）
		
		System.out.println(d);
		
		
		int h = (1+ (int)(6 * (Math.random())));
		System.out.println(h);
		if(h <= 3) {
			System.out.println("看起来是小于三");
			
		}else {
			System.out.println("看起来是大于三，现在立刻叫我爸爸");
		}
			
		
		System.out.println("###################################################################################");
	
		int i = 1+(int)(6 * (Math.random()));
		int j = 1+(int)(6 * (Math.random()));
		int k = 1+(int)(6 * (int)(Math.random()));
	   int count  = i + j+ k;
	   System.out.println(count);
	   if(count >15);{
	   System.out.println("今天的手气简直爆炸");
	   }
	   
	   if(count <=15 && count >=10);{
	   System.out.println("今天的手气还行但是也就那样吧");
	   }
	   
	   if(count <10);{
	   System.out.println("今天的手气差到爆炸了");
	   }
	
	
	
	}
	
}
