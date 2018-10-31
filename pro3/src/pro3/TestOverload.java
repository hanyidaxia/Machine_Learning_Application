package pro3;
/**
 * 测试重新加载方法
 * @author hanyi_gm
 *
 */
public class TestOverload {

  public static void main(String[] args) {
	  System.out.println(add(5,6));
	  System.out.println(add(5,6,7));
	  System.out.println(add(61.56,5,6));
	  
	  
      }
  
	  public static int add (int n1, int  n2) {
		  int sum = n1 + n2;
		  return sum;
	  
	  }
	  public static int add (int n1, int  n2,int n3) {
		  int sum = n1 + n2 + n3;
		  return sum;
	  
	  }
	  public static double add (double n1, int  n2, int n3) {
		  double sum = n1 + n2+ n3 ;
		  return sum;
	  
	  }
	  
	  
	  
	  
	  
	  
}
