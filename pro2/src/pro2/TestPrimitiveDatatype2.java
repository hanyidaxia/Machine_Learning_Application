package pro2;
import java.math.*;


/**
 * 浮点型数据类型的练习
 * @author hanyi_gm
 *
 */
public class TestPrimitiveDatatype2 {

	
	public static void main(String[] args) {
		
		float a  = 3.14F;
		double  b = 6.28;
		
		double  c = 6.28e-2;
		System.out.print(c);
		
		// float num is not exact , definetly not suitble for compare
		float f  = 0.1f;
		double e = 1/10;
		System.out.println(f == e);
		
		float d1 = 53446135434353f;
		float d2 = d1 + 1;
		if (d1 == d2) {
			System.out.println("d1 = d2");
		}else {
			System.out.println("d1 != d2");
		}
		System.out.println("#######################################################################");
		// when need use excat compare float num then use this package
		BigDecimal bd = BigDecimal.valueOf(1.0);
		bd = bd.subtract(BigDecimal.valueOf(0.1));
		bd = bd.subtract(BigDecimal.valueOf(0.1));
		bd = bd.subtract(BigDecimal.valueOf(0.1));
		bd = bd.subtract(BigDecimal.valueOf(0.1));
		bd = bd.subtract(BigDecimal.valueOf(0.1));
		System.out.println(bd);
		System.out.println(1.0-0.1-0.1-0.1-0.1-0.1);
		
		BigDecimal bd2  = BigDecimal.valueOf(0.1);
		BigDecimal bd3  = BigDecimal.valueOf(1.0/10);
		//BigDecimal bd2  = BigDecimal.valueOf(1.0);
		//BigDecimal bd2  = BigDecimal.valueOf(1.0);
		
				System.out.println(bd2.equals(bd3));
		
		
	}
}
