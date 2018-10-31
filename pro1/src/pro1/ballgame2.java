package pro1;

import java.awt.*;
import javax.swing.*;



public class ballgame2 extends JFrame {
	Image ball = Toolkit.getDefaultToolkit().getImage("image/ball.png");
	Image board = Toolkit.getDefaultToolkit().getImage("image/board.jpg");
	
	
	double x = 100;// the x axis of the ball
	double y = 100;// the y axis of the ball

	double degree  = 3.14/3; // start degree is 60
	// draw the window
	public void paint(Graphics g) {
		System.out.println("I draw a picture once");
		g.drawImage(board, 0, 0, null);
		g.drawImage(ball, (int)x, (int)y, null);
		
	x = x + 10*Math.cos(degree);
	y = y + 10*Math.sin(degree);
	
	
	if( y > 400 - 40 || y< 30) {
		degree = -degree;
		
	}
	if( x > 600 - 20 || x< 20) {
		degree = 3.14 -degree;
		
	}
	
	}
			
		
		
		
	// load the window
	void launchFrame() {
		setSize(600,400);
		setLocation(50,50);
		setVisible(true);
		
		// keep redraw the window
		while(true) {
			repaint();
			try {
				Thread.sleep(50);//stop per 50 ms, 1s = 1000 ms
			}catch(Exception e){
			 e.printStackTrace();	
			}
			
		
			
		}
	}
	public static void main(String[] args) {
		System.out.println("I am all your guys father");
		ballgame2 game  = new ballgame2();
		game.launchFrame();
	}
}
