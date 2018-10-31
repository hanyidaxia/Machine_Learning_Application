package pro1;

import java.awt.*;
import javax.swing.*;



public class ballgame extends JFrame {
	Image ball = Toolkit.getDefaultToolkit().getImage("image/ball.png");
	Image board = Toolkit.getDefaultToolkit().getImage("image/board.jpg");
	
	
	double x = 100;// the x axis of the ball
	double y = 100;// the y axis of the ball
	boolean right = true;//the direction is head to right
	
	// draw the window
	public void paint(Graphics g) {
		System.out.println("I draw a picture once");
		g.drawImage(board, 0, 0, null);
		g.drawImage(ball, (int)x, (int)y, null);
		
		
		if (right){
			x = x + 10;
		}else {
			x=  x - 10;
		}
		
		if (x >600-50){// 50 is the diameter of the ball 
		right = false;
	
		}
		
		if(x < 20){
			right = true;
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
		ballgame game  = new ballgame();
		game.launchFrame();
	}
}
