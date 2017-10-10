#include <gl/glew.h>
#include <gl/freeglut.h>
#include <gl/GLU.h>

#include <stdio.h>
#include <string>

#include <openvr.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

//-------------------------------------------------------------------------------
// Variables
//-------------------------------------------------------------------------------
VideoCapture g_webCam;
Mat g_camFrame;


//-------------------------------------------------------------------------------
// Purpose: Functions
//-------------------------------------------------------------------------------


//-------------------------------------------------------------------------------
// Purpose: Texture
//-------------------------------------------------------------------------------
bool SetupTexture() {
	return true;
}


//-------------------------------------------------------------------------------
// Purpose: glut callback function
//-------------------------------------------------------------------------------
void Display() {

}

void Keyboard(unsigned char key, int x, int y) {

}

//-------------------------------------------------------------------------------
// Purpose: Main Function
//-------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	// webCam setup
	VideoCapture webCam;
	if (!webCam.open(1)) {
		cout << "Cannot open the camera." << endl;
		return -1;
	}
	// opengl setup
	// glut setup
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	// glut window
	glutInitWindowSize(640, 320);
	glutCreateWindow("Openvr + WebCamera");
	// glew initialize
	glewInit();
	// glut callback function registration
	glutDisplayFunc(Display);
	glutKeyboardFunc(Keyboard);
	
	// openvr setup

	// main loop
	glutMainLoop();

	// shutdown

	return 0;
}