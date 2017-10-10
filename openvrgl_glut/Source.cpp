#include <gl/glew.h>
#include <gl/freeglut.h>
#include <gl/GLU.h>

#include <stdio.h>
#include <string>
#include <thread>

#include <openvr.h>
#include <opencv2\opencv.hpp>

#include <glm/matrix.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "OpenVRGL.h"

using namespace std;
using namespace cv;

//-------------------------------------------------------------------------------
// Variables
//-------------------------------------------------------------------------------
VideoCapture	g_webCam;
Mat				g_camFrame;
unsigned int	g_uUpdateTimeInterval = 10;
std::thread		g_threadLoadFrame;
bool			g_bUpdated = false;
bool			g_bRunning = true;

COpenVRGL* g_pOpenVRGL = nullptr;


//-------------------------------------------------------------------------------
// Purpose: Functions
//-------------------------------------------------------------------------------
void onExit()
{
	g_pOpenVRGL->Release();
	delete g_pOpenVRGL;

	//g_bRunning = false;
	//g_threadLoadFrame.join();
}



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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);



	glutSwapBuffers();
	imshow("WebCamera", g_camFrame);
}

void timer(int iVal)
{
	glutPostRedisplay();

	if (g_bUpdated)
	{
		//g_Ball.UpdateTexture(g_imgFrame);
		imshow("WebCamera", g_camFrame);
		g_bUpdated = false;
	}

	/*auto* pController = g_pOpenVRGL->GetController(vr::TrackedControllerRole_Invalid);
	if (pController != nullptr)
	{
		if (pController->m_eState.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Touchpad))
		{
			glm::mat4 matHMD = g_pOpenVRGL->GetHMDPose();
			glm::vec3 vecSide = COpenVRGL::GetCameraSide(matHMD);
			glm::vec3 vecUp = COpenVRGL::GetCameraUpper(matHMD);

			glm::vec2 vTP(pController->m_eState.rAxis[0].x, pController->m_eState.rAxis[0].y);

			g_matBall = glm::rotate(0.05f * vTP.x, vecUp) * glm::rotate(-0.05f * vTP.y, vecSide) * g_matBall;
		}
		if (pController->m_eState.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_Grip))
		{
			g_matBall = glm::mat4();
		}
	}*/

	glutTimerFunc(g_uUpdateTimeInterval, timer, iVal);
}
void Keyboard(unsigned char key, int x, int y) {

}

//-------------------------------------------------------------------------------
// Purpose: Main Function
//-------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
	// webCam setup
	if (!g_webCam.open(1)) {
		cout << "Cannot open the camera." << endl;
		return -1;
	}
	g_webCam >> g_camFrame;
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
	glutTimerFunc(g_uUpdateTimeInterval, timer, 1);
	glutKeyboardFunc(Keyboard);
	
	// openvr setup
	g_pOpenVRGL = new COpenVRGL();
	g_pOpenVRGL->Initial(0.1f, 30.f);

	g_threadLoadFrame = std::thread([]() {
		while (g_bRunning)
		{
			static std::chrono::system_clock::time_point tpNow, tpLUpdate = std::chrono::system_clock::now();
			tpNow = std::chrono::system_clock::now();
			if ((tpNow - tpLUpdate) >= std::chrono::milliseconds(40))
			{
				g_webCam >> g_camFrame;
				g_bUpdated = true;
				tpLUpdate = tpNow;
				
			}
		}
	});

	std::atexit(onExit);
	glutMainLoop();

	// shutdown

	return 0;
}