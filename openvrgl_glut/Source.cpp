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

GLuint m_frameTexture;

int frameWindowVertCount;
GLuint g_frameWindowVAO;
GLuint frameWindowVertexbuffer;
GLuint g_renderFrameWindowProgramID;

//-------------------------------------------------------------------------------
// Purpose: Functions
//-------------------------------------------------------------------------------
void SetupScene() {
	static const GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 0.0f,  0.0f,  1.0f, // B
		 0.5f, -0.5f, 0.0f,  1.0f,  1.0f, // A
		 0.5f,  0.5f, 0.0f,  1.0f,  0.0f, // D
		 0.5f,  0.5f, 0.0f,  1.0f,  0.0f, // D
		-0.5f,  0.5f, 0.0f,  0.0f,  0.0f, // C
		-0.5f, -0.5f, 0.0f,  0.0f,  1.0f, // B
	};
	frameWindowVertCount = 6;

	glGenVertexArrays(1, &g_frameWindowVAO);
	glBindVertexArray(g_frameWindowVAO);
	// Generate 1 buffer, put the resulting identifier in vertexbuffer
	glGenBuffers(1, &frameWindowVertexbuffer);
	// The following commands will talk about our 'vertexbuffer' buffer
	glBindBuffer(GL_ARRAY_BUFFER, frameWindowVertexbuffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// 1rst attribute buffer : vertices
	GLsizei stride = sizeof(GLfloat) * 5;
	uintptr_t offset = 0;
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		stride,                  // stride
		(const void*)offset            // array buffer offset
	);
	// 2nd attribute buffer : texture coordinates
	offset += sizeof(GLfloat) * 3;
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		2,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		stride,                  // stride
		(const void*)offset            // array buffer offset
	);

	glBindVertexArray(0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

void RenderScene() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// 1rst attribute buffer : vertices
	//glEnableVertexAttribArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, frameWindowVertexbuffer);
	//glVertexAttribPointer(
	//	0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
	//	3,                  // size
	//	GL_FLOAT,           // type
	//	GL_FALSE,           // normalized?
	//	0,                  // stride
	//	(void*)0            // array buffer offset
	//);
	// Draw the triangle !
	glUseProgram(g_renderFrameWindowProgramID);
	glBindVertexArray(g_frameWindowVAO);
	glBindTexture(GL_TEXTURE_2D, m_frameTexture);
	glDrawArrays(GL_TRIANGLES, 0, frameWindowVertCount); // Starting from vertex 0; 3 vertices total -> 1 triangle
	glBindVertexArray(0);
}

void onExit()
{
	g_pOpenVRGL->Release();
	delete g_pOpenVRGL;

	//g_bRunning = false;
	//g_threadLoadFrame.join();
}

//-------------------------------------------------------------------------------
// Purpose: Compile Shader
//-------------------------------------------------------------------------------
GLuint CompileShader(const char *pchShaderName, const char *pchVertexShader, const char *pchFragmentShader)
{
	GLuint unProgramID = glCreateProgram();

	GLuint nSceneVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(nSceneVertexShader, 1, &pchVertexShader, NULL);
	glCompileShader(nSceneVertexShader);

	GLint vShaderCompiled = GL_FALSE;
	glGetShaderiv(nSceneVertexShader, GL_COMPILE_STATUS, &vShaderCompiled);
	if (vShaderCompiled != GL_TRUE)
	{
		printf("%s - Unable to compile vertex shader %d!\n", pchShaderName, nSceneVertexShader);
		glDeleteProgram(unProgramID);
		glDeleteShader(nSceneVertexShader);
		return 0;
	}
	glAttachShader(unProgramID, nSceneVertexShader);
	glDeleteShader(nSceneVertexShader); // the program hangs onto this once it's attached

	GLuint  nSceneFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(nSceneFragmentShader, 1, &pchFragmentShader, NULL);
	glCompileShader(nSceneFragmentShader);

	GLint fShaderCompiled = GL_FALSE;
	glGetShaderiv(nSceneFragmentShader, GL_COMPILE_STATUS, &fShaderCompiled);
	if (fShaderCompiled != GL_TRUE)
	{
		printf("%s - Unable to compile fragment shader %d!\n", pchShaderName, nSceneFragmentShader);
		glDeleteProgram(unProgramID);
		glDeleteShader(nSceneFragmentShader);
		return 0;
	}

	glAttachShader(unProgramID, nSceneFragmentShader);
	glDeleteShader(nSceneFragmentShader); // the program hangs onto this once it's attached

	glLinkProgram(unProgramID);

	GLint programSuccess = GL_TRUE;
	glGetProgramiv(unProgramID, GL_LINK_STATUS, &programSuccess);
	if (programSuccess != GL_TRUE)
	{
		printf("%s - Error linking program %d!\n", pchShaderName, unProgramID);
		glDeleteProgram(unProgramID);
		return 0;
	}

	glUseProgram(unProgramID);
	glUseProgram(0);

	return unProgramID;
}

//-------------------------------------------------------------------------------
// Purpose: Create Shader
//-------------------------------------------------------------------------------
bool CreateAllShaders() {
	g_renderFrameWindowProgramID = CompileShader(
		"FrameWindow",

		// Vertex Shader
		"#version 410\n"
		"//uniform mat4 matrix;\n"
		"layout(location = 0) in vec4 position;\n"
		"layout(location = 1) in vec2 v2UVcoordsIn;\n"
		"//layout(location = 2) in vec3 v3NormalIn;\n"
		"out vec2 v2UVcoords;\n"
		"void main()\n"
		"{\n"
		"	v2UVcoords = v2UVcoordsIn;\n"
		"	//gl_Position = matrix * position;\n"
		"	gl_Position = position;\n"
		"}\n",

		// Fragment Shader
		"#version 410 core\n"
		"uniform sampler2D mytexture;\n"
		"in vec2 v2UVcoords;\n"
		"out vec4 outputColor;\n"
		"void main()\n"
		"{\n"
		"   outputColor = texture(mytexture, v2UVcoords);\n"
		"}\n"
	);
	cout << "Shader Created!" << endl;
	return g_renderFrameWindowProgramID != 0;
}

//-------------------------------------------------------------------------------
// Purpose: Texture
//-------------------------------------------------------------------------------
bool SetupTexture() {
	if (g_camFrame.data == NULL)
		return false;

	glGenTextures(1, &m_frameTexture);
	glBindTexture(GL_TEXTURE_2D, m_frameTexture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_camFrame.cols, g_camFrame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, g_camFrame.data);

	//glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	//GLfloat fLargest;
	//glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);

	glBindTexture(GL_TEXTURE_2D, 0);

	std::cout << "Texture loaded: " << m_frameTexture << std::endl;

	return (m_frameTexture != 0);
}

void UpdateFrameTexture() {
	glBindTexture(GL_TEXTURE_2D, m_frameTexture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_camFrame.cols, g_camFrame.rows, GL_BGR, GL_UNSIGNED_BYTE, g_camFrame.data);
	//imshow("WebCam", g_camFrame);
}

//-------------------------------------------------------------------------------
// Purpose: glut callback function
//-------------------------------------------------------------------------------
void Display() {
	static std::chrono::high_resolution_clock::time_point tpLast = std::chrono::high_resolution_clock::now(), tpNow;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	RenderScene();

	glutSwapBuffers();
	
	tpNow = std::chrono::high_resolution_clock::now();
	//std::cout << "FPS :" << 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(tpNow - tpLast).count() << "\n";
	tpLast = tpNow;
}

void timer(int iVal)
{
	glutPostRedisplay();

	if (g_bUpdated)
	{
		UpdateFrameTexture();//g_Ball.UpdateTexture(g_imgFrame);
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
	glutInitWindowSize(g_camFrame.cols, g_camFrame.rows);
	glutCreateWindow("Openvr + WebCamera");
	// glew initialize
	glewInit();
	// glut callback function registration
	glutDisplayFunc(Display);
	glutTimerFunc(g_uUpdateTimeInterval, timer, 1);
	glutKeyboardFunc(Keyboard);
	
	// scene setup
	SetupScene();
	SetupTexture();
	CreateAllShaders();

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