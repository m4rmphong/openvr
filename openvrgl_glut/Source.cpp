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

//#include "OpenVRGL.h"

using namespace std;
using namespace cv;

//-------------------------------------------------------------------------------
// Struct
//-------------------------------------------------------------------------------

struct VertexDataScene {
	glm::vec3 position;
	glm::vec2 texCoord;
	VertexDataScene(const glm::vec3& pos, const glm::vec2& tex) : position(pos), texCoord(tex) {}
};

struct VertexDataWindow { // companion window
	glm::vec2 position;
	glm::vec2 texCoord;
};

struct FramebufferDesc {
	GLuint m_nDepthBufferId;
	GLuint m_nRenderTextureId;
	GLuint m_nRenderFramebufferId;
	GLuint m_nResolveTextureId;
	GLuint m_nResolveFramebufferId;
};
FramebufferDesc leftEyeDesc;
FramebufferDesc rightEyeDesc;


//-------------------------------------------------------------------------------
// Variables
//-------------------------------------------------------------------------------
VideoCapture	g_webCam;
Mat				g_camFrame;
unsigned int	g_uUpdateTimeInterval = 10;
std::thread		g_threadLoadFrame;
bool			g_bUpdated = false;
bool			g_bRunning = true;

// openvr variable
//COpenVRGL* g_pOpenVRGL = nullptr;
vr::IVRSystem *m_pHMD;
vr::IVRRenderModels *m_pRenderModels;
vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
glm::mat4x4 m_rmat4DevicePose[vr::k_unMaxTrackedDeviceCount];

float m_fNearClip;
float m_fFarClip;

uint32_t m_nRenderWidth;
uint32_t m_nRenderHeight;

// pose matrix
glm::mat4 m_mat4HMDPose;
glm::mat4 m_mat4eyePosLeft;
glm::mat4 m_mat4eyePosRight;

glm::mat4 m_mat4ProjectionCenter;
glm::mat4 m_mat4ProjectionLeft;
glm::mat4 m_mat4ProjectionRight;

glm::mat4 m_mat4VPLeft;
glm::mat4 m_mat4VPRight;

GLuint m_frameTexture;

int frameWindowVertCount;
GLuint g_frameWindowVAO;
GLuint frameWindowVertexbuffer;
GLuint g_renderFrameWindowProgramID;
GLint m_frameWindowMatrixLocation;

//-------------------------------------------------------------------------------
// Purpose: Functions
//-------------------------------------------------------------------------------

glm::mat4 MVPmatrix() {
	glm::mat4 proj = glm::perspective(glm::radians(45.0f),(float)g_camFrame.cols/(float)g_camFrame.rows,0.1f,30.f);
	glm::mat4 view = glm::lookAt(glm::vec3(2,2,1),glm::vec3(0,0,0),glm::vec3(0,1,0));
	glm::mat4 model = glm::mat4(1.0f);
	glm::mat4 mvp = proj*view*model;
	return mvp;
}

glm::mat4 ConvertMat(const vr::HmdMatrix34_t& mat)
{
	return  glm::mat4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0,
		mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
		mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0,
		mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
	);
}

glm::mat4 ConvertMat(const vr::HmdMatrix44_t& mat)
{
	return  glm::mat4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
		mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
		mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
		mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
	);
}

bool CreateFrameBuffer(int nWidth, int nHeight, FramebufferDesc& framebufferDesc) {
	glGenFramebuffers(1, &framebufferDesc.m_nRenderFramebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nRenderFramebufferId);

	glGenRenderbuffers(1, &framebufferDesc.m_nDepthBufferId);
	glBindRenderbuffer(GL_RENDERBUFFER, framebufferDesc.m_nDepthBufferId);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, nWidth, nHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, framebufferDesc.m_nDepthBufferId);

	glGenTextures(1, &framebufferDesc.m_nRenderTextureId);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, nWidth, nHeight, true);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId, 0);

	glGenFramebuffers(1, &framebufferDesc.m_nResolveFramebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nResolveFramebufferId);

	glGenTextures(1, &framebufferDesc.m_nResolveTextureId);
	glBindTexture(GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nWidth, nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId, 0);

	// check FBO status
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		return false;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}

//-------------------------------------------------------------------------------
// Purpose: Setup
//-------------------------------------------------------------------------------
void SetupScene() {
	static const GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, -5.0f,  0.0f,  1.0f, // B
		 0.5f, -0.5f, -5.0f,  1.0f,  1.0f, // A
		 0.5f,  0.5f, -5.0f,  1.0f,  0.0f, // D
		 0.5f,  0.5f, -5.0f,  1.0f,  0.0f, // D
		-0.5f,  0.5f, -5.0f,  0.0f,  0.0f, // C
		-0.5f, -0.5f, -5.0f,  0.0f,  1.0f, // B
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

void SetupCamera() {
	if (!m_pHMD) return;
	m_mat4ProjectionLeft = ConvertMat(m_pHMD->GetProjectionMatrix(vr::Eye_Left, m_fNearClip, m_fFarClip));
	m_mat4ProjectionRight = ConvertMat(m_pHMD->GetProjectionMatrix(vr::Eye_Right, m_fNearClip, m_fFarClip));
	m_mat4eyePosLeft = ConvertMat(m_pHMD->GetEyeToHeadTransform(vr::Eye_Left));
	m_mat4eyePosRight = ConvertMat(m_pHMD->GetEyeToHeadTransform(vr::Eye_Right));
	// ViewProjection Matrix VP = Projection*View(position respect to origin)
	// MVP = VP*m_mat4HMDPose(current hmd pose); update HMD pose only
	m_mat4VPLeft = m_mat4ProjectionLeft*m_mat4eyePosLeft;
	m_mat4VPRight = m_mat4ProjectionRight*m_mat4eyePosRight;
}

void SetupStereoRenderTargets() {
	// setup frame buffer for the views of two eyes
	// get the proper width and height
	m_pHMD->GetRecommendedRenderTargetSize(&m_nRenderWidth, &m_nRenderHeight);
	// use the width and height information to create frame buffer
	CreateFrameBuffer(m_nRenderWidth, m_nRenderHeight, leftEyeDesc/*Struct to strore the framebuffer information*/); // send struct in
	CreateFrameBuffer(m_nRenderWidth, m_nRenderHeight, rightEyeDesc/*Struct to strore the framebuffer information*/);
}

void SetupCompanionWindow() {

}

glm::mat4x4 GetCurrentMVP(vr::Hmd_Eye eyeIdx) {
	glm::mat4x4 matMVP;
	if (eyeIdx == vr::Eye_Left) {
		matMVP = m_mat4VPLeft*m_mat4HMDPose;
	}
	else {
		matMVP = m_mat4VPRight*m_mat4HMDPose;
	}
	return matMVP;
}

void RenderScene() {
	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
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
	//glm::mat4 mvpMatrix = GetCurrentMVP(nEye); 
	glm::mat4 mvpMatrix = MVPmatrix();
	glUseProgram(g_renderFrameWindowProgramID);
	glUniformMatrix4fv(m_frameWindowMatrixLocation, 1, GL_FALSE, &mvpMatrix[0][0]);
	glBindVertexArray(g_frameWindowVAO);
	glBindTexture(GL_TEXTURE_2D, m_frameTexture);
	glDrawArrays(GL_TRIANGLES, 0, frameWindowVertCount); // Starting from vertex 0; 3 vertices total -> 1 triangle
	glBindVertexArray(0);
}

void RenderSceneOnEye(vr::Hmd_Eye nEye) {
	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
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
	glm::mat4 mvpMatrix = GetCurrentMVP(nEye); //glm::mat4 mvpMatrix = MVPmatrix();
	glUseProgram(g_renderFrameWindowProgramID);
	glUniformMatrix4fv(m_frameWindowMatrixLocation, 1, GL_FALSE, &mvpMatrix[0][0]);
	glBindVertexArray(g_frameWindowVAO);
	glBindTexture(GL_TEXTURE_2D, m_frameTexture);
	glDrawArrays(GL_TRIANGLES, 0, frameWindowVertCount); // Starting from vertex 0; 3 vertices total -> 1 triangle
	glBindVertexArray(0);
}

void RenderStereoTargets()
{
	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
	glEnable(GL_MULTISAMPLE);

	// Left Eye
	glBindFramebuffer(GL_FRAMEBUFFER, leftEyeDesc.m_nRenderFramebufferId);
	glViewport(0, 0, m_nRenderWidth, m_nRenderHeight);
	RenderSceneOnEye(vr::Eye_Left);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_MULTISAMPLE);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, leftEyeDesc.m_nRenderFramebufferId);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, leftEyeDesc.m_nResolveFramebufferId);

	glBlitFramebuffer(0, 0, m_nRenderWidth, m_nRenderHeight, 0, 0, m_nRenderWidth, m_nRenderHeight,
		GL_COLOR_BUFFER_BIT,
		GL_LINEAR);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glEnable(GL_MULTISAMPLE);

	// Right Eye
	glBindFramebuffer(GL_FRAMEBUFFER, rightEyeDesc.m_nRenderFramebufferId);
	glViewport(0, 0, m_nRenderWidth, m_nRenderHeight);
	RenderSceneOnEye(vr::Eye_Right);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_MULTISAMPLE);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, rightEyeDesc.m_nRenderFramebufferId);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, rightEyeDesc.m_nResolveFramebufferId);

	glBlitFramebuffer(0, 0, m_nRenderWidth, m_nRenderHeight, 0, 0, m_nRenderWidth, m_nRenderHeight,
		GL_COLOR_BUFFER_BIT,
		GL_LINEAR);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void onExit()
{
	//g_pOpenVRGL->Release();
	//delete g_pOpenVRGL;

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
		"uniform mat4 matrix;\n"
		"layout(location = 0) in vec4 position;\n"
		"layout(location = 1) in vec2 v2UVcoordsIn;\n"
		"//layout(location = 2) in vec3 v3NormalIn;\n"
		"out vec2 v2UVcoords;\n"
		"void main()\n"
		"{\n"
		"	v2UVcoords = v2UVcoordsIn;\n"
		"	gl_Position = matrix * position;\n"
		"	//gl_Position = position;\n"
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
	m_frameWindowMatrixLocation = glGetUniformLocation(g_renderFrameWindowProgramID, "matrix");
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

	GLfloat fLargest;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);

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
// Purpose: Initialize
//-------------------------------------------------------------------------------
bool InitVRGL() {
	// loading the  steamvr 
	vr::EVRInitError eError = vr::VRInitError_None; // error message
	m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);
	if (eError != vr::VRInitError_None) { // error happens
		m_pHMD = NULL;
		return false;
	}
	m_pRenderModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &eError);
	if (!m_pRenderModels)
	{
		m_pHMD = NULL;
		vr::VR_Shutdown();
		return false;
	}

	// glew initialization
	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK)
	{
		printf("%s - Error initializing GLEW! %s\n", __FUNCTION__, glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	// gl initialization
	// opengl-related parameter
	m_fNearClip = 0.1f;
	m_fFarClip = 30.0f;
	m_frameTexture = 0;
	frameWindowVertCount = 0;
	// setup list: texture, scene, camera, companion window( on PC monitor), device models.
	SetupTexture();
	SetupScene();
	SetupCamera();
	SetupStereoRenderTargets();
	//SetupCompanionWindow();
	//SetupRenderModels();	

	// compositor initialization
	vr::EVRInitError peError = vr::VRInitError_None;
	if (!vr::VRCompositor())
	{
		printf("Compositor initialization failed. See log file for details\n");
		return false;
	}

	return true;
}

void UpdateHMDPose() {
	if (!m_pHMD)
		return;

	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		if (m_rTrackedDevicePose[nDevice].bPoseIsValid)
		{
			m_rmat4DevicePose[nDevice] = ConvertMat(m_rTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
		}
	}

	vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
	if (m_rTrackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
	{
		m_mat4HMDPose = m_rmat4DevicePose[vr::k_unTrackedDeviceIndex_Hmd];
	}
}

//-------------------------------------------------------------------------------
// Purpose: glut callback function
//-------------------------------------------------------------------------------
void Display() {
	static std::chrono::high_resolution_clock::time_point tpLast = std::chrono::high_resolution_clock::now(), tpNow;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	RenderStereoTargets();
	//RenderScene();

	vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)leftEyeDesc.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
	vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
	vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightEyeDesc.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
	vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);

	glutSwapBuffers();
	
	tpNow = std::chrono::high_resolution_clock::now();
	//std::cout << "FPS :" << 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(tpNow - tpLast).count() << "\n";
	tpLast = tpNow;

	UpdateHMDPose();
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
	//g_pOpenVRGL = new COpenVRGL();
	//g_pOpenVRGL->Initial(0.1f, 30.f);

	if (!InitVRGL()) {
		return -1;
	}

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