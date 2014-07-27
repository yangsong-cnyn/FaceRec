// RythemDlg.cpp : implementation file
//
#include "stdafx.h"
#pragma  once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h" 
#include "Rythem.h"
#include "RythemDlg.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cassert>
#include "CvvImage.h"
#include "afxdialogex.h"
#include <cv.h>
#include <highgui.h>
#include "CUDA_Transfer.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
using namespace cv;

CvCapture *capture;
CRect      rect;  
CDC       *pDC;  
HDC        hDC;
CWnd      *pwnd;
CRect      face_rect; 
CDC       *face_pDC;  
HDC        face_hDC;
CWnd      *face_pwnd;
IplImage  *gray;
IplImage  *small_img;
IplImage  *iFace;

Mat eigenValues;
Mat eigenVector;
Mat eigenMean;

const char              *cascade_name = "haarcascade_frontalface_alt.xml";
CvHaarClassifierCascade *cascade = NULL;
CvMemStorage            *storage = NULL;
void                     faceDetect(IplImage *img, double scale =3.0);
void                     bwChange(IplImage *frame);

float delta_Qmax;
float *Q;
vector<Mat> images;
vector<int> labels;
PCA *pca;
Mat Mean;

// CAboutDlg dialog used for App About
class CAboutDlg : public CDialogEx
{
public:
    CAboutDlg();

// Dialog Data
    enum { IDD = IDD_ABOUTBOX };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
    DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CRythemDlg dialog

CRythemDlg::CRythemDlg(CWnd* pParent /*=NULL*/)
    : CDialogEx(CRythemDlg::IDD, pParent)
	, imgList(_T(""))
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CRythemDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, imgList);
}

BEGIN_MESSAGE_MAP(CRythemDlg, CDialogEx)
    ON_WM_SYSCOMMAND()
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_TRAIN, &CRythemDlg::OnBnClickedTrain)
	ON_BN_CLICKED(IDC_OPEN, &CRythemDlg::OnBnClickedOpen)
    ON_WM_TIMER()
    ON_WM_CLOSE()
END_MESSAGE_MAP()


// CRythemDlg message handlers

BOOL CRythemDlg::OnInitDialog()
{
    CDialogEx::OnInitDialog();

    // Add "About..." menu item to system menu.

    // IDM_ABOUTBOX must be in the system command range.
    ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
    ASSERT(IDM_ABOUTBOX < 0xF000);

    CMenu* pSysMenu = GetSystemMenu(FALSE);
    if (pSysMenu != NULL)
    {
        BOOL bNameValid;
        CString strAboutMenu;
        bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
        ASSERT(bNameValid);
        if (!strAboutMenu.IsEmpty())
        {
            pSysMenu->AppendMenu(MF_SEPARATOR);
            pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
        }
    }

    // Set the icon for this dialog.  The framework does this automatically
    // when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);         // Set big icon
    SetIcon(m_hIcon, FALSE);        // Set small icon

    // TODO: Add extra initialization here
    pwnd = GetDlgItem(IDC_STATIC);  
    //pwnd->MoveWindow(35,30,352,288);  
    pDC =pwnd->GetDC();  
    //pDC =GetDC();  
    hDC= pDC->GetSafeHdc();  
    pwnd->GetClientRect(&rect);
	////////////////////////////////
	face_pwnd = GetDlgItem(IDC_FACE);  
    //pwnd->MoveWindow(35,30,352,288);  
    face_pDC = face_pwnd->GetDC();  
    //pDC =GetDC();  
    face_hDC = face_pDC->GetSafeHdc();  
    face_pwnd->GetClientRect(&face_rect);

    cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
    assert(cascade != NULL);
    storage = cvCreateMemStorage(0);
    assert(storage != NULL);

	imgList = "lwjface";
	UpdateData(false);
    return TRUE;  // return TRUE  unless you set the focus to a control
}

void CRythemDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
    if ((nID & 0xFFF0) == IDM_ABOUTBOX)
    {
        CAboutDlg dlgAbout;
        dlgAbout.DoModal();
    }
    else
    {
        CDialogEx::OnSysCommand(nID, lParam);
    }
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CRythemDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CDialogEx::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CRythemDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

void CRythemDlg::OnBnClickedOpen()
{
    if(!capture)  
    {  
        capture = cvCreateCameraCapture( 0 );  
    }  
  
    if (!capture)  
    {  
        MessageBox("Fail to open the camera!");  
        return;  
    }  
  
    IplImage* m_Frame;  
    m_Frame=cvQueryFrame(capture);

    gray = cvCreateImage(cvGetSize(m_Frame), IPL_DEPTH_8U, 1);
	cvCvtColor( m_Frame, gray, CV_BGR2GRAY);
	iFace = cvCreateImage(cvGetSize(m_Frame), IPL_DEPTH_8U, 1);
	cvCopy(gray, iFace, 0);

    small_img = cvCreateImage(cvSize(cvRound(m_Frame->width / 3.0), cvRound(m_Frame->height / 3.0)), IPL_DEPTH_8U, 1);

    CvvImage m_CvvImage;  
    m_CvvImage.CopyOf(m_Frame,1);     
    if (true)  
    {  
        m_CvvImage.DrawToHDC(hDC, &rect);   
    }  
   
    SetTimer(1,50,NULL);  
}


void CRythemDlg::OnTimer(UINT_PTR nIDEvent)
{
    CDialogEx::OnTimer(nIDEvent);
    IplImage* m_Frame;  
	float delta_Qmin;
    m_Frame=cvQueryFrame(capture); 
	cvCvtColor( m_Frame, gray, CV_BGR2GRAY);
    faceDetect(m_Frame);
	//bwChange(gray);
	float *q_test = (float *)malloc(sizeof(float)* pca->eigenvectors.cols);
	//run_cuda(iFace, eigenVector, eigenValues, eigenMean);
	run_cuda_Cal_test_pic_Q(iFace, pca->eigenvectors, Mean, q_test);
	delta_Qmin = Q_compare_test(Q, q_test, pca->eigenvectors.rows, images.size());
	if (delta_Qmin > delta_Qmax)
		MessageBox("Recognization failed: This is not the face of database");
	else
		MessageBox("Recognization success: This is the face of database!");
    CvvImage m_CvvImage;  
    m_CvvImage.CopyOf(gray,1);  
	CvvImage m_faceImage;  
    m_faceImage.CopyOf(iFace,1);
    if (true)  
    {  
        m_CvvImage.DrawToHDC(hDC, &rect); 
		m_faceImage.DrawToHDC(face_hDC, &face_rect);
        //cvWaitKey(10);   
    }   
 
    CDialogEx::OnTimer(nIDEvent);  
}

void CRythemDlg::OnClose()
{
    CDialogEx::OnClose();
    cvReleaseCapture(&capture);  
    CDC MemDC;    
    MemDC.CreateCompatibleDC(NULL);  
    pDC->StretchBlt(rect.left,rect.top,rect.Width(),rect.Height(),&MemDC,0,0,48,48,SRCCOPY);  	
    cvReleaseImage(&gray);
	//run_cuda_deinitialization();
	if (iFace) cvReleaseImage(&iFace);
    cvReleaseImage(&small_img);
    cvReleaseMemStorage(&storage);
    cvReleaseHaarClassifierCascade(&cascade);
}

void faceDetect(IplImage *img, double scale)
{
    assert(img != NULL);
    static CvScalar colors[] = {
        {0, 0, 255}, {0, 128, 255}, {0, 255, 255}, {0, 255, 0},
        {255, 128, 0}, {255, 255, 0}, {255, 0, 0}, {255, 0, 255}
    };

    assert(gray != NULL);
    assert(small_img != NULL);

    cvCvtColor( img, gray, CV_BGR2GRAY);
    cvResize(gray, small_img,CV_INTER_LINEAR );
    cvEqualizeHist(small_img, small_img);

    cvClearMemStorage( storage );

    CvSeq* objects = cvHaarDetectObjects(
        small_img,
        cascade,
        storage,
        1.1,
        3,
        0,
        cvSize(50, 50)
    );
    if (objects)
    {
	   if (objects->total>0){
			CvRect* r = (CvRect*)cvGetSeqElem(objects, 0);
			cvRectangle(
				img,
				cvPoint(r->x * scale, r->y * scale),
				cvPoint(r->x * scale + r->width * scale, r->y * scale + r->height * scale),
				colors[2],
				3
			);
			cvReleaseImage(&iFace);
			cvSetImageROI(gray, cvRect(r->x * scale,r->y * scale, r->width * scale, r->height * scale));
			iFace = cvCreateImage(cvSize(r->width * scale, r->height * scale), gray->depth, gray->nChannels);
            cvCopy(gray, iFace, 0);
            cvResetImageROI(gray);
	   }
    }
}

static  Mat norm_0_255(cv::InputArray _src)  
{  
    Mat src = _src.getMat();  
    Mat dst;  
    switch(src.channels())  
    {  
    case 1:  
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);  
        break;  
    case 3:  
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);  
        break;  
    default:  
        src.copyTo(dst);  
        break;  
    } 
    return dst;  
} 

void CRythemDlg::OnBnClickedTrain()
{
	// TODO: 在此添加控件通知处理程序代码
	UpdateData(TRUE);
  /* vector<Mat> images; */ 
  /* vector<int> labels; */
	CString  folderName = imgList + "\\";

	CString  fileList = folderName+"list";

	ifstream iFile;
	int imgNum;
	iFile.open(fileList);
	if (!iFile)
	   {
	     MessageBox("Database folder not found!!!!!");
		 return;
	   }
	iFile>>imgNum;

	char    temp[50];
	int     labeltemp;
	for (int i=0; i<imgNum; i++)
	    {
	      if (iFile>>temp && iFile>>labeltemp){
		   CString filePath = folderName;
		   CString fileName(temp);
		   filePath += fileName;
		   //images.push_back(imread("lwjface\\lwj1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
		   images.push_back(cvLoadImage(filePath, CV_LOAD_IMAGE_GRAYSCALE));
           labels.push_back(labeltemp); 
		  }
	    }

	int totPixel= images[0].rows*images[0].cols;
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:
	Mat Mean = model->getMat("mean");
	Mean = norm_0_255(Mean.reshape(1, images[0].rows));
	Mat dataBase(totPixel, images.size(), CV_32FC1);

	for(int i = 0; i < images.size(); i++)
       {
         Mat col_tmp = dataBase.col(i);
         images[i].reshape(1, totPixel).col(0).convertTo(col_tmp, CV_32FC1, 1/255.);
       }

	pca = new PCA(dataBase, Mat(), CV_PCA_DATA_AS_COL, 10);

	Mat pca_face1 = norm_0_255(pca->eigenvectors.row(0)).reshape(1, images[0].rows);//第一个主成分脸
	//imshow(format("eigenface1"), pca_face1);
	Mat pca_face2 = norm_0_255(pca->eigenvectors.row(1)).reshape(1, images[0].rows);//第一个主成分脸
	//imshow(format("eigenface2"), pca_face2);
	Mat pca_face3 = norm_0_255(pca->eigenvectors.row(2)).reshape(1, images[0].rows);//第一个主成分脸
	//imshow(format("eigenface3"), pca_face3);
	Mat pca_face4 = norm_0_255(pca->eigenvectors.row(3)).reshape(1, images[0].rows);//第一个主成分脸
	//imshow(format("eigenface4"), pca_face4);
	Mat pca_face5 = norm_0_255(pca->eigenvectors.row(4)).reshape(1, images[0].rows);//第一个主成分脸
	//imshow(format("eigenface5"), pca_face5);


	Q = (float *) malloc(sizeof(float)* pca->eigenvectors.cols * images.size());   //109 * 170
	memset(Q, 0, sizeof(float)* pca->eigenvectors.cols * images.size());
	run_cuda_initialization(images[0], pca->eigenvectors, Mean);
	for (int i = 0; i < images.size(); i++)
		run_cuda_preparation(images[i], pca->eigenvectors, Mean, Q);
	delta_Qmax = Q_compare(Q, pca->eigenvectors.rows, images.size());
	MessageBox("Training complete!!!!!!!");
}