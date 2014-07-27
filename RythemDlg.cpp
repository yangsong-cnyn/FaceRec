// RythemDlg.cpp : implementation file
//
#include "stdafx.h"
#pragma  once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

#define WIDTH      256
#define HEIGHT     256

bool       ifTrained=0;

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
IplImage  *theFace;
char       formalFace[7][64] = {"formal\\unknown.jpg","formal\\LWJ.jpg","formal\\SY.jpg","formal\\ZYN.jpg", "formal\\yale1.jpg", "formal\\yale2.jpg", "formal\\yale3.jpg"}; 
int        faceFreq[7] = {0,0,0,0,0,0,0};
int        statNum = 10;

float     *curProj;
int icount =0;

int             imgNum;
vector<int>     labels;
vector<CString> pathList;
Mat             pRecon;
PCA            *decPCA;
float           thresQ;

int             statFrame=0;

const char              *cascade_name = "haarcascade_frontalface_alt.xml";
CvHaarClassifierCascade *cascade = NULL;
CvMemStorage            *storage = NULL;
void                     faceDetect(IplImage *img, double scale =3.0);
void                     bwChange(IplImage *frame);

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
	//;;;;;;;;;;;;;;//
    pwnd = GetDlgItem(IDC_STATIC);   
    pDC =pwnd->GetDC();   
    hDC= pDC->GetSafeHdc();  
    pwnd->GetClientRect(&rect);
	//;;;;;;;;;;;;;;//
	face_pwnd = GetDlgItem(IDC_FACE);  
    face_pDC = face_pwnd->GetDC();   
    face_hDC = face_pDC->GetSafeHdc();  
    face_pwnd->GetClientRect(&face_rect);

    cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
    assert(cascade != NULL);
    storage = cvCreateMemStorage(0);
    assert(storage != NULL);

	imgList = "ourFace";
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
	if (ifTrained == 0)
	   {
	     MessageBox("Training not yet conducted!");
	   }
    if (!capture)  
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
	iFace = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);
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
    m_Frame=cvQueryFrame(capture); 
	cvCvtColor( m_Frame, gray, CV_BGR2GRAY);
    faceDetect( m_Frame);
	//int result = run_cuda(iFace, decPCA, pRecon, threshold);
	/**********************************/
	statFrame = statFrame%statNum;

	if (ifTrained == 1)
	   {
		Mat mat_img(iFace);
		Mat mat_test(1, HEIGHT*WIDTH, CV_32FC1);
		for (int row=0; row<HEIGHT; ++row)
		{
			for (int col=0; col<WIDTH; ++col)
			{
				float f = (float)(mat_img.at<uchar>(row, col));
				mat_test.at<float>(0, (row*(WIDTH) + col) ) = f;
			}
		}
		Mat encoded_test(1, decPCA->eigenvectors.rows, CV_32FC1);
		decPCA->project(mat_test, encoded_test);

		float min_sum = CV_MAX_ALLOC_SIZE;
		int   min_index = 0;
		for (int s=0; s<imgNum; ++s)
		{
			float sum=0;
			for (int e=0; e<decPCA->eigenvectors.rows; ++e)
			{
				float fs = pRecon.at<float>(s,e);
				float fi = encoded_test.at<float>(0,e);
				sum += ((fs-fi)*(fs-fi));
			}
			if(sum < min_sum){
				min_sum = sum;
				min_index = s;
			}
		}
		if (min_sum>thresQ)
		   {
			 faceFreq[0]++;
			 //min_index=0;
			 //theFace = cvLoadImage(formalFace[0]);
		   }
		else
		   {
			 faceFreq[labels[min_index]]++;
			 //theFace = cvLoadImage(formalFace[labels[min_index]]);
		   }
		if (statFrame==0)
		   {
			 int tempFreq = faceFreq[0];
			 int freqLabel = 0;
		     for (int i=0; i<7; i++)
			     {
			       if (faceFreq[i]>tempFreq)
				      {
				        freqLabel=i;
						tempFreq = faceFreq[i];
				      }
				   faceFreq[i] = 0;
			     }
			 theFace = cvLoadImage(formalFace[freqLabel]);
		   }
	   }
	else
	   {
	    theFace = iFace;
	   }
	/************************************/
    CvvImage m_CvvImage;  
    m_CvvImage.CopyOf(m_Frame, 1);  
	CvvImage m_faceImage;  
    m_faceImage.CopyOf(theFace, 1);
    if (true)  
    {  
        m_CvvImage.DrawToHDC(hDC, &rect); 
		m_faceImage.DrawToHDC(face_hDC, &face_rect);
        //cvWaitKey(10);  
    }  
    statFrame++;
	icount ++;
	if (icount == 200)
	{
	  int ydfsdf=90;
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
	cvReleaseImage(&iFace);
    cvReleaseImage(&small_img);
    cvReleaseMemStorage(&storage);
    cvReleaseHaarClassifierCascade(&cascade);
	delete decPCA;
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
				colors[7],
				3
			);
			//cvReleaseImage(&iFace);
			cvSetImageROI(gray, cvRect(r->x * scale,r->y * scale, r->width * scale, r->height * scale));
			IplImage *faceVictim = cvCreateImage(cvSize(r->width * scale, r->height * scale), gray->depth, gray->nChannels);
            cvCopy(gray, faceVictim, 0);
			cvResize( faceVictim, iFace, CV_INTER_LINEAR );  
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
	UpdateData(TRUE);
	if (ifTrained == 1)
	   {
		 MessageBox("You have already trained!");
	     return;
	   }
	vector<Mat> images;
	
	CString  folderName = imgList + "\\";
	CString  fileList   = folderName+"list";

	ifstream iFile;
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
		   pathList.push_back(filePath);
		   images.push_back(cvLoadImage(pathList[i], CV_LOAD_IMAGE_GRAYSCALE));
           labels.push_back(labeltemp); 
		  }
	    }

	Mat dataBase(imgNum, WIDTH*HEIGHT, CV_32FC1);

	for (int s=0; s<imgNum; ++s)
	{
		for (int row=0; row<HEIGHT; ++row)
		{
			for (int col=0; col<WIDTH; ++col)
			{
				float f = (float)(images[s].at<uchar>(row, col));
				dataBase.at<float>(s, (row*WIDTH + col) ) = f;
			}
		}
	}


	PCA *pca = new PCA(dataBase, Mat(), CV_PCA_DATA_AS_ROW);

	int   index;  
	float sum=0, SOP=0, ratio;  
	for(int i=0; i<pca->eigenvalues.rows; i++)  
	   {  
		 sum += pca->eigenvalues.at<float>(i, 0);  
	   }  
	for(int i=0; i<pca->eigenvalues.rows; i++)  
	   {  
		 SOP += pca->eigenvalues.at<float>(i, 0);  
		 ratio = SOP/sum;
		 if(ratio > 0.95)
		   {
			 index = i;  
			 break;  
		   }  
	   }

	Mat evVictim;
	evVictim.create((index+1), WIDTH*HEIGHT, CV_32FC1);//eigen values of decreased dimension  
	for (int i=0; i<=index; i++)  
	{  
		pca->eigenvectors.row(i).copyTo(evVictim.row(i));  
	} 

	decPCA = new PCA();
	decPCA->mean = pca->mean;
	decPCA->eigenvectors = evVictim;

	pRecon.create(imgNum, decPCA->eigenvectors.rows, CV_32FC1);
	for(int s=0; s<imgNum; ++s)
	{
		Mat in = dataBase.row(s);
		Mat out = pRecon.row(s);
		decPCA->project(in, out);
	}

	float maxQ = 0;
	for(int i=0; i<imgNum; i++)
	{
       for(int j=i+1; j<imgNum; j++)
	   {
		 float curQ = 0;
	     for(int k=0; k<decPCA->eigenvectors.rows; k++)
			{
			   float fs = pRecon.at<float>(i,k);
			   float fi = pRecon.at<float>(j,k);
			   curQ += ((fs-fi)*(fs-fi));
			}
		 if (maxQ<curQ)
		    {
		       maxQ = curQ;
		    }
	   }
	}
	thresQ = maxQ/8;
	ifTrained = 1;
	MessageBox("Training complete!!!!!!!");
}