
// RythemDlg.h : header file
//

#pragma once


// CRythemDlg dialog
class CRythemDlg : public CDialogEx
{
// Construction
public:
	CRythemDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_RYTHEM_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOpen();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnClose();
	CString imgList;
	afx_msg void OnBnClickedTrain();
};
