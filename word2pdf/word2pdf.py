# -*- encoding: utf-8 -*-
import os
from win32com import client


def doc2pdf(doc_name, pdf_name):
    """
    :word文件转pdf
    :param doc_name word文件名称
    :param pdf_name 转换后pdf文件名称
    """
    try:
        word = client.DispatchEx("Word.Application")
        if os.path.exists(pdf_name):
            os.remove(pdf_name)
        worddoc = word.Documents.Open(doc_name, ReadOnly=1)
        worddoc.SaveAs(pdf_name, FileFormat=17)
        worddoc.Close()
        return pdf_name
    except:
        return 1


if __name__ == '__main__':
    # 只能是绝对路径
    doc_name = r"C:\pythonApp\word2pdf\1.docx"
    ftp_name = r"C:\pythonApp\word2pdf\1.pdf"
    doc2pdf(doc_name, ftp_name)