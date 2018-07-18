# -*- encoding: utf-8 -*-
import os
from win32com import client
from config import REPORT_DOC_PATH, REPORT_PDF_PATH


def word2pdf(filename):

    pdf_name = filename.split('.doc')[0] + '.pdf'

    if not os.path.isabs(filename):
        os.chdir(REPORT_DOC_PATH)
        filename = os.path.abspath(filename)

    if not os.path.isabs(pdf_name):
        os.chdir(REPORT_PDF_PATH)
        pdf_name = os.path.abspath(pdf_name)

    print(filename, pdf_name)
    # 2007 需要用 gencache
    word = client.DispatchEx("Word.Application")
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = word.Documents.Open(filename, ReadOnly=1)
    doc.SaveAs(pdf_name, FileFormat=17)
    doc.Close()
    return pdf_name


if __name__ == '__main__':
    # 只能是绝对路径

    for _, _, filenames in os.walk(REPORT_DOC_PATH):
        for file in filenames:
            word2pdf(file)
