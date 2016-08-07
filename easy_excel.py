#!/usr/bin/env python
# encoding:utf-8
import xlwt

def set_style(name='Times New Roman', bold=False):
    style = xlwt.XFStyle()
    # font
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    style.font = font
    # alignment
    alignment = xlwt.Alignment()
    alignment.horz = xlwt.Alignment.HORZ_LEFT
    alignment.vert = xlwt.Alignment.VERT_CENTER
    style.alignment = alignment
    return style

def save(results):
    try:
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('results')
        # To generate the first line
        row0 = [u'特征集', u'样本个数', u'分类器', u'Acc', u'Precision', u'Recall', u'SE', u'SP',
                u'Gm', u'F_measure', u'F_score', u'MCC', u'混淆矩阵', u'tp', u'fn', u'fp', u'tn']
        for i in range(2, 4):
            ws.col(i).width = 3333 * 2
        for i in range(0, len(row0)):
            ws.write(0, i+1, row0[i], set_style(bold=True))
        # Write results
        ws.write_merge(1, len(results), 1, 1, u'188D', set_style(bold=True))
        end = len(results[0])
        line = u'正：'+str(results[0][end-2])+u' 反：'+str(results[0][end-1])
        ws.write_merge(1, len(results), 2, 2, line, set_style(bold=True))
        for i in range(0, len(results)):
            for j in range(0, end-2):
                ws.write(i+1, j+3, results[i][j], set_style())
        wb.save('results.xls')
        return True
    except Exception, e:
        return False
