# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:51:58 2020

@author: peter Chungungo app
"""
from tkinter import scrolledtext
import webview
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, simpledialog, messagebox
import tkinter as tk
import pandas as pd
import numpy as np
import base64
from io import StringIO
from datetime import date
import webbrowser
import os.path
from os import remove

from tkintertable import TableCanvas as Table
import copy
from splash_chungungo import splash_screen, destroy_splash
import time
import tempfile
import os

splash_screen()

os.environ['PATH'] = 'R/bin/x64' + os.pathsep + os.environ['PATH']
print(os.environ['PATH'])
os.environ['R_HOME'] = 'R/'

# R modules needs environ set-------------------------------------------

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2 import robjects


# variables
version = '1.0 Beta'
buf = StringIO()
test = tempfile.TemporaryFile(mode='w+t')
today = date.today()
# Textual month, day and year
d2 = today.strftime("%B %d, %Y")
# seteo dataframe_original Open dialog --- df_correct Respuestas-dialog
NoneType = type(None)
dataframe_original = 0
menudisabled = "disabled"
filedir = '/'

# Start write html

test.write('<h1> Análisis de Psicométrico </h1>' + '<br>')


# -----------------------------------------------------------------------------------------------------------------
# FUNCTIONS
def open_web():
    with open('preview.html', 'w') as f:
        test.seek(0)
        for x in test.readlines():
            f.write(x)
        f.close()
    webview.create_window('Reporte', 'preview.html')
    webview.start(gui='mshtml')
    return


# Abrir archivo
def open_datatable():
    global dataframe_table
    app.exportTable()
    dataframe_table = pd.read_csv('tempCSV', encoding="Latin-1", dtype=str, index_col=[0])
    if os.path.exists('tempCSV'):
        remove('tempCSV')
    return


def open_file():
    if type(dataframe_original) == int:
        open_file2()
    else:
        remplace = messagebox.askyesno(message="Desea abrir otro archivo", title="Mensaje")
        if remplace:
            test.seek(0)
            test.truncate()
            test.write('<style>'
                       + 'body{font-family: \'Trebuchet MS\', Arial, Helvetica, sans-serif;'
                       + '  border-collapse: collapse;' + '  width: 100%;' + '}'
                       + 'h1 {font-family: \'Trebuchet MS\', Arial, Helvetica, sans-serif;}'
                       + ' td,  th {' + '  border: 1px solid #ddd;' + '  padding: 8px;' + '}'
                       + ' tr:nth-child(even){background-color: #f2f2f2;}' + ' tr:hover {background-color: #ddd;}'
                       + ' th {' + '  padding-top: 12px;' + '  padding-bottom: 12px;' + '  text-align: left;'
                       + '  background-color: #4CAF50;' + '  color: white;' + '}'
                       + '</style>' + d2 + '<br>' + '<h1>Analisis de Excel Google Forms</h1>' + '<br>')
            open_file2()
    return


def open_file2():
    global filedir
    filename_excel = filedialog.askopenfilename(initialdir=filedir, title="Abrir Archivo",
                                                filetypes=(("excel files", "*.xls"), ("excel files", "*.xlsx")))
    filedir = os.path.dirname(filename_excel)
    if filename_excel:
        global dataframe_original
        dataframe_original = pd.read_excel(filename_excel, dtype=str, index_col=[0])
        print(dataframe_original)
        # copy Excel  #fix missing data
        dataframe_original = dataframe_original.replace(np.nan, "NaN")
        buffer = dataframe_original.to_dict('index')
        # Global app permite que la tabla actualice la data correctamente
        global app
        app = MyTable(frame, data=buffer)
        app.redrawTable()
        # Activa menu
        global menudisabled
        menudisabled = "normal"
        # menubar.entryconfig("Edit", state= menudisabled)
        menubar.entryconfig("Análisis", state=menudisabled)
        root.title('Chungungo' + ' - ' + os.path.basename(filename_excel))
    return


# Guardar tabla
def save_excel():
    report_file = filedialog.asksaveasfilename(initialdir=filedir, title="Guardar excel",
                                               filetypes=[("Archivo excel", "*.xlsx")], defaultextension='.xlsx')
    if report_file:
        open_datatable()
        dataframe_table.to_excel(report_file, index=False, header=False)
    return


# Guardar reporte
def save_html():
    report_file = filedialog.asksaveasfilename(initialdir=filedir, title="Guardar Reporte",
                                               filetypes=[("Archivo html", "*.html")], defaultextension='.html')
    if report_file:
        with open(report_file, 'w') as f:
            test.seek(0)
            for x in test.readlines():
                f.write(x)
                print(test.read())
            f.close()
        webbrowser.open_new_tab(report_file)
    return


# Plots frecuency for item----------------------------------------------------------------------------------------
def dificultad_tct():
    def disablebutton():
        i = 0
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                i = i + 1
                if i >= 4:
                    apply_button.configure(state=NORMAL)
                else:
                    apply_button.configure(state='disabled')
        return

    def difficult_function():
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        dataframe_diff = dataframe_table[column_selected].applymap(int)

        writer = robjects.r['write.csv']
        with localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.conversion.py2rpy(dataframe_diff)
        print(scores)
        data = ro.r('data.matrix')
        item_analysis = importr('ShinyItemAnalysis')
        as_null = ro.r['as.null']
        table = item_analysis.ItemAnalysis(data(scores), y=as_null(), k=4, l=1, u=4, add_bin=FALSE)
        writer(table, 'filetempo')
        df_diff = pd.read_csv('filetempo')
        df_diff.rename(columns={'Unnamed: 0': 'Item', 'avgScore': 'Dificultad', 'SD': 'Desv. por Item',
                                'ULI': 'Indice de Discr.', 'RIR': 'Coef. Discr.'}, inplace=True)
        print(df_diff[['Item', 'Dificultad', 'Desv. por Item', 'Indice de Discr.', 'Coef. Discr.']])
        buf.truncate(0)
        df_diff[['Item', 'Dificultad', 'Desv. por Item', 'Indice de Discr.', 'Coef. Discr.']].to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Dificultad y Discriminación<h2>' + text + '<hr>')
        messagebox.showinfo(master=difficult_window, message="Diff y Disc Listo", title="Mensaje")
        open_web()
        return

    open_datatable()
    difficult_window = tk.Toplevel()
    ClassToplevel(difficult_window)
    difficult_window.geometry('500x280')
    difficult_window.title('Dificultad y Discriminación')
    apply_button = tk.Button(difficult_window, text="Aplicar", command=difficult_function, state='disabled')
    tk.Label(difficult_window, text='Dificultad, indice de Discriminación \n y coef. Discriminación',
             font='Arial 11 bold').grid(row=0, column=1, columnspan=4)
    info_label = tk.Label(difficult_window, wraplength=270, justify=LEFT,
                          text='Grado de dificultad por item, indice de discriminación y coeficiente de discriminación')
    info_label.grid(row=1, column=1, columnspan=4, rowspan=3)
    apply_button.grid(row=6, column=8, sticky=W)
    checklist = scrolledtext.ScrolledText(difficult_window, height=10, width=20, cursor='arrow', background='white')
    checklist.grid(row=0, column=8, rowspan=4)
    vars = []
    for column in dataframe_table:
        var = tk.IntVar()
        vars.append([var, column])
        checkbutton = tk.Checkbutton(checklist, text=column, variable=var, command=disablebutton, background='white')
        checklist.window_create("end", window=checkbutton)
        checklist.insert("end", "\n")
    checklist.configure(state="disabled")
    difficult_window.mainloop()
    return


# Funciones de analisis------------------------------------------------------------------
def irt_rasch():
    def disablebutton():
        i = 0
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                i = i + 1
                if i >= 3:
                    apply_button.configure(state=NORMAL)
                else:
                    apply_button.configure(state='disabled')
        return

    def rasch_function():
        analysis_rasch_window.config(cursor='wait')
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        dataframe_rasch = dataframe_table[column_selected].copy()
        ltm = importr('ltm')
        sia = importr('ShinyItemAnalysis')
        writer = robjects.r['write.csv']
        grdevices = importr('grDevices')
        with localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.conversion.py2rpy(dataframe_rasch)
        model = ltm.rasch(scores, IRT_param=True)
        coef = robjects.r['coef']
        plot = robjects.r['plot']
        coeff = coef(model, prob=True, order=False)
        grdevices.png('file.png', width=512, height=512)
        plot(model, type='ICC')
        grdevices.dev_off()
        encoded = base64.b64encode(open("file.png", "rb").read()).decode('utf-8')

        fscores = ltm.factor_scores(model, resp_patterns=scores)
        theta = fscores[0].rx(True, 'z1')
        b = coeff.rx(True, 'Dffclt')
        print(theta, b)
        wright = sia.ggWrightMap(theta, b)
        grdevices.png('file2.png', width=512, height=512)
        print(wright)
        grdevices.dev_off()
        encoded2 = base64.b64encode(open("file2.png", "rb").read()).decode('utf-8')
        godfit = ltm.GoF_rasch(model)
        p_value = np.array(godfit.rx2('p.value'))
        print(p_value[0])
        list_gof = {'p valor-Bondad de ajuste modelo': [p_value[0]]}
        df_gof = pd.DataFrame(list_gof, columns=['p valor-Bondad de ajuste modelo'])
        writer(coeff, 'filetempo')
        df_rasch = pd.read_csv('filetempo')
        print(df_rasch)
        df_rasch.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
        buf.truncate(0)
        df_rasch.to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>ITR-RASCH<h2>' + text + '<hr>')
        test.write('<img src=\'data:image/png;base64,{}\'>'.format(encoded))
        test.write('<img src=\'data:image/png;base64,{}\'>'.format(encoded2))
        remove('file.png')
        remove('file2.png')
        remove('filetempo')
        buf.truncate(0)
        df_gof.to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + text + '<hr>')
        analysis_rasch_window.config(cursor='')
        messagebox.showinfo(master=analysis_rasch_window, message="Rasch Listo", title="Mensaje")
        open_web()
        return

    def irt_function():
        analysis_rasch_window.config(cursor='wait')
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        dataframe_rasch = dataframe_table[column_selected].copy()
        ltm = importr('ltm')
        writer = robjects.r['write.csv']
        grdevices = importr('grDevices')
        with localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.conversion.py2rpy(dataframe_rasch)
        print(scores)
        fmla = Formula('scores ~ z1')
        env = fmla.environment
        env['scores'] = scores
        model = ltm.ltm(fmla, IRT_param=True)
        coef = robjects.r['coef']
        plot_ltm = robjects.r['plot']
        grdevices.png('file.png', width=512, height=512)
        plot_ltm(model, type='ICC')
        grdevices.dev_off()
        encoded = base64.b64encode(open("file.png", "rb").read()).decode('utf-8')
        coeff = coef(model, prob=True, order=False)
        writer(coeff, 'filetempo')
        df_irt = pd.read_csv('filetempo')
        print(df_irt)
        df_irt.rename(columns={'Unnamed: 0': 'Item'}, inplace=True)
        buf.truncate(0)
        df_irt.to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>ITR-2PL<h2>' + text + '<hr>')
        test.write('<img src=\'data:image/png;base64,{}\'>'.format(encoded))
        remove('file.png')
        remove('filetempo')
        analysis_rasch_window.config(cursor='')
        messagebox.showinfo(message="IRT-2PL Listo", title="Mensaje")
        open_web()
        return

    open_datatable()
    analysis_rasch_window = tk.Toplevel()
    ClassToplevel(analysis_rasch_window)
    analysis_rasch_window.geometry('550x300')
    analysis_rasch_window.title('Análisis IRT')
    apply_button = tk.Button(analysis_rasch_window, state='disabled', text="Aplicar",
                             command=lambda: irt_function() if check_pl.get() else rasch_function())
    tk.Label(analysis_rasch_window, text='Análisis Teoria Respuesta al Item', font='Arial 11 bold').grid(row=0,
                                                                                                         column=1,
                                                                                                         columnspan=4)
    info_label = tk.Label(analysis_rasch_window, wraplength=270, justify=LEFT,
                          text='Test IRT modelo Rasch de 1 - 2PL, al aplicar este test obtienes parámetro'
                               ' Dificultad por item y en el caso de 2PL Discriminación, además de la probabilidad de '
                               'responder correctamente dado Rasgo latente equilibrado. Se adjunta curva '
                               'característica de item y mapa persona item.\n \n Seleccione entre 2PL y '
                               'un 1PL')
    info_label.grid(row=1, column=1, columnspan=4, rowspan=3)
    apply_button.grid(row=6, column=8, sticky=W)
    checklist = scrolledtext.ScrolledText(analysis_rasch_window, height=10, width=20, cursor='arrow',
                                          background='white')
    checklist.grid(row=0, column=8, rowspan=4)
    vars = []
    for column in dataframe_table:
        var = tk.IntVar()
        vars.append([var, column])
        checkbutton = tk.Checkbutton(checklist, text=column, variable=var, command=disablebutton, background='white')
        checklist.window_create("end", window=checkbutton)
        checklist.insert("end", "\n")
    check_pl = tk.IntVar()
    checkbutton_pl = tk.Checkbutton(analysis_rasch_window, text='IRT-2PL (por defecto 1PL)', variable=check_pl)
    checkbutton_pl.grid(row=6, column=1, columnspan=4, sticky=W)
    checklist.configure(state="disabled")
    analysis_rasch_window.mainloop()
    return


# Alpha Cronbach
def alpha_cronbach():
    def disablebutton():
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                apply_button.configure(state=NORMAL)
                return
            else:
                apply_button.configure(state='disabled')
        return

    def function_alpha():
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        dataframe_alpha = dataframe_table[column_selected].copy()
        with localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.conversion.py2rpy(dataframe_alpha)
        ltm = importr('ltm')
        sapply = ro.r('sapply')
        data = ro.r('as.matrix')
        as_numeric = ro.r('as.numeric')
        writer = robjects.r['write.csv']
        datito = sapply(scores, as_numeric)
        print('convertido a matrix', data(datito))
        item_analysis = importr('ShinyItemAnalysis')
        as_null = ro.r['as.null']
        table = item_analysis.ItemAnalysis(data(datito), y=as_null(), k=4, l=1, u=4, add_bin=FALSE)
        print(table)
        alpha_table = ltm.cronbach_alpha(scores)
        print(alpha_table)
        alpha = np.array(alpha_table.rx2('alpha'))
        items = alpha_table.rx2('p')
        n = alpha_table.rx2('n')
        print(round(alpha[0], 3), items, n)
        list_alphac = {'Coeficiente Alpha Cronbach': [round(alpha[0], 5)],
                       'N de item': [items[0]],
                       'N de casos': [n[0]]}
        df_alphac = pd.DataFrame(list_alphac, columns=['Coeficiente Alpha Cronbach', 'N de item', 'N de casos'])
        # export df to html
        if 0.6 <= alpha[0] < 0.65:
            interpretation = 'Alpha Cronbach se considera indeseable, se recomienda revisar instrumento.'
        elif 0.65 <= alpha[0] < 0.7:
            interpretation = 'Alpha Cronbach es minimamente aceptable. No es suficiente para tomar decisiones y menos' \
                             ' aun las que influyan en el futuro de las personas como por ejemplo Test de Admisión'
        elif 0.7 <= alpha[0] < 0.8:
            interpretation = 'Alpha Cronbach es suficientemente bueno para cualquier investigación. Deseable en la' \
                             ' mayoria de los casos (Ej: Test de habilidades)'
        elif 0.8 <= alpha[0] < 0.9:
            interpretation = 'Alpha Cronbach se considera muy buena.'
        elif 0.9 <= alpha[0]:
            interpretation = 'Alpha Cronbach es un nivel elevado de confiabilidad.'
        else:
            interpretation = 'Alpha Cronbach se considera inaceptable, se recomienda revisar instrumento.'
        buf.truncate(0)
        df_alphac.to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Test de Confiabilidad Alpha Cronbach<h2>' + text + '<br><h3>Interpretación:<h3>'
                   + interpretation + '<hr>')
        writer(table, 'filetempo')
        alpha_drop = pd.read_csv('filetempo', index_col=[0])
        buf.truncate(0)
        alpha_drop[['alphaDrop']].to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Alpha Cronbach eliminando Item<h2>' + text + '<hr>')
        messagebox.showinfo(message="Alpha Cronbach listo", title="Mensaje")
        open_web()
        return

    open_datatable()
    alpha_window = tk.Toplevel()
    ClassToplevel(alpha_window)
    alpha_window.geometry('500x280')
    alpha_window.title('Análisis')
    apply_button = tk.Button(alpha_window, text="Aplicar", command=function_alpha)
    tk.Label(alpha_window, text='Alpha Cronbach', font='Arial 11 bold').grid(row=0,
                                                                             column=1,
                                                                             columnspan=4)
    info_label = tk.Label(alpha_window, wraplength=270, justify=LEFT,
                          text='Permite calcular el Alpha Cronbach de su instrumento.'
                               ' (Para ver su resultado e interpretación debe'
                               ' "Guardar Reporte" en el menú Archivo).\n Selecciona las columnas de los item a '
                               'analizar los datos deben ser de tipo dicotómico, politómico o continuo. (El ingresar'
                               ' tablas con datos no aceptados arrojará error).')
    info_label.grid(row=1, column=1, columnspan=4, rowspan=3)
    apply_button.grid(row=6, column=8, sticky=W)
    apply_button.configure(state=DISABLED)
    checklist = scrolledtext.ScrolledText(alpha_window, height=10, width=20, cursor='arrow', background='white')
    checklist.grid(row=0, column=8, rowspan=4)
    vars = []
    for column in dataframe_table:
        var = tk.IntVar()
        vars.append([var, column])
        checkbutton = tk.Checkbutton(checklist, text=column, variable=var, command=disablebutton, background='white')
        checklist.window_create("end", window=checkbutton)
        checklist.insert("end", "\n")
    checklist.configure(state="disabled")
    alpha_window.mainloop()


# Splithalf RULON BROWN------------------------------------------------

def analysis_splithalf():
    def disablebutton():
        i = 0
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                i = i + 1
                if i >= 2:
                    apply_button.configure(state=NORMAL)
                else:
                    apply_button.configure(state='disabled')
        return

    def splithalf_function():
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        i = 0
        k1 = 0
        k2 = 0
        score = 0
        pair = 0
        impair = 0
        for column in column_selected:
            i = i + 1
            score = score + dataframe_table[column].apply(int)
            if i % 2 == 0:
                pair = pair + dataframe_table[column].apply(int)
                k1 = k1 + 1
            else:
                impair = impair + dataframe_table[column].apply(int)
                k2 = k2 + 1
        pair = pair.astype(float)
        impair = impair.astype(float)
        dif_halfsplit = pair - impair
        corr = pair.corr(impair, method='pearson')
        spearman = corr * (pow(corr ** 2 + 4 * (k1 * k2 / (i * i)) * (1 - corr ** 2), 0.5) - corr) / (
                2 * (k1 * k2 / (i ** 2)) * (1 - corr ** 2))
        print(spearman)
        rulon = 1 - (dif_halfsplit.var(ddof=0) / score.var(ddof=0))
        scale = {'Conf. Spearman-Brown': [spearman], 'Conf. Rulón': [rulon]}
        df_scale = pd.DataFrame(scale, columns=['Conf. Spearman-Brown', 'Conf. Rulón'])
        buf.truncate(0)
        df_scale.to_html(buf)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Método de dos partes<h2>' + text + '<hr>')
        messagebox.showinfo(message="Metodo de dos partes listo", title="Mensaje")
        open_web()
        return

    open_datatable()
    global vars
    window_splithalf = tk.Toplevel()
    ClassToplevel(window_splithalf)
    window_splithalf.geometry('500x250')
    window_splithalf.title('Análisis dos mitades')
    apply_button = tk.Button(window_splithalf, text="Aplicar", command=splithalf_function, state="disabled")
    info_label = tk.Label(window_splithalf, wraplength=270, justify=LEFT,
                          text='Permite calcular la fiabilidad del instrumento por método de dos mitades (La división '
                               'se hace por pares e impares).\n(Para ver su resultado debe guardar '
                               'reporte en el Menú Archivo)')

    tk.Label(window_splithalf, text='Análisis de dos mitades', font='Arial 11 bold').grid(row=0, column=1,
                                                                                          columnspan=4)
    info_label.grid(row=1, column=1, columnspan=4, rowspan=3)
    apply_button.grid(row=6, column=8, sticky=W)
    checklist = scrolledtext.ScrolledText(window_splithalf, height=10, width=20, cursor='arrow', background='white')
    checklist.grid(row=0, column=8, rowspan=4)
    vars = []
    for column in dataframe_table:
        var = tk.IntVar()
        vars.append([var, column])
        checkbutton = tk.Checkbutton(checklist, text=column, variable=var, command=disablebutton, background='white')
        checklist.window_create("end", window=checkbutton)
        checklist.insert("end", "\n")
    checklist.configure(state="disabled")
    window_splithalf.mainloop()


# Transformar Datos ---------------------------------------------------------------

def transform_data():
    def disablebutton():
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                print('activado', vars[row][0].get)
                check_column.configure(state=NORMAL)
                assign_but.configure(state=NORMAL)
                return
            else:
                print('desactivado')
                check_column.configure(state='disabled')
                assign_but.configure(state='disabled')
        return

    def cuali_replace():
        def apply_trans():
            print(array_valores)
            global app
            if var_replace.get() == 1:
                for row in range(len(array_valores)):
                    if array_valores[row][1].get():
                        dataframe_table[select_column] = dataframe_table[select_column].replace([array_valores[row][0]],
                                                                                                array_valores[row][
                                                                                                    1].get())
                    else:
                        dataframe_table[select_column] = dataframe_table[select_column].replace(array_valores[row][0],
                                                                                                array_valores[len(
                                                                                                    array_valores) - 1][
                                                                                                    1].get())
                print(dataframe_table[select_column])
                buffer = dataframe_table.to_dict('index')
                app = MyTable(frame, data=buffer)
                app.redrawTable()
            if var_replace.get() == 0:
                for row in range(len(array_valores)):
                    if array_valores[row][1].get():
                        dataframe_table[select_column] = dataframe_table[select_column].replace([array_valores[row][0]],
                                                                                                array_valores[row][
                                                                                                    1].get())
                    else:
                        dataframe_table[select_column] = dataframe_table[select_column].replace(array_valores[row][0],
                                                                                                array_valores[len(
                                                                                                    array_valores) - 1][
                                                                                                    1].get())
                temporal = dataframe_table[select_column].copy()
                print('temporal', temporal)
                open_datatable()
                for column in select_column:
                    for c in range(len(list(dataframe_table))):
                        if not 'copy_%d(%s)' % (c, column) in dataframe_table:
                            dataframe_table['copy_%d(%s)' % (c, column)] = temporal[column].values
                            break
                print(dataframe_table)
                buffer = dataframe_table.to_dict('index')
                app = MyTable(frame, data=buffer)
                app.redrawTable()
            tk.messagebox.showinfo(title='Transformar', message='Datos transformados')

        cuali_window = tk.Toplevel()
        ClassToplevel(cuali_window)
        input_window.title('Reemplazar valores')
        input_window.geometry('550x300')
        select_column = []
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                select_column.append(vars[row][1])
        unique_values = (pd.unique(dataframe_table[select_column].values.ravel())).tolist()
        print(unique_values)
        unique_values.sort()
        unique_values.append('else')
        array_valores = []
        i = 0
        for values in unique_values:
            variable_remplazada = StringVar()
            tk.Label(cuali_window, text=values).grid(row=i, column=0, columnspan=4, sticky=W)
            tk.Entry(cuali_window, textvariable=variable_remplazada).grid(row=i, column=5, columnspan=4, sticky=W)
            array_valores.append([values, variable_remplazada])
            i = i + 1
        button_apply = tk.Button(cuali_window, text="aplicar",
                                 command=lambda: apply_trans() if array_valores[len(array_valores) - 1][
                                     1].get() else tk.messagebox.showinfo(title='Información',
                                                                          message='Ingrese valor a else'))
        button_apply.grid()
        button_apply.configure(DISABLED)
        cuali_window.mainloop()
        return

    open_datatable()
    input_window = tk.Toplevel()
    ClassToplevel(input_window)
    input_window.title('Transformar Data')
    input_window.geometry('500x300')
    info_label = tk.Label(input_window, wraplength=270, justify=LEFT,
                          text='Permite transformar los datos de las columnas seleccionadas. Selecciona las columnas'
                               ' en el cuadro blanco y elige si deseas reemplazar las columnas originales por las '
                               'transformadas. Por defecto se agregará una nueva columna de nombre Copy"numero"'
                               '(nombre de la columna).')

    tk.Label(input_window, text='Transformar datos', font='Arial 11 bold').grid(row=0, column=1, columnspan=4)
    info_label.grid(row=1, column=1, columnspan=4, rowspan=3)
    var_replace = IntVar()
    checklist = scrolledtext.ScrolledText(input_window, height=10, width=20, cursor="arrow")
    checklist.grid(row=0, column=8, rowspan=4, columnspan=4, sticky=W)
    vars = []
    for column in dataframe_table:
        var = tk.IntVar()
        vars.append([var, column])
        checkbutton = tk.Checkbutton(checklist, text=column, variable=var, command=disablebutton, background='white')
        checklist.window_create("end", window=checkbutton)
        checklist.insert("end", "\n")
    check_column = tk.Checkbutton(input_window, text="Reemplazar columnas", variable=var_replace)
    check_column.grid(row=6, column=1, columnspan=4, sticky=W)
    check_column.configure(state="disabled")
    checklist.configure(state="disabled")
    assign_but = tk.Button(input_window, text="Asignar valores", command=cuali_replace)
    assign_but.grid(row=6, column=8, sticky=W)
    assign_but.configure(state="disabled")
    input_window.mainloop()
    return


# --------------------------------------------------------------------------------------------------
def about_chungungo():
    about_window = tk.Toplevel()
    ClassToplevel(about_window)
    about_window.overrideredirect(True)
    about_window.configure(bg='#ebe8e3')
    # Gets both half the screen width/height and window width/height
    positionright = int(about_window.winfo_screenwidth() / 2 - 400 / 2)
    positiondown = int(about_window.winfo_screenheight() / 2 - 250 / 2)
    about_window.geometry('400x250')
    about_window.geometry("+{}+{}".format(positionright, positiondown))
    img = PhotoImage(
        data='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')
    tk.Label(about_window, bg='#ebe8e3', image=img).grid(row=0, column=0, columnspan=4, sticky=W)
    s = 'Chungungo es un programa que permite a profesores que diseñen instrumentos realizar' \
        ' analisis psicométrico desde una Hoja de calculo. Entregando información' \
        ' estadística, con su respectiva interpretación.'
    tk.Label(about_window, bg='#ebe8e3', text='Acerca de Chungungo', font='Arial 11 bold').grid(row=1, column=1)
    tk.Label(about_window, bg='#ebe8e3', text=f'Versión: {version}', font='Arial 9').grid(row=1, column=3)
    tk.Label(about_window, bg='#ebe8e3', text=s, font='Arial 9', justify=LEFT,
             wraplength=350).grid(row=2, column=0, columnspan=4, rowspan=3)
    tk.Button(about_window, bg='#ebe8e3', text="Aceptar", command=about_window.destroy).grid(row=6, column=3)
    about_window.mainloop()


# TKINTER Y TKINTERTABLE--------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

class MyTable(Table):
    """  Custom table class inherits from Table.This overrides the right click menu. """

    def __init__(self, parent=None, **kwargs):
        Table.__init__(self, parent, rows=30, cols=27, rowheight=20, **kwargs)
        self.cellbackgr = 'white'
        self.show()
        self.rows = self.model.getRowCount()
        self.cols = self.model.getColumnCount()
        return

    def copyCell(self, rows, cols=None):
        row = rows[0]
        col = cols[0]
        self.clipboard = copy.deepcopy(self.model.getCellRecord(row, col))
        return

    def cutCell(self, rows, cols=None):
        row = rows[0]
        col = cols[0]
        self.clipboard = copy.deepcopy(self.model.getCellRecord(row, col))
        self.model.setValueAt('', row, col)
        self.redrawTable()
        return

    def pasteCell(self, rows, cols=None):
        row = rows[0]
        col = cols[0]
        val = self.clipboard
        self.model.setValueAt(val, row, col)
        self.redrawTable()
        return

    def findValue(self, searchstring=None, findagain=None):
        if searchstring == None:
            searchstring = simpledialog.askstring("Search table.",
                                                  "Enter search value",
                                                  parent=self.parentframe)
        found = 0
        if findagain == None or not hasattr(self, 'foundlist'):
            self.foundlist = []
        if self.model != None:
            for row in range(self.rows):
                for col in range(self.cols):
                    text = str(self.model.getValueAt(row, col))
                    if text == '' or text == None:
                        continue
                    cell = row, col
                    if findagain == 1 and cell in self.foundlist:
                        continue
                    if text.lower().find(searchstring.lower()) != -1:
                        print('found in', row, col)
                        found = 1
                        # highlight cell
                        self.delete('searchrect')
                        self.drawRect(row, col, color='red', tag='searchrect', delete=0)
                        self.lift('searchrect')
                        self.lift('celltext' + str(col) + '_' + str(row))
                        # add row/col to foundlist
                        self.foundlist.append(cell)
                        # need to scroll to centre the cell here..
                        x, y = self.getCanvasPos(row, col)
                        self.xview('moveto', x)
                        self.yview('moveto', y)
                        self.tablecolheader.xview('moveto', x)
                        self.tablerowheader.yview('moveto', y)
                        return row, col
        if found == 0:
            self.delete('searchrect')
            print('nothing found')
            return None

    def popupMenu(self, event, rows=None, cols=None, outside=None):
        popupmenu = Menu(self, tearoff=0)
        popupmenu.add_command(label="Cortar", command=lambda: self.cutCell(rows, cols))
        popupmenu.add_command(label="Copiar", command=lambda: self.copyCell(rows, cols))
        popupmenu.add_command(label="Pegar", command=lambda: self.pasteCell(rows, cols))
        popupmenu.add_command(label="Buscar", command=lambda: self.findValue())

        def popupFocusOut(event):
            popupmenu.unpost()

        popupmenu.bind("<FocusOut>", popupFocusOut)
        popupmenu.focus_set()
        popupmenu.post(event.x_root, event.y_root)
        return popupmenu


# override close chungungo--------------------------------------------------------------------------

def closechungungo():
    msg_box1 = tk.messagebox.askquestion(title='Salir', message='Cerrara la ventana de la aplicación '
                                                                '¿Desea continuar?', icon='warning')
    if msg_box1 == 'yes':
        msg_box2 = tk.messagebox.askquestion(title='Guardar', message='Al cerrar la ventana perderá los Datos '
                                                                      '¿Desea guardar reporte antes de salir?',
                                             icon='question')
        if msg_box2 == 'yes':
            save_html()
        print('bye')
        root.destroy()
    return


# ---------------------------------------------------------


class ClassToplevel:
    def __init__(self, master):
        self.master = master
        self.master.resizable(0, 0)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(5, weight=1)
        self.master.grid_rowconfigure(5, weight=1)
        self.master.grid_rowconfigure(7, weight=1)
        self.master.focus()
        self.master.grab_set()

# ----------------------------------------------------------------------------------------------------------------
# creating tkinter window and define property--------------------------------------------------------------------
destroy_splash()
root = tk.Tk()
root.iconphoto(False, tk.PhotoImage(
    data='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'))
frame = Frame(root)
root.geometry('700x400+200+100')
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
frame.grid(sticky=tk.N + tk.S + tk.E + tk.W)
root.title('Chungungo')
# var
path_answers = tk.StringVar()
# Reemplacing Close window
root.protocol('WM_DELETE_WINDOW', closechungungo)  # root is your root window
# Creating Menubar
menubar = Menu(root)
# Adding File Menu and commands
file = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Archivo', menu=file)
file.add_command(label='Abrir...', command=open_file)
file.add_command(label='Guardar Excel', command=save_excel)
file.add_command(label='Guardar Reporte', command=save_html)
file.add_separator()
file.add_command(label='Salir de Chungungo', command=closechungungo)

# Adding Edit Menu and commands 
# edit = Menu(menubar, tearoff = 0)
# menubar.add_cascade(label ='Edit', menu = edit)
# edit.add_command(label ='Cut', command = self.copyCell(rows,cols))
# edit.add_command(label ='Copy', command = self.copyCell(rows,cols))
# edit.add_command(label ='Paste', command = self.pasteCell(rows,cols))
# edit.add_command(label ='Select All', command = None)
# edit.add_separator()
# edit.add_command(label ='Find...', command = None)
# edit.add_command(label ='Find again', command = None)

# Adding analysis Menu and commands 
analysis = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Análisis', menu=analysis)
analysis.add_command(label='Dificultad y Discriminación', command=dificultad_tct)
analysis.add_command(label='Test de Fiabilidad', command=alpha_cronbach)
analysis.add_command(label='Metodo de dos partes', command=analysis_splithalf)
analysis.add_command(label='IRT-RASCH..', command=irt_rasch)
analysis.add_command(label='Transformar Datos', command=transform_data)

# Adding Help Menu 
help_ = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Ayuda', menu=help_)
help_.add_command(label='Chuchungo Ayuda', command=None)
help_.add_separator()
help_.add_command(label='Acerca de...', command=about_chungungo)

# display Menu 
root.config(menu=menubar)

# menubar.entryconfig("Edit", state= menudisabled)
menubar.entryconfig("Análisis", state=menudisabled)

# -----------------------------------------------------------------------------------------------------------------
# Widget Table
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)
app = MyTable(frame)
root.mainloop()
