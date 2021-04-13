# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:51:58 2020
    Item stats tools for psychometrics analysis.
    Created November 2020
    Copyright (C) Peter Gonzalez

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import webview
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, simpledialog, messagebox, scrolledtext
import tkinter as tk
import pandas as pd
import numpy as np
import base64
from io import StringIO
from datetime import date
import webbrowser
import os.path
from os import remove
from math import ceil
from tkintertable import TableCanvas as Table
import copy
from splash_chungungo import splash_screen, destroy_splash
from CSS import css_report
from text_functions import text_descriptions
import time
import tempfile
import os
from wrightmap import wright
import configparser

splash_screen()

os.environ['PATH'] = 'R/bin/x64' + os.pathsep + os.environ['PATH']
os.environ['R_HOME'] = 'R/'
config = configparser.ConfigParser()
config.read('FILE_CONFIG.INI')
try:
    print(config['DEFAULT']['path'])
except:
    config['DEFAULT']['path']='/'
    pass

# R modules needs environ set-------------------------------------------
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2 import robjects

# variables##############################
version = '1.0 Beta'
buf = StringIO()
test = tempfile.TemporaryFile(mode='w+t')
today = date.today()
# Textual month, day and year
d2 = today.strftime("%B %d, %Y")
# seteo dataframe_original Open dialog --- df_correct Respuestas-dialog
NoneType = type(None)
dataframe_original = 0
closecount = 0
menudisabled = "disabled"
pd.options.display.html.border = 0

style, script = css_report()

# Start write html

test.write(style + '<page size="A4"> <h3 align="right">' + d2 + '</h3>' + '<h1 align="center"> Análisis Psicométrico </h1>' + '<br>')


# -----------------------------------------------------------------------------------------------------------------
# FUNCTIONS
def open_web():
    global closecount
    closecount = 1
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
    dataframe_table = pd.read_csv('tempCSV', encoding="Latin-1", dtype=str)
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
            test.write(style + '<page size="A4"> <h3 align="right">' + d2 + '</h3>' + '<h1 align="center"> Análisis Psicométrico </h1>' + '<br>')
            open_file2()
    return


def open_file2():
    filename_excel = filedialog.askopenfilename(initialdir=config['DEFAULT']['path'], title="Abrir Archivo",
                                                filetypes=(("excel files", "*.xls"), ("excel files", "*.xlsx")))
    config['DEFAULT']['path'] = os.path.dirname(filename_excel)
    with open('FILE_CONFIG.ini', 'w') as configfile:
        config.write(configfile)
    if filename_excel:
        global dataframe_original
        dataframe_original = pd.read_excel(filename_excel, dtype=str)
        dataframe_original = dataframe_original.replace(np.nan, "NaN") # copy Excel  #fix missing data
        buffer = dataframe_original.to_dict('index')
        global app # Global app permite que la tabla actualice la data correctamente
        app = MyTable(frame, data=buffer)
        app.redrawTable() # Activa menu
        global menudisabled
        menudisabled = "normal" # menubar.entryconfig("Edit", state= menudisabled)
        menubar.entryconfig("Análisis", state=menudisabled)
        root.title('Item Stats Tools' + ' - ' + os.path.basename(filename_excel))
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
    report_file = filedialog.asksaveasfilename(initialdir=config['DEFAULT']['path'], title="Guardar Reporte",
                                               filetypes=[("Archivo html", "*.html")], defaultextension='.html')
    if report_file:
        with open(report_file, 'w') as f:
            test.seek(0)
            for x in test.readlines():
                f.write(x)
                print(test.read())
            f.close()
        global closecount
        closecount = 0
        webbrowser.open_new_tab(report_file)
    return


# Funcion analisis TCT----------------------------------------------------------------------------------------
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

        print (dataframe_diff)

        Item = dataframe_diff.columns
        score = 0
        for col in Item:
            score = score + dataframe_diff[col].apply(int)
        dataframe_diff['score'] = score
        upper = dataframe_diff.nlargest(ceil(len(dataframe_diff)//4), 'score', keep='first')
        lower = dataframe_diff.nsmallest(ceil(len(dataframe_diff)//4), 'score', keep='first')
        ##def columns
        avgScore = pd.DataFrame(columns=['avgScore'])
        SD = pd.DataFrame(columns=['SD'])
        gULI = pd.DataFrame(columns=['gULI'])
        RIR = pd.DataFrame(columns=['RIR'])
        ##fuctions
        for col in Item:
            avgScore = avgScore.append({'avgScore': dataframe_diff[col].mean()}, ignore_index=True)
            SD = SD.append({'SD': dataframe_diff[col].std(ddof=1)}, ignore_index=True)
            gULI = gULI.append({'gULI': upper[col].mean() - lower[col].mean()}, ignore_index=True)
            score_correg= score - dataframe_diff[col]
            RIR = RIR.append({'RIR': score_correg.corr(dataframe_diff[col])}, ignore_index=True)
        df_diff = pd.concat([avgScore, SD, gULI, RIR], axis=1); df_diff['Item'] = Item
        df_diff.rename(columns={'avgScore': 'Dificultad', 'SD': 'Desv. por Item',
                                'gULI': 'Indice de Discr.', 'RIR': 'Coef. Discr.'}, inplace=True)
        print(df_diff)
        it_dificultad = df_diff[['Dificultad']].copy(); it_dificultad.rename(columns={'Dificultad': 'col'}, inplace=True)
        it_dificultad.loc[it_dificultad.col >= 0.8, 'Interpretación Dificultad'] = 'Muy Fácil'
        it_dificultad.loc[(0.8 > it_dificultad.col) & (it_dificultad.col >= 0.66), 'Interpretación Dificultad'] = 'Relativamente Fácil'
        it_dificultad.loc[(0.66 > it_dificultad.col) & (it_dificultad.col >= 0.5), 'Interpretación Dificultad'] = 'Dificultad Adecuada'
        it_dificultad.loc[(0.5 > it_dificultad.col) & (it_dificultad.col>= 0.3), 'Interpretación Dificultad'] = 'Relativamente difícil'
        it_dificultad.loc[(0.3 > it_dificultad.col) & (it_dificultad.col >= 0.1), 'Interpretación Dificultad'] = 'Difícil'
        it_dificultad.loc[0.1 > it_dificultad.col, 'Interpretación Dificultad'] = 'Muy difícil'
        print(it_dificultad)
        df_interpreted = pd.concat([df_diff, it_dificultad], axis=1)
        print(df_diff[['Indice de Discr.']])
        it_discrim = df_diff[['Indice de Discr.']].copy(); it_discrim.rename(columns={'Indice de Discr.': 'col'}, inplace=True)
        it_discrim.loc[it_discrim.col >= 0.4, 'Interpretación Discriminación'] = 'Alta discriminación'
        it_discrim.loc[(0.4 > it_discrim.col) & (it_discrim.col >= 0.3), 'Interpretación Discriminación'] = 'Aceptable'
        it_discrim.loc[(0.3 > it_discrim.col) & (it_discrim.col >= 0.2), 'Interpretación Discriminación'] = 'Baja. Debe revisar el ítem'
        it_discrim.loc[(0.2 > it_discrim.col) & (it_discrim.col >= 0), 'Interpretación Discriminación'] = 'Mala. Se sugiere eliminar el ítem o reformularlo'
        it_discrim.loc[(0 > it_discrim.col), 'Interpretación Discriminación'] = 'Inaceptable. Se debe eliminar el ítem'
        print(it_discrim)
        df_interpreted = pd.concat([df_interpreted, it_discrim], axis=1)
        buf.truncate(0)
        df_interpreted[['Item', 'Dificultad', 'Interpretación Dificultad', 'Desv. por Item', 'Indice de Discr.', 'Interpretación Discriminación', 'Coef. Discr.']].to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Dificultad y Discriminación<h2>'+ text_descriptions('dificultad_tct') + text + '<hr>')
        messagebox.showinfo(master=difficult_window, message="Análisis TCT Listo", title="Mensaje")
        open_web()
        return
    open_datatable()
    difficult_window = tk.Toplevel()
    ClassToplevel(difficult_window)
    difficult_window.geometry('500x280')
    difficult_window.title('Análisis TCT')
    apply_button = tk.Button(difficult_window, text="Aplicar", command=difficult_function, state='disabled')
    tk.Label(difficult_window, text='Análisis TCT',
             font='Arial 11 bold').grid(row=0, column=1, columnspan=4)
    info_label = tk.Label(difficult_window, wraplength=270, justify=LEFT,
                          text='Permite hacer un análisis bajo la teoría clásica de test(TCT). Cálcula el grado de dificultad, indice de discriminación y coeficiente de discriminación por item. Con su respectiva interpretación.')
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
        ##### Extraer los thetas en una lista
        with localconverter(ro.default_converter + pandas2ri.converter):
            fscores_df = ro.conversion.rpy2py(fscores[0])
        theta = {}
        x = len(dataframe_rasch.columns)
        for row in range(len(dataframe_rasch)):
            for rows in range(len(fscores_df)):
                if (dataframe_rasch.loc[row].astype(int) == fscores_df.iloc[rows, 0:x].astype(int)).all():
                    theta[f'P{row:0>2d}'] = fscores_df['z1'].iloc[rows]
        with localconverter(ro.default_converter + pandas2ri.converter):
            b = ro.conversion.rpy2py(coeff.rx(True, 'Dffclt'))
        betas = {}
        for item in range(len(dataframe_rasch.columns)):
            betas[dataframe_rasch.columns[item]] = b[item]
        encoded2 = wright(theta, betas)
        godfit = ltm.GoF_rasch(model)
        p_value = np.array(godfit.rx2('p.value'))
        if p_value[0] >= 0.05:
            interpretation = 'La distribución de los datos ajustan al modelo Rasch'
        else:
            interpretation = 'La distribución de los datos no se ajustan al modelo Rasch'
        list_gof = {'P-valor Bondad de ajuste': [p_value[0]]}
        df_gof = pd.DataFrame(list_gof, columns=['P-valor Bondad de ajuste'])
        writer(coeff, 'filetempo')
        df_rasch = pd.read_csv('filetempo')
        df_rasch.rename(columns={'Unnamed: 0': 'Item', 'Dffclt': 'Dificultad', 'Dscrmn': 'Discriminación',
                                 'P(x=1|z=0)': 'Prob. de acertar item'}, inplace=True)
        buf.truncate(0)
        df_rasch[['Item', 'Dificultad', 'Prob. de acertar item']].to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Análisis IRT-RASCH 1PL<h2>' + text_descriptions('irt_rasch') + text +'<h4>Prob. de acertar item, es la probabilidad de tener el item correcto dado un nivel de habilidad medio P(x<sub>i</sub>=1|z=0), i= número de item </h4>' + '<br>')
        buf.truncate(0)
        df_gof.to_html(buf, index=False)
        text = buf.getvalue()
        test.write(text +'<h3>' + '<br>' + 'Interpretación:' + '<br>' + interpretation + '<h3>' + '<br>')
        test.write('<h2> <br>Gráfico curva característica del item (ICC) <br> <img src=\'data:image/png;base64,{}\' height="380px" class="center"> </h2>'.format(encoded))
        test.write('<h2> <br>Mapa Item-Persona (Wrightmap) <br> <img class="zoom" src=\'data:image/png;base64,{}\'> </h2>'.format(encoded2) + '<hr>')
        remove('file.png')
        remove('filetempo')
        analysis_rasch_window.config(cursor='')
        messagebox.showinfo(master=analysis_rasch_window, message="Análisis Rasch Listo", title="Mensaje")
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
        ###############wrightmap
        fscores = ltm.factor_scores(model, resp_patterns=scores)
        ##### Extraer los thetas en una lista
        print(coeff)
        with localconverter(ro.default_converter + pandas2ri.converter):
            fscores_df = ro.conversion.rpy2py(fscores[0])
        theta = {}
        x = len(dataframe_rasch.columns)
        for row in range(len(dataframe_rasch)):
            for rows in range(len(fscores_df)):
                if (dataframe_rasch.loc[row].astype(int) == fscores_df.iloc[rows, 0:x].astype(int)).all():
                    theta[f'P{row:0>2d}'] = fscores_df['z1'].iloc[rows]
        with localconverter(ro.default_converter + pandas2ri.converter):
            b = ro.conversion.rpy2py(coeff.rx(True, 'Dffclt'))
        betas = {}
        for item in range(len(dataframe_rasch.columns)):
            betas[dataframe_rasch.columns[item]] = b[item]
        encoded2 = wright(theta, betas)



        print(df_irt)
        it_discrim = df_irt[['Dscrmn']].copy(); it_discrim.rename(columns={'Dscrmn': 'col'}, inplace=True)
        it_discrim.loc[it_discrim.col >= 1.7, 'Interpretación Discriminación'] = 'Muy Alta'
        it_discrim.loc[(1.7 > it_discrim.col) & (it_discrim.col >= 1.35), 'Interpretación Discriminación'] = 'Alta'
        it_discrim.loc[(1.35 > it_discrim.col) & (it_discrim.col >= 0.65), 'Interpretación Discriminación'] = 'Moderada'
        it_discrim.loc[(0.65 > it_discrim.col) & (it_discrim.col>= 0.35), 'Interpretación Discriminación'] = 'Baja'
        it_discrim.loc[(0.35 > it_discrim.col) & (it_discrim.col >= 0.1), 'Interpretación Discriminación'] = 'Muy baja'
        it_discrim.loc[0.1 > it_discrim.col, 'Interpretación Discriminación'] = 'Ninguna'
        df_irt = pd.concat([df_irt, it_discrim], axis=1)
        df_irt.rename(columns={'Unnamed: 0': 'Item', 'Dffclt': 'Dificultad', 'Dscrmn': 'Discriminación',
                                 'P(x=1|z=0)': 'Prob. de acertar item'}, inplace=True)
        buf.truncate(0)
        df_irt[['Item','Dificultad', 'Discriminación', 'Interpretación Discriminación', 'Prob. de acertar item']].to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2> Análisis IRT-2PL </h2>' + text_descriptions('irt_rasch') + text+ '<h4>Prob. de acertar item, es la probabilidad de tener el item correcto dado un nivel de habilidad medio P(x<sub>i</sub>=1|z=0), i= número de item </h4>')
        test.write('<h2> <br>Gráfico curva característica del item <br> <img src=\'data:image/png;base64,{}\' height="380px" class="center"> </h2>'.format(encoded) + '<hr>'+'<button onclick="myFunction()">ICC item por separado</button>'+script)
        remove('filetempo')
        test.write('<div id="oculto" style="display: none">')
        for item in range(1, len(df_irt[['Item']])+1):
            print(item)
            grdevices.png('file.png', width=512, height=512)
            plot_ltm(model, type='ICC', items=item)
            grdevices.dev_off()
            encoded = base64.b64encode(open("file.png", "rb").read()).decode('utf-8')
            test.write('<br>ICC item {item} <br> <img src=\'data:image/png;base64,{}\' height="380px" class="center">'.format(encoded, item=item))
        test.write('</div>')
        test.write(
            '<h2> <br>Mapa Item-Persona (Wrightmap) <br> <img class="zoom" src=\'data:image/png;base64,{}\'> </h2>'.format(
                encoded2) + '<hr>')
        analysis_rasch_window.config(cursor='')
        remove('file.png')
        messagebox.showinfo(message="Análisis IRT-2PL Listo", title="Mensaje")
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
                          text='Se realiza un análisis bajo la teoría de respuesta al item (IRT), para esto se estiman los parámetros'
                               ' Dificultad y Discriminación por item (en el modelo de 1 parámetro solo se estima dificultad, en el otro caso ambos). Además de calcular la probabilidad de que un  '
                               'individuo con nivel de habilidad 0 (o Medio) responda el item correctamente.'
                               '\n Grafíca la curva característica de item (ICC) y el Mapa Persona Item (WrightMap).\n \n Seleccione entre modelo de 1 parámetro (1PL)y '
                               '2 parámetros (2PL).')
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
        i = 0
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                i = i + 1
                if i >= 4:
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
        scale = {'Coef. Spearman-Brown': [spearman], 'Coef. Rulon': [rulon]}
        df_scale = pd.DataFrame(scale, columns=['Coef. Spearman-Brown', 'Coef. Rulon'])
        buf.truncate(0)
        df_scale.to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<br><h2>Método de dos partes<h2>' + text +'<h4>Para el Método de dos partes (Half-split), la división entre grupos es entre pares e impares.</h4>'+ '<hr>')
        return

    def function_alpha():
        column_selected = []
        for i in range(len(vars)):
            if vars[i][0].get() == 1:
                column_selected.append(vars[i][1])
        dataframe_alpha = dataframe_table[column_selected].copy().astype(float)

        def alpha_calc(data):
            score = 0
            sum = 0
            for col in data.columns:
                score = score + data[col]
                sum = sum + data[col].var(ddof=0)
            st= score.var(ddof=0)
            n = len(data.columns)
            alph = (n/(n-1))*(1-(sum/st))
            print(alph)
            return alph

        alpha = alpha_calc(dataframe_alpha)
        list_alphac = {'Coeficiente Alpha Cronbach': [round(alpha, 5)],
                       'N de item': [len(dataframe_alpha.columns)],
                       'N de casos': [len(dataframe_alpha)]}
        df_alphac = pd.DataFrame(list_alphac, columns=['Coeficiente Alpha Cronbach', 'N de item', 'N de casos'])
        print(df_alphac)
        alpha_drop = pd.DataFrame(columns=['alphaDrop'])
        for column in dataframe_alpha.columns:
            temporal_alpha = dataframe_alpha.drop(column, 1)
            alpha_drop = alpha_drop.append({'alphaDrop': alpha_calc(temporal_alpha)}, ignore_index=True)
        alpha_drop['Item'] = dataframe_alpha.columns

        # export df to html
        if 0.6 <= alpha < 0.65:
            interpretation = 'Alpha Cronbach se considera indeseable, se recomienda revisar instrumento.'
        elif 0.65 <= alpha < 0.7:
            interpretation = 'Alpha Cronbach es minimamente aceptable. No es suficiente para tomar decisiones y menos' \
                             ' aun las que influyan en el futuro de las personas como por ejemplo Test de Admisión'
        elif 0.7 <= alpha < 0.8:
            interpretation = 'Alpha Cronbach es suficientemente bueno para cualquier investigación. Deseable en la' \
                             ' mayoria de los casos (Ej: Test de habilidades)'
        elif 0.8 <= alpha < 0.9:
            interpretation = 'Alpha Cronbach se considera muy buena.'
        elif 0.9 <= alpha:
            interpretation = 'Alpha Cronbach es un nivel elevado de confiabilidad.'
        else:
            interpretation = 'Alpha Cronbach se considera inaceptable, se recomienda revisar instrumento.'
        buf.truncate(0)
        df_alphac.to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<hr>' + '<br><h2>Test de Confiabilidad Alpha Cronbach<h2>' + text_descriptions('alpha_cronbach') + text + '<br><h3>Interpretación:<h3>'
                   + interpretation + '<br>')
        print(alpha_drop)
        buf.truncate(0)
        alpha_drop.rename(columns={'alphaDrop': 'Alpha sin Item'}, inplace=True)
        alpha_drop[['Item', 'Alpha sin Item']].to_html(buf, index=False)
        text = buf.getvalue()
        test.write('<br><h2>Alpha Cronbach eliminando Item<h2>' + text + '<h4>Mientras mayor sea el Alpha mas confiable es la medición, se recomienda evaluar si es necesario eliminar un item en favor de la confiabilidad del test. Un Alpha mayor a 0,7 es idóneo en la mayoria de los casos.</h4><br>')
        if var_replace.get() == 1: splithalf_function()
        messagebox.showinfo(message="Alpha Cronbach listo", title="Mensaje")
        open_web()
        return

    open_datatable()
    alpha_window = tk.Toplevel()
    ClassToplevel(alpha_window)
    alpha_window.geometry('500x280')
    alpha_window.title('Análisis de Fiabilidad')
    apply_button = tk.Button(alpha_window, text="Aplicar", command=function_alpha)
    tk.Label(alpha_window, text='Cofiabilidad del Test', font='Arial 11 bold').grid(row=0,
                                                                             column=1,
                                                                             columnspan=4)
    info_label = tk.Label(alpha_window, wraplength=270, justify=LEFT,
                          text='Permite medir la confiabilidad de su instrumento mediante el cálculo  del Alpha de Cronbach.'
                               '\n Selecciona las columnas de los item a analizar. Los datos deben ser de tipo '
                               'dicotómico, politómico o continuo. (El ingresar tablas con datos no aceptados arrojará error).')
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
    var_replace = tk.IntVar()
    check_column = tk.Checkbutton(alpha_window, text="Método de dos mitades (Spearman-Brown | Rulon)", variable=var_replace)
    check_column.grid(row=6, column=1, columnspan=4, sticky=W)
    checklist.configure(state="disabled")
    alpha_window.mainloop()


# Transformar Datos ---------------------------------------------------------------

def transform_data():
    def disablebutton():
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                print('activado', vars[row][0].get)
                check_column.configure(state=NORMAL)
                assign_but.configure(state=NORMAL)
                assign_cuanti.configure(state=NORMAL)
                return
            else:
                print('desactivado')
                check_column.configure(state='disabled')
                assign_but.configure(state='disabled')
                assign_cuanti.configure(state='disabled')
        return
### CUALIREPLACE#####################################################3
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
        cuali_window.title('Asignar valores')
        cuali_window.geometry('570x300')
        label_frame = LabelFrame(cuali_window, text="Ayuda")
        info_cuali = tk.Label(label_frame, wraplength=150, justify=LEFT,
                               text='   Ingresa los valores para reemplazar la variable.\n\n'
                                    'La categoría ELSE permite asignar un valor por defecto a todas las categorías que estén vacías. Es obligatorio asignarle un valor.')
        frame_cuali = ScrollableFrame(cuali_window)
        select_column = []
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                select_column.append(vars[row][1])
        unique_values = (pd.unique(dataframe_table[select_column].values.ravel())).tolist()
        unique_values.sort()
        unique_values.append('ELSE')
        array_valores = []
        i = 0
        for values in unique_values:
            variable_remplazada = StringVar()
            tk.Label(frame_cuali.scrollable_frame, text=values).grid(row=i, column=0, columnspan=4, sticky=W)
            tk.Entry(frame_cuali.scrollable_frame, textvariable=variable_remplazada).grid(row=i, column=5, columnspan=4, sticky=W)
            array_valores.append([values, variable_remplazada])
            i = i + 1
        button_apply = tk.Button(cuali_window, text="Aplicar",
                                 command=lambda: apply_trans() if array_valores[len(array_valores) - 1][
                                     1].get() else tk.messagebox.showinfo(title='Información',
                                                                          message='Ingrese valor a ELSE'))
        frame_cuali.grid(row=1, column=1)
        label_frame.grid(row=1, column=2)
        info_cuali.pack()
        button_apply.grid(row=2, column=2)
        button_apply.configure(DISABLED)
        cuali_window.mainloop()
        return
####CUANTI REPLACE#######################
    def cuanti_replace():
        def apply_trans():
            try:
                global app
                testing = dataframe_table.copy()
                if var_replace.get() == 1:
                    for rows in array_valores:
                        functions = rows[1].get().split(';')
                        for f in functions:
                            equal = f.split(':')
                            i = 0
                            for x in testing[rows[0]]:
                                if eval(equal[0], {"x": float(x)}):
                                    dataframe_table[rows[0]][i] = equal[1]
                                i += 1
                    buffer = dataframe_table.to_dict('index')
                    app = MyTable(frame, data=buffer)
                    app.redrawTable()

                if var_replace.get() == 0:
                    for row2 in array_valores:
                        functions = row2[1].get().split(';')
                        print(functions)
                        for f in functions:
                            equal = f.split(':')
                            i = 0
                            for x in testing[row2[0]]:
                                if eval(equal[0], {"x": float(x)}):
                                    dataframe_table[row2[0]][i] = equal[1]
                                i += 1
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
            except Exception as error:
                print(error.args[1][3])
                tk.messagebox.showinfo(title='Error', message=f'Error en la sintaxis: {error.args[1][3]}')

#######################################################################################################
##############VENTANA CUANTI###########################################################
        cuanti_window = tk.Toplevel()
        ClassToplevel(cuanti_window)
        cuanti_window.title('Asignar valores')
        cuanti_window.geometry('680x300')
        label_frame = LabelFrame(cuanti_window, text="Ayuda")
        info_cuanti = tk.Label(label_frame, wraplength=250, justify=LEFT,
                              text='   Ingresa los intervalos para transformar la variable.\n\n'
                                   'Ejemplo: x <= 6.5 : 4 ,    Los valores menores o igual a 6.5 serán igual a 4\n\n'
                                   'Ejemplo: 2 > x > 1 : 0 ,    Los valores entre 2 y 1 serán igual a 0\n\n'
                                   'Operadores disponibles: [>, mayor][>=, mayor o igual][<, menor][<=, menor o igual]\n[==, igual] [!=, distinto]\n'
                                   'Los intervalos deben estar separados por el símbolo ";" y los decimales por punto.\n\n Ejemplo: x > 4 : 1 ; x <= 4 : 0')

        frame_cuanti = ScrollableFrame(cuanti_window)
        select_column = []
        for row in range(len(vars)):
            if vars[row][0].get() == 1:
                select_column.append(vars[row][1])
        array_valores = []
        i = 0
        for values in select_column:
            variable_remplazada = StringVar()
            tk.Label(frame_cuanti.scrollable_frame, text=values).grid(row=i, column=0, columnspan=4, sticky=W)
            tk.Entry(frame_cuanti.scrollable_frame, textvariable=variable_remplazada).grid(row=i, column=5, columnspan=4, sticky=W)
            array_valores.append([values, variable_remplazada])
            i = i + 1

        def missing_value():
            count = 1
            for miss in array_valores:
                if len(miss) == 1:
                    count = 0
            return count
        button_apply = tk.Button(cuanti_window, text="Aplicar",
                                 command=lambda: apply_trans() if missing_value() else tk.messagebox.showinfo(title='Información',
                                                                          message='Ingrese valores'))

        label_frame.grid(row=1, column=2)
        info_cuanti.pack()
        frame_cuanti.grid(row=1, column=1)
        button_apply.grid(row=2, column=2)
        button_apply.configure(DISABLED)
        cuanti_window.mainloop()
        return
####################ventana transform#########################33
    open_datatable()
    input_window = tk.Toplevel()
    ClassToplevel(input_window)
    input_window.title('Transformar Data')
    input_window.geometry('500x300')
    info_label = tk.Label(input_window, wraplength=270, justify=LEFT,
                          text='Permite transformar los datos de las columnas seleccionadas. Para seleccionar las columnas'
                               ' haz click en el cuadro blanco. Puedes eligir reemplazar las columnas originales por las '
                               'transformadas. Por defecto se agregará una nueva columna de nombre Copy'
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
    assign_but = tk.Button(input_window, text="Valores categoricos", command=cuali_replace)
    assign_but.grid(row=6, column=8, sticky=W)
    assign_but.configure(state="disabled")
    assign_cuanti = tk.Button(input_window, text="Valores continuos", command=cuanti_replace)
    assign_cuanti.grid(row=7, column=8, sticky=W)
    assign_cuanti.configure(state="disabled")
    input_window.mainloop()
    return


# --------------------------------------------------------------------------------------------------
def about_chungungo():
    about_window = tk.Toplevel()
    ClassToplevel(about_window)
    about_window.overrideredirect(True)
    about_window.configure(bg='#efefef')
    # Gets both half the screen width/height and window width/height
    positionright = int(about_window.winfo_screenwidth() / 2 - 400 / 2)
    positiondown = int(about_window.winfo_screenheight() / 2 - 250 / 2)
    about_window.geometry('400x250')
    about_window.geometry("+{}+{}".format(positionright, positiondown))
    img = PhotoImage(
        data='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JIjU1bXr23db/uNntnh0w388eVuhd6+eu+++m+aHoTFXnh+qnh9KzQk7R4yaZI6BCmT2dAb/0Y9+9Na3vnXnnXdWYL9+/fTHOHmqH3nkEZLXddddN3HiRLZjgwcPvvzyy5nV/v37q8XGxsaLL754hx126Nq1K2/7K6+88pe//OXtt98+a9asYcOGsQ38y1/+Qiv777//L37xC7IheWTffffV9pDeTps27ec///lvfvObf/7zn9wp/U1ThnDppZcOHz78hRde4OWPJi3uscceL774IucmnGfOnOm/SIj4jTfeiPKf/vSnuXPn7rPPPhgffvhhshXhPITI9uzZ86GHHlq2bFnv3r0ffPDBRYsWde/e/fvf/z5X9QfoFyxYQJoG99577+LFi5kKRsdBiTPR1Vdf/fjjjzNjDAdPpkvTCF555RWu0p+///3vjGXUqFGy05mxY8fyeNNbOj9jxgz9bVQCp0yZcs011xDyt7/9jZ7vuuuuUvvd7343cODAI488kha7devWp08f2Rnpr371K9yYk8suu4y5vfPOO+ntyJEjGReckdIWzgyQ+8idYp2Td/r27asWKQ0axef6669nZhhj586dWSSkPPqvuzl69GiFaNmgyTRy78hWd911FzPDjRswYIA26YwFNXrFIqE/TDKxTz31FNPFfo0xMnzmk8MmK+faa6/99re/TUPMkvomqCFArDnEPRdnrTLDSjqCnYGdgTkky5tiNGBB6x7gZDtGt4FR8YClYK7jDG44+7Nwqj6A4Cx/gLN1mEE36mMggfp71lSJ6tihJDhp0lOMvOS3JfQIrVrV8g12EY466sjKf8SpBeDexiOh54ph5h4JcdCcgKLf9+hA/UUspo5n+x3veAf8iCOOYK2QuQjhlUs28R255ZZbSGEs1iVLlhx33HFkJZ6EXXbZ5e677+bBkM+zzz57+OGHv/baa4cddhir+T3veY/sLHR2dqRCHglaf9e73vWtb30LO7eSLPaRj3zklFNOIaHwFJ1zzjlUjz32WC7tvffe7Cz0d5h5GXzqU5/ircmRlvRK6vzgBz+InVzz0ksvcaN53njS1O3vfve7DAf/5557jtxKKiTj4EwePOigg3gm99xzT55GUgzZ85lnnuGAvHDhQsbO6rrooouyy4b0wRBolxTD5pTnU3ZS4fve9773vve95Mf99tsPO+G65BDG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P1p1vgUMU7lJ4P6yS+INwV2DE44OJZCDA4EWgxoS16RddNFFj2wGm0FOf35FSceriKqWpexedSA+leY4sGmwiAhltldCbCjy7ApsSliq1wgNXogcEVctGC1yEGRMEVcB3CKQXr2T0wpggT7zzDNJJYD387Dhw3ccuVOv3r2jVMuwNd+RqYyePXp4aCAOUwTAo93cyDWCrJH3Nu981hCLkiovZ05MnOY4C/CClf9b3/pWHsW4jIwoKE45duxYctbVV18tO+CEMm/ePH3An5iCf7ybLERxzjhPPfWUPu5NgV2JdsopRHHAjobTlnY0Quq+6wdePKI6aXJVDuQvns8xY8aUvZLMa3ETnM1feeUVtr0Oyf6zNimzYyULv/e972WY+qGH7IAd8dlnn83M33LLLTYCuKqc6G+++WaW98SJpX9t5v0ps8r+S0d7gPPvf//7/fffn4QlHykAOTgQC3kNQsmrTpxt9ahRo3hdlV1KwJ9heqQichaUy5RBLA6c4wCcQFphFdlHIhKMgVEcrs4D2+H2h1AtEdVBKl4kG0PDEqUEGHUklI9PbVjE7SPOkDzCeOThRqohfEg6EqT0BxNEHXLIwd27N/3j5zvuuMvzmws/FS3Ghg1NL4RtiPbt2x155OE+BjIJnrc4P0ya5wcH34jUJHuuHEiUAsePH88u4M4777z//vsvuOACLHvuuWfZpRWP06RJk9iDkLwU+IUvfIF3+0knnfSXMr785S/7a1CIqxWgbjCxNI04p6qvfOUrbOI4H33oQx/irET6cFeB/Skt4lvD88xugiMJOYV8d80115xyyikPP1w6gHPyIolwBEPcgSBygY0SHeaMxr7mxhtv5GjMRobkcv7553M6ZtfGjoazp79IIey3336U7Hdw+M53voOz7Fl4FR1wwAH0/Etf+hL7oO9+97vMmOwpMGT2rezsjj/+eBIKFhQ4rV911VUPPPAAuRXCC0POGg436MILL2Tjdu+995533nlE+TsWAu8VzvUnnngieYrBcmykD/7ZJfrZaUlh9OjRpOybbrrpsccew/l73/sec8UtYxvILHG7P/axjyWuAR67oISVMhpcrXwMLAqs2nkFUlb/HhbNi2OkijSQs+x0ESI7xxytfjyxK08RiL8eIQiCcDljEWcvA0GHQLiSIJ5sjPV8Erhu7dp99t571uzZehlyacSIEZwp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')
    tk.Label(about_window, bg='#ebe8e3', image=img).grid(row=0, column=0, columnspan=4, sticky=W)
    s = 'Es un programa que permite a profesores realizar' \
        ' analisis psicométrico con sus respectiva interpretaciones.\n\n Errores y sugerencias al correo ✉peter.gonzalez2018@umce.cl'
    tk.Label(about_window, bg='#efefef', text='Acerca de Item Stats Tools', font='Arial 11 bold').grid(row=1, column=0)
    tk.Label(about_window, bg='#efefef', text=f'Versión: {version}', font='Arial 9').grid(row=1, column=3)
    tk.Label(about_window, bg='#efefef', text=s, font='Arial 9', justify=LEFT,
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

#
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# override close chungungo--------------------------------------------------------------------------

def closechungungo():
    if closecount == 1:
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
    else:
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
    data='iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAN1wAADdcBQiibeAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAACAASURBVHic7N15cGX5dR/2713evm/Aw8O+9go0uhu9znAWkpKGEkWKDBWxLNtK2YlqIsVlK4mlGiki6dgl/mHHcZUTi065oiRKUk5ZiWSJIkVSIocznKX3fcW+Pjzg4e373fIHuoc9PUBju+t751PVhZlu4N7T6If3O/e3nMMoigK1fPNbV20AXgHwGoBOAHEAHU9+xQCwqt2MEEIIaT4ygHUAySe/VgEsA3gbwDtvvTkhqHUj5qAJwDe/dTUI4HMAvvDkY0CFuAghhBDycXkA3wXw5wC++9abE7mDXGzfCcA3v3U1AuAtAL8JwHmQIAghhBCyJzUA/zOAb7715sTGfi6w5wTgm9+66gbwWwD+MehpnxBCCDFSHsA/B/A/vvXmRGUvX7inBOCb37r6FQD/Gptr+4QQQggxh1UA/+CtNyf+ZLdfsKsE4JvfusoA+CcAfn//sRFCCCFEY/8UwNffenNix8F9xwTgm9+66gHwxwC+pE5shBBCCNHQnwL4O2+9OVF+0Se9MAH45reudgP4NoAxdWMjhBBCiIZuA/j8W29OLG73CdsmAE+e/N8HDf6EEEKIFd0GcHG7mYAtC/M8WfP/Y9DgTwghhFjVGIA/fjKmf8J2lfn+CWjNnxBCCLG6L2FzTP+ETywBPDnq9x90CIoQQggh+vjl548IfiwBeFLkZxp0zp8QQghpJqsABp8tFvT8EsBvgQZ/QgghpNnEsTnGf+SjGYAntf2nQeV9CSGEkGaUx+YswAbw8RmAt0CDPyGEENKsAtgc6wE8mQF40tI3CerqRwghhDSzGoCOt96cyD2dAfgcaPAnhBBCmp0Tm2P+R0sAXzAuFkIIIYTo6AsAwPzBH16xAVgHrf8TQgghrSAPIMYCeAU0+BNCCCGtIgDgFRbAawYHQgghhBB9vcYC6DQ6CkIIIYToqpMFVf4jhBBCWk2cBdBhdBSEEEII0VUHJQCEEEJI6+lgAcSMjoIQQgghuoqx+GRHQEIIIYQ0N5YGf0IIIaQFUQJACCGEtCBKAAghhJAWRAkAIYQQ0oIoASCEEEJaECUAhBBCSAuiBIAQQghpQbzRARBzYBQFNqkBmyj89KPYgE1qwP7s70kN/I3XjUVOAc/y4BgO3DMf+Wf+n2dtcNvc4Fmb0X89Qj6BlWU4hRocQnXzY6MGp1CFQ6jhocOG6zbAxtlgY+0//cja4LZ54ORdRodvKoyiwFctIFTKIFjOwNWoGB2SKSkMUHL6kfWGkfOEUXF4DI2HEoAWwigKvLUCguUcAuUsApUsvNUC7GIDvCTu+joiF0cS5V1/vo21w23zfPwXv/mRZ+klSPThFGpoyyXRnksilk/BIdS2/dwHbR1YqZe2/XMH50DAEULAGULQEYLfEQDLcFqEbXqRwjpOT38AT2337wlk03qgHdcHzxmWCNC7b5PiJQGBSg6BZwZ7fyUPTpZ0j0WQG8jXG8jXs5/4Mztnh8fmRdAZRsgZQcARoqSAqIJVZISLabQ/GfQD5U++/varLtWxVlnFWmUVAMAwDIKOEBLebrR7OsC1wGuYk2UcW7iJwdXHgKIYHY4lxfIpfObWd3C79xTm2wd1v3/zv0pbhLdWREdmCeHiBgKVLDy17Z9ezKQhNdCQMsjWMpjFFBgw8Dn8CDkjCDnDCDojsNESAtkDd72EQ8sP0JWeBy8JutxTURRka5uv44eZe2j3dKDT242gM6zL/Y1wfO46BlKTRodhebwk4tTMZUi8DUuRHn3vrevdiKpCpQw6MktIZJbgq+aNDkcVChQU6nkU6nnM52cAAF67DyFnBBFXDBFXDCxDe1fJJ/mqBYws30d3eg6MgU+kkixipbiIleIi3DYPevz96PL1gmEYw2JSWyy/ioG1KaPDaConZq9i3deGut2p2z0pAbAQRlEQLawhkVlCR2apZTbalBpFlBpFLBbmwLM2tHs6EPckEHJFwKB53lTJ/gQqORxauofExiIYmGsquiKU8XDjLhYKsxgOH0GbO250SAfGyTJOT1+maX+V2YU6xuau4crIS7rdkxIAk+NkCe25JDoyS+jILsMmNowOyVCiLGC5uIDl4gLsnANxTwJxbycCjqDRoRGdeasFHJ+/iY7sstGh7KgilHErdRVBZxgj4SMIOEJGh7RvgXIGrjpt+NNCR3YZjKJA0Wm2iBIAkwoX0xhKPkI8u2zIxj0raEh1LBRmsVCYhYt3I+7tRIc3AY/NZ3RoRGO9azM4MXsNnLz70ytmkKtlcHnlPcQ9CRyOjlpyf0uwnDE6hKbFyRL8lRzyHn0SREoATISBgo6NJQwnHyJcTBsdjqVUxQpmc5OYzU0i7IqiLzCIiCtmdFhEZbwk4OTMFXSl540O5UBWyyvI1bMYazttudmrUEm90xTkk0LlDCUArYSTRfStzWAw+cgyu/fNLFNNI1NNw2v3oy8wgLins6k2YLWqUCmDM5PvNc3PSE2s4kryPYyEj6LH3290OLum18mKVqXnMi8lAAZyNqoYXH2M/tRUy6/ta6HUKODu+k1MZR+ixz+ATl8P1RiwqOGVBzi6cBusIhsdiqoURcGjjXvI1jZwLDpuiddn1htGIrNodBhNK+ON6nYv87/amlCgksPQykN0peeb7g3NjGpiDY8z9zGTe4wuXy96Av1wcPodtSH7xygKJibfR9fGgtGhaGqtvIpS411MdFww/Wsz57XuBkazU8Agp9P0P0AJgK6C5QyOzd9CW37V6FBakiiLmMtPY6Ewi25/PwaCw5Z44mpVDBScmv6w6Qf/pypCGVeTH2Ci4yIcnMPocLaV9UQgsbzlNmBaQc4XhsTp955EFVV0YBMbODF7Fa/d/j4N/iYgKzLm89N4b+lHWC4uQDHZ2XGyaXzmCnrW54wOQ1cVoYxryQ/QkOpGh7ItgbfjXs+Y0WE0HYVhcKtvQtd7UgKgsZ71GfzMzW9jYHXSdEVKWl1DquN++jYuLf8E2RodbTKTsdlr6EtNGx2GIcpCCVeTH5o6CZjpGMGGn07ZqOlx4giyXn1LR1MCoJFAJYdX7v41Tk9dgkMw7w8yAYqNPK4m38ftteuoiVWjw2l5x+efNJhpYWWhiGurH0KQzbnjXgGDa4MXkHdb6wijWS1HuvGwa1T3+9ICqMp4ScCRxTsYXH1saD1ysnep8grWKyn0BQbRFxwE16LtXY10eOkuhlceGB2GKZQaRdxbv4nx9jNGh7KlstODt0d/DoeX72Jk+T693+1Dg7fjVv8ElqK9htyfEgAVdaXnMTp/A84GPUValaxImMk9xkppEcdjJxFq4m5uZhMtrOHI4h2jwzCV9UoK8/kZ9AYGjA5lSzLL4n73GJYiPWjPrSJYziBUysDZKINRqPbG8xSGQcnlQ84TRtYTwkqkG3Wbcac+KAFQga9awInZq4jlU0aHQlRSE6u4lvwA/cEhDARHqJCQxjhZxKnpS0aHYUqT2QcIOsOmrhhYcAdRoOUAy6E9AAfUn5rCp299lwb/JqRAwUxuEleS76MqtkbnRaMcn7/VNBX+1KYoCu6sXYdIx+6IyigB2CdOlnB66hLGZ65QMZ8ml69n8eHyO0iWlowOpSlFC2sYaPFNfzupihXcT98yOgzSZCgB2Ad3vYRX7v4APeszRodCdCLKIu6u36QnMZVxsojTUzT1vxupchIb1XWjwyBNhBKAPWrPJfHp299DsEwdsVrRankFHyz/GDmqG6CK4/M34a7T1P9uTWUeGh0CaSKUAOzB4aW7uPDgx9S4p8XVxCquJj/AUrE1StRqJVDOYWB10ugwLKXQyGO1vGJ0GKRJUAKwCzaxgQsPf4wji3eomh8BsLlB8EH6Nqay9ES2X0NJ+t7tx1T2IRQ6c09UQAnADgLlLF6/8z3Es5R1k0+azU3hzvoNyLQRdE8cQg1d6Xmjw7CkqlDBUpG+d+TgqA7AC/Ssz2J85go4WTI6FGJiq6Vl1MUaxtsnwLM2o8OxhIHVSTo9cwB1aRks0wNZoWc4sn/06tnGUPIRTk99SIM/2ZVsbQOXV96jXgK7wMoy+lNTRodhOSzD4lDYh985W8J/f/EqOjyLRodELI5mALYwlHyE0bnrRodBLKYslHBp5Sc42X4WfkfA6HBMqzs9B4dQMzoMy7BzNpxqt+MrwzNo9/z0xESPbwrLJWNqyJPmQAnAc4aSDzE6d8PoMIhFNaQ6ribfx4XO03DZ2owOx5SGko+MDsESfHYXXumS8MWhKbj4T3YF7PFN48Pkp6GAylST/aEE4BnDKw9xfJ4Gf3Iw7R43/u7R/w/vJ9/AcqnP6HBMJVJch7+SMzoMU+vwevFGXwGv99x94dDu4Kpo9yxhtdytW2ykuVAC8AQN/kQNbW4vvn7hHly8gNe6vo0fLn4ByXKP0WGZBvXM2BrLsBgOefCl4SSORaZ3/XW9vilKAMi+UQIAYHjlAY7P3zQ6DGJxEZcH37j4EG7b5nQtx0h4vesv8DeLv4RUpdPg6MwhUqRSts96ur7/yyMzaHPvvSJij28al1Zf1yAy0gpaPgGgwZ+oIeh04xsXJuGz1z/2+zwr4jPd/xE/WPgS1qsdBkVnDoyiIFzcMDoMU/DbXXilW8QXh6bg5PbfW8LFl9HmWsFaNaFidKRVtHQCQIM/UYPf7sLXLswg6Nz6CCDPCvhsz5/hBwtfRrrarnN05hGo5MBLn9zM1ko6vF58ri+P13ZY39+LHv8UJQBkX1o2ARhZvo9jC9RekxyMx+bE712YR8xVfuHn2dgGPtv9p/je/FeQrUd1is5cWnX6n2VYjIQ9+PLQCo7sYX1/txIe8/SkEGUBhXoeNakKKAdLcRgGcNs88Nr94BhOpQjJs1oyAejaWKDBnxyYi3fg984vI+Ep7urz7Vwdr3f/Bb4z+1XUJJfG0ZlPpNBaCYCds2OincdXRmYR28f6/m65eWO7KYqyiOnsI6xX11AVXpwI7wcDBh67DwlvF3oD/QAde1RNyyUA/koep6j/ODkgB2fHb59Nodu3tyNtXlsBr3X9Jb6/8OWWK+PaKjMAfocbr3YJ+MLQ5IHW93fLztXBsyJEWf+380xtA/fXb6KqYQVMBQpKjQIeZ+5jrbKKY7ETcPMeze7XSloqAbBJAs49ehecrP0PJWleds6G/2ZiA0PB/W1oa3Mv41z8R/gg+RmVIzMvh1CDs9HcZZITXh8+15/Da913dL+3iy+h2Ajqes9kaQn30rd07UyYq2XwwdI7OJd4CV67X7f7NquWSgBOT34Ab21307WEbIVnefzDU3kciawd6DrDwbvI1SN4kBlXKTJzszXp5r/N+vwefGl4BUfC6q/v75abL+uaANTEGh5u3DOkLbGsSLi7fhNnEy+DZVprFk1tLZMAHFq+h47sstFhEAvjWA6/MV7CWCypyvUm2t9Brh5uiUJBvNRcs24Ozo7TcR5fGdZ2fX+3XLz6a+8vcj99G6JsXFJXbBQwm5/EYPCQYTE0g5ZIANpzSRxZ0H9ajjQPlmHxX4zVcCauXhLJQMGrnd/BX859VffpW71xTZIABBxuvNot4AuDk3DosL6/W3puBCwJRWxUDzYDpob5/CwGAiNgGNoUuF9NnwC462VMTL4PBvpPVZHmwDIsfu2YiJcS6h+3snN1vNL5XXx37leaelMgb/F9N50+H97oz+K1LnM+SLht+s0AFBsF3e71IpIsoiKW4bF5jQ7Fspo6AeBkCecevQu72DA6FGJRDBj8ymEFn+6Z1eweEecaxmMf4PraS5rdw2hWLADEMiwOhz348vAyDhm4vr8bes4AFOrmaeZUqOcoATiApk4AxmeuIljOGh0GsSgGDL40wuLn+x9rfq9jkWtYKfVitdKl+b2MYLU9AIN2Gf/iUys7FngyCztX0+1eVaGi2712UjFRLFbUtHOO/akp9KzPGB0GsbCfH+TxpSHtB39gcz/AS4nvw87Vd/5kC7La0duunoJlBn8AqOtYWMpMT9xmisWKmjIB8NaKGJu7ZnQYxMI+2+vEVw891PWeHlsR5+N/o+s99cIacFxsvxSGQbR/1egw9qQq6lcYx+8wz4bVgIlisaKmTABOzF4FK8tGh0Es6lNdbvzasXuG3LvPP4nBwAND7q2lit1tdAi7Vgz6YXNYayamIuiZAAR0u9eL2FkHXDbrvK7MqOkSgM6NRbTlrJW9E/M42+HBr48Zu9P7bPxtw+u7q63stM5UrdRhvWNles4AuHg34p5O3e63nYHgkNEhWF5TJQCcLGJ07rrRYRCLGm/z4h+cvG10GLCxDUy0v2t0GKqyUgLg788YHcKe6ZkAAMCRyHE4OIeu93xWyBlBl7/PsPs3i6ZKAI4s3oWrQbtCyd4djfjwX0+Yp0Nkn/8x4p5Fo8NQjcjZ0OCNGzB2q253ItieNjqMPavonADwnA1HomNgDCjFa2cdOBYdowJAKmiaBMBXzWMw+cjoMIgFDYd8+J2zt0zXZPRc+9tgmObZy2KFWYBKmzXXlPWeAQCAmLsd5xIvw6djU542TxwXul6By0bdANXQNAnAidlrYJXmebMk+ugL+PC75+6AZcy3Sz3gyOBI6KbRYaimZIEEwNal33l6tYgyD0G2G3Jvn92Ps4mXMRQ6jKAzDI5Vv7SMjbUj4ophtO0kTrRNwG7g0kOzaYpCQN3pecTyKaPDIBbT6fPh98/fBc9KRoeyrROxS5gtHDLkCU9tFZMnAArDIDJgvQ3ERveRYBkW/cEh9AeHoCgKymIJdbEKRTnYnBrLAC6bFy5evxoHrcbyCQAvCTg+f8PoMIjFxD1efP3CPdhN1NBlK083BL67/IbRoRxYzh0yOoQXKgb88Dut9yCxWBowOoSPMAwDr80Hr81ndChkFyy/BHBk8Q6cjarRYRALibo8+MbFB3Dx1qhP3+9/hJjLek+mz0uFOiCxnNFhbEtKmG0XyO4sFgeNDoFYlKUTgEAlh8FVfUq1kuYQcnrwjZcm4bFZq0HUidgHRodwYBLLYy3YYXQY2/L1Wq9vSFnwYaPWZnQYxKIsnQAcm78JxkIlRomx/A43vn5hGgG79WaMEp4FxFxJo8M4sJVwt9EhbKludyLUsW50GHu2WDTP9D+xHssmAP5KHu05678hEn147U587fwsIhZq8PK88diHRodwYMlQArIBZ8d3YtXjfwslmv4n+2e+n8RdGkrq26iFWJfb5sTvnltGu8fa5XU7PAtoc68YHcaBCLwd64F2o8P4BCse/2tIDqQqxpfkJdZlyQTAIdTQnZ4zOgxiAU7ejt85m0S3L2d0KKoYj1p/FmAlYq5lAIVhELFY9z8AWCr1Q1Es+RZOTMKSr57B1cfU7Y/syM7Z8d9OpDEQsF5t9+3EPYtody8bHcaBJENdploGKPr9sLusNwMwVxg2OgRiceb5KdwlTpbQvzpldBjE5Gwsj390OotDYett7NrJWPSS0SEcSN3mwFy7edaupQ7LvQ0iU2vDkonO/xNrstwrv2d9BnbRWr26ib44lsNvnixiNGq9ad3d6PAswm+33pG1Zz3qOg5Jg7Kx++Hrs9738traS0aHQJqApRIABgqGVqjhD9kex3D49bEqTrdbe7PcTkZCd4wO4UBqNiemOg4ZHQbqdgdCiTWjw9iTlXIPkuUeo8MgTcBSCUA8swxvrWh0GMSkWIbFf3a8gYuJ5mmju53BwANwjHl7GOzGZOcRNHhjmtg8VYlZr8fC9bWXjQ6BNAlLJQDDdPSPbINhGPytIzJe654zOhRdOLga+vzWroIpcDY87jxmaAx8l7WWE2fyh5GpxYwOgzQJyyQAoVIGkULzbegiB8eAwVdGWPxc37TRoejK6ssAADATH0bVbkwRHoVhEB2wTjExWeFwc/2C0WGQJmKZBGBo5YHRIRCT+sUhHl8YtPbT8H7EXEmEHGmjwzgQieVwv+eEIfe22vG/h9kxlAS/0WGQJmKJBICXRCSyS0aHQUzoZ/sc+OWR1l0aaoZZgIVYH+bb9D/SJnVYp/tfoRHC7fVzRodBmow5zuHsoD2XpMI/5BNe7Xbh7xy9a3QYhurzP8bl1KuWrwh3q38CgXIWwbJ+R/K8fXnd7nUQDcmBHy5+AQ3ZYXQou6TAxjbAWCe/gijzkBXztqrWiiUSgI4MPf2Tjzuf8OA/H71tdBiGc3A1xN1Llj8WJrEcLh36FF6//Vewi9q3am7YHIhY4LSIorD48fLPo9AIGh3KC8U9i+j0zCPqSiHkWIOds1a7bUVhka1HkKm1Y62SwHT+MBRYKIPZJ9MnAIyiIJ5r7jPdZG9OtXvxm+O3jA7DNHp9U5ZPAACg4vDg6vBFXHjwYzDQts13OeZBRNM7qONK6hVT/9vauTrOtL2DweB9o0M5EIaREXauI+xcx1DwLg6FbuEnyZ9Fvh42OjRNmX7eMFpYg02HJwJiDcejPvzWaRr8n9Xjm9Z8wNRLKtiBh93HNb8P323+43+Ps6N4mDVmg+RuhJ3r+GL//2n5wX8rEVcKv9j/f2Mw0Nybz02fACRo+p88MRL24bfP3jQ6DNNx8hXLtwl+1sOu40iGuzS7vgLzd/9brXThcuo1o8PYFstIeDnxPbhs1m6x/SIsI+F8/Ifw25ujk+hWTJ8A0Po/AYD+gA+/e+5WC6zK7U+Pr7kaZF0aeRkLsX5Nrl3y++BwVzW5thryjRB+vPQLkE28sfNE9BKCjg2jw9Acx4p4KfH9pplhe555X2EAguUMXI2K0WEQg3X7fPj9C3fAMc35Q6iG3iZLABSGwbWh83jceVT1a4sd5t3tvVLuwXfnfgV1yWl0KNtycRUci1wzOgzdxFxJ9AWas86IqRMAmv4nHV4vvnbxLmystevea81tKyHqMve09n7c6zmBW/0Tqu7I9vSZc0r3QeYk/mbhl9CQzH3cL+JaBcu01rHsmLN5ltieZeoEoCOzbHQIxEAxtxdfv/AATk40OhRLiLubM2GeiQ/j8qGXIbEHf3Jv2BwIm6z7n6xweD/5WVxJvWKJo2cRp7m+f3qIuJqzDL1pEwBPrQR/xZyZOtFe2OnBNy4+gsdGJ0B2q93dvAnzSrgL7x19HQ3+YE/H5ZgHjImWkmqiG9+f/zKmcsY2RdqLcJMOhi8SdqwDTbgPwLQJQAeV/m1ZAYcbX784Bb/dOnXazSDmTjbtZiUA2PDF8Nfjv3CgzYFcp3kSykwthr+c+yrWqgmjQ9mTmmBM8yYj1SQXYIHZmb0ybQIQzbfeNBMBvHYXvnZhFmEnbf7cKztbR8hp7eZAO6nbHLg2dB7vHPssCu7Anr5WAYPooPH7JBqSA9fWXsZ3534FZcFndDh7lq63Gx2C7jK1NqND0IRpKwEGaPq/5bhtTvx35xbR5m7es8Vaa3cvt0S/+A1/DD8aewODyUc4sngXnLzzPpGS3we/O6VDdFuTFQ4Ps2O4kz5r6l3+O8lUm//19bw0JQD6sYkNuOtlo8MgOnLyDrx1bgWdPms0aDGrNvcyHmTGjQ5DFzLDYjJxBEvRXozNXtvx1JAYN+7432zhEG6sXWyKdr7Zegy5eqQl6gAAgKRwWCgMGx2GJkyZANDTf2txcHb89tk19Pn16wTXrNpdzXlc6UWqdjcuHfoU/JU8+tam0b0+B7v4yVK/RnT/W6104VrqZWzUmmfaXFZYvLfys/hc3//TEscBb65dRL4RMjoMTZgzAdCxJSgxlo2z4bcmMhgONvfatdYUMChVAygXfPBKRZQ4660tH1TBHcDtvlO42zOORHYJvalptOU31/wbvB3hTn1OSeTrYSyWBrBQHES6GtflnnrbqLXh3sZpjEavGB2KptaqCdzPnjQ6DM2YMwGgGYCWwLM8/qvxPI5FjFuXtaJnB3tpwwZuVYZrpQa2IcODOryHCyiFWi8BeEpmWSxFerAU6YG7Xkbv2gxCfBoRRpuTRQoYrFc6sFgaxGJxwPSte9Vyc/0CGoodJ6MfgmWar1DXbP4QLqVeh6I03+7/p8yZANAMQNPjGA5vnijjVHvS6FBMTQGDYjWISsG75WAPbDHVXSvqH6hJVRwePOgeBQDcnDyDiCuFqCuFqDOFiDMFO7e3roCKwqIquVERPCgJfqyU+7BU7H9yTKy1KGBwLz2BpeIAXur4QdNUoqyJbny4+joWikNGh6I50yUAjKLAX6WNYM2MZVj8vdE6znVQrYdn7Wew34q3SgnAViqiF5WiF4vFwY9+z2fLw8lX4OBqcHA12J9+ZOtoyE5UBC8qogcV0YOq6EVNdFmiWp+e8vUwvjP3K3DyVYQda4i4UvDYSpaqmyNIDmzU27BRbUNR2NvxUiszXQLgq+bBys2/saRVMQyDv31Uwitd80aHYqiPDfZpO7iUDNdKdc+D/VZoBmD3ikKgpd7wtVQTXVgRe7FS7jU6FLJLpksAAmVa/29WDBj8p4eAn+mdMToUXW052C9XwQoHH+y3QjMAhJDdMF8CUKH1/2b1xWEOnx94ZHQYmtJ7sN+Kq1EBJ0uqNM8hhDQv8yUANAPQlD43YMd/MvzA6DBUpSgsirUAKvkna/YpRffBfjueWmnPpXIJIa3FdAkAbQBsPq/3OPG3Dt8zOowDMfNgvxVvrUgJACHkhUyXAGxVwYtY18VON/7e8TtGh7Ennxjsn+7GN+lgvxVvtWB0CIQQkzNVAsDJEp0AaCITcS/+yxO3jA7jhRSFRaEaRLXghbTBW3Kw34qnRg2VCCEvZqoEwCYJRodAVDIa8+IfnjLX4P/iwb5mdHiqctBMGiFkB+ZKAMSG0SEQFRyO+PCPz9w0NIaPDfZpG7hVCa5kcw72W+EpmSaE7IASAKKqgaAPb529pWutNFlhUXzhYN/8A/7zKAEghOzEXAkAvWlZWo/fh98/fwcso10N0K0H+ypYQWnZwX4rNkk0OgRCiMlRAkBUkfD68LULd8Gz6nUF+2iwzz+zZk+D/a7QDAAhZCfmSgBoCcCS2txefP3CKWqqfQAAIABJREFUPTi4/T917jzYk73gaQaAELIDSgDIgURcHnzj4kO4bbt/4vzYYJ+2gUs9P41PDoqjBIAQsgNzJQA0bWkpQacb37gwCZ99+yNnTwf7St4LecvBngZ8LTBQwEsiRM5UP+KEEBMx1buDTaQEwCp8dhe+dmEGQWf1o9+TFQ7FauCjwZ5PSnAmq2AlBV4a7HXHSwIlAISQbZnq3cEm0RKAFXhtHH7v/AJsihPJlQjkDRrszYiXaRmAELI9UyUAjKLd8THyYgLLomCzocTZUeJ5lDkOZY5DheVQ4VhUGQZVdnNYr/PAv/twEJwMcABYbH7k4gCnAHIiDqcsw1urI1itIFaqoK1UQLRaMfYv2WKorDYhxuMYEQ6uBjtXg52rQ5R5NCQnGrITDclhaGymSgAEzmZ0CE1HYhgsuLyYc7uxZLdhjWNQhoyGIqMBCYIiQZBFyMqzg4X45NcTypNf8jN/vA2G4zA88jKKDIP15/5MkWXIggBFEABBACeI4BsCHI0GguUKBtNpHE+twkEb2FQhsZzRIRDSshhGRsiRhs+e3fZzGpIT6WocgmxMImCqBECkBGDfPj7Q25HigA1FQFGqPhncK5sDuMYPhU5fEGC2rgPIsCw4hwNw/PTFLm1GhgqAFQDvKAqkShlcroDEo0kMV2s4UszBJalXX6BVSLT+T4ghHFwVUdcqePbFy9p2roYO7zzy9Qjy9YhO0f2Uqd4hBJ4SgJ3sbqCvaD7Qb8cZCB7o6xmGAe/xolQp4cdKAT92AozTAz/vRhw29AkyhitlHCrlwdGS0QuJNANAiO44RkSbexkss7uHFgYKgo40JIVHqRHQOLqPM1UCQDMAW5v0BnDd58UkD6TEMiTlyXOzgQP9dlz+kCrXqeYyH/23AgV5sYw8gEcM8D0PwHr9CHEeJBQO/Q0Rh8ol9FWKuvYgMDuJNdWPNyEtIexK7Xrwf1bIsYaa6IYo6zcOmuodgvYAbMraHbgSCOO+g8eCUkVNqgNKAbDAKUlnQP0EYCuyImNDLGIDwB0OgB+wBUOIcG70SizOFwo4VMqrEosVySwLZZulGEKINty2Itx8aV9fyzIyws41rFU6VY5qe5QAmIDAsLgVCOOWx4UZRkRWLAEovXCznRnxDid4h/PA11FkCfXi3gdvQRaxKhewCuCSF/AE23BIceBcsYzRwosTimYj0tM/Ibpz8eUDfb2TL4OBAkWnuUxTvUuILbQHYM7tw1W/H49tDJJiCZJSB6TtK+pZgdN/sPX/p6r5HBTl4GsbZbGK66jiuhtw+WIYggtnyxWczG2ARXPvH6Dpf0L0Z2cPVv+EgQIbV0dDOviD1G6Y6l2i2WcA7vpDeNvvxZxSQ0WqAbDGtP5uqTX9X9th+n8/qlINd1DDHSfg6AxjgHFjolLHmWwavArJhtnQCQBC9MVAgZ07eDE7B1ejBKBZ5Hk7fhCN4RovIi9WADFndEiacaq1ATC/ocp1tlOXGniABh7YgX/fEUQf68GpWgPns2k4muS4YTP+LBFiZhwrAirMLHKMfk+FpkoAmuUUgALgWjCCH3tdmJWKkJWC5dbz94xh4PSrcIRFUVDNbV84Q22CLGBSzmGSB/6kzYce3oczVQGvbKTAWviYYdnhMToEQsg+6Ll111QJgMDbjQ7hQNYdTnw/HMFNVkBZqgIt1N7Y4fGBVWHauV4qQDaoEqCkSJgVcpjlgb9MxHBGtuMX1lPwWLBJVcXpNToEQojJmSoBkFgOMsOCtdCarAzg/XAMP3HbsSgVoSiFzfJ2LUa1DYAarP/vR1mq4m1U8ZM2L46xPnx+I4PO6v6O9xihQjMAhJAdmCoBAICy0wtftWB0GDtadrrx/XAYd5gaalINEFu7+51q5//z5kgAnhJlEbfkLG4HWPRGu/GzhQrGNd6joIayg2YACCEvZroEoOAOmjoB+N7wCK5ywHIuCcitW2jmec02A/A8BQrmhBz+FxcQ8XfgUzXgM+lV05YjrjhpBoAQ8mKmSwDyniA6NxaMDuMTvjNyGPdGBmALBOESBbhuN1DNmv9JUC9298EHHEloQKybfyZlQyjhzzjge4kozsgO/Hx6DX7BPPs9FDC0BEAI2ZH5EgC3Ok+SalAAfPvwUTwcHoDN78fTMwosb0PX+Hkk715DaX3VyBBNgbM7wKjQeKZRsc4aO7BZW+Ad1PBe1I0jXDs+n8mixwR/h5rdBZlhjQ6DEGJypksACiZIABQA//HocUwN9YP3+bDV4USGZZEYnUDq4W3kV8w3Y6Enm9OlynWEysHKaBpFUiTcFXO452fRHenCL2cKGCwbt4xF0/+EkN0wXQJQcXggcjbwkv5HrxQAf3p8DDODfeC93p2/OQyD9iMnwNnsyMxP6RChOdlcblWu07BoAvCUAgULQh7/0s/geLAbv7qWMmRpoOT0635PQoj1mC4BADaXASLFdV3v+WdHj2NqZBC8x7Pnb0p06Ag4ux3rk/c1ic3sbE51EgCrzgA8T1EU3JFy+FrMi1clF76wtqLrZsGMN6LbvQgh1mXKBKCgYwJwvaMLPzw5Cj4UOtA3I9QzCM5mR+rBLSgm3RmuFV6lJYCGhc7Z70ZDFvADRsDlznZ8uSziTDaty30zPkoACCE7M2UCkPeoUFJ2B2tuL/79mQlInR3gVeqb7u/oBmezY+XONShy61QDUmsJoFlmAJ6XF8v4Iwfwo+4u/Go6i86qdn9PkbOh6Nb+54cQYn2m3Cqs5UZAkWHxv5+awB9//g3IXQkwKg3+T3mi7eg6eR5sC7U2VmMToFivQW6SRjzbmRPy+GaQxx8lulHlDn5qYitZb1i3XuKEEGsz5wyARgnAjWAYP3jpHOzBELR5+93kCobRffoilm98CLFR1/BO5sCrsAegWZ/+nycrMq4oOdyNh/AzDRveWE+qev2MN6rq9QghzcuUMwAiZ1O1kIkEBv9bRxf+nVPEyoMbaJSLql17Ow6vH90TL8Pmau4jWZzNDlaFp9mGhtPiZlSV6vhzroSvdSdwT6U2ygCt/xNCds+UCQCwOZWphkWXB9/oiuOykocCBY1KGQtXfoLSmrpPXluxudzomXgJDl/zHstq9RoAB5UWivg3bgn/qqsbGw7nga+XpRkAQsgumTYBWA11Hvga343F8c9DdmyIH99dLksiVu5cRXr6IaDxjn3O7kD3qYuqNcsxG161GgDNdQJgLxQoeCzm8E8jHrwXbtv3dcpOL+o2h4qREUKamYkTgASUfW7QK/I2/IuuLvwFV4Yob99bPjM3ieVblyFr3O+d5W3oHD8Hh9en6X2MoNYMgM3lafrlkp005Ab+L3sV/7arGwK79x/NtL9dg6gIIc2K+8znf/0bRgexFYnlEcun4K7vbWr4ejCC/ynkwpq0u68TqmUU15Jwh6Lg7do9PbEsB28sjtLaquYJh5787Z2qzG54IjGEuvsR6hmAOxKDw+MHZ7NBkeWm+n7tRkqu4VIgjEGRQXAPlQQfdI+h5Gq+JJMQK2AZGX579sDXqUsu1CR9HoZMeQrgqWS4E9HC2q4+V2IY/B8dXbgq56FIe5vWF6plLF59F+1HT8LX1rGfUHeFdzjRefI8Fq/9BFLDPN3j9oPleQQSPfC1J1S+rg3uUBTu0E/XsmVRQDWXQTG1glJ6FbK4/axOs8iKJfwPAQ5veBP4hbWVHT9f5HisBeI6REYIaRbmTgBCXRidu7Hj5xVsdvzLeBRrQm7f95IlCck7VyEOH0WoZ3Df19mJ3e1B1/h5LF57H7JkvYHM5nIj2N2PQKIHLKfPy4flbfBE2+GJtkORZZQzayilVlBKp5o6GZAUCX/JFvGwuxtvJlfhecFMSCrYAXkfywaEkNZl6gSg7PSi4A7AX8lv+zmrThf+VTSAgqDO0b71yfsQazXERo6pcr2tOHwBJE6cwfLNS1BkWbP7qMkViiDUPQBvtB1QuXjSXjAsC280Dm80vpkMbKyhtLaC0nrKkgnVbkwLOXyjPYS/WxQwWth6ijEZ7tY5KkKI1Zk6AQA2TwNslwBMeQL4w6AdVVHdI2TZxRmIjRriR0+C0eipyh2KouPYKazcvab5SYT9YlgWvvZOhLoHTHmUkWFZeGNxeGNxKJKEfHIB2fkZCLWK0aGprixV8S0Pg5c9Xfhqculjtf5khsVqSN2lGEJI8zP9nOFKuGvL378ejOBfBzhUpZom9y2mVrB885KmU8zetg60HxrT7PoH4W3rQP/FzyB+dNyUg//zGI5DsKsf/Rc/jY5jp+Dwmj/mvVIUBe8qefyz7k6sP1MzIB1og8C1TulpQog6TJ8AZL0R1OwfP2r2drQd/6tLhCBruzu8kk1j8dp7EOvaJBkAEOjsQXTwsGbX3yub043OE2eRGJ0Ar0JhGt0xDHzxTvSeexWd4+c+tpmwWSSFAv7gmZoB2yXJhBDyIqY9BvgsX62IYDkDAPjT9k58mytDgT7T5lKjjtJaEp5IGzi7XZN7uIIRyKKI2jbru3pgGBbh3kEkRk/D3iT1CuxuD/wd3fBE2yGJAhpNVG1QUiTc4UQsBdvBRg9BohkAQgxlxWOApp8BAIDkk6qAf5Toxg+Ygm6D/1NCrYrFa++hVtj/KYOdxEaOwR835knOFQyj5+wriA4dAaNRlzojOf1BJEYn0HfuVbjDzTUjcEvM4YPMHUhKc3dSJISozxIzAGWHG3/FFnFH0m4A3okiS5sFg8JRzabGPbE46oUcBJ0a43A2G9oOjaLt0KimRZDMgrM74O/oht3rQy2fbZpTA1Wxgkw1jTZ3HBzbfAkcIVZAMwAaeZR7jMeCcdPjT8migKUbH2o2E8AwDBKjE3AF1GmE9CL+eCf6zn8agUSP5vcyG19bAn0XPo1w37Bmpzz0lq/ncDn5Hqpi852AIIRow/TvfsvFBcznp40O4yOyKGD5xoeoF7evTXAQDMchMX4Wdo826/A2pxud4+cRP3ZKsz0NVsByHKKDh9F77jV4IvtvwGMmFaGMK8n3UWwUjA6FEGIBpk4AMtU0HmzcMTqMT5BEAUs3PtAsCeCeNA/i1JyWZxiEegbRd/41eCIx9a5rcXa3B53j55AYOwObU53OhkaqizVcS76PbG3D6FAIISZn2gSgLJRwa+0qFJMWyZGEJ0lASZunLZvThc4TZ1SZonYFw+g98ynEho825SY/NXhjcfRdeA3hvmFDKx2qQZBF3Fi9hFQ5aXQohBATM2UC0JAauLF6+YWtfM1AEgQsXdcuCXD6Q4gfO7nvr7e53OgYnUD36Zfg8AVUjKw5MezmskDXyQvWrIHwDEmRcWf9OhYLc0aHQggxKdMlALIi49baFctsZpKExpMkQJ1eBM/ztSX2XCiI5XhEh46g7/zrmnY3bFbuUAS9516FJ9pudCgHoigKHm7cxXT2kdGhEEJMyHQJwL30LeRqxu/43wtJaGD55oeaVQwM9w3D37GLZi8Mg0BnD/ovfhrh3qGm2eFuBM5mR+eJs2gbOW757+NMbhIPN27rXj+DEGJupmoGNJ19hNXSstFh7ItYr2H55iV0n34JLK/+t7X9yBjEWgWV7Cc3d7Ecj0CiB8Hufthc5tjIJgkCKtk0Kpl1VDLpLWsbsBwHhy8Apz/40S+bS5/zr7sV7O6HKxRB8u41NMolo8PZt8XCAsAqOBw6YXQohBCTYP7gD6+Y4rEgWVrG3fUbRodxYO5wFJ3j58Aw6j81SqKAxSvvflTSlnc4EezuR7CzFyxvfClYWRRRWF1CIbmIWjG/ry6HLG+D0x+ENxaHrz0BzmaOo4qKJGHt8V3kVxaMDuVAjrQPo8t9yOgwCGk6PCug0ztz4OsU6mFk6/qc1DJFApCtZXB99UPIimx0KKrwx7sOtHnvRYRqGakHt+FPdMPXntAk0diraj6D/PICimsrUCT1StIyDAt3JAZ/vBPeaNwUJxiKqRWkHt6GLGrbiEpLp7vHEOZbrwAUIVqyYgJg+BJARSjjVupq0wz+AFBYXQLvdGnS5c/m8qDr1AXVr7sf1XwG6akHqOYymlxfUWSU0ymU0ymwHL85KxDvgiccNeyonq89AYfPj+WblyBUrbFR9XnXF2/jfJ8DXsbamxwJIQdj6OOjKIu4mboCQW4YGYYmMnOTyC/PGx2GJhrlElZuX8Hi1fc0G/yfJ0ubywvLNz/E3KW3UVhd3tcSgxrsbi96Jl6G0x805P4HpQC4vHANNVhrsy0hZsaYtGbNixiaADzO3EdZsO7Gqp2kHt1BOZ0yOgzVKLKM9cf3MHfpbZTWVw2Lo1EuYfXedcx9+CMUkouGFIvi7A50n7oIbyyu+73VIMkyLi9dhsA0788fIXoJlrM4/+hdo8PYM8OWADaq61guWntDFQDwThccHh9Y3gaW58E9+bj5ywamSbqz1Yt5JO/dQKOsTb2D/WhUyli9fxMbs4+fHJXs0nVPBMNxSIxOYG3yHnKLs7rdVy11QcDVlQ9xLvEyWMXahY8IMQInSziyeBtDyUcQnDYAxrR03y9DEgBRFnBv/ZYRtz4Q3umC88mxNYcvAKcv2PwNdRQFmfkpbMw8hmLSfRpCtYLUg1vIPEkEAoke/fYIMAzaRo7D5nRjfeq+YcsS+1Wq1XBz/RJOxi6CUYw/SUKIVbTnkhifuQJ3XZ/27VowJAF4uHEXdUmbojlqcwUjCHb1wh2KNf9g/5xGuYjV+zc1a3+sNqFWRerhbeSW5xE/ckLX8sehngHYnC4k792AIqt3EkIPG8UiHtiu4GjwHKA0x4wVIVpxCDWMzV1HV9r6e7x0TwDWKqtImrzYD8vb4O/oQrCzV7O2vGamKDIyc1PIzE1Ckc351P8i9WIe81feRai7H5GBw2B1Oj7obetAt8OJ5VuXIQnW2ti6nMnAZb+JfvcpQLF2MyRCtNK7NoPR+Ruwidb6+d6OrglAQ2rgQdp87X2f4h1ORAYOwd/eaYoz50aoFXJIPbilWYMj3SgKsgszKK0l0XZ4DJ5Imy63dQZC6DnzMpZufGi5Y4IL6XW8/oqMudnWfO0Tsh1vrYiT05cRLawZHYqqdE0AHmzcQUOq63nL3WEYBDt7ER08okkZXyuQGg1szD1GbmnOcuvYLyLUqli+eQm+9gRiI8fB2x2a33OzVsNFLF57D2Ktqvn91NAW8OO3fvEN9MQi+EDJ4tEcnQ4ghFVkjCzfx6Hle2AtOBu6E91Gu9XSMtZM2J/c7vEhfuQEnIGQ0aEYQhYFZBamkVuYhSyZu/3yQRRTKyhvrCM2dBSBTu2r4NmcLnSdvICla+9BbJgw6X3Gyf5e/MbnPgO3YzM5OjsWRDrXwEauOaY5CdmPcDGNU9OX4avmjQ5FM7okAHWpjocbd/W41e4xDCL9Iwj3DZminK7eFElCdmkW2bkpSBYua7sXsigg9fAWyukU4sfGNe+fYHd70HnyApauv2/KPQEMw+DL5yfwpfMTeHbVn2MZvHYmgr94O4WG0HxPPYS8iE0ScGz+JvpTU0aHojldEoD76VsQZPMMMgzLIn7sJHxtCaND0Z0iy8ivzCMzO2n6J1OtlNKrmL/8DhKjZ+Dw+TW9l8PrQ+fJ81i6/j5k0TwzLF6nA7/xuc/iRN/WsyE+D4+XT4Xxw0tpnSMjxDiJjUWcmLsGZ8MaS3cHpXkCsFxcRLpino0TLG9DYuwM3KGI0aHoS1FQWF3CxswjCBZZl9aSUK1g4eq7aD88Bn9Ht6b3cvoC6DxxDss3P4SsYrOk/epri+If/eIbiPlffMKlp8OFY0M+3JsyT/EnQrTgalRwYuYqOrLmPqGmNk0TgJpYxePMPS1vsSe8w4nO8XNweLV96jOb4toKNmYeWbqfvRYUWcbq/Zuo5jJoOzQKhtVuKcgVDCMxdhbLty4ZdrSSAfBzJ8fw1ZfPw8bvbqf/6WNBrGcbWNtozdki0twYKBhYncTRhdvgJfPMUutF0wTgwcYdiLI5pj15pws9p18C73QZHYpuKtk01ifvo15s3k0sasivLKBWzCMxOgGby63ZfdzhKBKjE1i5fVX3qooRnxdv/tyncbS7c09fxzLAa2ci+PMfraJWp/0ApHkEKjmcnL6MUGnD6FAMo1kCkK1lTDP1z/I8Ok+cbanBHwCquQwN/rtUL+axcPkdxI+dhCeqXZtcT7Qd8WMnkbx3Xbfjlp86egi/9vrLcO2zkqXbyeGV0xH84IP1Zjohui/uegntuVVE8yk4xDpYWUbd5kTWG0Yq2IG8pzVPE1kJJ0s4vHQXwysPLNnBT02aJQBT2YdaXXpPGIZBx/GJlpv2B4BAogcbs4+b6ly/liRRwPKty4gOHEa4f1iz+/jaE5AlEakH2vbD8Ltd+PuffRUTg/0HvlaizYnRET9uP7J4gah9YBUZHZkl9KemEMtv3d0zkVnEsYVbWAvG8aBrFBlfVOcoyW605VcxPnMFnhothwIaJQDpyhpyNX36xO+k7dAoPJGY0WEYgnc44Y22G9q614rSMw8hCQ3ERo5pdo9AogeNUhHZxRlNrj8x2I+//zOvwu9Sb9Zr/JAf8ytV5IutsVbqrpfRl5pG39o0HMLuepe05VbRllvFXPsQ7vacgMC3Vv8Qs7KLdYzOXUfP+pzRoZiKJgmAWZ7+Qz0DCHT2Gh2GoQKdvZQA7EN2cQaSKCB+5IRmnQWjw0dRK+VRzaq3Bhn2evGrr17E+ZFB1a75FMsyuDgewnffNcfSnhYYKGjPrqA/NYX2bBIM9jd71peaQjyzhNv9p7Ec0b7wFNlez/osRuduwC7SRtbnqZ4ArJZXUGwYP01od3sQHTxidBiG84Rj4J0uy5SkNZNCchGyJKLj2ClNTggwDIPE8dOYv/Lugf99OJbFz58+gS+dOw2HTbsCR+0RB0b6vHjcZKWCnY0qetdm0Lc2BXddnR4OTqGGs4/fw2poDrf6J1BxaLfBlHySp1bC+MwVtOXpAWg7qiYAChRMZx+pecl9azs8pumxrt1SFBmKLEORJMiSBFkSIYvixz8+/W9BQKNSRL1cQmzwCHzxve3Y3hLDbO4FmDHHv4vVlNaSWJEuIzF6RpMGUZzdgcToBBavvbfv44GjvV34tdc/hY5QUOXotjZxLIDF1SqqNeNrGhxULJ/CQGoSHZklzTaExbPLiBZSuN89hpmOESigbotaYhQFwysPcHjpLjiLtebWm6oJwEpxERWhrOYl98Xf0Q13SPtNOIoso1Epo1EuolEuov7ko9SoQ5Y3B/79bsBbm7wHb1uHKkkMbQY8mPLGOpZufojOE2c1KR/s9AfRfngMq/dv7unrIj4v/varF3F2WP3p/hex21icGw3i7SvWPD5lF+voXZtFX2oK3po+RY54ScTY3HV0p+dwY+AsnRbQSKi0gZPTlxGo5IwOxRJUSwBkRcZsflKty+0bZ7MjNnxU03tU8xlkF2ZQXl+FotGgKjXqKKZW4O/oOvC1eIcTnkgbyumtdzCTnVVzGSxe/wBd4+fB7fM43Yv4O7pRK+Q2uzHuwGGz4XOnxvDFs6dgN6h7ZV+nG92LFSyuWmdpKVxMYyA1ic6NBcM6u4VKGbx+53uY6jiMB92jkFhqvawGXhJwdOE2BlYn971voxWp9u6xVJxHVTD+zSA2fBScTZudt7V8FmuP76FWyGpy/eflludUSQAAINjZSwnAAdWLeSxefw9dJy+AdzhVv35s5BjqpQKqua1P0Nh4Dp8dO44vnDkJv9v4mhbnT4Swmq5BEM37hstLAnrW59CfmoLfJE+FT6eoOzcWcWPgDNaCcaNDsrSO7DJOzFyFq6HO3o1WokoCIMki5gvGd05yh6La1HVXFGzMPsbG3KSu0+i1fBb1UkGVGgaeSBt4hxNifXfHmcjWGuUSFq+9h56Jl8HZHapem2FYJEYnMH/5nY/9O/Eci9eOH8EvnT2NkNej6j0PwuPicOpIAJfumGNgfVawnEV/ahLd6/PgTFKN9HnuegkvPfgRFqN9uNN3CnWbuq+nZudsVHFi7hoSG4tGh2JZqiQA84VZ1ARjj1gwLIu2w2OqX1eWRCzfvLTtU5nWcktzaFfj78Uw8Mbiu5piJi8mVCtYvnUZXacuglV5YyBndyAxNoHFa++DBfDykRF8+fwEojs07jHK4QEfppcqSGeNb3fMyRK60vPoT01Zqrxrd3oO7bkk7vSNYyE2YHQ4ltCfmsSx+VuwtWD9fjUdOAEQZAGLBW2KmexFuG8Ydre6T0eKoiB556phgz8AFFeXERs+CpY7eK7mDscoAVBJrZBD8u41dI6dUb1OgMMbwMT4KXz1xAjiwYCq11Ybw2wuBXz7beOWl3zVAvpTU+hZn4VNND4R2Q+7WMfpqUvoWZ/DjYEzKDvNmfAZzV/J4+TMZYSL1KZaDQceVeZyU2gYnIXZPV6Ee4dUv+7aw9sob6yrft29kCURxdVlVQoauUNRMAyj2cbFVlNOp7D2+C7aDo2qcj1ZFBCTa/jSSDc6/dYpHhMN2tHT4cJCUr89QKwiI7GxiP7UFKKF5ilMFMun8Jlb38WjrmOYTByBzBh/lNkMWFnG4eW7GF5+AFbnRlrN7EAJgCSLWC7NqxXLvsWGj6l+5j8zO4n8yoKq19yv3NKcKgkAy/Nw+kOo5s1RprkZ5JbmYHO6Eerd/1E8uV7FkB348vE+eOzaFfHR0qmjm7UBtM4t3fUS+lPT6F2b2XV5XqvhZAlHF26jKz2PGwNnW76vQLSwhpPTl3U7stlKDpQApCorECRjN9g4/UF4Im2qXrOwuoT0jDnKGQNAvVRALZ+FM3Dws8PucJQSAJWtT90H73TB157Y09cxlSImwm68MToEVqNyw3oJ+mzo73JjZlH9ndiMoiCeXd4sz5tLqn59s/JX8njl7l9jNj6Eez0nIHLWTA73yy42cHz+BnrXjF9iblYHSgDWa8tqxbFv4T51u7ZVsmnNu7TtRz65qFICENssCkRUtXr/BniWLa8LAAAgAElEQVSHE65g+IWfJwsNBKUaXu2M4tRYc5WqHj8cwNxSBbJKswDORhV9a9PoS0237BEvBgoGVjcrFd7qn0AyrM6xYLPrSs9jbO56087ymMW+E4CyUMRa0didtg6vH96YemdoFUnC6r0b+y7JqqVKRp1NL85ACCzHQzZ45qbZKLKMlduX0T3xMuxu73N/JsFZK2M84sVnjvXBxjXnuq7fw2Oox4PH8werBtqWW0V/ahId2eWW79f+lKtRxflH72Il3IXb/ROo2o2vA6EFd72M8ZkrLTXTY6R9JwAlxfh/ILV7tm/MTZr2nLxQLUOsVcE7D/aDzzAMXKEIFQXSgCQIWL55abNGgM0OtlrCIZ8db4x0w+9sjbawJw4HML1YgbTHaYDN8rwz6E9NUa/2F0hkltCWT+FezwnMxoeapq8AoygYSj7CkcXbVL9fR/tKABTImE4bu/nP7vHCF+tQ7XpCtYLswrRq19NCJZtWpdCRJxyjBEADNo7D4bYwDnM1TPS1oSugQVEqk/O4OIz0efBgZneDeKSwjv7UJDozi4aV57UaXhJwYvbqR30FCm5zHxXdSbCcwcnpywiW9amwSn5qXwlAg9tAuW5s4Z9w37Cq56/XH98z5dT/syoZdRIAdzimQjQE2GzIc6KvByf7e3Gsp1PTVrxWMTbix+R8GaK09SyATRLQvT77pDxvXufomke4mMbrt/8Kk4kjeNR1zHJ9BThZxNGF2xhcfUxLPQbZVwKwXDT2eJzN5YG/XYVWuU+UN9ZRSpu/Z3Qlq84+ALvHS2WB94ljWQx3xDHe34Px/h50RyNGh2Q6LieHIwM+3JksfOz3Q6UM+lNT6ErP0TSvSlhFxqHle+jcWMDNgTNYD7QbHdKuxLMrODF7Fe668d1jW9meEwCFq2EmZez6f7hvSLWnf0WRsf74rirX0ppYr6FRLsHu8e78yTtwh6IorC6pEFVz4zkWPdEIBuJtONLVibHeLrgdVLN9J8eHfXg4W4LcENCVnntSnpeOn2rFWyvi5fs/xHzbAO72nkSDN+eeE4dQw9jsNXRtmKPGSqvbcwJQlJOGNlu0OV2qdcgDgNziLBoV62w6qmTXVUkAXKEIJQDP4VgWneEQBuJtGGiPob+9DT3RCPgm3bWvJYedxdmEgtCf/CnVa9dR79oM4tll3Ok7jcXowYuHqalvbRrH529atlxzM9pTAqBAwaPUrFax7EqodwiMSuUxZUlEZm5SlWvppZJJI9jVf+DruEOtXV3M63Qi6vehKxLCQHsbBuJt6I1FYOdV65Dd8joPx9EATfXrzSHUMTH5PrrXZ3FzYAIVx8EfGA7CWy3g5MyVpirZ3Cz29G4n23LIlY0ryME7nAgk1KuRnluagyRY6+mkkt3YbEl8wCUQm8sN3umCWNOvfrteGIZByONB1O9F1O9D1Pfko9+HqM+HqN9Lm/V04HbzyJ6agPfKZaNDaUntuSQ+e/M7eNA9iqmOw1B0rjbJKjJGlu/j0PI9OuFhUntKAJJlY9dtQj2DqtX8V2QJ2QXrlZiURQH1UhEOn39PX6fIMhShDlaWwCsyHAzw6ROjcMjmT4AYADaeh8PGw87zcPA87Dbbk49P/v+Z//a7XeBU7g1B9sd+fhygBMAwnCzh+PxNdD/pK5D1vrhSpVoixXWcnL4MX7Ww8ycTw+w6AWB4AY/njSv9y9nsqjTEeSq/vACpYexRxv2q5jZ2TAAUWQJXqyBmY3A04seZzjhcNpreJvqKdgax3h6HI2X+UzbNLFDO4tW738d0fAT3e8Ygsdq8F9jEBo4v3EJfakqT6xN17fpVUEEKkoHTOKGeAbCcOudcFVlGZt66L9BKbgPB7q33ASi1MkZcHL54tNeyneVI82AYQHjlJTj+w/9rdCgt72m1vURmEbf6z2A1tLfmVTvp3FjA2Ow1OKl+v2XsOgGYyyxqGccL/f/t3XeQXOd5JvrnnM5xZrp7uicHAINBBkhEggokrUhZokRTDrKviTVXrmtf26Vd27Vrey1bu6u1Ldsq171eS+sVSoBtrdd2eVeUHFaSxSCJCTnHyam7Z6Zzjuf+MQBEEANgwun+Tp9+flUoFamZ7zwkgTlvf+H9ZKMJrT0Dqo2XDE439Bn4XPzeOxgsuRTe29mGx/qGBSQiuj/Pjn4U/pcBUoUbArXAXsjisWuvYtbbhwuDe5E3Wdc1nq2YxZ6xU+iIib8cjlZnRQWAbCpgJiLu4p/WngHIRnU+zSqKguhE4376B4BKsYhiJgWzwwUln8EHO1rwmM5uliP9sFmNiBw4CNcbr4uOQm/THZmCPxHCpb49mAhsXPX3S1CwIXgD26YvwMjLxRrSigqALBZqneO+ZKMJbX0bVBsvFZpBKd/4V4sWYosYthvxM/s2Q5b1cSEI6Zf1wC6ABYDmmMpFPDJ2An2L4zi74QBStpVtLm7JxPHI2Fts7tTgVlQATEbFTe34Ng7DYFKpq5WiNNy5/+X0t/vwq+9+BF6X2PO9RCvl63JjvqsbljlOE2uRN7mAp87/M0Y7N2O0cxg5s33Zr7MX0hiau4bB8Aj79+vAQwsA2VTE1KKYGQCL043W7gHVxkvNz6GYbeze0weGNuD//uBTPMdODaf0nsdh+Z9/KzoG3YesVDE0dw0bgzcQ8nQjYW9F2uqCBAWOfBqt6Sg64nN88evIQwuAnMDpf//wTlVv/Gv0tf9PHNyLHzt8QCc3gFOz8W3rRdZoglzWfu+JZiYrVXRFptEVEbfxm+rjod1SpgTt7HR39MDWql7TivRCCIV0YzalkCUJv/T0+/EcX/7UwCwWAzKHDomOQUS3PLAAkI0lTC7Uv3+zbDTCN7RN1TEbee3/p99zGI8NbxIdg2jdLI+q++eaiNbugQVAXl4UcvOfd3AYRrN6V65mIgvIJ+OqjVdPT+zYig89ukt0DCJVeDrcqFrWd+6ciNTxwAIg6LKifdM2mGzL7witBYfXf98ud2sVnbih6nj1MtzdiX/11HtExyBSjcEgIbuLBS2RFtx/E6CkoOp0o63Fg7a+DchE5hGfmUAmUrslAYfXj65d+yGpuPEvF4sgF2+8s6rtbhc+89EP8i560h1p2xAvCCLSgPsWACVnHrLh1id/SYLDF4DDF0Apl0F8ZgLJuWlUVNzNe+flr/Itbotj11Udrx4kAL/8kQ/AbbOJjkKkutYBP3gOgEi8+xYAKdPyf0RNNgfah7bDt3EL0gthZBbDyETmUSkV1xTA7HDCM7AZ7kCXqkf+ACAZmlm2b77WHRzehI0dftExiGrCbjchNDAI28S46ChETe2+BUDa+OCXsSQb4Ap0wRXoAhQF+WQc2dgiCukkCukUStk0lPs0jDDZHLA4XUvf71f/xQ8AlXIJCzevqD5urRlkGT9++IDoGEQ1Vd61A2ABQCTU8gWApKBitT+8ScCdr5dgbWmDtaXtzt9SqlWU8jko1QqUahVKtQpJlmF2uFS71vdBFkeuolIs1Pw5anty51YEWltExyCqKfumXtERiJresgVA2V6AbFzf+rMkyzDbHesaY63yyRgSc1NCnr0eZqMRnzi4T3QMoppr8zsRdbpgTKdERyFqWst+yE9a1raerwmKgvC1i0AD9qv+wJ6daHXU78glkSiSBOR27xYdg6ipLVsApFd0R6A2xWbGUUglRMdYk8Nb2O2Pmodx6+rvoCci9SxbAFSsjfkptFzII9KAx/4AwOd2ob/dJzoGUd209fmgSOxzQSTKPX/6So48ZGNjXjU7f/0iquWy6BhrsnfjgOgIRHVlsRiQG9osOgZR07qnAEibG3P9f+HmFaQXQqJjrNnejeq2PyZqBJXtW0RHIFq3ksOEzHsa746Le1b7G3H9PzY5itjUqOgYa+awWrClu1N0DKK6s/QGREcgWrOqQUb6oB3tQ7OwQ8Z4qLH2cd3zui9bbSs//68ByeA0FkYar+HP223t7oJB5RbIRI3A3e5EXnQIojVIbXPC/cgiui1LM8/5SuO1br+rACjbC5BNjTONkVkMI3z1vOgY6+ZxiemXQCSaxWxAoqsHlrkZ0VGIViTbZYPxsTy6WsZER1m3uwqAtLUAoDEKgFw8irmLp+/bbriRtDlYAFDzKg5tZAFAmldymVB43ICOrilIaPz3DvCOAiBjaIB/KEVBbHoci6PXoFQrotOogs1/qJnJfd2iIwihSBKyA3ZUeiRIliokYxXVtBHSAuAYz0Eu6uPnW6OrmiSkDzjgH5qBR27MTfL3c1cBkM/nYHS21ORyHjUUMymErpxHPhkTHUVVrU7OAFDzsnd6REeoq5LLhNx+M7w9YbiN4Xu/YDNQPGTBwnQXbCeLMKV4ebIoqR0utOyeR7dlmf9OOnBXATB79RwqshEtXX1o6eqD0aqNTQ2KUkV0YgTRiZtQqlXRcVTHGQBqZu42GxJmE+Sivl90BY8Fpf0y/J2z8MgP7ldiNhTQPTCOYq8FC5d74D7FOxPqKdtjh+lQFl3uxj1dthJ3CgDZWEShVAZQRmT8BiITN+HwtKOlux8Orx+SoF3q+VQC4SvnUEgnhTy/HlgAUDOTZQm5jUNwXG3s0zz3k+u0QXm0Cr9/FpK0ug8wZkMB3btGER3wo/J9K6xhnpmopaLbhNK7DAh0TOpmnf9B7hQAiuEdv7EUBZnIPDKReUiyAfY2LxxePxw+P0y22k5Zl4sFpMNzSIZnkU/oa7r/nQyyDJdNGzMtRKJUBgcAnRUAmX47DI8U4fdMrnssj3selacNCI32wflaDnJFfzOhIlVNMtKH7AhsnIZR1vdM1NvdKQAKSua+X6RUK3eKAdwATDYH7B4frK4WWFxumB1uyAbDuoJUyyWk5oNIheeQjS025G1+a9HqsEObOy6I6sfco5+GQKkhJyy70+hwT6g6rkGqoHvTOFLdrci80QbHxP1/ZtPKJXc50bZrHt3mxu0ku1Z3CoBMMb3ibyrlMkjMZnDnzj1Jgtlmh9nphtnuhMFshsFkgdFshsG09EuSZZSLRVRKBVSKBVSKRZSLBVRKBZTzOWRjEV2u7z8MjwASAVaPS3SEdVEkCekdDti3J9Blr+35cJctDudTCYRm+2B7tQRDvjHvPxEt02eH5VAG3c7GP8+/VncKgER+HZtMFAXFbAbFLCvS1Wp1cv2fyOYwIyc6xBpUDTLSexxwDy+iy1q/neISFHR2TyL3SQdiZzrgusxNgitVaDWj/LiEjsCE6CjC3SkAFpOJB30d1QhnAIgAs1lG2maHIZcVHWVFqmYZ6b12tG0Ko9skburYZsrAdnAU8xu7YXhVgjmhr3PqaqqaZKQP2xAYnGmqdf4HMQKALFcQy/DTuwhel1N0BCJNKPoDsE2Oi47xQGW7Edl9VvgG5tBi1M6asd83i9LHzZi/0gv3Sc4GvFNyjwttO0NCizUtMgKAYiyIztG0dvb3iI5ApAllnxfQaAFQbDGjsN+I9p5ZtGn006PJUET3zlFEB9pR/p4NNh4ZRGbAAcvBFLod+j7Pv1ZGAKigMabd9MbrcmLA3y46BpEmKJ420RHukfdZUN4nwd85C4PUGK15Pa4FVJ82IDjWB+cPspArzXGi6u2KrWaU3w10tGuzoNQKIwDkKpz+F2HfxkHREYg0Q2p1i45wR7bLBuytwN8+05ANYWSpgu6NS0cG02+0wTneHD/jq2YDMoetCAxMw/CQbot0qwBIFrhmJMLeTSwAiG4ztIg/CpgedMC4J49A2/qb92iByxqH68k4gsP9sL5ahDHXGLMYa5F81AXP9iBaTEHRURqGEQCiOm6zq1UOqwVbujtFxyDSDJNb3JHY1LAT1l1pdLr0OWXc2TWJ3HN2xM52wnVJXx/4MoMOWA8kuc6/BrcKAH39hmgEB4c2wiDofgUiLbI6Lajn59Oi24T8Rgucw/GaN+/RApspC9uBUUSGAyhccsB1feXN37Qos8EBeWcBHV59Fm31YISkIF/S5q5WvTIZDfj4wb2iYxBpit1mxOwuF1DLNXeLBIO/CFdrAl7LfO2eo2HeljDwOJDa24pM3I1KygikJKCq8b0OMgAHIDsrcHoT6LA2538/NRllWb9rQlr1wT27eP6f6B1kWULH3omG2W3f6FzWOFwdcaBDdBISRYaBOyXryWG14GP7HxEdg4iImpysgAVAPT1z4FE4rBbRMYiIqMnJVYkFQL20t7jxgd07RccgIiKCXAU3ANaDxWTCv/3oh2AyGkRHISIiglxROANQaxKAX/zQj6Cv3Ss6ChEREQBALld5fWStPXf4APax6x8REWmIXKxwCaCWDg8P8cw/ERFpjrFY4QxArezbOIif/8CTomNQk1EUBcViEaVSCeVyGZVyGZIsQZYNkGUZBlmGbJAgSUt/bTKZIEmS6NhEVGfGYpkzALXw8YN78dzhA+CPVVKboijIZDJIp1NIp9LIZNIolZZe+KVSCZXK6hrpSJIEh8MBp8sNt9sNl8sNl8vFooBI54z5UkF0Bl0xGQ34+Q88icPDQ6KjkA4oioJkMol0KolUOoV0KoV0Og1FUa9tq6IoSKfTSKfTCAXnAACyLMNud8DtXioKvD4fzOba9q8oFCswsQsgUd0Yc0UuAail1WHHrz7zYWwI+EVHoQZWqVQQjS4isriIxcVFlATc1VGtVpdmGNIpzM3NQpIkuN0t8Pv9aPcHYLGoXwxks3m0qD4qEd2PsbzK6UK6l8lgwPt278DHDz4Kp9UqOg41oHw+j8XFRUQWFxCLRVX9hK8GRVGQSMSRSMRx8+YNuN0taPf70d7uh81mU+UZxTxvJSWqJ6PMK2nXTALw+NbNeO7wAbS7XaLjUAOpVqtIxGOIRqOIRiNIpxvratZkMoFkMoHRkZtwudzo7u5BoKMD6/l5UilGVUxIRA9jlLnRZ9UkALsG+vAT7zqI/naf6DjUIDKZDKLRCKKRKOLxKKrVquhIqkilkrh27QpGRm6gq7sb3d29sK5hJkwu8XpXonoyGjgDsCIGWca23i7s27gBezcOoM3pEB2JGoCiKAgG5zA1NYlcNis6Tk2Vy2VMTU5iemoKXm87enp70NbmWfH3myuzNUxHRO/EGYBlGGQZrQ472hwO+FvdeGSwH3sG+2G3mEVHowYyNzeL8fExFAvNddJGURQsLs5jcXEeDocDPb29CAQ6YTA8+B4MW3UKPDdLVD9GWar/DIDbZkOnpxWtDjta7Xa47Ta0OuxLG+gE/QAwGQxoczjQ6rTDZbPx5xCty9zcLK5fuyo6hnCZTAbXr13D+NgYevv60d3ds2whUCxVYZEiAhISNS+jLNf+Ved1ObGluxNberqwpbsLXZ7Wmj+TSJTI4iJuXL8mOoamFItFjI7cxPTUJPr6+9Hd3XvXhsFsNg+3wHxEzahmMwB2iwXv3rYZT+3chh7vytcBiRpZpVLBlSuXNHeMTyuKxSJGbt7E1OQk+voH0N3dA1mWeQSQSACj2u0+N3UG8CM7t+HQ8CaYjUZVxybSusXFeZTLvGL7YZYKgRu3CoF+GJWM6EhETUe1GYABfzt+5r2HsbWnS5XxiBpRMBgUHaGhFIsFjNy8gd72KsA9tkR1te5TAK0OO3788YN4z7ZhXh5CTU1RFMSibGazFn7jRdERiJrOmgsAWZLwo/v24JmDe2E1mVSORdR4JEmCLMu6afBTL1arDU6ZMydE9WaU1rAE4Lbb8EtPvx/be7trEImocbEAWD2rlR8giERY9QzAcHcnfvnp97MTHtEyjEYjNwGukt3IDYBEIhjNxpXvvHl672785LsOge2DH27pOtU0UqkkkskkSsUCJEkGJECCBEmWlv5XkmC2mOFyueFyuVW7WY3E8AcCmJqcFPJsk8kEs8UCi9kCi8UMs8UCs8kC6T69PnLZLOLxGNLptNBji17DdWHPJmpmRqtxZZd2fPLwAXz84N4ax2lc+XwOsWgUyVQSqWRyzT9UTSYTXC7XUkHgdqOtzQMjj1M2jN7efsxMT9d0GUCSJNhsdrjdS0Wj2+2Cw+l6aKvd+6lUKkgm4ojHE0gkYkgkEnVbxjCZzPijNx34uR1ebGplJ0CiejKaZctDv+gTB/fy5b8MRVEQiSxidnYGsag6d7iXSqVbV8Qu7SaXZRk+XzsCHR3wen08aaFxZrMZXV1dmJmZUXVcq9UKX7sfPq8P7paWNb/sl2MwGNDm8aLN4wWwNHuVSqWQSMQRi8YQi0VqNkMgmx2YTqbxH99ow+72bvzcjptos+Zq8qx6qioSplKtiBesSBbMcFuK2OULQZbYIIq0w2jEgwuAj+5/BM8dPlCnOI0jHAphbGwE+Xy+ps+pVquYnw9jfj4Mk8kEvz+AQEcnWlpaavpcWrsNG4eQzeYRjS6uaxyXywVfezvafX44nE6V0j2cLMtoaWlBS0sL+vr6USjkEQzOITg3p/rv90i+BGCpmD43n8GvvtKPJ/pkfGrLdRjliqrPqodQ1o1vjPTjdLiKbOnt/66MMBmG0eOy4oMDUTzeNSUsI9Ft0u995VXl2zdeWvb/fP/uHTjy1LvrHEnbUqkkbt64jkQiITSH0+XC4OAG+HztQnPQ8qrVKq5dvYJwOLTi7zGZTGjzeOHxeOH1eGC2PHx2rp4URUEsFsXc7BwikQVVlgm+N5fCjYV7/x25zDY8O5TB+/rH1/2MesiWzPjzi9twJryypb/tPhf+nz1X4TI3102RepSvGHE23IlrUS82tHWse7xkwYNYoT4/16X/8t/eUL4z+n/u+T82dQbw2R//ODf83aIoCsZGRzA9PaWpPu8ulxuDGzbA6/WJjkLLmA+HEAyGlp1Gl2UZbrcbHo8XHq8PLpdLUMrVKxaLCAWDmAvOIpfNrmkMs8WKPzvx4AZAnU4nnt8exHbv/JqeUQ+vTg/ir685kCmtbunCbrLi1/aHMdS6vpkiqq943oZT8524sujEeFJCNJdFVanCZTbhJ7YOrHv8ehYARlSNMBpklCs/rObtFgt+6en38+V/S7FYxOVLFxGPx0RHuUcqlcSF8+fgdrsxuGETPB5evKQl/kAH/IEOFIsFxGNxSLIEg8EAo8Gwro17opnNZvT196Ovvx+xWAyTE+OIxVbXBVEy2R/6NcF0Gn/wlhtbfZ14YecI/DbtHBlczDnxpfNDuBFNAlj9voVsKY8/eKsdn9lrwA5fWP2ApIqZVCvOzPtxNWrDdLKKROF2waud34trZQQAt82OaDp9529++v1PoN3dOJ9GaimRiOPSxQsoFouiozxQMpnE+XNnEOjoxNDQZpjYnVFTzGYL/IGA6Bg10dbWhra2NsRiMUyMjyIej6/o+xazpRV9nQIFVxbT+Hev9uBd3Ub89LbrsBrE9lr4+sgWfHNUQrGSXNc4hUoRXzzVil98xIB9gTmV0tFaKQBuRNtxdt6HGzELZtOlW3s5KgDSD/nuxnNPAfAju7bjwNAGoaG0IhaL4uKF86hUGmczUjgURCwawfDwVvjauT+A6mepENiHWDSKsfFRJB+yT+ZyaHXT+uVqGa9Ml3EiNIyPbizgRzeMrCfumkwkPfjyuT7MptW7vrhULeNPzzrx6V193BxYZ6WqAecXAriw0IbRuAmhTB7FSglA8dYvfTMCgMOy1HzGZbPip959SGggrWjEl/9txWIRFy+ehz8QwJYt2xp2mpkaU5vHg70eDyKRCCbGR5FM3vsp2Wgy4VLw5prGz5by+JtrwEtTu/B/bVvEI/7af3KuKBL+8sp2vDJVREVR7+V/Z/xqBX9+3oJceQPe1zem+vi0JFMy43S4ExcXWzCekLCYzaGiVADkb/1qLkYAsJmWmgE9c+BR2My8kzMWizXsy//t5sNhlIol7Nq9BzL3c1Cdeb1eeL1eRBYXMTY2ivTbPjWbLOtvJb6QzeCLp2wYatuDF3aOo9tZm5M5Fxc78JULfkTztV3zrSpV/MVlGfnyJiGzG3o0n3XiVLgDVyMOTCUVxPJZKFCgh/V7NRgBwGKwwud24X27d4jOI1w+n8eli43/8r8tFovi2tUr2Lad/21JDK/PB6/Ph4WFeUyMjy21yF7Z8v+K3Iyl8FvfD+BAZx+e334NDpM6g2dLJvz3i9twOpyBotTnhaEoCv72moR8eRjPbWaL5NUai3txdsGHa1EbZlJlpIs5LK3s62/9Xg1GADDJFjz32H6YmnyqWFEUXL1ySXeXuYTDIbhbWtDT0ys6CjWx9nY/2tv9mJ8P4+/euKTq2BWlgjfmsjg7P4QPDVbw7NB1rKdn5vdn+vG1q05kSvV/cShQ8OJIGdnyNvzstit1f36jqCgSLi8GcG7Bg5GYGXPpAgqVIoAyAPWXafTICACtFi/etbVfdBbhJicnVryDudGEgnMsAEgT/P4ALga/VZOx8+UCvn4T+N70LvzU1hgOdU6v6vujeTv+7NxmXI+mIHpN+DsTBRTKO/DpXeoWS40qVzbh7HwHLi62YixuwHw2h3K1DKBw6xetlhEAqmVz0/eYTyYTmBjX7+abVCqFXDYLm/3hZ6+JailXqCCVq+2UejSfwX89a8Y/je3BkR1T2NDy4B4F5aoB3xjdhH8ck1GsaOfT4/dmcsiVd+FXHr0gOkrdRfN2nA534ErEhYmEgmg+h6pSBbC2xlN0LyMA5PIVlMoKTMbmLAIqlQquXL6kqQ5/tbC4uIjevj7RMajJRZP1+2Q9nkjhd1/zoMvVhyd6U9gbCKH9VjOhYsWI8UQbXpsL4M25KnJlbX6KPBnK4Asnd+PX959f17KG1s2kWnF63o9rETumU5W3Ndzh+n2t3LlnNpUpwdPSnCcAbt68gVyu8W8ge5il3a9EYi0k6vsDXYGC2VQKX7sCfO1KBwySAWaDEflK8VbRr/0/+xcX0vj8m3vwGwfPw6CDGwUVANeifpyb9+JmzIKZVPFWAVYB1+/r504BkEiXm7IAKBQKCAXr34Hr9pJLPWcd2A+AtCAUE3uRVjk5sZwAACAASURBVEWpIFduvFM+16MpfO713fgPhy7BLLgT4mqVqgacm+/AxcU2jMSNCGXyKDVRwx2tulMAJNON9RtKLXOzMzV9CbvdbrS3B9Du98Nqtd6116JSqSCVSiGVTCKZSiARj6NQqN00pJEFAGnATDQiOkLDGk+k8NnXd+B3HrsMm1HFs5QqSxUtOD3fgcv3NNzR/mxLM7lTAMSS2v3NVCvVahVzc7M1Gdvj8WB4yzZYrdb7fo3BYEBraytaW1sBLM0GLC4uYGZ6WvWLh2RZRpvHq+qYRGsxtajdm/0awWwqhd9+bRt+57FrmrlOOJxx4nS4A1ejDkwmFcTZcKch3CkAFqLa+I1UTwvzYdUv+TEaTdi+Ywc8a3jZSpJ056x0Op3C9PQ05sMhVe5d7+rqhpldHkmwqgJMRnjz3XqFM2n89mvD+OxjI/BY678rfiTuxbl5H67HbJhJlZAu5sGGO43nTgGQyVWQy1dgszbPNPHMzOrOCD+Mw+HEzl27YbPZ1j2W0+nC1q3bsHHjJszNzWJ2ZnrNxYosy+jrH1h3JqL1SmWKKOuky6ZokVwGn31tIz772Dj89tq9eJca7nTg3EIbG+7ojPHtf7EQK6Kvc/0vr0aQzWSWvaRkrbxeH7bv2Kn6Rjuz2YyBgUH09PRidPQm5mZXt2QhSRK2bN0Gi8Wiai6itUhkmm+msZYShSx+5/UB/NbBGfS41GliliubcGa+E5cWW9hwR+eatgBIqXidp9Vmw7btO2q6y95oNGJ4eCs6Ap24fv0qMpmHr60ZDAbs2LkbHo+nZrmIVqNcWf9yFt0tXczhc2904V/tcOFw1+pnNW833LkccWIyAUTy2Vsbo9lwR+/uLgCaaB9AJq3OlJkkSdi+fQeMRuPDv1gFLa2t2H/gEBYW5jE5OYF06t5CxmAwwOvzob9/AE6nqy65iFai3IDH7xpBvlzAl89ZcCK0C58cmkS36/5HLadTrTgd9uNadKnhTvJOwx1u2Gs2d721IvEiFAVohq7AK/kEvRJ+fwBud4sqY62UJEnw+wPw+wOIxWLI57IolysoV8qw2+zwtbfzzD9pUokzADWjQMHpUAZnQu3ocg2ix1VFwF6ELCkIZ8yYzxowl2bDHfqhuwqAUllBPFVCm9skKk/dpDPqzAD09IptrdvW1ga0tQnNQLRSJW4ArLnbnQ9n73q/N2efF3ow+Z1/YyGq/65MlUoFeRVa/9rtdrjdbhUSETWHEpcAiDTj3gIgpv99AEWVuu1xfZ1odTgDQKQdTTkDIMnqbHJwOByqjEPULDgDQKQd9xQAiXQJpbK+N+pI0j3/2GsbR1ZnHKJmUWQBQKQZ97zBFAVYjOl7FkBWaQagUubGGqLVKFX4Z4ZIK5b9CKv3ZQC1ZgBK5ea7QIloPVoc978ci4jqa/kCQOczAJJKjQ6SCbH3mhM1mk4PN84SaUVTFgAGgwEm0/p7HaTTadVvEyTSs0AbCwAirVi2AMgXKkhn9b1Wp9YRvngspso4RM3AYTWh1WEXHYOIcJ8CAADmdb4PwOl0qjJONBZRZRyiZtHr4+VURFpw3wJA7ycB1CoAYtGoKuMQNYten1d0BCLCAwoAvXcEVGsJIJ/PI5fjtZlEK9XHAoBIE+5bAETjJVSrSj2z1JXd4YCsUiOfSITLAEQr1dfuEx2BiPCAAqBSVbAY1+8ygCzLaFXpFr3g3Jwq4xA1gz6fB26bTXQMoqb3wI/As+F8vXII4fO1qzJOOp1Cgj0BiFZEkiQ8unFAdAyipvfAAmBG5wWA16veVOTc7IxqYxHp3f5NG0RHIGp6DywAIvEicnn9Xt5htVpV2ww4Px9GqcTWwEQrsaOvG1bz+ptxEdHaPXQX3Oy8vmcBfD51ZgGq1SpCQe4FIFoJo8GAPYP9omMQNbWHFgB6XwbwBzpUG2t2dla1sYj0bv/GQdERiJraQwuAufk8dHwaEA6HA62traqMlctlEWVjIKIVeWTDAEwGg+gYRE3roQVAsVTFQlTfTYG6unpUG4ubAYlWxmIyYmd/r+gYRE1rRZ1wZkL6XgZo9/tVuR0QABYXF1Ao6LtgIlLLj+57VHQEoqa1sgIgnKt1DqFkWUZnZ5cqYymKgpnpKVXGItK74e4AhjrV24dDRCu3ogIgliwhk9PvcUAA6OruVm2sublZlMv6vk6ZSC0/efjdoiMQNaUVN8PXe1dAm82ONo8615SWy2UE53gigGgltvT5sKVjQHQMoqaz4gJA78sAANCt4mbA6ZlpKIqOj08QqeiTjx2GQeKJAKJ6Mo6cPraiL5wym/Hefb8Gg46P7fja22E2m1Esrv8SpEI+j/lwGIEOrm8SPcxQrwvbK3nEkynRUYjWRJIkvBp/yDFwRYH8kA+GlUoVlUp9ltyN5dLKPtmXSzmMj09g06aNNY4kjiRJ6OzqxuTEuCrjTU1NsgAgWgGDQca7Du3DN7/5D6KjEK2JAiDfYCvlK14CAIDr16/XKodmdHV1Q5IkVcZKp1NIp/mJhmglDh48AJ/PKzoGUdNYVQFw7Zr+CwCr1QqPV70fQvPhsGpjEemZ0WjEs88+q1oBTkQPtqoCYGFhAdForFZZNKO7W73uZPPz86qNRaR3GzYMYv/+/aJjEDWFVRUAQHMsA3g8HthsNlXGyuWySKe4DEC0Uk8//SG43W7RMYh0b9UFwKVLl2qRQ1MkSVL1foD5eS4DEK2U1WrFM898THQMIt1bdQEwNjaORCJRiyya0tnVBVle9b+eZbEAIFqd7du3YceOHaJjEOnaqt9wiqLg3LlztciiKSaTCe3+gCpj5XI5pNNpVcYiahbPPPNR2GxW0TGIdGtNH3HPnNF/AQAA3d3qLQMkEnHVxiJqBi6XCx/5yEdExyDSrTUVAOFwGMFgUO0smtPS0gKn06XKWM2wbEKktn379uq6+RiRSGte5D5z5qyaOTSru0edWYAkZwCI1uTZZ5+FyWQSHYNId9ZcAJw/f6EpLrvx+wOqNCbJ5XKq3DFA1Gw8njZ88IMfEB2DSHfWXAAkk0mMjIyqmUWTjEYjHA6nKmMlk1wGIFqLxx8/jN5e9Rp0EdE6CgAAOHu2OZYB2traVBknyX0ARGsiSRKee+5ZXd9GSlRv6yoALl26jFKppFYWzWptVakASCVVGYeoGQUCATz55BOiYxDpxroKgGKxiMuXr6iVRbNaWltVGSffaHdFEmnMk08+gUBAnf4cRM1u3a3ummEZwGQywWa3r3ucAgsAonUxGAx47rkf442BRCpYdwFw8+ZIU3S5s6tQAFSrVZ4EIFqn3t4evOtdj4uOQdTw1l0AVKtVnD9/QY0smqZGAQAAhQJnAYjW6wMfeD88Ho/oGEQNTZXbbpqhKZDd7lBlHO4DIFo/k8mEZ5/9hOgYRA1NlQJgdnYWCwsLagylWarNALAAIFLFpk0bsX37NtExiBqWOvfdAjh58pRaQ2mSzWpTZRzOABCp58Mf/hB7AxCtkWoFwKlTp3TdE8CoUi9yFgBE6vH5fDh48IDoGEQNSbUCIJvN4exZ/V4TbDAYVLsTgIjU8773/QisVqvoGEQNR7UCAABef/0NNYfTHKPRuO4x8nkWAERqstvteOqpJ0XHIGo4qhYAoVAIo6Njag6pKWpcSVoul5FKpVRIQ0S3HTx4AGazWXQMooaiagEA6HsWQK1bAefmZlQZh4iWWCwW7Nq1U3QMooaiegFw9epVxGJxtYfVBJfbrco44VAIlUpFlbGIaMn+/ftFRyBqKKoXANVqFW+8oc9ZAJdLnQKgUqlgdoazAERq6u/vg9/vFx2DqGGoXgAA+j0S6HK5VBtrbGwEySSvByZS0549u0VHIGoY69/WvozbRwIPHNDXlJzJZEJLSwsSicS6x1IUBZcvXcD+AwdhNK5tc2EkEkFkcQGxWAyVShmSJMFut8Pj8cLvD8DCo1HUZNra2kRHIGoYNSkAgKXNgHorAADAH+hQpQAAlpoCXbxwATt27lrVCYNUKoWRm9cRj9+71yKfzyMajWJsbBR9ff3oHxiELNdkoodIc5xOdTbqEjWDmr0Z9Hok0O8PqHoXeTwew6mTJ5BOP/xooKIoGB8bxelTJ5Z9+b9dtVrFxMQ4zp87i2q1qlZcIk1zONS5tIuoGdT0o6EejwSazWa0tal7DWk+n8PpUycxPT21bKvgUqmEYHAOp06dwMTEOBRFWfHY8XgMly6eVzMukWaxIyDRytVsCQD44ZHAtrbWWj6m7nr7+hCNRlQds1qtYuTmDYzcvAGbzQabzYZqVUG1WkEqlVrVS/+dIpEIwuEQAoEOFRMTac8MT9cQrVhNZwCq1Sp+8IMf1PIRQng8XrS0tNRs/Fwuh2g0ing8hmQyua6X/21joyMqJCPStgsXLoiOQNQwar477K23Tuiy9e3A4EbREVYln88jrcP/DkS3jY6O4dKly6JjEDWMmhcA5XIZr7zyaq0fU3cejwetrY115CgWi4qOQFQT4XAYf/M3fys6BlFDqcv5sLfeOqHLpjdbtmyFwWAQHWPFSuWy6AhEqrt69Rq+/OU/1+XPGKJaqksBoNdZAJvdjo0bN4mOsWLqHV4kEi8YDOKrXz2G48f/Arkcr9kmWq2angJ4uxMnTuKJJ94Lt0oX6mhFd08vFhcXEI1qf3rdarWJjkC0ZoqiYGJiEleuXMGVK1cQiWj/zxyRltWtACiXy3j55VfwzDMfq9cj62brth04ffok8hr/FNLmUbd/AVGtVCoVpNNpZDJZxGJRXL16DVevXkMmkxEdjUg36lYAAMDJk6fwxBPvrekROhHMZjP27HkEZ06fQrFYFB1nWW53C5ukkKaVSiWcOXMW165dw+jomGb/LBHpRV0LgNt7AfQ4C2Cz2bFr9yM4e+YUKpWK6Dj3GNq8WXQEomUpioK33jqB7373JV0eGSbSqrrfEnPixEnVLtPRGpfLhV27H1nVxT71MDAwCLdbX7MupA+FQgF/+Zd/ha9//UW+/InqrO4FQKVSwUsvvVLvx9ZNa2sr9u7bD7vdLjoKAGBgcAMGNzRW0yJqDrlcDl/60n/DlStXRUchakpC7ok9derUQ2+za2Q2mx179x0QuunOYrFg567dGBzcICwD0f1Uq1V87Wt/jVAoJDoKUdMSUgBUKhW8/PIrIh5dN0ajEbt3P4KNm4bq3iyoo7MTBw4+Bp+vva7PJVqpf/mX72JkhPdTEIkkpAAAgFOnTiMW0+8sAABIkoS+vn4cOPgYvF5fzZ/ndLqwa/cebN26HUZjXfd3Eq1YMpnE97+vv0vCiBqNYfeeR35XxIMVRUGpVMTWrVtFPL6ujEYjAh0dcLlcKJfLyOfzqo7v9XoxvGUrNm4a0szeA6L7+ed//j+YmpoSHYOo6Qn9mHj69Bk8+eQTaGtrrEt11srna4fP145isYhwKIhQKIR0em07n50uFzweLzo6OuFwOFROSlQbiqLg4sVLomMQEQQXALdPBPzYj31CZIy6M5vN6O3rR29fP9LpNOLxGPL5PAr5PPL5HPL5PIrFIiRJgslkgtlsgdlshtVqRWtbGzweD0wms+h/DKJVm56eZjc/Io0wAsgBENYk/syZpVkAj6c5ZgHeyel0wul03vP3q9UqJEmCJPEKH9KPubmg6AhEtCQnA5gTmWBpFuAlkRE0SZZlvvxJd7LZrOgIRLRkTgYgvCQ/c+ZsQ9ymR0Trw2t7iTQjKAMQ3omjWq3ipZdeFh2DiGpMURTREYhoSUgTMwDA0iwA7/cmIiKqi6AM4LroFMDtWQDuBSAiIqqD6zKAfxCd4razZ89hYWFRdAwiIiK9+wf5+LGjkwAuik4CLM0CfOc73xEdg4iISM8uHj92dPL2XQDfEBrlbS5evMSzwkRERLXzDeCHlwF9U2CQuyiKgm9/+9uiYxAREenVN4EfFgAnAFwVl+Vu165d52UhRERE6ruKpXf+UgFw/NhRBcBvikz0Tt/6FmcBiIiIVPabt975d2YAcPzY0a8DeFNYpHcYHR3DyMiI6BhERER68eatdz2AtxUAt/z7Ood5oG99iycCiIiIVHLXO/6uAuD4saOvAvjHusZ5gOnpaVy5opmtCUS0TtVqVXQEomb1j7fe8Xe8cwYAAH4OgGZ24L38Mu8IINKLRCIpOgJRM5rC0rv9LvcUAMePHZ0H8HEAmri3c3p6BmNjY6JjEJEKeOsnUd1lAXz81rv9LsvNAOD4saNnARwBoImru1555dWHfxHdl6IoSKdTSCYTSCYTqFQqoiNRk2IBQFRXCoAjt97p9zDe77uOHzv6d88feWEbgN+tUbAVu3HjJubmgujq6hQdpaEk4nFMTk4gHo/d9dKXJAmtra3o7ulFe7tfYEJqJplMBoVCQXQMombyuePHjv7d/f7PZWcAbjt+7OjnAPwKgLLaqVbr1Vc5C7BSlUoFly9fxJkzpxCJLN7ziV9RFMRiMVy6eAHnzp1BsVgUlJSaCT/9E9VNGcCv3HqH39cDCwAAOH7s6P8H4MMAYioFW5OLFy/xB8gKVCoVXLxwHvPh8Iq+PhaN4tzZ0yiVWARQbfHPL1FdxAB8+Na7+4EeWgAAwPFjR/8FwAEIbBdcrVbx/e//QNTjG8bNG9cRi63uB20mk8G5s2dRLguf6CEdi0RYABDV2FUAB269sx9qRQUAABw/dnQES0XAfwSQWlu29Tl58hTy+byIRzeEZDKBYHBuTd+bTqdw6eIFntOmmuEMAFHNpLD0bj5w6129IvfdBLic48eOpgH8zvNHXvhTLN0d8AsALKsZYz3K5TIkSarX4xrO1OTkur4/Fovi2tUr2LZ9h0qJiH4oGhW6ikikRwUAXwLwX44fO7qw2m9eVQFw260H/Zvnj7zwRQC/CuATAPrWMtZq2O12WCx1qzcaytLGvvV/wgqHQzAajRjaPMxii1TFGQAi1UwB+N8A/vj4saPTax1kTQXAbbce/BkAn3n+yAu7AXzs1q+9AFR/e7S1tak9pG5ks1nV1vBnZ2dQLBaxbfsOyPKKV4mI7qtSqSCRSIiOQdSoFACnAXwDwDeOHzt6Xo1B11UAvN2tQOcB/Kfnj7zgBNB961fXrV8BAIb1PGPTpo1DAJ5eZ1RdUhR11+4XFuZx7uwZ7Ny1GyaTSdWxqfnE43H09fbU/DkKlMrU1Myf1vxBRLVTARAGMHfr1yyA2VtL8KqSFEUTzf5WJBgK/zqAL4jOoUWZTAYn3npD9XHtdgd27NwFh8Oh+tjUPG7euI6ZmTXPVK5G4VOf+mlrPR5E1OgabX53QHQArbLb7TAY1jXBsqxsNoNTJ9/C5OQEGqlYJO0ol8trPp1CRLXTaAXAoOgAWrXU3rc2eySq1SrGRkdw5vQpZLOZmjyD9Gtudpb3TxBpEAsAHenr66/p+MlkAidPvIXpqSnOBtCKVKtVzMxo5nZxInqbhikAgqGwDKC2b7gG19rWVvPLfarVKkZGbuDsmdPI5TRxYzRp2M2b13kBEJFGNUwBAOBRADbRIbRueMtW2O2137CXSMRx8sRb9drYRQ1odnYGc7OzomMQ0X00UgHwQdEBGoHJZMLuPXtgNptr/qxKpYKbN67j7NnTyOVyNX8eNY54PI6bN66LjkFED9BIBcAHRAdoFFarDbt2P1KTUwHLicdieOvN13H92lXe1UDIpNO4fOkC94kQaVxDFADBUNgF4DHRORqJy+XCjp276tbOV1EUzM3N4s03XsP16ywEmlU4FMTp0ydRLPJ6aSKtU60TYI09CYDt6FbJ4/Fiy5atuHr1St2eqSgK5mZnEQoG0dnZhf6BQd7f0AQy6TQmpyYQDoVERyGiFWqUAoDr/2vU0dkFRQGuX79a1ynZarWK2dkZBINz6OzqRn//AAsBnclmM4hGo1hcWFDlIioiqi8WAE2gs6sLFqsFly5eqHtDlmq1itmZaQTnZtHV1Y0+FgINLZVKYnZmBtFohMf7iBqc5u8CCIbCGwCMis6hB+l0ChfOnxP6g1uWZXR196C/vx9mMwuBRnLt2lUE5zR/rI93ARCtUCNsAuSnf5U4nS7s3bsfDodTWIZqtYqZ6Sm8/toPcOniBUQjEe4WbwATE+ON8PInolVohCUAHv9TkcVqxaN79+HSpQuIRcWt2yqKgoWFeSwszMNqtaKzswudnV2wWPnhTWsURcH01KToGESkMk0XAMFQ2AjgKdE59MZoNGL37kcwNTWJ8bFR4Z/A8/k8xsfHMDExDo/Hg86ubvh87XU7wkgPFotFUS6XRccgIpVpugAA8AQAt+gQeiRJEvr7B+DxeHDl8iVks+L7+iuKgkgkgkgkArPZjI7OTnR1dsNmt4uO1tS4w59In7S+B+BTogPoncvlxr79B9HV3S06yl2KxSKmJifx5puv4+yZ0wiHQqhWq6JjNaVSsSQ6AhHVgGZnAIKhsAXAs6JzNAODwYDh4a3wen24dvUqSiVtdXGLx2OIx2Mw3jCho6ODRwnrrFRiAUCkR1qeAfgIgBbRIZqJz9eOAwcPIdDRKTrKssrlEmZmpvHWm69jYmKcMwJ1Ui6zACDSIy0XAJz+F8BsNmPbtu145NG9cDjFHRd8kEqlgvGxUZw+dYKb0+qAhRaRPmmyAAiGwi0AflR0jmbW2tqG/fsPYtPQZhiN2lwpSqfTuHz5ovBTDHpnNPIaDiI90mQBgKW1fy7yCiZJEnp7+3Dw0GEEOjpEx1lWNBLBxPiY6Bi6ZjKxACDSI60WAJz+15ClZYEdeHTvfni8XtFx7jE3N8tZgBpiAUCkT5orAIKhcAeWrv8ljWlpacHu3Y9g3/4D8PnaRce5o1gsIhJZFB1Dt4wsAIh0SXMFAICfAGAQHYLuz+VyY+eu3dh/4BD8/oAmOvZFIxHREXSLMwBE+qTF3V2c/m8QTqcT23fsRDabxeTEOMLhkLCp+CqXAGrGxE2ARLqkqRmAYCi8CcAB0Tlodex2O7Zu246Dhw6jq6sbsizgtxULgJrhEgCRPmmqAADwU6ID0NrZbDYMb9mKQ489jp6e3roWAlo9qqgHXAIg0ifNFADBUFgCcER0Dlo/i8WCoc3DeOzwu9DX1w+DofZbOtr9/po/o1mZTCyuiPRIS3+yPwRgg+gQpB6z2YyNm4bQPzCIcDiE4NwcUqmk6s+xWq1oaWlVfVxawkZARPqkpQLgF0QHoNowGo3o7u5Bd3cP0uk0gnOzCIdDql0y09vXr8o4tDwurxDpkyb+ZAdD4T4AT4vOQbXndDoxtHkYGzcNYXFhAcHgLGKx2JpPD3R2daOnp1fllPR2kiTBaDTy3gUindFEAQDg58Gz/01FlmX4AwH4AwHk83kEg3MIBeeQz+dXPIavvR3Dw1tqmJJuM5lMLACIdEZ4ARAMhU0A/rXoHCSO1WrF4OAGDAwMIhaLIRicxeLCwn1vobPZbNg0tFlT3Qj1TsjRTiKqKeEFAIBPAAiIDkHiSZIEj8cDj8eDUqmEeCyGdCaNTDoNWZbhdLrgdDnR2trGF1KdVSq8EphIb7RQAPyi6ACkPSaTCe1+P9rB431aUKlw+p9Ib4R+jAqGwlsBvFdkBiJ6uEqlIjoCEalM9Dwqj/4RaVylUrnvfgwialzCCoBgKGwH8LOink9EK5OIx0VHIKIaEDkD8CkALQKfT0QrEInyqmUiPRJZAHD6n2ppCsCnAbwMgPPX6xCNsAAg0iMhBUAwFD4M4FERz6am8fnOjsBXOjsCTwHoBfBvAZwUnKnhZLNZZLMZ0TGIqAaktbZgXY9gKPwigI/V/cHUFPL5/NTnP/97/6ZUKt3zyX/37l2dhw4dfHdHR+dhm83aB0ASELFhvPrKywiFQqJjrJgClM6du/jjKgxVBbAAYA5A8Pixo0UVxiTSlLoXALeO/l0Gf/BSjRw9+lXcvHnzoV/ncDgwNLQJQ0NDGBraBLfbXYd0jePkyVP4+7//X6JjaMUigCCWCoIZAK8A+Mfjx47GRIYiWg8RBcBXARyp60OpaVy/fh1f/erxNX1vIBDA5s1DGBoawuDgAEym5r0GN5lM4otf/JNV3c3QhMoAXgXwIoAXjx87OiU4D9Gq1LUACIbCPQDGADTvT1aqmWq1ij/5k/8X8/Pz6x7LaDRiYKD/1uzAEDo7OyBJzTFplU6n8Rd/8VeYmuL7bJXOAPgagD87fuwoKyfSvHoXAH+Mpc1YRKp744038eKL36jJ2E6n867lApfLVZPniDY9PYO/+quvIZFIiI7SyGYAfBbA8ePHjvIECmlW3QqAYCjchqWjWc66PJCaSj6fxx/+4R8jk6nPjvWOjg4MDW3C5s1DGBjQx3LByZMn8eKL3+S1v+q5BOA3jh87+g+igxAtp54FwH8A8J/q8jBqOv/0T/+M733v+0KebTAYEAgE0NnZgc7OTnR1daKzsxM2m01IntWKRmP47ne/i9Onz4iOolffA/CZ48eOnhUdhOjt6lIABENhG4BJALzAnVSXSCTwhS/8keYurGlpablTDNz+5fV6NLOXYGxsHK+99jquXLkCEceBm0wewAvHjx39H6KDEN1WrwLgFwH815o/iJrSt7/9Hbz00suiY6yI2WxGR8fdMwUdHQGYzea6PL9cLuP8+Qt47bXXMDcXrMsz6S6/B+C3jh87yoqLhKt5ARAMhQ0AbgIYrOmDqClVKhX8/u9/AalUSnSUNZMkCV6vF+3tPjgcDtjtdjgcdtjtdtjt7/xrO2R55Q088/k8ZmZmMT09jZmZWUxMTNRtnwTd14sAfub4saNp0UGoudWjAPhJAH9d04dQ07p48RK+9rXmmVWVJAlWq+WewkCSJBQKBRQKBeTzt/83j1Qqxel9bboI4GPHjx2dEB2Emlc9CoCzAPbU9CHUtL7ylaMYGRkVHYNoLWYA7D9+7Gjj9FomXanpZUDBUPgZ8OVPNbK4GMHo6JjoGERr1QPgfz9/5AWL6CDUnGpWAARDYRnAXHxDIwAAAoZJREFUf67V+ERvvfUWp7ep0R0C8OeiQ1BzquUMwKcA7Kjh+NTEyuUyz62TXvzs80de+HXRIaj51KQACIbCJgCfq8XYRMDS5r9sNis6BpFafv/5Iy98RHQIai61mgH41wA21GhsIpw4cUJ0BCI1yQD+x/NHXgiIDkLNQ/UC4FbXv99We1yi23K5HCYmJkXHIFKbG0uXCBHVRS1mAH4ZQGcNxiUCAIyOjnLzH+nVp58/8sKQ6BDUHFQtAIKhcAuAf6fmmETvdPMmz/2TbpkAfF50CGoOas8A/BoAj8pjEt1lZOSm6AhEtfTJ54+8cEB0CNI/1QqAYCjsB/AZtcYjWk4sFkMkEhUdg6jW/kB0ANI/NWcAfhOAU8XxiO4xMjIiOgJRPTzx/JEXDosOQfqmSgEQDIW3AvgFNcYiehCu/1MT+YToAKRv6y4AgqGwBODLAOpzoTk1LUVRMDrKAoCaxsdFByB9U2MG4AiA96gwDtEDBYNB3mVPzWTT80de2C46BOnXugqAYCjsA/CHKmUheqDJySnREYjq7RnRAUi/1jsD8EcAvGoEIXqYSCQiOgJRvXEZgGpmzQVAMBR+EsDzKmYheqBolMf/qOnse/7IC12iQ5A+rakACIbCFgBfUjkL0QPx/D81IQnALtEhSJ/WOgPw7wEMqxmE6EEUReEMADUr3q1CNbHqAiAYCj8K4DdqkIXovlKpFEqlkugYRCJwCYBqYlUFQDAU3gDgnwBYahOHaHmc/qcmxgKAamLFBUAwFG4H8C0AgdrFIVoep/+pibEAoJpYUQEQDIUdWPrkv6m2cYiWxyOA1MRYAFBNPLQACIbCJgB/D2Bf7eMQLY9LANTEWABQTfz/E4BW+TO5aCEAAAAASUVORK5CYII='))
frame = Frame(root)
root.geometry('700x400+200+100')
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
frame.grid(sticky=tk.N + tk.S + tk.E + tk.W)
root.title('Item Stats Tools')
# var
path_answers = tk.StringVar()
# Reemplacing Close window
root.protocol('WM_DELETE_WINDOW', closechungungo)  # root is your root window
# Creating Menubar
menubar = Menu(root)
# Adding File Menu and commands
file = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Archivo', menu=file)
file.add_command(label='Abrir Excel', command=open_file)
file.add_command(label='Guardar Excel', command=save_excel)
file.add_command(label='Guardar Reporte', command=save_html)
file.add_separator()
file.add_command(label='Salir', command=closechungungo)

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
analysis.add_command(label='Analisis TCT', command=dificultad_tct)
analysis.add_command(label='Test de Fiabilidad', command=alpha_cronbach)
analysis.add_command(label='Analisis IRT', command=irt_rasch)
analysis.add_command(label='Transformar Datos', command=transform_data)

# Adding Help Menu
help_ = Menu(menubar, tearoff=0)
menubar.add_cascade(label='Ayuda', menu=help_)
help_.add_command(label='Ayuda', command=None)
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
