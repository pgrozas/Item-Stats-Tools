import pandas as pd
import numpy as np
import base64
from os import remove


def wright(thetas, betas):
    image_text = ''
    intervalos = []
    for i in range(0, 81):
        intervalos.append((-4 + i*0.1, -4 + (i+1)*0.1))

    mapeo = dict(zip(intervalos, [''] * len(intervalos)))
    mapeo2 = dict(zip(intervalos, ['  '] * len(intervalos)))
    mapeo_desc = dict(zip(intervalos, [' | '] * len(intervalos)))
    #####descriptives
    media = np.array(list(thetas.values())).mean()
    median = np.median(np.array(list(thetas.values())))
    std = np.array(list(thetas.values())).std()
    descriptive = [media, media + std, media-std, median]
    i = 0
    for measure in descriptive:
        for intervalo in mapeo_desc:
            if intervalo[0] <= measure < intervalo[1]:
                i += 1
                mapeo_desc[intervalo] = ''
                if i == 1:
                    mapeo_desc[intervalo] = f'{mapeo_desc[intervalo]}' + ' M '
                if i == 2:
                    mapeo_desc[intervalo] = f'{mapeo_desc[intervalo]}' + '+S'
                if i == 3:
                    mapeo_desc[intervalo] = f'{mapeo_desc[intervalo]}' + '-S'
                if i == 4:
                    mapeo_desc[intervalo] = f'{mapeo_desc[intervalo]}' + ' m'
                break
    maximo = 0
    for theta in thetas:
        for intervalo in mapeo:
            if intervalo[0] <= thetas[theta] < intervalo[1]:
                mapeo[intervalo] = f'{mapeo[intervalo]}' + f'  {theta}'
                if len(mapeo[intervalo]) > maximo:
                    maximo = len(mapeo[intervalo])
                break

    maxstring = 0
    for beta in betas:
        print(betas[beta])
        for intervalo in mapeo2:
            if intervalo[0] <= betas[beta] < intervalo[1]:
                mapeo2[intervalo] = f'{mapeo2[intervalo]}' + f'  {beta}'
                if len(mapeo2[intervalo]) > maxstring:
                    maxstring = len(mapeo2[intervalo])
                break
    start = 80
    ability = 4
    image_text = image_text + f'N° de item:{len(betas)} N° personas:{len(thetas)}\n{"-" * (maximo + 35 + maxstring)}'
    image_text = image_text + '\n' + f'  Nivel de{" " * (maximo + 3)}  PERSONA - ITEM\n  habilidad'
    for valor in sorted(mapeo, reverse=True):
        spaces = maximo + 15 - len(mapeo[valor])
        if start % 10 == 0:
            num = ability
            if num >= 0:
                num = f'{" " * 6}{ability}'
            else:
                num = f'{" " * 5}{ability}'
            ability = ability - 1
        else:
            num = f'{" " * 7}'
        start = start - 1
        image_text = image_text + '\n' + f'{num}{" " * spaces}{mapeo[valor]} {mapeo_desc[valor]}{mapeo2[valor]}'

    image_text = image_text + '\n' + f'\n{"-" * (maximo + 35 + maxstring)}\nDescriptivos nivel de habilidad de individuos \nM=Media | +S=Desv.Estándar Superior | -S=Desv.Estándar Superior | m=Mediana'
    image_text = image_text + '\n' + f'\nM={media:.2f} | +S={(media+std):.2f} | -S={(media-std):.2f} | m={median:.2f}\n{"-" * (maximo + 35 + maxstring)}'
    ancho = maximo + 25 + maxstring
    width = 500
    if ancho > 75:
        width = (500 * ancho) // 75

    from PIL import Image, ImageDraw

    img = Image.new('RGB', (width, 1020), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), image_text, fill=(0, 0, 0), spacing=0)
    img.save('wrightmap.png')
    encoded2 = base64.b64encode(open("wrightmap.png", "rb").read()).decode('utf-8')
    remove('wrightmap.png')
    return encoded2
