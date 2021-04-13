description_tct = '''
<br><h3 style="width:90%">En la siguiente tabla se presenta un análisis de los items seleccionados bajo la Teoría Clásica de Test (TCT o CTT).
Es importante identificar items que puedan estar dando información irrelevante o contraproducente del constructo que se desea medir, para esto, 
se presenta información que permita al investigador decidir que item puede ser necesario mejorar o eliminar, en post de una mejor medición.<br>

<br><b>Dificultad</b>: Los item que tienen dificultades muy bajas o muy altas no tiene sentido aplicarlos.
Si tiene altos niveles de dificultad puede ser que el item fuese confuso o necesitaba conocimiento previo necesario que 
no posee la población. Para medirla se usa la proporción de aciertos por item.<br>

<br><b>Discriminación</b>: Los item deben diferenciar entre personas que poseen el constructo y quienes no, de no ser 
asi, el item no cumple su función. Para medirla se usa la diferencia de proporción de aciertos entre el grupo de mejores
notas y el grupo de notas mas bajas, siendo cada grupo una cuarta parte del total. Además de la correlación del Item con
 el puntaje sin el item.</h3><br>
'''

description_fiability = '''<br><h3 style="width:90%">Confiabilidad permite ver el grado en que un test mide
consistentemente a través de sucesivas aplicaciones.Tiene relación con el error de medida, a mayor margen de error, 
menor confiabilidad y a menor margen de error, mayor confiabilidad.

<br>Se aplica para su medida el Alfa de Cronbach y el Método de dos partes para el cálculo de Spearman-Brown y Rulon.</h3><br>'''

decription_irt = '''<br><h3 style="width:90%">En la siguiente tabla y gráficos se presenta un análisis de los item 
seleccionados bajo la Teoría de Respuesta al Item (TRI o IRT).
Es importante identificar items que puedan estar dando información irrelevante o contraproducente del constructo que se 
desea medir, para esto, se presenta información que permita al investigador decidir que item puede ser necesario mejorar
 o eliminar, en post de una mejor medición.<br>
   
<br><b>Dificultad</b>:Los item que tienen dificultades muy bajas o muy altas no tiene sentido aplicarlos.
Si tiene altos niveles de dificultad puede ser que el item fuese confuso o necesitaba conocimiento previo necesario que 
no posee la población. Para medir la Dificultad se estima como parámetro del modelo logístico y con la probabilidad de 
responder correctamente el item dado un nivel de habilidad medio.

<br></b>Discriminación</b>: Los item deben diferenciar entre personas que poseen el constructo y quienes no, de no ser 
asi, el item no cumple su función.
Para medir la Discriminación se estima como parámetro del modelo logístico.</h3><br>'''


def text_descriptions(function):

    if function == 'dificultad_tct': return description_tct
    if function == 'alpha_cronbach': return description_fiability
    if function == 'irt_rasch': return decription_irt
