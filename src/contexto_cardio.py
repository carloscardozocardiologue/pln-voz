"""
Contexto médico cardiológico por categoría.
El modelo QA extrae respuestas de estos textos según la pregunta del médico.
"""

CONTEXTOS = {
    "Síntomas": """
El dolor torácico de origen cardíaco se describe típicamente como una presión o
opresión en el centro del pecho, que puede irradiarse al brazo izquierdo, mandíbula
o espalda. Dura más de 20 minutos en el infarto agudo de miocardio. El dolor anginoso
dura menos de 10 minutos y cede con nitroglicerina o reposo. El dolor pleurítico
aumenta con la respiración y sugiere pericarditis o embolia pulmonar.

La disnea cardiogénica aparece con el esfuerzo en estadios iniciales (disnea de
esfuerzo) y en reposo en fases avanzadas de insuficiencia cardíaca. La ortopnea
es la disnea que aparece al tumbarse y mejora incorporándose. La disnea paroxística
nocturna despierta al paciente de madrugada con sensación de ahogo.

Las palpitaciones son la percepción anormal del propio latido cardíaco. Pueden
indicar taquicardia sinusal, fibrilación auricular, flutter auricular, taquicardia
ventricular o extrasístoles. Las extrasístoles se perciben como un latido fuerte
seguido de pausa. Las palpitaciones de inicio y fin bruscos sugieren taquicardia
paroxística supraventricular.

El síncope cardiogénico es una pérdida de conciencia brusca y breve por hipoperfusión
cerebral transitoria. En la estenosis aórtica aparece con el esfuerzo. En los
bloqueos auriculoventriculares completos produce el síndrome de Stokes-Adams.
El presíncope es la sensación de pérdida inminente de conciencia sin llegar a perderla.

El edema de origen cardíaco es bilateral, declive, con fóvea, y empeora a lo largo
del día. Comienza en tobillos y asciende hasta muslos y región sacra en casos graves.
Se asocia a insuficiencia cardíaca derecha o global. El edema agudo de pulmón produce
disnea intensa, crepitantes bilaterales y esputo rosado.

La cianosis central indica hipoxemia grave con saturación de oxígeno inferior al 85%.
La cianosis periférica afecta a dedos y labios por vasoconstricción periférica.
La presión venosa yugular elevada indica sobrecarga de volumen o insuficiencia cardíaca
derecha. El pulso alternante, con latidos fuertes y débiles alternos, indica disfunción
ventricular grave.
""",

    "Medicamentos": """
El enalapril es un inhibidor de la enzima conversora de angiotensina indicado en
hipertensión arterial e insuficiencia cardíaca. La dosis inicial es de 5 mg al día,
ajustable hasta 40 mg al día en una o dos tomas. Está contraindicado en el embarazo
y en la estenosis bilateral de arterias renales. Sus efectos adversos más frecuentes
son la tos seca y la hiperpotasemia.

El ramipril se usa en hipertensión arterial, insuficiencia cardíaca y nefropatía
diabética. La dosis habitual es de 2.5 a 10 mg al día. El lisinopril se administra
una vez al día, con dosis de 5 a 40 mg. Los IECA reducen la mortalidad en pacientes
con infarto de miocardio y fracción de eyección reducida.

El bisoprolol es un betabloqueante cardioselectivo indicado en insuficiencia cardíaca,
angina de pecho, hipertensión y fibrilación auricular. La dosis inicial en insuficiencia
cardíaca es de 1.25 mg al día, titulando hasta 10 mg al día. El metoprolol succinate
se usa en las mismas indicaciones con dosis de 25 a 200 mg al día. Los betabloqueantes
están contraindicados en el asma grave y el bloqueo AV de segundo y tercer grado.

La atorvastatina reduce el colesterol LDL y se usa en prevención cardiovascular primaria
y secundaria. Las dosis van de 10 a 80 mg al día, administrada por la noche. La
rosuvastatina es la estatina más potente con dosis de 5 a 40 mg al día. Las estatinas
pueden producir mialgias y, raramente, rabdomiólisis.

La aspirina en dosis de 100 mg al día es el antiagregante de primera línea en prevención
de eventos cardiovasculares. El clopidogrel se usa a dosis de 75 mg al día como
alternativa o en combinación con aspirina tras síndrome coronario agudo o implante de
stent. El ticagrelor se administra a 90 mg dos veces al día en el síndrome coronario
agudo, con mayor potencia antiagregante que el clopidogrel.

La furosemida es el diurético de asa de elección en insuficiencia cardíaca con retención
de líquidos. Las dosis orales van de 20 a 500 mg al día. Por vía intravenosa se usa
en el edema agudo de pulmón a dosis de 40 a 80 mg en bolo. La espironolactona a dosis
de 25 a 50 mg al día reduce la mortalidad en insuficiencia cardíaca con fracción de
eyección reducida.

La amiodarona es el antiarrítmico más eficaz disponible, indicado en fibrilación
auricular, flutter auricular y taquicardia ventricular. La dosis de carga es de 200 mg
tres veces al día durante una semana, seguida de 200 mg al día de mantenimiento. Requiere
monitorización de función tiroidea, hepática y pulmonar por sus efectos adversos.

La nitroglicerina sublingual se administra en la angina aguda a dosis de 0.4 mg,
repetible cada 5 minutos hasta tres veces. Si el dolor persiste tras tres dosis, se
debe sospechar infarto agudo de miocardio. La nitroglicerina intravenosa se usa en el
síndrome coronario agudo y el edema agudo de pulmón a dosis de 5 a 200 microgramos
por minuto.

La digoxina controla la frecuencia ventricular en la fibrilación auricular crónica y
mejora los síntomas en la insuficiencia cardíaca. La dosis habitual es de 0.125 a 0.25
mg al día. Requiere monitorización de niveles plasmáticos por su estrecho margen
terapéutico. La toxicidad digitálica produce náuseas, visión amarilla y arritmias.

El apixabán y el rivaroxabán son anticoagulantes orales de acción directa indicados en
la fibrilación auricular no valvular para prevenir el ictus. El apixabán se administra
a 5 mg dos veces al día. La warfarina requiere control periódico del INR con objetivo
entre 2 y 3 en fibrilación auricular.
""",

    "Protocolos": """
El protocolo de actuación ante el síndrome coronario agudo sin elevación del ST
incluye: administrar aspirina 300 mg y clopidogrel 300 mg o ticagrelor 180 mg de carga,
anticoagulación con heparina no fraccionada o enoxaparina, monitorización continua,
troponinas seriadas a las 0, 3 y 6 horas, y cateterismo cardíaco urgente o diferido
según el riesgo. En el SCACEST la reperfusión con angioplastia primaria debe realizarse
en menos de 90 minutos desde el primer contacto médico.

La reanimación cardiopulmonar avanzada ante parada cardiorrespiratoria comienza con
30 compresiones torácicas por cada 2 ventilaciones a ritmo de 100 a 120 compresiones
por minuto con una profundidad de 5 a 6 centímetros. En el ritmo desfibrilable como la
fibrilación ventricular se aplica desfibrilación con 200 julios bifásico lo antes posible,
seguida de dos minutos de RCP. La adrenalina 1 mg intravenosa se administra cada 3 a 5
minutos. La amiodarona 300 mg intravenosa se usa en la fibrilación ventricular refractaria
a tres choques.

El protocolo de manejo de la fibrilación auricular de nueva aparición incluye control
de frecuencia cardíaca con betabloqueantes o digoxina para mantenerla por debajo de 110
latidos por minuto. La cardioversión eléctrica sincronizada se realiza con 150 a 200
julios bifásicos si hay inestabilidad hemodinámica. La anticoagulación debe iniciarse
antes de la cardioversión si la fibrilación auricular lleva más de 48 horas o se
desconoce su duración. El riesgo embólico se calcula con la escala CHA2DS2-VASc.

El protocolo de insuficiencia cardíaca aguda descompensada incluye posición
semisentada, oxígeno para mantener saturación superior al 94%, furosemida intravenosa
40 a 80 mg, nitroglicerina intravenosa si la presión arterial sistólica es mayor de
90 mmHg, y monitorización continua de diuresis, tensión arterial y frecuencia cardíaca.
En el edema agudo de pulmón grave se valora la ventilación mecánica no invasiva con
CPAP o BIPAP.

El protocolo de crisis hipertensiva distingue entre urgencia y emergencia hipertensiva.
La urgencia hipertensiva con presión arterial mayor de 180/120 mmHg sin daño orgánico
agudo se trata con captopril 25 mg sublingual o labetalol oral. La emergencia
hipertensiva con daño orgánico agudo requiere reducción de la presión arterial en menos
de una hora con labetalol intravenoso, nitroprusiato sódico o nitroglicerina intravenosa
según el órgano afectado.

La escala GRACE estratifica el riesgo en el síndrome coronario agudo considerando edad,
frecuencia cardíaca, presión arterial sistólica, creatinina, clase Killip, parada
cardíaca al ingreso, desviación del segmento ST y elevación de troponinas. Una puntuación
mayor de 140 indica alto riesgo y requiere cateterismo en menos de 24 horas.

El implante de marcapasos definitivo está indicado en el bloqueo auriculoventricular
completo sintomático, el síndrome del seno enfermo con bradicardia sintomática y la
bradicardia sinusal severa con frecuencia menor de 40 latidos por minuto sintomática.
El desfibrilador automático implantable se indica en pacientes con fracción de eyección
menor del 35% a pesar de tratamiento médico óptimo durante al menos tres meses.
"""
}
