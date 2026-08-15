# Auditoría de novedad: Corbino Xene transitorio con AB, Rashba y Kane–Mele

Fecha de corte: 2026-08-03. Esta auditoría es una búsqueda primaria dirigida,
no una prueba de ausencia de precedentes. El objetivo es separar la novedad
del régimen transitorio de la pregunta secundaria sobre protección topológica.

## Dictamen corto

Los ingredientes individuales ya tienen precedentes claros:

- corrientes persistentes de carga/espín en anillos o Corbino de grafeno con
  flujo AB y Rashba;
- fases QSH/Kane–Mele y transporte de borde/bulto en Xenes;
- transporte transitorio Keldysh/NEGF de dos tiempos;
- anillos AB transitorios y corrientes de espín con interacción espín-órbita;
- bombeo dinámico y corrientes transitorias en grafeno, siliceno, staneno y
  otros aislantes QSH.

Por tanto, no es defendible vender como novedad el método Keldysh, el
Hamiltoniano Kane–Mele–Rashba, el Corbino, el flujo AB, ni la existencia de
una corriente transitoria de espín tomados por separado.

La candidata de novedad es más estrecha: un protocolo reproducible que
resuelva un anillo Corbino finito de un Xene con rampa AB de duración finita y
contactos de reservorio, usando funciones de Green de dos tiempos/EOM, y que
separe en el mismo observable la respuesta persistente, la respuesta inyectada
por reservorios, las capas borde interior/borde exterior/bulk y el torque de
espín de Rashba, con oráculos finitos, controles de tamaño y balance de
continuidad. No se encontró una publicación primaria que reúna todos esos
elementos en una sola realización; este resultado queda **UNCONFIRMED** hasta
repetir la búsqueda en Scopus, Web of Science, INSPIRE y categorías
especializadas de arXiv.

## Matriz de solapamiento

| Bloque | Precedente localizado | Solapamiento | Estado para nuestro proyecto |
|---|---|---:|---|
| Corbino/AB + corrientes persistentes de carga y espín | Grafeno Corbino con ISO/Rashba; anillos con SOI y flujo | Alto | Prior art; usar como baseline estático |
| Kane–Mele/Xene + borde/bulk/QSH | Siliceno/staneno y fase QSH; interferometría en anillos de siliceno | Alto | Prior art; no reclamar descubrimiento de borde |
| AB transitorio | Anillos AB genéricos, wave packets, interferómetros Hall | Medio-alto | Prior art de la dinámica AB, pero geometría/material distintos |
| Espín transitorio + AB/SOI | Interferómetro AB de dos puntos cuánticos con SOI; corrientes transitoriamente polarizadas | Alto en mecanismo, bajo en geometría | No es nuevo el mecanismo; sí puede cambiar la partición bulk/borde |
| Xene transitorio | Bombeo espín/valle en siliceno y corrientes ac/transitorias en QSH/staneno | Parcial | El régimen Xene transitorio existe; no demuestra el mismo protocolo |
| Keldysh/Kadanoff–Baym de dos tiempos e interacciones | KBE/NEGF transitorio, memoria de reservorio y aproximaciones conservantes | Alto | Novedad de método rechazada |
| Persistente vs reservorio + borde/bulk + torque en Corbino Xene | No encontrado en la búsqueda dirigida | Bajo precedente directo | Candidato integrado, aún no confirmado |
| Protección topológica de la corriente transitoria | No es una consecuencia automática de Z2 ni de robustez numérica | — | Pregunta secundaria; exigir controles o reportar ausencia/indeterminación |

## Qué afirma y qué no afirma el resultado

### Claim permisible después de los controles

> Estudiamos una respuesta transitoria de carga y espín en un Corbino Xene
> finito con flujo AB de encendido/apagado suave y reservorios, en un
> formalismo EOM/Keldysh de dos tiempos. El análisis separa las contribuciones
> circulante/persistente y de contactos y cuantifica la redistribución entre
> bulk, borde interno y borde externo, incluyendo el torque no conmutante de
> Rashba.

Este claim describe una integración y una observable, no una teoría nueva. Su
interés publicable depende de mostrar una firma dinámica que no se reduzca a
la suma de un cálculo estacionario más un transitorio genérico.

### Claims que deben rechazarse

- “Keldysh/EOM transitorio es un método nuevo”.
- “El primer cálculo de corrientes persistentes en un anillo Xene”.
- “Rashba produce por primera vez una corriente de espín transitoria”.
- “Una respuesta robusta es protección topológica”.
- “Z2=1 implica que la corriente transitoria del Corbino está protegida”.
- “No existe ningún precedente”: la búsqueda actual no tiene cobertura
  bibliométrica exhaustiva.

## Qué puede ser realmente nuevo en la física

El hueco que conviene atacar no es la existencia de transitorios sino su
descomposición causal en una geometría de anillo:

1. **Memoria del flujo:** latencia, overshoot, fase y relajación tras una rampa
   AB finita, incluyendo el protocolo de volver a flujo cero o invertirlo.
2. **Competencia de canales:** diferencia temporal entre corriente persistente
   atrapada en el anillo y corriente inyectada/extraída por reservorios.
3. **Redistribución espacial:** cuándo la señal ocupa borde interno, borde
   externo o bulk, y cómo cambia con ancho, contacto y energía de Fermi.
4. **Espín no conservado:** torque de Rashba separado de la inyección de
   reservorio, con balance de continuidad visible.
5. **Interacciones como corrección acotada:** comparar el oráculo no
   interactuante exacto con el cierre Hubbard aproximado, sin presentarlo como
   teorema conservante del continuo.

La novedad será débil si sólo se muestran curvas transitorias para una malla.
Será más defendible si se presentan escalados con duración de rampa, ancho,
tamaño de lead, acoplamiento de contacto, Rashba y un control trivial de la
misma geometría.

## Papel de la topología

La topología debe entrar como hipótesis comprobable y como control, no como
objetivo impuesto. El parent bulk periódico puede tener `Z2=1`, pero eso no
garantiza que el observable compacto sea de borde ni que sobreviva al contacto,
al ancho finito, al flujo AB que rompe reversión temporal instantáneamente o a
Rashba. En nuestro estado actual el criterio compacto ya dio un dictamen
negativo para protección; fuera de ese régimen la conclusión permanece
indeterminada. El manuscrito debe mostrar el control trivial y reportar tanto
un resultado positivo como uno negativo si aparecen.

## Siguiente umbral para llamar al resultado “novedad publicable”

Antes de usar “first” o “new”, hacen falta:

1. auditoría Scopus/Web of Science/INSPIRE y búsqueda por sinónimos
   (`annulus`, `Corbino disk`, `Xene`, `silicene`, `stanene`, `germanene`,
   `Kane–Mele`, `Keldysh`, `Kadanoff–Baym`, `transient`, `quench`, `flux ramp`);
2. una comparación explícita contra los seis baselines primarios del registro
   JSON adjunto;
3. un mapa de parámetros con al menos control trivial, Rashba=0, flujo fijo,
   rampa reversible, ancho y tamaño de lead;
4. una figura de descomposición persistente/reservorio y otra de carga/espín/
   torque con residuos de continuidad;
5. un texto que diga “no encontramos una combinación exacta en la búsqueda
   dirigida” en lugar de convertir una ausencia de resultados en prioridad.

## Fuentes primarias de la búsqueda dirigida

El registro machine-readable en
`docs/evidence/transient_novelty_matrix_20260803.json` conserva las URLs y la
clasificación de cada fuente. Entre los precedentes más cercanos están el
Corbino de grafeno en equilibrio, el transporte transitorio de un interferómetro
AB con SOI, el bombeo espín/valle dinámico en siliceno, el transporte transitorio
de grafeno en el formalismo de tiempo real y las corrientes transitorias
polarizadas en aislantes QSH.

