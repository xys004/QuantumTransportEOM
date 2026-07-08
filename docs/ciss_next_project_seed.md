# CISS next-project seed

Fecha: 2026-07-08

## Motivacion

El manuscrito CISS-ladder dejo una conclusion util: un modelo coherente minimalista con SOC quiral no basta, por si solo, para producir una senal robusta directamente comparable con la polarizacion experimental CISS. Esa limitacion no es un fracaso sino una brujula: el siguiente aporte debe formular con precision que ingrediente fisico adicional convierte una estructura quiral en un observable experimental robusto.

La herramienta local `QuantumTransportEOM` ya permite trabajar en esa direccion: combina EOM simbolico, Green functions retarded/advanced/lesser/greater, Keldysh, Landauer, Meir-Wingreen, contactos ferromagneticos y observables spin-resueltos.

Nota complementaria con Hamiltonianos candidatos y rutas modulares:

`docs/ciss_hamiltonian_candidates.md`

Nota de desarrollo analitico/simbolico previo a la senal:

`docs/ciss_symbolic_preanalysis.md`

## Estado del arte condensado

1. Restricciones de simetria y dos terminales.

   La literatura reciente insiste en que chirality + SOC no resuelve automaticamente CISS. Utsumi, Entin-Wohlman y Aharony muestran que puede haber spin selectivity en junctions helicoidales time-reversal symmetric con dos canales orbitales, sin violar Bardarson, porque las transmisiones conservan degeneraciones pero los canales spin/orbital se reorganizan.

   Fuente: https://arxiv.org/abs/2005.04041

2. La comunidad se movio de "spin-polarized transmission" hacia observables de dispositivo.

   En mediciones de transporte, muchas veces el observable real es una respuesta anisotropica de corriente/carga al invertir magnetizacion, mas cercana a magnetoresistencia que a una polarizacion spin normalizada directa. Esto sugiere que hay que modelar explicitamente contactos, condiciones de frontera y protocolo experimental.

   Fuente: https://pubs.aip.org/aip/aco/article/1/2/020903/3368954/Should-it-really-be-that-hard-to-model-the

3. Disipacion/interacciones aparecen como ingredientes centrales, pero no cualquier aproximacion sirve.

   Fransson formula una tesis fuerte: chirality rompe degeneracion de spin y dissipation es necesaria para una spin polarization no nula.

   Fuente: https://arxiv.org/abs/2501.00781

   Upadhyay y Levy hacen un benchmark NEGF self-consistent con electron-phonon en SCBA y encuentran que weak e-ph coupling, tratado conservativamente, da polarizacion despreciable en dos terminales. Esto es importante: nos advierte contra mecanismos "dephasing-like" demasiado fenomenologicos si no estan conectados a un observable y a una self-energy fisicamente consistente.

   Fuente: https://pubs.aip.org/aip/jcp/article/164/16/164303/3387590/Weak-electron-phonon-coupling-is-insufficient-to

4. La ruta no-Hermitiana/interfacial esta muy viva.

   Zhao et al. proponen que la CISS magnetoresistance puede originarse en charge trapping que modifica la barrera de tunelamiento, gobernado por non-Hermitian skin effect en la interfaz ferromagneto-molecula quiral. Esta ruta separa "MR de dispositivo" de "polarizacion intrinseca de transmision".

   Fuente: https://www.nature.com/articles/s41467-024-55433-1

5. Polaron/strong electron-phonon sigue siendo una veta, pero distinta del weak-coupling SCBA.

   Klein y Michaeli argumentan que corrientes cargadas por polarons pueden introducir una escala energetica efectiva mucho mayor que el SOC organico, afectando transmision spin-dependiente y magnetoresistencia asimetrica.

   Fuente: https://arxiv.org/abs/2208.02530

## Hipotesis de trabajo

La veta mas prometedora para nosotros no es "demostrar CISS con otro tight-binding coherente", sino construir una taxonomia computacional de ingredientes minimos:

- geometria quiral + SOC + canales orbitales;
- contactos spin-activos o ferromagneticos;
- self-energies disipativas tipo Keldysh;
- interacciones efectivas o electron-vibration;
- charge trapping / potencial auto-consistente en la interfaz;
- comparacion estricta entre tres observables: spin current, current spin polarization y magnetocurrent/magnetoresistance.

## Primer test local

Se agrego:

`examples/ciss_seed_transport.py`

El script construye una cadena helicoidal spinful toy con eje SOC rotante, calcula corrientes Keldysh con:

- leads normales;
- contacto izquierdo ferromagnetico y derecho normal;
- inversion de quiralidad;
- inversion de magnetizacion;
- broadening fenomenologico de equilibrio.

Resultado inicial: el modelo no genera senal CISS espuria. Con leads normales la corriente de spin sale cero a precision numerica, y con contacto ferromagnetico la asimetria al invertir magnetizacion tambien sale cero. Esto confirma que el banco de pruebas respeta las restricciones basicas y que necesitamos introducir un ingrediente no trivial.

## Primeros experimentos Keldysh nuevos

Se agregaron dos bancos de prueba adicionales:

`examples/ciss_interfacial_trap_scf.py`

Este script implementa una geometria FM lead -> trap interfacial -> cadena quiral SOC -> lead normal. La ocupacion de la trampa se obtiene de

`n_trap = -i int dE Tr[G^<_trap(E)] / (2*pi)`

y alimenta un shift Hartree `eps_trap + U_trap (n_trap - n0)`. Se resuelve self-consistently para `M` y `-M`.

Lectura: la trampa Hartree genera acumulacion local de spin que se invierte con la magnetizacion del contacto, pero no produce asimetria de corriente de carga bajo `M -> -M`. El resultado descarta que un campo medio escalar local, por si solo, convierta la estructura quiral en una magnetoresistencia CISS.

`examples/ciss_ladder_keldysh.py`

Este script traslada el ladder de dos canales/hibridacion al flujo Keldysh de `QuantumTransportEOM`. Mantiene controles directos del manuscrito:

- `gamma_hybrid = 0` suprime la corriente de spin fisica;
- `lambda_soc = 0` tambien la suprime;
- con leads normales, aun con `gamma_hybrid != 0` y SOC, el spin current fisico Meir-Wingreen integrado sigue cancelandose a precision numerica;
- con contacto izquierdo ferromagnetico aparece spin current de signo opuesto al invertir `M`, pero la corriente de carga no cambia, de modo que la magnetocurrent asymmetry sigue siendo cero.

Lectura: el kernel proyectado tipo manuscrito no debe confundirse automaticamente con un observable Keldysh de dispositivo. Para obtener una senal experimental tipo MR hace falta un ingrediente de contorno/disipacion/interaccion que rompa la cancelacion a nivel de corriente de carga, no solo un kernel spin-proyectado no nulo.

`examples/ciss_ladder_elastic_probe.py`

Este script implementa probes elasticos de Buttiker en los rungs internos del ladder. Para cada energia se resuelve una ocupacion ficticia `f_p(E)` imponiendo:

`I_p(E) = 0`

De este modo se pierde fase localmente sin absorber corriente neta. En el barrido inicial, `max|I_p(E)|` queda en el orden de `1e-16`, confirmando conservacion numerica.

Lectura: la decoherencia elastica conservativa reduce la corriente de carga y tambien reduce el spin current inyectado por el contacto ferromagnetico, pero no genera magnetocorriente CISS: `I(M)` e `I(-M)` siguen iguales a precision numerica. Esto sugiere que phase breaking elastico, al menos en esta version spin-independiente y localmente equilibrante, no basta. El siguiente paso natural es comparar contra probes spin-resueltos/canal-resueltos y luego pasar a voltage probes inelasticos o a una self-energy electron-vibracion conservativa.

`examples/ciss_rho_tau_sigma_ladder.py`

Este script reescribe el ladder en una base explicita por sitio:

`rho` = hebra/cadena, `tau` = canal orbital, `sigma` = spin fisico.

El Hamiltoniano usa productos de Pauli `rho_i`, `tau_i`, `sigma_i`, de modo que el SOC actua sobre spin fisico y el canal orbital queda separado. Con leads normales, los controles vuelven a dar corriente de spin integrada nula. Con contacto FM aparece spin current que cambia de signo con `M`, pero todavia no aparece magnetocorriente de carga en el problema coherente.

`examples/ciss_rho_tau_sigma_voltage_probe.py`

Este script agrega una sonda inelastica tipo voltage probe al ladder `rho/tau/sigma`. La sonda tiene una sola `mu_p` auto-consistente, determinada por:

`int dE I_p(E) = 0`

Con detunings `chain_detuning=0.5`, `channel_detuning=1.2`, probe orbital `tau_plus` y `gamma_probe=0.8`, aparece una magnetocorriente pequena pero robusta:

`A_current = -3.3365e-3` para `chi=+1`;

`A_current = +3.3365e-3` para `chi=-1`.

Controles refinados:

- `lambda_soc = 0` da `A_current = 0`;
- `p_FM = 0` da `A_current = 0`;
- invertir quiralidad invierte el signo de la respuesta;
- el residual de conservacion de la sonda queda pequeno, tipicamente `|I_p| ~ 1e-11` o menor.

Lectura: esta es la primera senal CISS-like del banco nuevo. No sale de una perdida de fase elastica, sino de la combinacion de SOC quiral, frontera FM spin-activa, estructura orbital/hebra detunada y redistribucion inelastica conservativa. En este Hamiltoniano, `gamma_hybrid=0` no anula necesariamente la senal porque el propio termino SOC usa `tau_x` y ya introduce mezcla inter-canal en el hopping; por eso el control fundamental aqui es apagar `lambda_soc`, no solo `gamma_hybrid`.

## Tres proyectos posibles

### A. CISS as boundary-condition sensitivity

Objetivo: estudiar cuando un modelo quiral con contacto ferromagnetico produce magnetocurrent no reciproco al agregar una self-energy interfacial dependiente de ocupacion o trapping.

Ventaja: conecta directamente con los observables experimentales de transporte.

Primer experimento:

- agregar un impurity/trap site cerca de la interfaz FM-molecula;
- calcular ocupacion con `G^<`;
- alimentar un shift Hartree-like `U_trap n_trap`;
- resolver self-consistently para `M` y `-M`;
- medir `I(M)`, `I(-M)`, `MR`, `G^<`, spin density local.

### B. Conserving vs non-conserving dephasing approximations

Objetivo: comparar self-energies fenomenologicas, diagonal approximations y esquemas conservativos tipo SCBA simplificado para ver que artefactos producen polarizacion falsa.

Ventaja: seria una contribucion metodologica fuerte y alineada con la critica reciente de NEGF self-consistent.

Primer experimento:

- implementar una self-energy electron-phonon local en Born/self-consistent Born para el toy helix;
- comprobar current conservation;
- comparar contra diagonal broadening y probe no self-consistente.

### C. Current-induced spin density in chiral open systems

Objetivo: cambiar el foco de transmision spin-proyectada hacia densidad de spin fuera de equilibrio, `S_i = -i Tr[s_i G^<_ii]`, y su relacion con corriente/campo magnetico.

Ventaja: conecta con current-induced spin polarization y permite usar directamente Keldysh.

Primer experimento:

- calcular `G^<` local en cadena quiral bajo bias;
- mapear spin density por sitio y por quiralidad;
- comparar con spin current y magnetocurrent;
- identificar condiciones en que hay spin accumulation aunque no haya polarizacion transmitida.

## Decision recomendada

Empezar por A + C en paralelo:

1. usar C para diagnosticar donde aparece spin accumulation local;
2. usar A para convertir esa acumulacion en un observable de dispositivo medible;
3. mantener B como control metodologico para evitar artefactos.

Actualizacion luego de los primeros tests: A con Hartree escalar no basta, C en una cadena de un canal tampoco muestra spin accumulation quiral con leads normales, y probes elasticos conservativos en el ladder no generan magnetocorriente. La primera senal CISS-like aparece al pasar a una condicion de contorno inelastica conservativa tipo voltage probe sobre el ladder explicito `rho/tau/sigma`. El proximo paso es fortalecer esta senal: mapas de parametros, mas de una sonda local, comparacion con SCBA electron-vibracion y verificacion estricta de conservacion de corriente.
