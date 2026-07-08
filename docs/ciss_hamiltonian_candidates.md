# Hamiltonianos candidatos para el proximo proyecto CISS

Fecha: 2026-07-08

## Punto de partida

La evidencia numerica actual sugiere que la senal CISS de transporte no sale de un unico ingrediente aislado:

- SOC quiral coherente: no basta.
- Hibridacion inter-canal: necesaria para kernels proyectados tipo manuscrito, pero no garantiza magnetocorriente Keldysh.
- Trampa Hartree escalar: genera spin accumulation local inducida por contacto FM, pero no magnetocorriente.
- Probes elasticos conservativos spin-independientes: reducen corriente y spin current, pero no rompen `I(M)=I(-M)`.
- Voltage probe inelastico conservativo sobre el ladder `rho/tau/sigma`: produce la primera senal CISS-like pequena y controlada.

La apuesta razonable es una combinacion minima: molecula quiral multicanal + interfaz spin-activa + mecanismo conservativo de relajacion/decoherencia + posible trapping/interaccion local.

## Estructura modular

Conviene escribir el problema como:

```text
H_total = H_mol + H_interface + H_trap + H_vib + H_leads
```

y tratar `H_leads`, decoherencia y vibraciones como self-energies Keldysh cuando sea posible:

```text
G^r(E) = [E I - H_mol - Sigma_L^r - Sigma_R^r - Sigma_probe^r - Sigma_int^r(E)]^{-1}
G^<(E) = G^r(E) [Sigma_L^< + Sigma_R^< + Sigma_probe^< + Sigma_int^<(E)] G^a(E)
```

Esto permite medir por separado:

- corriente de carga `I`;
- spin current `I_s`;
- densidad local de spin `S_i = -i Tr[s_i G^<_{ii}] / (2*pi)`;
- magnetocorriente `A_M = [I(M)-I(-M)]/[|I(M)|+|I(-M)|]`;
- conservacion de corriente en cada sonda/proceso.

## Hamiltoniano molecular transparente

Usar indices:

- `n`: sitio/rung;
- `a = 1,2`: cadena/hebra;
- `tau = +/-`: canal orbital interno;
- `s = up,down`: spin fisico.

Una base natural es:

```text
Psi_n = (c_{n,1,+,up}, c_{n,1,+,down}, c_{n,1,-,up}, c_{n,1,-,down},
         c_{n,2,+,up}, c_{n,2,+,down}, c_{n,2,-,up}, c_{n,2,-,down})^T
```

con matrices de Pauli:

- `rho_i`: espacio de cadena `a`;
- `tau_i`: espacio de canal `+/-`;
- `sigma_i`: spin fisico.

Un modelo minimo legible seria:

```text
H_mol =
sum_n Psi_n^dag [epsilon I + Delta rho_z + delta_tau tau_z] Psi_n
+
sum_n { Psi_{n+1}^dag [t I
        + i lambda_so tau_x (d_n(chi) . sigma)] Psi_n + h.c. }
+
sum_n Psi_n^dag [gamma_parallel rho_x + gamma_hyb rho_x tau_x] Psi_n
```

donde:

```text
d_n(chi) = (cos(phi_n), chi sin(phi_n), eta_z)
phi_n = 2*pi*n/q
```

Lectura:

- `lambda_so` introduce SOC quiral real en spin fisico;
- `gamma_hyb` es la hibridacion inter-canal que el manuscrito identifico como necesaria;
- `Delta` o `delta_tau` permiten romper simetrias de forma controlada;
- `chi = +/-1` invierte la quiralidad.

Este Hamiltoniano es mas transparente que el legacy ladder, porque explicita que parte vive en spin fisico y que parte vive en canales orbitales.

Implementacion inicial:

`examples/ciss_rho_tau_sigma_ladder.py`

Resultado: el Hamiltoniano coherente con contactos normales o FM reproduce los controles de no-senal espuria. La magnetocorriente de carga sigue anulandose sin relajacion inelastica.

Desarrollo simbolico asociado:

`docs/ciss_symbolic_preanalysis.md`

`examples/ciss_symbolic_preanalysis.py`

## Contacto ferromagnetico

El contacto FM no deberia meterse necesariamente como termino Hermitiano del Hamiltoniano molecular, sino como self-energy:

```text
Sigma_L^r = -i Gamma_L(M)/2
Gamma_L(M) = Gamma_0 P_L [I + p (m . sigma)] P_L
Sigma_L^< = i f_L(E) Gamma_L(M)
```

`P_L` proyecta sobre los grados de libertad de la interfaz izquierda.

El contacto derecho puede empezar normal:

```text
Gamma_R = Gamma_R0 P_R
```

El observable experimental tipo CISS seria:

```text
A_M = [I(m)-I(-m)]/[|I(m)|+|I(-m)|]
```

no solo un kernel spin-proyectado.

## Ruta 1: probes conservativos

### Probe elastico

Ya probado en version spin-independiente:

```text
Sigma_p^r = -i Gamma_p/2
Sigma_p^< = i f_p(E) Gamma_p
I_p(E) = 0
```

Resultado inicial: conserva corriente pero no genera magnetocorriente.

Variantes utiles antes de descartarlo:

- probes por canal `tau`;
- probes por cadena `rho`;
- probes con acoplo espacialmente asimetrico cerca de la interfaz;
- probes spin-resueltos como control, aunque fisicamente son menos inocentes.

### Voltage probe inelastico

Siguiente ruta:

```text
Sigma_p^< = i f(E; mu_p, T) Gamma_p
int dE I_p(E) = 0
```

Esto permite redistribuir energia. Es mas parecido a relajacion con ambiente electronico/termico que el probe elastico.

Implementacion inicial:

`examples/ciss_rho_tau_sigma_voltage_probe.py`

Resultado candidato:

```text
chain_detuning = 0.5
channel_detuning = 1.2
probe_kind = tau_plus
gamma_probe = 0.8
temperature = 0.03

chi = +1: A_current = -3.3365e-3
chi = -1: A_current = +3.3365e-3
```

Controles:

```text
lambda_soc = 0  -> A_current = 0
p_FM = 0        -> A_current = 0
chi -> -chi     -> A_current cambia de signo
```

Nota importante: en el Hamiltoniano `rho/tau/sigma`, apagar `gamma_hyb` no es equivalente a eliminar toda mezcla inter-canal, porque el SOC elegido contiene `tau_x`. Por eso el control `lambda_soc = 0` es el test decisivo de quiralidad/spin-orbit para este modelo.

## Ruta 2: trampa interfacial no puramente escalar

Una trampa escalar Hartree no basto. La version mas fisica para MR interfacial debe modificar la barrera o el acoplamiento:

```text
H_trap = epsilon_d d^dag d + V_d (d^dag P_L Psi_1 + h.c.)
epsilon_d -> epsilon_d + U (n_d - n0)
Gamma_L -> Gamma_L[n_d]
```

Posible extension spin-activa:

```text
H_trap,ex = J_d d^dag (m . sigma) d
```

Esta extension debe tratarse con cuidado: si se introduce `J_d` demasiado libremente, la senal puede ser simplemente una barrera magnetica, no CISS molecular.

## Ruta 3: electron-vibracion conservativo

Agregar modos locales:

```text
H_vib = sum_q Omega_q b_q^dag b_q
H_e-v = sum_q Psi^dag M_q Psi (b_q + b_q^dag)
```

En SCBA:

```text
Sigma_ev^<(E) ~ sum_q M_q [
    N_q G^<(E - Omega_q) + (N_q + 1) G^<(E + Omega_q)
] M_q
```

La prioridad no seria obtener una gran polarizacion de inmediato, sino verificar:

- conservacion de corriente;
- si la parte even de la respuesta bajo `M -> -M` deja de cancelarse;
- si la senal requiere simultaneamente `lambda_so`, `gamma_hyb`, contacto FM e inelasticidad.

## Propuesta concreta de avance

1. Reescribir el ladder nuevo con matrices `rho`, `tau`, `sigma`, para no depender de ambiguedades del legacy basis.
2. Repetir controles:
   - `lambda_so = 0`;
   - `gamma_hyb = 0`;
   - `p_FM = 0`;
   - `chi -> -chi`;
   - `M -> -M`.
3. Probar probes elasticos canal-resueltos.
4. Implementar voltage probes inelasticos. Estado: primera senal encontrada.
5. Fortalecer el resultado con mapas de parametros, sondas locales multiples y comparacion contra una self-energy electron-vibracion conservativa.

La tesis posible no tiene que ser "encontramos una receta numerica que da CISS", sino algo mas fuerte:

```text
Una magnetocorriente CISS robusta requiere la coexistencia de hibridacion inter-canal,
spin-orbit quiral, una condicion de frontera spin-activa y un mecanismo conservativo
de relajacion/interfaz que convierta spin accumulation local en respuesta de carga.
```
