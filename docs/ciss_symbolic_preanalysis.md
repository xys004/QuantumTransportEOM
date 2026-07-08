# Desarrollo analitico previo para la senal CISS-like

Fecha: 2026-07-08

## Objetivo

Ahora que el banco numerico encontro una senal CISS-like con el ladder `rho/tau/sigma` y una sonda inelastica conservativa, conviene separar que se puede argumentar antes del calculo numerico y que queda como resultado de evaluacion.

La meta analitica no es invertir simbolicamente el Hamiltoniano completo de dimension `8N`. Eso seria poco transparente. La meta util es:

1. mostrar que los operadores elegidos contienen los ingredientes necesarios;
2. derivar la formula efectiva de una sonda de voltaje conservativa;
3. identificar que terminos pueden ser impares bajo `M -> -M` y `chi -> -chi`;
4. demostrar controles de anulacion: `lambda_soc = 0`, `p_FM = 0`, y ausencia de feedback de sonda.

Script asociado:

`examples/ciss_symbolic_preanalysis.py`

## Hamiltoniano local

La base por sitio es:

```text
Psi_n = (rho, tau, sigma)
rho   = hebra/cadena
tau   = canal orbital interno
sigma = spin fisico
```

El Hamiltoniano molecular minimo usado en el banco numerico es:

```text
H_mol =
sum_n Psi_n^dag [epsilon I + Delta rho_z + delta tau_z
                 + gamma_parallel rho_x + gamma_hyb rho_x tau_x] Psi_n
+
sum_n { Psi_{n+1}^dag [t I + i lambda tau_x (d_n(chi).sigma)] Psi_n + h.c. }
```

con:

```text
d_n(chi) = (cos phi_n, chi sin phi_n, eta_z)
```

Aqui `sigma` es spin fisico. El canal orbital `tau` y la hebra `rho` son grados de libertad internos, no spin.

## Identidades de operadores

SymPy verifica exactamente:

```text
[tau_z, tau_x d.sigma] = 2 i tau_y d.sigma
```

Esto dice que un detuning/filtro de canal `tau_z` no conmuta con el SOC quiral, porque el SOC usa `tau_x`. Por eso un probe selectivo en `tau_plus` puede detectar mezcla inter-canal inducida por SOC.

```text
[sigma_z, tau_x d.sigma] = 2 i tau_x (d_x sigma_y - d_y sigma_x)
```

Esto dice que un contacto ferromagnetico polarizado en `z` no conmuta con la parte transversal del SOC. Si `lambda = 0`, o si el contacto no tiene polarizacion `p_FM`, desaparece el mecanismo spin-activo.

```text
[P_tau+, tau_x d.sigma] = i tau_y d.sigma
```

Un voltage probe acoplado solo al canal `tau_plus` no es un bano neutro respecto al SOC: mide/relaja una combinacion que el Hamiltoniano mezcla durante la propagacion.

```text
[P_tau+, rho_x tau_x] = i rho_x tau_y
```

La hibridacion inter-hebra/inter-canal tambien es visible para un probe orbital selectivo.

Estas identidades no prueban por si solas una magnetocorriente, pero explican por que la combinacion `SOC + contacto FM + probe tau-selectivo` puede producir una respuesta impar.

## Formula de Landauer-Buttiker con una sonda de voltaje

En respuesta lineal, para tres terminales `L`, `R`, `P`:

```text
I_alpha = (e^2/h) sum_beta G_alpha,beta (V_alpha - V_beta)
```

con:

```text
G_alpha,beta = int dE [-df0/dE] T_alpha,beta(E)
```

La condicion de voltage probe es:

```text
I_P = 0
```

Por lo tanto:

```text
V_P = (G_PL V_L + G_PR V_R) / (G_PL + G_PR)
```

y la corriente izquierda queda:

```text
I_L = (e^2/h) G_eff (V_L - V_R)

G_eff = G_LR + G_LP G_PR / (G_PL + G_PR)
```

Este segundo termino no existe en un problema coherente de dos terminales. Es la primera formula compacta que muestra como la sonda puede convertir informacion orbital/local en una contribucion de carga.

## Parte impar bajo inversion de magnetizacion

Escribamos:

```text
G_ab(M) = G_ab^0 + m delta G_ab + O(m^2)
```

donde `m` cambia de signo bajo `M -> -M`. Entonces:

```text
delta G_eff =
delta G_LR
+ delta G_LP G_PR^0 / D
+ G_LP^0 delta G_PR / D
- G_LP^0 G_PR^0 (delta G_PL + delta G_PR) / D^2

D = G_PL^0 + G_PR^0
```

Si el problema coherente de dos terminales cumple `delta G_LR = 0`, aun queda una contribucion impar si alguna de las transmisiones que involucran la sonda tiene parte impar:

```text
delta G_LP, delta G_PL, delta G_PR != 0
```

Esto explica por que el voltage probe puede desbloquear magnetocorriente aunque el problema coherente no la tenga.

Controles inmediatos:

```text
lambda_soc = 0  -> delta G_ab = 0 -> delta G_eff = 0
p_FM = 0        -> delta G_ab = 0 -> delta G_eff = 0
```

El cambio de quiralidad `chi -> -chi` debe invertir la parte quiral de `delta G_ab`, y por tanto invertir la respuesta si el resto de la geometria se mantiene.

## Formula no lineal de feedback de voltaje

El calculo numerico actual no es estrictamente lineal: usa bias finito y una `mu_p` determinada por:

```text
I_P(m, mu_p(m)) = 0
```

Diferenciando implicitamente:

```text
d mu_p / dm = - (partial_m I_P) / (partial_mu I_P)
```

y la corriente efectiva queda:

```text
d I_L[m, mu_p(m)] / dm =
partial_m I_L - (partial_mu I_L) (partial_m I_P)/(partial_mu I_P)
```

El segundo termino es el feedback de voltaje de la sonda. Este termino no existe si:

- no hay sonda;
- la sonda tiene `mu_p` fija;
- la sonda elastica ajusta `f_p(E)` energia por energia de modo que no redistribuye energia;
- `partial_m I_P = 0`, que ocurre en los controles sin SOC o sin polarizacion FM.

## Que queda demostrado antes de numerica

Se puede sostener analiticamente:

1. El Hamiltoniano separa spin fisico `sigma` de canales orbitales `tau` y hebras `rho`.
2. SOC quiral, contacto FM y probe orbital selectivo son mutuamente no conmutantes.
3. Un problema coherente de dos terminales no tiene el termino de feedback `G_LP G_PR/(G_PL + G_PR)`.
4. Una sonda de voltaje conservativa agrega un canal analitico para convertir una asimetria local/orbital en magnetocorriente.
5. La respuesta debe anularse si `lambda_soc = 0` o `p_FM = 0`.
6. Si la parte impar es quiral, debe cambiar de signo con `chi`.

## Que queda necesariamente numerico

Todavia requiere calculo numerico:

- el signo y magnitud de `delta G_ab(E)`;
- el valor de `mu_p` a bias finito;
- la integral energetica y posible cancelacion espectral;
- la dependencia en `chain_detuning`, `channel_detuning`, `gamma_probe`, temperatura y bias;
- la comparacion cuantitativa contra probes elasticos y contra una futura self-energy electron-vibracion.

## Lectura para el proyecto

La senal numerica encontrada no aparece como accidente aislado. Tiene una ruta analitica:

```text
SOC quiral + contacto FM + estructura orbital asimetrica
    -> transmisiones a la sonda con parte impar en M y chi
    -> mu_p(M, chi) se ajusta por conservacion de corriente
    -> el feedback de voltaje modifica I_L
    -> aparece magnetocorriente CISS-like
```

Esto sugiere que el proximo desarrollo fuerte no debe ser solo "buscar parametros", sino derivar y medir los objetos:

```text
delta G_LP, delta G_PL, delta G_PR, d mu_p / dM
```

en mapas de energia y parametros.
