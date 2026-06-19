import tkinter as tk
from tkinter import ttk, messagebox
import math
import cmath

C1 = "#1a237e"; C2 = "#3949ab"; C3 = "#5c6bc0"
FONDO = "#d2d8e9"; BLANCO = "#ffffff"; VERDE = "#2e7d32"
GRIS = "#e8eaf6"; NEGRO = "#1a1a2e"; ROJO = "#c62828"
F1 = ("Segoe UI", 10, "bold"); F2 = ("Consolas", 10); F3 = ("Segoe UI", 9)


# ---- punto 1: eliminacion gaussiana con pivoteo parcial ----

def gauss(A, b):
    n = len(b)
    mat = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = col
        for f in range(col+1, n):
            if abs(mat[f][col]) > abs(mat[piv][col]): piv = f
        mat[col], mat[piv] = mat[piv], mat[col]
        if abs(mat[col][col]) < 1e-12: continue
        for f in range(col+1, n):
            if abs(mat[f][col]) < 1e-15: continue
            fac = mat[f][col] / mat[col][col]
            for j in range(col, n+1):
                mat[f][j] -= fac * mat[col][j]
    rango_A  = sum(1 for i in range(n) if any(abs(mat[i][j]) > 1e-10 for j in range(n)))
    rango_Ab = sum(1 for i in range(n) if any(abs(mat[i][j]) > 1e-10 for j in range(n+1)))
    if rango_A != rango_Ab:
        return 'incompatible', None
    if rango_A < n:
        return 'infinitas', None
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        s = mat[i][n] - sum(mat[i][j] * x[j] for j in range(i+1, n))
        x[i] = s / mat[i][i]
    return 'unica', x


# ---- punto 2: valores y vectores propios ----

# determinante por cofactores 
def det(M):
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    resultado = 0.0
    for j in range(n):
        sub = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        resultado += ((-1)**j) * M[0][j] * det(sub)
    return resultado


# coeficientes del polinomio caracteristico usando interpolacion de Newton
def pol_car(A):
    n = len(A)
    xs = list(range(n+1))
    ys = []
    for k in xs:
        B = [[A[i][j] - (k if i == j else 0) for j in range(n)] for i in range(n)]
        ys.append(det(B))

    # tabla de diferencias divididas
    tabla = [ys[:]]
    for nivel in range(1, n+1):
        nueva = [(tabla[-1][i+1] - tabla[-1][i]) / (xs[i+nivel] - xs[i])
                 for i in range(len(tabla[-1])-1)]
        tabla.append(nueva)
    cn = [tabla[k][0] for k in range(n+1)]

    # paso de forma Newton a coeficientes estandar
    coef = [0.0] * (n+1)
    prod = [1.0]
    for k in range(n+1):
        for i in range(len(prod)):
            coef[n - (len(prod)-1-i)] += cn[k] * prod[i]
        if k < n:
            np2 = [0.0] * (len(prod)+1)
            for i in range(len(prod)):
                np2[i] += prod[i]
                np2[i+1] += prod[i] * (-xs[k])
            prod = np2
    return coef


def evalp(c, x):
    r = 0.0
    for ci in c:
        r = r * x + ci
    return r

def deriv_real(c):
    # derivada de un polinomio real (lista de coeficientes, con el mayor grado primero)
    n = len(c) - 1
    return [c[i] * (n - i) for i in range(n)]


# CORRECCION 1 
def buscar_raices(c_orig):
    n = len(c_orig) - 1
    if n == 0:
        return []
    c = [x / c_orig[0] for x in c_orig]  # normalizar coef principal a 1

    def evalp_c(coefs, x):
        r = 0j
        for ci in coefs:
            r = r * x + ci
        return r

    radio = 1.0 + max(abs(ci) for ci in c[1:]) if n > 0 else 1.0
    z = [(0.4 + 0.9j) + radio * 0.5 * cmath.exp(2j * math.pi * k / n)
         for k in range(n)]

    for _ in range(300):
        z_nuevo = z[:]
        max_delta = 0.0
        for i in range(n):
            num = evalp_c(c, z[i])
            den = 1 + 0j
            for j in range(n):
                if j != i:
                    den *= (z[i] - z[j])
            if abs(den) < 1e-14:
                den = 1e-14
            delta = num / den
            z_nuevo[i] = z[i] - delta
            max_delta = max(max_delta, abs(delta))
        z = z_nuevo
        if max_delta < 1e-12:
            break

    # acomodar 1
    cd = deriv_real(c_orig)
    resultado = []
    for zi in z:
        if abs(zi.imag) < 1e-3:
            x = zi.real
            for _ in range(60):
                fp = evalp(cd, x)
                if abs(fp) < 1e-14: break
                paso = evalp(c_orig, x) / fp
                x -= paso
                if abs(paso) < 1e-15: break
            resultado.append((x, 0.0))
        else:
            resultado.append((zi.real, zi.imag))
    return resultado


#  CORRECCION 2
def vec_propio(A, lam):
    n = len(A)
    M = [[A[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)]
    fil = 0; col_piv = []
    TOL = 1e-4
    for col in range(n):
        mejor = next((f for f in range(fil, n) if abs(M[f][col]) > TOL), -1)
        if mejor == -1: continue
        M[fil], M[mejor] = M[mejor], M[fil]
        col_piv.append(col)
        p = M[fil][col]
        M[fil] = [v/p for v in M[fil]]
        for f in range(n):
            if f != fil and abs(M[f][col]) > TOL:
                fc = M[f][col]
                M[f] = [M[f][j] - fc*M[fil][j] for j in range(n)]
        fil += 1
    libres = [j for j in range(n) if j not in col_piv]
    if not libres:
        libres = [col_piv[-1]]; col_piv = col_piv[:-1]
    v = [0.0]*n; v[libres[0]] = 1.0
    for i, col in enumerate(col_piv):
        v[col] = -sum(M[i][j]*v[j] for j in range(n) if j != col)
    norma = math.sqrt(sum(vi**2 for vi in v))
    if norma > 1e-12:
        v = [vi/norma for vi in v]
    return v


def leer(entry, ph):
    val = entry.get().strip()
    if val == "" or val == ph: return 0.0
    return float(eval(val, {"__builtins__": {}}, vars(math)))

def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4f}"


# ---- interfaz ----

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algebra Lineal - Taller")
        self.configure(bg=FONDO)
        self.geometry("920x680")
        self.resizable(True, True)
        self._build()

    def _build(self):
        cab = tk.Frame(self, bg=C1, pady=12)
        cab.pack(fill="x")
        tk.Label(cab, text="Algebra Lineal", font=("Segoe UI", 18, "bold"),
                 bg=C1, fg=BLANCO).pack()
        tk.Label(cab, text="Eliminacion de Gauss   |   Valores y Vectores Propios",
                 font=("Segoe UI", 9), bg=C1, fg="#9fa8da").pack()

        est = ttk.Style(self); est.theme_use("clam")
        est.configure("TNotebook", background=FONDO, borderwidth=0)
        est.configure("TNotebook.Tab", font=F1, padding=[16, 7], background=GRIS, foreground=C1)
        est.map("TNotebook.Tab", background=[("selected", C2)], foreground=[("selected", BLANCO)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=14, pady=12)
        p1 = tk.Frame(nb, bg=FONDO); p2 = tk.Frame(nb, bg=FONDO)
        nb.add(p1, text="  Punto 1 - Gauss  ")
        nb.add(p2, text="  Punto 2 - Valores Propios  ")
        self._ui_gauss(p1); self._ui_propios(p2)

    # pestaña gauss

    def _ui_gauss(self, parent):
        bar = tk.Frame(parent, bg=BLANCO, highlightbackground=C3, highlightthickness=1)
        bar.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(bar, text="Tamano n =", font=F1, bg=BLANCO, fg=NEGRO).pack(side="left", padx=(12,4), pady=8)
        self.gn = tk.IntVar(value=3)
        tk.Spinbox(bar, from_=2, to=10, textvariable=self.gn, width=4,
                   font=F1, fg=C1, relief="flat", bd=2, justify="center").pack(side="left", padx=4)
        tk.Button(bar, text="Generar", font=F1, bg=C2, fg=BLANCO, relief="flat",
                  padx=10, pady=3, command=self._gen_gauss).pack(side="left", padx=12)
        # NUEVO: boton para limpiar la matriz sin tener que borrar casilla por casilla
        tk.Button(bar, text="Limpiar matriz", font=F1, bg=ROJO, fg=BLANCO, relief="flat",
                  padx=10, pady=3, command=lambda: self._limpiar_matriz(self.eg, self.pg)).pack(side="left", padx=4)
        tk.Label(bar, text="2 <= n <= 10", font=F3, bg=BLANCO, fg="#78909c").pack(side="right", padx=12)

        cont = tk.Frame(parent, bg=FONDO)
        cont.pack(fill="both", expand=True, padx=10)
        self.feg = tk.Frame(cont, bg=FONDO)
        self.feg.pack(side="left", fill="both", expand=True)

        der = tk.Frame(cont, bg=BLANCO, highlightbackground=C3, highlightthickness=1)
        der.pack(side="right", fill="both", expand=True, padx=(8, 0))
        tk.Label(der, text="Resultado", font=("Segoe UI", 14, "bold"),
                 bg=BLANCO, fg=C1).pack(pady=(10, 4))
        self.tg = tk.Text(der, font=F2, bg=BLANCO, fg=NEGRO, relief="flat",
                          state="disabled", wrap="word", padx=8, pady=6)
        self.tg.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        tk.Button(der, text="Limpiar resultado", font=F3, bg=GRIS, fg=C1, relief="flat",
                  padx=8, pady=2, command=lambda: (self.tg.config(state="normal"),
                  self.tg.delete("1.0", "end"), self.tg.config(state="disabled"))).pack(pady=(0, 6))
        

        self.eg = []; self.pg = []; self._gen_gauss()

    def _gen_gauss(self):
        n = self.gn.get()
        for w in self.feg.winfo_children(): w.destroy()
        self.eg = []; self.pg = []
        tk.Label(self.feg, text=f"Sistema {n}x{n}", font=F3, bg=FONDO, fg="#546e7a").pack(pady=(10,5))
        gr = tk.Frame(self.feg, bg=FONDO); gr.pack(pady=4)

        for j in range(n):
            tk.Label(gr, text=f"x{j+1}", font=F1, bg=C1, fg=BLANCO,
                     width=6, pady=3, relief="flat").grid(row=0, column=j+1, padx=2, pady=2)
        tk.Label(gr, text="b", font=F1, bg=VERDE, fg=BLANCO,
                 width=6, pady=3, relief="flat").grid(row=0, column=n+2, padx=2, pady=2)
        tk.Label(gr, text="=", font=F1, bg=FONDO, fg=C1).grid(row=1, column=n+1, rowspan=n, sticky="ns")

        for i in range(n):
            tk.Label(gr, text=f"Ec{i+1}", font=F1, bg=C1, fg=BLANCO,
                     width=6, pady=3, relief="flat").grid(row=i+1, column=0, padx=2, pady=2)
            fila = []; phs = []
            for j in range(n+1):
                ph = f"a{i+1}{j+1}" if j < n else f"b{i+1}"
                phs.append(ph)
                bg = BLANCO if j < n else "#e8f5e9"
                borde = C3 if j < n else VERDE
                col = j+1 if j < n else n+2
                fr = tk.Frame(gr, bg=borde, padx=1, pady=1)
                fr.grid(row=i+1, column=col, padx=2, pady=2)
                e = tk.Entry(fr, width=6, font=F2, bg=bg, fg="#aaaaaa", relief="flat", justify="center")
                e.pack(); e.insert(0, ph)
                e.bind("<FocusIn>",  lambda ev, en=e, p=ph: (en.delete(0,"end"), en.config(fg=NEGRO)) if en.get()==p else None)
                e.bind("<FocusOut>", lambda ev, en=e, p=ph: (en.insert(0,p), en.config(fg="#aaaaaa")) if en.get().strip()==""  else None)
                fila.append(e)
            self.eg.append(fila); self.pg.append(phs)
        tk.Button(self.feg, text="Resolver sistema", font=F1, bg=C1, fg=BLANCO,
                  relief="flat", padx=14, pady=6, command=self._resolver).pack(pady=12)

    # boton de limpiar matriz
    # (a11, a12, ... / b1, b2, ...)
    def _limpiar_matriz(self, entries, placeholders):
        for i in range(len(entries)):
            for j in range(len(entries[i])):
                e = entries[i][j]; ph = placeholders[i][j]
                e.delete(0, "end")
                e.insert(0, ph)
                e.config(fg="#aaaaaa")

    def _resolver(self):
        n = self.gn.get()
        try:
            A = [[leer(self.eg[i][j], self.pg[i][j]) for j in range(n)] for i in range(n)]
            b = [leer(self.eg[i][n], self.pg[i][n]) for i in range(n)]
        except:
            messagebox.showerror("Error", "Revisa que todos los valores sean numeros validos.")
            return
        tipo, sol = gauss(A, b)
        self.tg.config(state="normal"); self.tg.delete("1.0", "end")
        if tipo == 'incompatible':
            self.tg.insert("end", "Sistema incompatible.\n", "tit")
            self.tg.insert("end", "No existe solucion.\n(las ecuaciones se contradicen entre si)")
        elif tipo == 'infinitas':
            self.tg.insert("end", "Infinitas soluciones.\n", "tit")
            self.tg.insert("end", "El sistema tiene infinitas soluciones.\n(ecuaciones linealmente dependientes)")
        else:
            self.tg.insert("end", "Solucion:\n", "tit")
            self.tg.insert("end", "-"*28 + "\n")
            for i, xi in enumerate(sol):
                self.tg.insert("end", f"  x{i+1} = {fmt(xi)}\n", "sol")
            self.tg.insert("end", "\nVerificacion Ax = b:\n")
            for i in range(n):
                calc = sum(A[i][j]*sol[j] for j in range(n))
                ok = "OK" if abs(calc - b[i]) < 1e-6 else "revisar"
                self.tg.insert("end", f"  Ec{i+1}: {fmt(calc)} = {fmt(b[i])}  [{ok}]\n")
        self.tg.tag_config("tit", font=("Segoe UI", 10, "bold"), foreground=C1)
        self.tg.tag_config("sol", font=("Consolas", 11, "bold"), foreground=VERDE)
        self.tg.config(state="disabled")

    #  pestaña valores propios

    def _ui_propios(self, parent):
        bar = tk.Frame(parent, bg=BLANCO, highlightbackground=C3, highlightthickness=1)
        bar.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(bar, text="Orden n =", font=F1, bg=BLANCO, fg=NEGRO).pack(side="left", padx=(12,4), pady=8)
        self.en = tk.IntVar(value=2)
        tk.Spinbox(bar, from_=2, to=5, textvariable=self.en, width=4,
                   font=F1, fg=C1, relief="flat", bd=2, justify="center").pack(side="left", padx=4)
        tk.Button(bar, text="Generar", font=F1, bg=C2, fg=BLANCO, relief="flat",
                  padx=10, pady=3, command=self._gen_prop).pack(side="left", padx=12)
        # NUEVO: boton para limpiar la matriz de esta pestaña tambien
        tk.Button(bar, text="Limpiar matriz", font=F1, bg=ROJO, fg=BLANCO, relief="flat",
                  padx=10, pady=3, command=lambda: self._limpiar_matriz(self.ee, self.pe)).pack(side="left", padx=4)
        tk.Label(bar, text="2 <= n <= 5", font=F3, bg=BLANCO, fg="#78909c").pack(side="right", padx=12)

        cont = tk.Frame(parent, bg=FONDO)
        cont.pack(fill="both", expand=True, padx=10)
        self.fee = tk.Frame(cont, bg=FONDO)
        self.fee.pack(side="left", fill="both", expand=True)

        der = tk.Frame(cont, bg=BLANCO, highlightbackground=C3, highlightthickness=1)
        der.pack(side="right", fill="both", expand=True, padx=(8, 0))
        tk.Label(der, text="Resultado", font=("Segoe UI", 14, "bold"),
                 bg=BLANCO, fg=C1).pack(pady=(10, 4))
        self.te = tk.Text(der, font=F2, bg=BLANCO, fg=NEGRO, relief="flat",
                          state="disabled", wrap="word", padx=8, pady=6)
        self.te.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        tk.Button(der, text="Limpiar resultado", font=F3, bg=GRIS, fg=C1, relief="flat",
                  padx=8, pady=2, command=lambda: (self.te.config(state="normal"),
                  self.te.delete("1.0", "end"), self.te.config(state="disabled"))).pack(pady=(0, 6))

        self.ee = []; self.pe = []; self._gen_prop()

    def _gen_prop(self):
        n = self.en.get()
        for w in self.fee.winfo_children(): w.destroy()
        self.ee = []; self.pe = []
        tk.Label(self.fee, text=f"Matriz {n}x{n}", font=F3, bg=FONDO, fg="#546e7a").pack(pady=(10,5))
        gr = tk.Frame(self.fee, bg=FONDO); gr.pack(pady=4)

        for j in range(n):
            tk.Label(gr, text=f"col {j+1}", font=F1, bg=C1, fg=BLANCO,
                     width=7, pady=3, relief="flat").grid(row=0, column=j+1, padx=2, pady=2)
        for i in range(n):
            tk.Label(gr, text=f"fila {i+1}", font=F1, bg=C1, fg=BLANCO,
                     width=7, pady=3, relief="flat").grid(row=i+1, column=0, padx=2, pady=2)
            fila = []; phs = []
            for j in range(n):
                ph = f"a{i+1}{j+1}"; phs.append(ph)
                fr = tk.Frame(gr, bg=C3, padx=1, pady=1)
                fr.grid(row=i+1, column=j+1, padx=2, pady=2)
                e = tk.Entry(fr, width=7, font=F2, bg=BLANCO, fg="#aaaaaa", relief="flat", justify="center")
                e.pack(); e.insert(0, ph)
                e.bind("<FocusIn>",  lambda ev, en=e, p=ph: (en.delete(0,"end"), en.config(fg=NEGRO)) if en.get()==p else None)
                e.bind("<FocusOut>", lambda ev, en=e, p=ph: (en.insert(0,p), en.config(fg="#aaaaaa")) if en.get().strip()==""  else None)
                fila.append(e)
            self.ee.append(fila); self.pe.append(phs)
        tk.Button(self.fee, text="Calcular", font=F1, bg=C1, fg=BLANCO,
                  relief="flat", padx=14, pady=6, command=self._calcular).pack(pady=12)

    def _calcular(self):
        n = self.en.get()
        try:
            A = [[leer(self.ee[i][j], self.pe[i][j]) for j in range(n)] for i in range(n)]
        except:
            messagebox.showerror("Error", "Revisa que todos los valores sean numeros validos.")
            return

        coefs = pol_car(A)
        raices = buscar_raices(coefs)

        self.te.config(state="normal"); self.te.delete("1.0", "end")

        # polinomio caracteristico
        self.te.insert("end", "Polinomio caracteristico:\n", "tit")
        grado = len(coefs) - 1
        terminos = []
        for i, c in enumerate(coefs):
            exp = grado - i
            if abs(c) < 1e-10: continue
            cf = fmt(c)
            if exp == 0:   terminos.append(cf)
            elif exp == 1: terminos.append(f"({cf})L")
            else:          terminos.append(f"({cf})L^{exp}")
        ps = " + ".join(terminos).replace("+ (-", "- (")
        self.te.insert("end", "  p(L) = " + ps + "\n\n")

        # valores propios
        self.te.insert("end", f"Valores propios ({len(raices)} encontrados):\n", "tit")
        for k, (re, im) in enumerate(raices):
            if abs(im) < 1e-4:
                self.te.insert("end", f"  L{k+1} = {fmt(re)}\n", "vr")
            else:
                sg = "+" if im >= 0 else "-"
                self.te.insert("end", f"  L{k+1} = {fmt(re)} {sg} {fmt(abs(im))}i\n", "vc")

        # vectores propios para valores reales
        reales   = [(k, re) for k, (re, im) in enumerate(raices) if abs(im) < 1e-4]
        complejos = [(k, re, im) for k, (re, im) in enumerate(raices) if abs(im) >= 1e-4]

        self.te.insert("end", "\nVectores propios (valores reales):\n", "tit")
        if not reales:
            self.te.insert("end", "  Todos los valores son complejos.\n")
        for k, lam in reales:
            vp = vec_propio(A, lam)
            vs = "[" + ", ".join(fmt(v) for v in vp) + "]"
            self.te.insert("end", f"  v{k+1} (L={fmt(lam)}) = {vs}\n", "vec")

        if complejos:
            self.te.insert("end", "\nValores complejos:\n", "tit")
            for k, re, im in complejos:
                sg = "+" if im >= 0 else "-"
                self.te.insert("end", f"  L{k+1} = {fmt(re)} {sg} {fmt(abs(im))}i\n", "vc")

        # verificacion
        if reales:
            self.te.insert("end", "\nVerificacion Av = Lv:\n", "tit")
            for k, lam in reales:
                vp = vec_propio(A, lam)
                Av = [sum(A[i][j]*vp[j] for j in range(n)) for i in range(n)]
                lv = [lam*vp[j] for j in range(n)]
                ok = all(abs(Av[i]-lv[i]) < 1e-3 for i in range(n))
                self.te.insert("end", f"  L{k+1}={fmt(lam)}: {'correcto' if ok else 'revisar'}\n")

        self.te.tag_config("tit", font=("Segoe UI", 10, "bold"), foreground=C1)
        self.te.tag_config("vr",  font=("Consolas", 11, "bold"), foreground=VERDE)
        self.te.tag_config("vc",  font=("Consolas", 10, "bold"), foreground="#e65100")
        self.te.tag_config("vec", font=("Consolas", 10),         foreground=C2)
        self.te.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()