"""
ui/teclado.py
Teclado virtual táctil para CustomTkinter.
Se instancia una vez en FaceAccess y se reutiliza en todos los campos.

Uso:
    self._teclado = TecladoVirtual(root=self)
    entry.bind("<FocusIn>", lambda e: self._teclado.abrir(entry))
"""

import customtkinter as ctk
from ui.constantes import (
    C_BG, C_FRAME, C_BORDE, C_OK, C_TXT, C_TXT2, C_ERROR,
    KB_APP_W, KB_COLS, KB_PAD, KB_BH, KB_FS,
)


class TecladoVirtual:

    def __init__(self, root):
        self._root   = root
        self._target = None
        self._mayus  = False

        # Dimensiones base (se recalculan en abrir)
        self._APP_W = APP_W
        # Separacion horizontal minima entre teclas.
        self._PAD   = max(KB_PAD, 1)
        # Separacion vertical de teclas: 2x respecto a la horizontal.
        self._PAD_Y = self._PAD * 2
        self._BH    = max(KB_BH, 48)
        self._FS    = max(KB_FS, 16)
        self._BW    = 52
        self._SPC_W = 320
        self._OK_W  = 180
        self._KBD_W = KB_APP_W
        self._KBD_H = 360

        self._frame = ctk.CTkFrame(root, fg_color="#0A1520", corner_radius=0)

    def _recalcular_dimensiones(self):
        """Ajustar teclado al ancho real de ventana para centrar y agrandar teclas."""
        try:
            self._root.update_idletasks()
            win_w = self._root.winfo_width()
            if win_w <= 1:
                win_w = self._APP_W
        except Exception:
            win_w = self._APP_W

        # Ancho del teclado reducido para evitar que quede cargado a los bordes.
        self._KBD_W = max(480, int(win_w * 0.52))
        self._BW = max(44, (self._KBD_W - self._PAD * (KB_COLS + 1)) // KB_COLS)
        self._OK_W = max(150, int(self._KBD_W * 0.28))
        self._SPC_W = max(220, self._KBD_W - self._OK_W - (self._PAD * 6))
        # Alto total considerando la separacion vertical ampliada.
        self._KBD_H = (self._BH * 4) + (self._PAD_Y * 8) + (self._PAD * 3) + 66

    # ── API pública ───────────────────────────────────────────────────────────

    def abrir(self, entry_target):
        """Muestra el teclado ligado al entry dado."""
        if self._frame.winfo_ismapped():
            self._frame.place_forget()
        self._target = entry_target
        self._mayus  = False
        for w in self._frame.winfo_children():
            w.destroy()
        self._renderizar()
        self._frame.place(relx=0, rely=0.45, relwidth=1, relheight=0.55)
        self._frame.lift()
        self._frame.tkraise()

    def cerrar(self):
        """Oculta el teclado."""
        self._frame.place_forget()

    # ── Renderizado ───────────────────────────────────────────────────────────

    def _renderizar(self):
        for w in self._frame.winfo_children():
            w.destroy()

        filas = [
            ["1","2","3","4","5","6","7","8","9","0"],
            ["Q","W","E","R","T","Y","U","I","O","P"],
            ["A","S","D","F","G","H","J","K","L","⌫"],
            ["⇧","Z","X","C","V","B","N","M","-","_"],
        ]

        for r_idx, fila in enumerate(filas):
            for c_idx, tecla in enumerate(fila):
                texto = tecla if not tecla.isalpha() else \
                    (tecla if self._mayus else tecla.lower())

                if tecla == "⌫":
                    b = ctk.CTkButton(
                        self._frame, text=tecla,
                        width=self._BW + 10, height=self._BH,
                        font=("Helvetica", self._FS),
                        fg_color=C_FRAME, text_color=C_ERROR,
                        hover_color=C_BORDE, border_width=1,
                        border_color=C_BORDE, corner_radius=8,
                        command=self._del)

                elif tecla == "⇧":
                    b = ctk.CTkButton(
                        self._frame, text=tecla,
                        width=self._BW + 10, height=self._BH,
                        font=("Helvetica", self._FS),
                        fg_color=C_OK if self._mayus else C_FRAME,
                        text_color=C_BG if self._mayus else C_TXT,
                        hover_color=C_BORDE, border_width=1,
                        border_color=C_BORDE, corner_radius=8,
                        command=self._toggle_mayus)

                else:
                    b = ctk.CTkButton(
                        self._frame, text=texto,
                        width=self._BW, height=self._BH,
                        font=("Helvetica", self._FS, "bold"),
                        fg_color=C_FRAME, text_color=C_TXT,
                        hover_color=C_BORDE, border_width=1,
                        border_color=C_BORDE, corner_radius=8,
                        command=lambda t=texto: self._press(t))

                # Mantiene separacion horizontal compacta y vertical mas respirada.
                b.grid(row=r_idx, column=c_idx, padx=self._PAD, pady=self._PAD_Y)

        # Fila inferior: Espacio + Listo
        fb = ctk.CTkFrame(self._frame, fg_color="transparent")
        fb.grid(row=4, column=0, columnspan=11,
                padx=self._PAD, pady=(4, 8), sticky="ew")

        ctk.CTkButton(
            fb, text="Espacio",
            width=self._SPC_W, height=70,
            font=("Helvetica", self._FS),
            fg_color=C_FRAME, text_color=C_TXT,
            hover_color=C_BORDE, border_width=1,
            border_color=C_BORDE, corner_radius=8,
            command=lambda: self._press(" ")
        ).pack(side="left", padx=self._PAD)

        ctk.CTkButton(
            fb, text="Listo ✓",
            width=140, height=70,
            font=("Helvetica", self._FS, "bold"),
            fg_color=C_OK, text_color=C_BG,
            hover_color="#00A88A", corner_radius=8,
            command=self.cerrar
        ).pack(side="left", padx=self._PAD)

    # ── Acciones de teclas ────────────────────────────────────────────────────

    def _press(self, t):
        if self._target:
            self._target.insert("end", t)

    def _del(self):
        if self._target:
            val = self._target.get()
            self._target.delete(0, "end")
            self._target.insert(0, val[:-1])

    def _toggle_mayus(self):
        self._mayus = not self._mayus
        self._renderizar()