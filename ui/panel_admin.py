"""
ui/panel_admin.py
Panel administrativo — se abre tras login facial exitoso.
Tabs: ➕ Nuevo usuario | 👥 Usuarios | 📋 Accesos
"""

import customtkinter as ctk
from ui.constantes import (
    C_BG, C_FRAME, C_FOOT, C_BORDE, C_OK, C_WARN, C_ERROR,
    C_TXT, C_TXT2, C_TXT3, C_ADMIN, px, fs,
)
from database.db_manager import listar_usuarios, listar_accesos, desactivar_usuario


class PanelAdmin(ctk.CTkFrame):
    """
    Frame que ocupa todo el frame_video.
    Se instancia una sola vez en _build_overlays y se muestra/oculta con place/place_forget.
    """

    def __init__(self, parent, on_nuevo_usuario, on_cerrar, **kwargs):
        super().__init__(parent, fg_color="#080F16", corner_radius=0, **kwargs)
        self._on_nuevo_usuario = on_nuevo_usuario
        self._on_cerrar        = on_cerrar
        self._tab_actual       = None
        self._operador_nombre  = ""
        self._build()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build(self):
        # ── Cabecera del panel ────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=C_FRAME, corner_radius=0, height=px(46))
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        self.lbl_op = ctk.CTkLabel(hdr, text="Panel Admin",
                                    font=("Helvetica", fs(13), "bold"),
                                    text_color=C_OK, fg_color="transparent")
        self.lbl_op.pack(side="left", padx=px(14), anchor="center")

        ctk.CTkButton(hdr, text="✕ Salir", width=px(80), height=px(30),
                       fg_color="transparent", text_color=C_TXT2,
                       hover_color=C_BORDE, font=("Helvetica", fs(11)),
                       command=self._on_cerrar).pack(side="right", padx=px(10))

        # ── Barra de tabs ─────────────────────────────────────────────────────
        tab_bar = ctk.CTkFrame(self, fg_color=C_FOOT, corner_radius=0, height=px(44))
        tab_bar.pack(fill="x"); tab_bar.pack_propagate(False)

        self._btn_tabs = {}
        tabs = [
            ("nuevo",   "➕  Nuevo"),
            ("usuarios","👥  Usuarios"),
            ("accesos", "📋  Accesos"),
        ]
        for key, label in tabs:
            b = ctk.CTkButton(tab_bar, text=label, height=px(44),
                               fg_color="transparent", text_color=C_TXT2,
                               hover_color=C_BORDE, corner_radius=0,
                               font=("Helvetica", fs(12), "bold"),
                               command=lambda k=key: self._cambiar_tab(k))
            b.pack(side="left", expand=True, fill="both")
            self._btn_tabs[key] = b

        # ── Contenedor de contenido ───────────────────────────────────────────
        self._contenido = ctk.CTkFrame(self, fg_color="#080F16", corner_radius=0)
        self._contenido.pack(fill="both", expand=True)

        # ── Frames de cada tab ────────────────────────────────────────────────
        self._frame_nuevo    = self._build_tab_nuevo()
        self._frame_usuarios = self._build_tab_usuarios()
        self._frame_accesos  = self._build_tab_accesos()

    # ── Tab: Nuevo usuario ────────────────────────────────────────────────────

    def _build_tab_nuevo(self):
        f = ctk.CTkFrame(self._contenido, fg_color="transparent")
        inner = ctk.CTkFrame(f, fg_color="transparent")
        inner.place(relx=0.5, rely=0.45, anchor="center")

        ctk.CTkLabel(inner, text="➕", font=("Helvetica", fs(40)),
                     fg_color="transparent").pack(pady=(0, px(10)))
        ctk.CTkLabel(inner, text="Registrar nuevo usuario",
                     font=("Helvetica", fs(15), "bold"),
                     text_color=C_TXT, fg_color="transparent").pack(pady=(0, px(6)))
        ctk.CTkLabel(inner,
                     text="Se abrirá el formulario de registro.\nLuego se capturará el rostro.",
                     font=("Helvetica", fs(12)), text_color=C_TXT2,
                     fg_color="transparent", justify="center").pack(pady=(0, px(24)))
        ctk.CTkButton(inner, text="Ir al formulario  →",
                       width=px(220), height=px(46),
                       fg_color=C_OK, text_color=C_BG, hover_color="#00A88A",
                       font=("Helvetica", fs(13), "bold"), corner_radius=12,
                       command=self._on_nuevo_usuario).pack()
        return f

    # ── Tab: Usuarios ─────────────────────────────────────────────────────────

    def _build_tab_usuarios(self):
        f = ctk.CTkFrame(self._contenido, fg_color="transparent")

        # Barra superior con contador y refresh
        top = ctk.CTkFrame(f, fg_color=C_FRAME, corner_radius=0, height=px(36))
        top.pack(fill="x"); top.pack_propagate(False)
        self.lbl_u_total = ctk.CTkLabel(top, text="", font=("Helvetica", fs(11)),
                                         text_color=C_TXT2, fg_color="transparent")
        self.lbl_u_total.pack(side="left", padx=px(12), anchor="center")
        ctk.CTkButton(top, text="↻", width=px(36), height=px(28),
                       fg_color="transparent", text_color=C_OK,
                       hover_color=C_BORDE, font=("Helvetica", fs(14)),
                       command=self._cargar_usuarios).pack(side="right", padx=px(6))

        # Cabecera de columnas
        cols = ctk.CTkFrame(f, fg_color=C_FOOT, corner_radius=0, height=px(28))
        cols.pack(fill="x"); cols.pack_propagate(False)
        for texto, ancho in [("Nombre", 200), ("Matrícula", 90), ("Rol", 100), ("", 70)]:
            ctk.CTkLabel(cols, text=texto, font=("Helvetica", fs(10), "bold"),
                          text_color=C_TXT2, fg_color="transparent",
                          width=px(ancho)).pack(side="left", padx=px(4))

        # Lista scrollable
        self._scroll_usuarios = ctk.CTkScrollableFrame(
            f, fg_color="#080F16", corner_radius=0)
        self._scroll_usuarios.pack(fill="both", expand=True)
        return f

    def _cargar_usuarios(self):
        for w in self._scroll_usuarios.winfo_children():
            w.destroy()

        usuarios = listar_usuarios()
        activos  = sum(1 for u in usuarios if u["estatus"] == 1)
        self.lbl_u_total.configure(
            text=f"{activos} activos  ·  {len(usuarios) - activos} inactivos")

        for u in usuarios:
            activo = u["estatus"] == 1
            nombre = f"{u['nombre']} {u['apellido_p']}"
            if u.get("grado") and u.get("grupo"):
                nombre += f"  ({u['grado']}{u['grupo']})"

            row = ctk.CTkFrame(self._scroll_usuarios,
                                fg_color=C_FRAME if activo else "#1a1a1a",
                                corner_radius=6)
            row.pack(fill="x", padx=px(6), pady=px(2))

            ctk.CTkLabel(row, text=nombre, font=("Helvetica", fs(11)),
                          text_color=C_TXT if activo else C_TXT3,
                          fg_color="transparent", width=px(200),
                          anchor="w").pack(side="left", padx=px(8), pady=px(6))

            ctk.CTkLabel(row, text=u["matricula"], font=("Helvetica", fs(10)),
                          text_color=C_TXT2 if activo else C_TXT3,
                          fg_color="transparent", width=px(90)).pack(side="left")

            ctk.CTkLabel(row, text=u["nombre_rol"], font=("Helvetica", fs(10)),
                          text_color=C_ADMIN if activo else C_TXT3,
                          fg_color="transparent", width=px(100)).pack(side="left")

            if activo:
                ctk.CTkButton(row, text="Dar baja", width=px(68), height=px(26),
                               fg_color=C_ERROR, text_color="white",
                               hover_color="#a00000", font=("Helvetica", fs(10)),
                               corner_radius=6,
                               command=lambda uid=u["id_usuario"], n=nombre:
                               self._confirmar_baja(uid, n)).pack(side="right", padx=px(6))
            else:
                ctk.CTkLabel(row, text="Inactivo", font=("Helvetica", fs(10)),
                              text_color=C_TXT3, fg_color="transparent",
                              width=px(68)).pack(side="right", padx=px(6))

    def _confirmar_baja(self, id_usuario, nombre):
        """Mini-diálogo de confirmación dentro del scroll."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("Confirmar baja")
        dlg.geometry(f"{px(300)}x{px(160)}")
        dlg.resizable(False, False)
        dlg.configure(fg_color=C_FRAME)
        dlg.grab_set()

        ctk.CTkLabel(dlg, text=f"¿Dar de baja a\n{nombre}?",
                     font=("Helvetica", fs(12)), text_color=C_TXT,
                     justify="center").pack(pady=(px(20), px(16)))

        fb = ctk.CTkFrame(dlg, fg_color="transparent")
        fb.pack()

        def _si():
            desactivar_usuario(id_usuario)
            dlg.destroy()
            self._cargar_usuarios()

        ctk.CTkButton(fb, text="Sí, dar baja", width=px(110), height=px(36),
                       fg_color=C_ERROR, text_color="white",
                       hover_color="#a00000", font=("Helvetica", fs(11), "bold"),
                       command=_si).pack(side="left", padx=px(6))
        ctk.CTkButton(fb, text="Cancelar", width=px(110), height=px(36),
                       fg_color="transparent", text_color=C_TXT2,
                       hover_color=C_BORDE, font=("Helvetica", fs(11)),
                       command=dlg.destroy).pack(side="left", padx=px(6))

    # ── Tab: Accesos ──────────────────────────────────────────────────────────

    def _build_tab_accesos(self):
        f = ctk.CTkFrame(self._contenido, fg_color="transparent")

        top = ctk.CTkFrame(f, fg_color=C_FRAME, corner_radius=0, height=px(36))
        top.pack(fill="x"); top.pack_propagate(False)
        self.lbl_a_total = ctk.CTkLabel(top, text="", font=("Helvetica", fs(11)),
                                         text_color=C_TXT2, fg_color="transparent")
        self.lbl_a_total.pack(side="left", padx=px(12), anchor="center")
        ctk.CTkButton(top, text="↻", width=px(36), height=px(28),
                       fg_color="transparent", text_color=C_OK,
                       hover_color=C_BORDE, font=("Helvetica", fs(14)),
                       command=self._cargar_accesos).pack(side="right", padx=px(6))

        cols = ctk.CTkFrame(f, fg_color=C_FOOT, corner_radius=0, height=px(28))
        cols.pack(fill="x"); cols.pack_propagate(False)
        for texto, ancho in [("Nombre", 190), ("Fecha", 80), ("Hora", 60), ("Tipo", 70)]:
            ctk.CTkLabel(cols, text=texto, font=("Helvetica", fs(10), "bold"),
                          text_color=C_TXT2, fg_color="transparent",
                          width=px(ancho)).pack(side="left", padx=px(4))

        self._scroll_accesos = ctk.CTkScrollableFrame(
            f, fg_color="#080F16", corner_radius=0)
        self._scroll_accesos.pack(fill="both", expand=True)
        return f

    def _cargar_accesos(self):
        for w in self._scroll_accesos.winfo_children():
            w.destroy()

        accesos = listar_accesos(50)
        self.lbl_a_total.configure(text=f"Últimos {len(accesos)} accesos")

        for a in accesos:
            nombre = f"{a['nombre']} {a['apellido_p']}"
            metodo = a["metodo"].upper()
            color_tipo = C_OK if metodo == "FACIAL" else C_WARN

            row = ctk.CTkFrame(self._scroll_accesos, fg_color=C_FRAME, corner_radius=6)
            row.pack(fill="x", padx=px(6), pady=px(2))

            ctk.CTkLabel(row, text=nombre, font=("Helvetica", fs(11)),
                          text_color=C_TXT, fg_color="transparent",
                          width=px(190), anchor="w").pack(side="left", padx=px(8), pady=px(5))

            ctk.CTkLabel(row, text=a["fecha"], font=("Helvetica", fs(10)),
                          text_color=C_TXT2, fg_color="transparent",
                          width=px(80)).pack(side="left")

            ctk.CTkLabel(row, text=a["hora_entrada"][:5],
                          font=("Helvetica", fs(10)),
                          text_color=C_TXT2, fg_color="transparent",
                          width=px(60)).pack(side="left")

            ctk.CTkLabel(row, text=metodo, font=("Helvetica", fs(10), "bold"),
                          text_color=color_tipo, fg_color="transparent",
                          width=px(70)).pack(side="left")

    # ── Control de tabs ───────────────────────────────────────────────────────

    def _cambiar_tab(self, key):
        # Ocultar todos
        for f in [self._frame_nuevo, self._frame_usuarios, self._frame_accesos]:
            f.place_forget()

        # Resaltar botón activo
        for k, b in self._btn_tabs.items():
            b.configure(
                fg_color=C_ADMIN if k == key else "transparent",
                text_color=C_TXT  if k == key else C_TXT2,
            )

        # Mostrar tab seleccionado y cargar datos
        if key == "nuevo":
            self._frame_nuevo.place(relx=0, rely=0, relwidth=1, relheight=1)
        elif key == "usuarios":
            self._frame_usuarios.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._cargar_usuarios()
        elif key == "accesos":
            self._frame_accesos.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._cargar_accesos()

        self._tab_actual = key

    # ── API pública ───────────────────────────────────────────────────────────

    def abrir(self, operador: dict):
        """Muestra el panel. Llamar desde ReconocimientoFacial tras login exitoso."""
        nom = f"{operador['nombre']} {operador['apellido_p']} · {operador['nombre_rol']}"
        self.lbl_op.configure(text=f"⚙  {nom}")
        self._cambiar_tab("nuevo")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

    def cerrar(self):
        self.place_forget()