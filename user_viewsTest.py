import customtkinter as ctk

class UsersView(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        self.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            self,
            text="VISTA DE USUARIOS (PRUEBA)",
            font=("Inter", 30, "bold")
        )
        label.pack(pady=100)

        btn = ctk.CTkButton(
            self,
            text="Volver al registro",
            command=self.volver
        )
        btn.pack()

    def volver(self):
        self.destroy()