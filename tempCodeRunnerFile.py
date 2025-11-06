import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import numpy as np

class CarPriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Car Price Analysis (Regression + Clustering)")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f8ff")

        self.model = None
        self.kmeans = None
        self.scaler = None
        self.df = None

        # Create notebook (tab layout)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="🏠 Home / Prediction")
        self.notebook.add(self.tab2, text="📊 Visualizations")

        self.create_home_tab()
        self.create_visualization_tab()

    # ---------------- HOME TAB ----------------
    def create_home_tab(self):
        ttk.Label(self.tab1, text="🚗 Car Price Analysis", font=("Helvetica", 18, "bold")).pack(pady=10)
        ttk.Button(self.tab1, text="📂 Load Dataset & Train Models", command=self.load_and_train).pack(pady=10)

        frame = ttk.LabelFrame(self.tab1, text="Enter Car Details", padding=10)
        frame.pack(pady=10, padx=10, fill="x")

        # Placeholder helper
        def add_placeholder(entry, text):
            entry.insert(0, text)
            entry.config(foreground="gray")
            def on_focus_in(event):
                if entry.get() == text:
                    entry.delete(0, "end")
                    entry.config(foreground="black")
            def on_focus_out(event):
                if not entry.get():
                    entry.insert(0, text)
                    entry.config(foreground="gray")
            entry.bind("<FocusIn>", on_focus_in)
            entry.bind("<FocusOut>", on_focus_out)

        # Input fields with examples
        self.engine_entry = ttk.Entry(frame)
        self.hp_entry = ttk.Entry(frame)
        self.mileage_entry = ttk.Entry(frame)
        self.width_entry = ttk.Entry(frame)
        self.length_entry = ttk.Entry(frame)

        labels = ["Engine Size (cc):", "Horsepower:", "Mileage (mpg):", "Car Width (inches):", "Car Length (inches):"]
        placeholders = ["e.g. 1500–4000", "e.g. 70–350", "e.g. 10–35", "e.g. 60–70", "e.g. 150–200"]

        for i, (label, entry, hint) in enumerate(zip(labels,
                                                    [self.engine_entry, self.hp_entry, self.mileage_entry, self.width_entry, self.length_entry],
                                                    placeholders)):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry.grid(row=i, column=1, padx=5, pady=5)
            add_placeholder(entry, hint)

        ttk.Button(self.tab1, text="🔮 Predict Car Price", command=self.predict_price).pack(pady=10)

        self.result_label = ttk.Label(self.tab1, text="", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

        self.cluster_label = ttk.Label(self.tab1, text="", font=("Helvetica", 12))
        self.cluster_label.pack(pady=5)

    # ---------------- VISUALIZATION TAB ----------------
    def create_visualization_tab(self):
        ttk.Label(self.tab2, text="📊 Simple Data Visualizations", font=("Helvetica", 18, "bold")).pack(pady=10)

        ttk.Button(self.tab2, text="🚘 View Car Groups (Clusters)", command=self.show_clusters).pack(pady=10)
        ttk.Button(self.tab2, text="💰 Average Price by Cluster", command=self.show_avg_price).pack(pady=10)
        ttk.Button(self.tab2, text="⚙️ Horsepower vs Price", command=self.show_scatter).pack(pady=10)

        self.canvas_frame = ttk.Frame(self.tab2)
        self.canvas_frame.pack(fill="both", expand=True)

    # ---------------- LOAD & TRAIN ----------------
    def load_and_train(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Select CarPrice_Assignment.csv",
                filetypes=[("CSV Files", "*.csv")]
            )
            if not file_path:
                return

            df = pd.read_csv(file_path)
            features = ["enginesize", "horsepower", "citympg", "carwidth", "carlength"]
            df = df[features + ["price"]].dropna()

            self.df = df

            X = df[features]
            y = df["price"]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            df["Cluster"] = clusters

            # Linear Regression
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            joblib.dump(model, "car_price_model.pkl")
            joblib.dump(kmeans, "car_kmeans.pkl")
            joblib.dump(scaler, "car_scaler.pkl")

            self.model = model
            self.kmeans = kmeans
            self.scaler = scaler

            messagebox.showinfo("Training Complete", f"✅ Models trained successfully!\n\nR² Score: {r2:.2f}\nMAE: {mae:.2f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- PREDICTION ----------------
    def predict_price(self):
        if not self.model or not self.kmeans:
            if os.path.exists("car_price_model.pkl"):
                self.model = joblib.load("car_price_model.pkl")
                self.kmeans = joblib.load("car_kmeans.pkl")
                self.scaler = joblib.load("car_scaler.pkl")
            else:
                messagebox.showwarning("Warning", "Please train the models first!")
                return

        try:
            eng = float(self.engine_entry.get().split()[0])
            hp = float(self.hp_entry.get().split()[0])
            mpg = float(self.mileage_entry.get().split()[0])
            width = float(self.width_entry.get().split()[0])
            length = float(self.length_entry.get().split()[0])

            features = [[eng, hp, mpg, width, length]]
            scaled = self.scaler.transform(features)
            cluster = self.kmeans.predict(scaled)[0]
            prediction = self.model.predict(features)[0]

            cluster_names = ["🚗 Budget Cars", "🚙 Mid-Range Cars", "🏎️ Luxury Cars"]
            cluster_name = cluster_names[cluster]

            self.result_label.config(text=f"💰 Estimated Price: ${prediction:,.2f}")
            self.cluster_label.config(text=f"Category: {cluster_name}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")

    # ---------------- VISUALIZATIONS ----------------
    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def embed_plot(self):
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close()

    def show_clusters(self):
        self.clear_canvas()
        if self.df is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        plt.figure(figsize=(6, 5))
        colors = ["#007bff", "#ff9900", "#28a745"]
        labels = ["Budget Cars", "Mid-Range Cars", "Luxury Cars"]
        for i in range(3):
            cluster_data = self.df[self.df["Cluster"] == i]
            plt.scatter(cluster_data["enginesize"], cluster_data["price"], color=colors[i], label=labels[i])
        plt.xlabel("Engine Size (cc)")
        plt.ylabel("Price ($)")
        plt.title("Car Groups Based on Price Range")
        plt.legend()
        self.embed_plot()

    def show_avg_price(self):
        self.clear_canvas()
        if self.df is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        avg_price = self.df.groupby("Cluster")["price"].mean()
        labels = ["Budget Cars", "Mid-Range Cars", "Luxury Cars"]
        colors = ["#007bff", "#ff9900", "#28a745"]

        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, avg_price, color=colors)
        plt.bar_label(bars, fmt="%.0f")
        plt.ylabel("Average Price ($)")
        plt.title("Average Price per Car Category")
        self.embed_plot()

    def show_scatter(self):
        self.clear_canvas()
        if self.df is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        plt.figure(figsize=(6, 5))
        plt.scatter(self.df["horsepower"], self.df["price"], color="#ff7f0e")
        plt.xlabel("Horsepower")
        plt.ylabel("Price ($)")
        plt.title("Relation Between Horsepower and Price")
        self.embed_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = CarPriceApp(root)
    root.mainloop()
