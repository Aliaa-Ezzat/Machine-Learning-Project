import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)
from sklearn.impute import (
    SimpleImputer,
    KNNImputer
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def create_sample_data():

    np.random.seed(42)

    n = 500

    data = {
        "Age": np.random.randint(18, 70, n),
        "Salary": np.random.randint(30000, 120000, n),
        "Experience": np.random.randint(0, 40, n),
        "Education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"],
            n
        ),
        "Department": np.random.choice(
            ["HR", "IT", "Sales", "Marketing"],
            n
        ),
        "Purchased": np.random.choice(
            [0, 1],
            n,
            p=[0.4, 0.6]
        )
    }

    df = pd.DataFrame(data)

    df.loc[
        np.random.choice(n, 20, replace=False),
        "Salary"
    ] = np.nan

    df.loc[
        np.random.choice(n, 15, replace=False),
        "Experience"
    ] = np.nan

    return df

class ModernMLApp:

    def __init__(self):

        self.root = ctk.CTk()

        self.root.title(" Machine Learning Project")

        self.root.geometry("1150x650")

        self.data = create_sample_data()

        self.model = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.setup_ui()

    def setup_ui(self):

        container = ctk.CTkFrame(self.root)

        container.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        title = ctk.CTkLabel(
            container,
            text="🤖 Machine Learning Professional GUI",
            font=ctk.CTkFont(
                size=30,
                weight="bold"
            )
        )

        title.pack(pady=15)

        self.tabs = ctk.CTkTabview(
            container
        )

        self.tabs.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.tabs.add("📁 Dataset")
        self.tabs.add("📊 Visualization")
        self.tabs.add("⚙️ Preprocessing")
        self.tabs.add("🧠 Model Training")
        self.tabs.add("📈 Evaluation")

        self.dataset_tab()
        self.visualization_tab()
        self.preprocessing_tab()
        self.model_tab()
        self.evaluation_tab()

    def dataset_tab(self):

        tab = self.tabs.tab("📁 Dataset")

        top = ctk.CTkFrame(tab)

        top.pack(
            fill="x",
            padx=10,
            pady=10
        )

        upload_btn = ctk.CTkButton(
            top,
            text="📂 Upload CSV",
            command=self.upload_dataset
        )

        upload_btn.pack(
            side="left",
            padx=10,
            pady=10
        )

        reset_btn = ctk.CTkButton(
            top,
            text="🔄 Reset Data",
            command=self.reset_data,
            fg_color="red"
        )

        reset_btn.pack(
            side="left",
            padx=10
        )

        export_btn = ctk.CTkButton(
            top,
            text="💾 Export CSV",
            command=self.export_data,
            fg_color="green"
        )

        export_btn.pack(
            side="left",
            padx=10
        )

        self.info_label = ctk.CTkLabel(
            tab,
            text=""
        )

        self.info_label.pack(pady=5)

        frame = ctk.CTkFrame(tab)

        frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.tree = ttk.Treeview(frame)

        self.tree.pack(
            fill="both",
            expand=True
        )

        self.update_preview()

    def upload_dataset(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV Files", "*.csv")
            ]
        )

        if not path:
            return

        try:

            self.data = pd.read_csv(path)

            self.update_preview()

            self.update_comboboxes()

            messagebox.showinfo(
                "Success",
                "Dataset Uploaded Successfully"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

    def reset_data(self):

        self.data = create_sample_data()

        self.update_preview()

        self.update_comboboxes()

        messagebox.showinfo(
            "Success",
            "Dataset Reset Successfully"
        )

    def export_data(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".csv"
        )

        if path:

            self.data.to_csv(
                path,
                index=False
            )

            messagebox.showinfo(
                "Success",
                "Dataset Exported"
            )

    def update_preview(self):

        for item in self.tree.get_children():

            self.tree.delete(item)

        cols = list(self.data.columns)

        self.tree["columns"] = cols

        self.tree["show"] = "headings"

        for col in cols:

            self.tree.heading(
                col,
                text=col
            )

            self.tree.column(
                col,
                width=120
            )

        for _, row in self.data.head(15).iterrows():

            self.tree.insert(
                "",
                "end",
                values=list(row)
            )

        self.info_label.configure(
            text=f"Rows: {self.data.shape[0]} | Columns: {self.data.shape[1]} | Missing Values: {self.data.isnull().sum().sum()}"
        )

    def visualization_tab(self):

        tab = self.tabs.tab("📊 Visualization")

        controls = ctk.CTkFrame(tab)

        controls.pack(
            fill="x",
            padx=10,
            pady=10
        )

        self.plot_type = ctk.CTkComboBox(
            controls,
            values=[
                "Scatter Plot",
                "Histogram",
                "Box Plot",
                "Heatmap"
            ]
        )

        self.plot_type.pack(
            side="left",
            padx=10,
            pady=10
        )

        self.x_col = ctk.CTkComboBox(
            controls,
            values=list(self.data.columns)
        )

        self.x_col.pack(
            side="left",
            padx=10
        )

        self.y_col = ctk.CTkComboBox(
            controls,
            values=list(self.data.columns)
        )

        self.y_col.pack(
            side="left",
            padx=10
        )

        plot_btn = ctk.CTkButton(
            controls,
            text="📈 Generate Plot",
            command=self.generate_plot
        )

        plot_btn.pack(
            side="left",
            padx=10
        )

    def generate_plot(self):

        plot = self.plot_type.get()

        plt.figure(figsize=(10, 6))

        try:

            if plot == "Scatter Plot":

                sns.scatterplot(
                    data=self.data,
                    x=self.x_col.get(),
                    y=self.y_col.get()
                )

            elif plot == "Histogram":

                self.data[
                    self.x_col.get()
                ].hist(
                    bins=30
                )

            elif plot == "Box Plot":

                sns.boxplot(
                    data=self.data,
                    x=self.x_col.get(),
                    y=self.y_col.get()
                )

            else:

                numeric = self.data.select_dtypes(
                    include=np.number
                )

                sns.heatmap(
                    numeric.corr(),
                    annot=True
                )

            plt.tight_layout()

            plt.show()

        except Exception as e:

            messagebox.showerror(
                "Plot Error",
                str(e)
            )

    def preprocessing_tab(self):

        tab = self.tabs.tab("⚙️ Preprocessing")

        frame = ctk.CTkScrollableFrame(tab)

        frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # Missing Values
        ctk.CTkLabel(
            frame,
            text="Missing Values",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(pady=10)

        self.imputer_combo = ctk.CTkComboBox(
            frame,
            values=[
                "Mean",
                "Median",
                "KNN",
                "Iterative"
            ]
        )

        self.imputer_combo.pack(pady=5)

        ctk.CTkButton(
            frame,
            text="Apply Imputer",
            command=self.apply_imputer
        ).pack(pady=5)

        ctk.CTkLabel(
            frame,
            text="Encoding",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(pady=10)

        self.encoding_col = ctk.CTkComboBox(
            frame,
            values=list(self.data.columns)
        )

        self.encoding_col.pack(pady=5)

        self.encoding_type = ctk.CTkComboBox(
            frame,
            values=[
                "Label Encoder",
                "One Hot Encoding"
            ]
        )

        self.encoding_type.pack(pady=5)

        ctk.CTkButton(
            frame,
            text="Apply Encoding",
            command=self.apply_encoding
        ).pack(pady=5)

        ctk.CTkLabel(
            frame,
            text="Scaling",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(pady=10)

        self.scaler_combo = ctk.CTkComboBox(
            frame,
            values=[
                "StandardScaler",
                "MinMaxScaler"
            ]
        )

        self.scaler_combo.pack(pady=5)

        ctk.CTkButton(
            frame,
            text="Apply Scaling",
            command=self.apply_scaling
        ).pack(pady=5)

        ctk.CTkLabel(
            frame,
            text="PCA",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(pady=10)

        self.pca_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Number of Components"
        )

        self.pca_entry.pack(pady=5)

        ctk.CTkButton(
            frame,
            text="Apply PCA",
            command=self.apply_pca
        ).pack(pady=5)

        ctk.CTkLabel(
            frame,
            text="SMOTE",
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        ).pack(pady=10)

        ctk.CTkButton(
            frame,
            text="Apply SMOTE",
            command=self.apply_smote
        ).pack(pady=5)

    def apply_imputer(self):

        method = self.imputer_combo.get()

        num_cols = self.data.select_dtypes(
            include=np.number
        ).columns

        if method == "Mean":

            imputer = SimpleImputer(
                strategy="mean"
            )

        elif method == "Median":

            imputer = SimpleImputer(
                strategy="median"
            )

        elif method == "KNN":

            imputer = KNNImputer()

        else:

            imputer = IterativeImputer()

        self.data[num_cols] = imputer.fit_transform(
            self.data[num_cols]
        )

        self.update_preview()

        messagebox.showinfo(
            "Success",
            f"{method} Imputer Applied"
        )

    def apply_encoding(self):

        col = self.encoding_col.get()

        enc = self.encoding_type.get()

        try:

            if enc == "Label Encoder":

                self.data[col] = LabelEncoder().fit_transform(
                    self.data[col].astype(str)
                )

            else:

                dummies = pd.get_dummies(
                    self.data[col],
                    prefix=col
                )

                self.data = pd.concat(
                    [
                        self.data.drop(columns=[col]),
                        dummies
                    ],
                    axis=1
                )

            self.update_preview()

            self.update_comboboxes()

            messagebox.showinfo(
                "Success",
                "Encoding Applied"
            )

        except Exception as e:

            messagebox.showerror(
                "Encoding Error",
                str(e)
            )

    def apply_scaling(self):

        num_cols = self.data.select_dtypes(
            include=np.number
        ).columns

        scaler_type = self.scaler_combo.get()

        if scaler_type == "StandardScaler":

            scaler = StandardScaler()

        else:

            scaler = MinMaxScaler()

        self.data[num_cols] = scaler.fit_transform(
            self.data[num_cols]
        )

        self.update_preview()

        messagebox.showinfo(
            "Success",
            f"{scaler_type} Applied"
        )

    def apply_pca(self):

        try:

            n = int(
                self.pca_entry.get()
            )

            target = self.target_col.get()

            X = self.data.drop(
                columns=[target]
            ).copy()

            y = self.data[target]

            for col in X.select_dtypes(
                include="object"
            ).columns:

                X[col] = LabelEncoder().fit_transform(
                    X[col].astype(str)
                )

            X = X.fillna(
                X.mean(numeric_only=True)
            )

            pca = PCA(
                n_components=n
            )

            reduced = pca.fit_transform(X)

            self.data = pd.DataFrame(
                reduced,
                columns=[
                    f"PC{i+1}"
                    for i in range(n)
                ]
            )

            self.data[target] = y.values

            self.update_preview()

            self.update_comboboxes()

            messagebox.showinfo(
                "Success",
                "PCA Applied"
            )

        except Exception as e:

            messagebox.showerror(
                "PCA Error",
                str(e)
            )

    def apply_smote(self):

        try:

            target = self.target_col.get()

            X = self.data.drop(
                columns=[target]
            )

            y = self.data[target]

            for col in X.select_dtypes(
                include="object"
            ).columns:

                X[col] = LabelEncoder().fit_transform(
                    X[col].astype(str)
                )

            smote = SMOTE(
                random_state=42
            )

            X_res, y_res = smote.fit_resample(
                X,
                y
            )

            self.data = pd.concat(
                [
                    pd.DataFrame(X_res),
                    pd.Series(y_res, name=target)
                ],
                axis=1
            )

            self.update_preview()

            self.update_comboboxes()

            messagebox.showinfo(
                "Success",
                "SMOTE Applied"
            )

        except Exception as e:

            messagebox.showerror(
                "SMOTE Error",
                str(e)
            )

    def model_tab(self):

        tab = self.tabs.tab("🧠 Model Training")

        frame = ctk.CTkFrame(tab)

        frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.problem_type = ctk.CTkComboBox(
            frame,
            values=[
                "Classification",
                "Regression",
                "Clustering"
            ]
        )

        self.problem_type.pack(pady=10)

        self.algorithm = ctk.CTkComboBox(
            frame,
            values=[
                "Random Forest",
                "Logistic Regression",
                "SVM",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Linear Regression",
                "KMeans"
            ]
        )

        self.algorithm.pack(pady=10)

        self.target_col = ctk.CTkComboBox(
            frame,
            values=list(self.data.columns)
        )

        self.target_col.pack(pady=10)

        self.target_col.set(
            self.data.columns[-1]
        )

        train_btn = ctk.CTkButton(
            frame,
            text="🚀 Train Model",
            command=self.train_model
        )

        train_btn.pack(pady=20)

        self.model_info = ctk.CTkTextbox(
            frame,
            height=350
        )

        self.model_info.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

    def train_model(self):

        try:

            target = self.target_col.get()

            X = self.data.drop(
                columns=[target]
            ).copy()

            y = self.data[target].copy()

            full = pd.concat(
                [X, y],
                axis=1
            ).dropna()

            X = full.drop(
                columns=[target]
            )

            y = full[target]

            for col in X.select_dtypes(
                include="object"
            ).columns:

                X[col] = LabelEncoder().fit_transform(
                    X[col].astype(str)
                )

            if y.dtype == "object":

                y = LabelEncoder().fit_transform(
                    y.astype(str)
                )

            scaler = StandardScaler()

            X = scaler.fit_transform(X)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            algo = self.algorithm.get()

            problem = self.problem_type.get()

            if problem == "Classification":

                if algo == "Random Forest":

                    self.model = RandomForestClassifier()

                elif algo == "Logistic Regression":

                    self.model = LogisticRegression(
                        max_iter=1000
                    )

                elif algo == "SVM":

                    self.model = SVC()

                elif algo == "Decision Tree":

                    self.model = DecisionTreeClassifier()

                elif algo == "KNN":

                    self.model = KNeighborsClassifier()

                else:

                    self.model = GaussianNB()

            elif problem == "Regression":

                if algo == "Random Forest":

                    self.model = RandomForestRegressor()

                else:

                    self.model = LinearRegression()

            else:

                self.model = KMeans(
                    n_clusters=3
                )

            self.model.fit(
                self.X_train,
                self.y_train
            )

            self.model_info.delete(
                "1.0",
                "end"
            )

            self.model_info.insert(
                "1.0",
                f"""
Model Trained Successfully

Problem Type: {problem}

Algorithm: {algo}

Training Samples: {len(self.X_train)}

Testing Samples: {len(self.X_test)}
"""
            )

            messagebox.showinfo(
                "Success",
                "Model Trained Successfully"
            )

        except Exception as e:

            messagebox.showerror(
                "Training Error",
                str(e)
            )

    def evaluation_tab(self):

        tab = self.tabs.tab("📈 Evaluation")

        btn = ctk.CTkButton(
            tab,
            text="📊 Evaluate Model",
            command=self.evaluate_model
        )

        btn.pack(pady=20)

        self.eval_text = ctk.CTkTextbox(
            tab
        )

        self.eval_text.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

    def evaluate_model(self):

        try:

            if self.model is None:

                messagebox.showerror(
                    "Error",
                    "Train Model First"
                )

                return

            self.eval_text.delete(
                "1.0",
                "end"
            )

            problem = self.problem_type.get()

            if problem == "Classification":

                pred = self.model.predict(
                    self.X_test
                )

                acc = accuracy_score(
                    self.y_test,
                    pred
                )

                report = classification_report(
                    self.y_test,
                    pred
                )

                matrix = confusion_matrix(
                    self.y_test,
                    pred
                )

                self.eval_text.insert(
                    "1.0",
                    f"""
Accuracy: {acc:.4f}

Classification Report:

{report}

Confusion Matrix:

{matrix}
"""
                )

            elif problem == "Regression":

                pred = self.model.predict(
                    self.X_test
                )

                r2 = r2_score(
                    self.y_test,
                    pred
                )

                mse = mean_squared_error(
                    self.y_test,
                    pred
                )

                self.eval_text.insert(
                    "1.0",
                    f"""
R2 Score: {r2:.4f}

MSE: {mse:.4f}
"""
                )

            else:

                self.eval_text.insert(
                    "1.0",
                    f"""
KMeans Inertia:

{self.model.inertia_}
"""
                )

        except Exception as e:

            messagebox.showerror(
                "Evaluation Error",
                str(e)
            )

    def update_comboboxes(self):

        cols = list(self.data.columns)

        self.x_col.configure(
            values=cols
        )

        self.y_col.configure(
            values=cols
        )

        self.encoding_col.configure(
            values=cols
        )

        self.target_col.configure(
            values=cols
        )

    def run(self):

        self.root.mainloop()

if __name__ == "__main__":

    app = ModernMLApp()

    app.run()