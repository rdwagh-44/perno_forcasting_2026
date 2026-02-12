import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Kalman Beta Forecasting", layout="wide")

# tab1, tab2 = st.tabs(["Kalman Regression", "Post-Modeling Analysis"])

# with tab1:

menu = st.sidebar.selectbox(
    "Select Module",
    ["Kalman Regression", "Post-Modeling Analysis"]
)

if menu == "Kalman Regression":
    st.header("Kalman Regression")

    # =====================================================
    # ----------- KALMAN REGRESSION CLASS -----------------
    # =====================================================


    # class TimeVaryingKalmanRegression:
    #     """
    #     Time-Varying Kalman Regression with:
    #     - Per-variable process noise
    #     - Mean reversion to base beta (EXCEPT intercept)
    #     - Adaptive intercept (beta0)
    #     - Covariance-aware sign constraints
    #     - Per-variable max deviation from base beta
    #     """

    #     def __init__(
    #         self,
    #         q=1e-5,
    #         r=50.0,
    #         init_cov=10.0,
    #         q_scale=None,
    #         anchor_strength=0.05,
    #         non_negative_idx=None,
    #         non_positive_idx=None,
    #         max_dev=None              # ⬅️ NEW
    #     ):
    #         self.q = q
    #         self.r = r
    #         self.init_cov = init_cov
    #         self.q_scale = q_scale or {}
    #         self.anchor_strength = anchor_strength

    #         self.non_negative_idx = non_negative_idx or []
    #         self.non_positive_idx = non_positive_idx or []

    #         # max allowed deviation from base beta (standardized scale)
    #         # {x_idx: delta}
    #         self.max_dev = max_dev or {}

    #     # --------------------------------------------------

    #     def fit(self, X, y, base_beta=None, beta0_init=0.0):
    #         T, p = X.shape
    #         n_state = p + 1  # intercept + betas

    #         # ----- Initial state -----
    #         self.beta_ = np.zeros(n_state)
    #         self.beta_[0] = beta0_init
    #         if base_beta is not None:
    #             self.beta_[1:] = base_beta

    #         self.base_beta_ = self.beta_.copy()

    #         # ----- Covariances -----
    #         self.P_ = np.eye(n_state) * self.init_cov
    #         self.Q_ = self._build_Q(n_state)
    #         self.R_ = self.r
    #         self.I_ = np.eye(n_state)

    #         betas = np.zeros((T + 1, n_state))
    #         y_pred = np.zeros(T)
    #         betas[0] = self.beta_

    #         for t in range(T):
    #             y_pred[t] = self._kalman_step(X[t], y[t])
    #             betas[t + 1] = self.beta_

    #         self.betas_ = betas
    #         self.y_pred_ = y_pred
    #         return self

    #     # --------------------------------------------------

    #     def _build_Q(self, n_state):
    #         Q = np.eye(n_state) * self.q

    #         # Intercept: smoother but not frozen
    #         Q[0, 0] = self.q * 0.2

    #         for idx, scale in self.q_scale.items():
    #             Q[idx + 1, idx + 1] = self.q * scale

    #         return Q

    #     # --------------------------------------------------

    #     def _kalman_step(self, x_t, y_t):
    #         x_aug = np.concatenate(([1.0], x_t)).reshape(-1, 1)

    #         # ---------- PREDICT ----------
    #         beta_pred = self.beta_.copy()

    #         # Mean reversion ONLY for slopes
    #         beta_pred[1:] += -self.anchor_strength * (
    #             beta_pred[1:] - self.base_beta_[1:]
    #         )

    #         P_pred = self.P_ + self.Q_

    #         y_hat = float(beta_pred @ x_aug)

    #         # ---------- UPDATE ----------
    #         residual = y_t - y_hat
    #         S = float(x_aug.T @ P_pred @ x_aug + self.R_)
    #         S = max(S, 1e-12)

    #         K = (P_pred @ x_aug) / S
    #         beta_upd = beta_pred + K.flatten() * residual

    #         temp = self.I_ - K @ x_aug.T
    #         P_upd = temp @ P_pred @ temp.T + K @ K.T * self.R_

    #         # ---------- SIGN CONSTRAINTS ----------
    #         if self.non_negative_idx or self.non_positive_idx:
    #             self._project_state(beta_upd, P_upd)

    #         # ---------- DEVIATION CONTROL ----------
    #         if self.max_dev:
    #             self._limit_beta_deviation(beta_upd)

    #         self.beta_ = beta_upd
    #         self.P_ = P_upd
    #         return y_hat

    #     # --------------------------------------------------

    #     def _limit_beta_deviation(self, beta_vec):
    #         """
    #         Hard limit beta deviation from base beta
    #         (intercept excluded)
    #         """
    #         for idx, max_delta in self.max_dev.items():
    #             k = idx + 1  # shift for intercept

    #             base = self.base_beta_[k]
    #             lower = base - max_delta
    #             upper = base + max_delta

    #             if beta_vec[k] < lower:
    #                 beta_vec[k] = lower
    #             elif beta_vec[k] > upper:
    #                 beta_vec[k] = upper

    #     # --------------------------------------------------

    #     def _project_state(self, beta_vec, cov_mat):
    #         active = []
    #         seen = set()

    #         for idx in self.non_negative_idx:
    #             k = idx + 1
    #             if beta_vec[k] < 0 and k not in seen:
    #                 active.append(k)
    #                 seen.add(k)

    #         for idx in self.non_positive_idx:
    #             k = idx + 1
    #             if beta_vec[k] > 0 and k not in seen:
    #                 active.append(k)
    #                 seen.add(k)

    #         if not active:
    #             return

    #         active = np.array(active, dtype=int)
    #         P_cc = cov_mat[np.ix_(active, active)]

    #         jitter = 1e-9
    #         for _ in range(6):
    #             try:
    #                 reg = P_cc + jitter * np.eye(len(active))
    #                 solve_beta = np.linalg.solve(reg, beta_vec[active])
    #                 solve_cov = np.linalg.solve(reg, cov_mat[active, :])
    #                 break
    #             except np.linalg.LinAlgError:
    #                 jitter *= 10
    #         else:
    #             reg = P_cc + jitter * np.eye(len(active))
    #             solve_beta = np.linalg.pinv(reg) @ beta_vec[active]
    #             solve_cov = np.linalg.pinv(reg) @ cov_mat[active, :]

    #         beta_vec -= cov_mat[:, active] @ solve_beta
    #         beta_vec[active] = 0.0

    #         cov_mat -= cov_mat[:, active] @ solve_cov
    #         cov_mat[:] = 0.5 * (cov_mat + cov_mat.T)

    
    class TimeVaryingKalmanRegression:
        """
        Time-Varying Kalman Regression with:
        - Per-variable process noise
        - Mean reversion to base beta (EXCEPT intercept)
        - Adaptive intercept (beta0)
        - Covariance-aware sign constraints
        - Per-variable max deviation from base beta
        - Ridge regularization for multicollinearity
        """

        def __init__(
            self,
            q=1e-5,
            r=50.0,
            init_cov=10.0,
            q_scale=None,
            anchor_strength=0.05,
            non_negative_idx=None,
            non_positive_idx=None,
            max_dev=None,
            ridge_lambda=0.0          # ⬅️ NEW
        ):
            self.q = q
            self.r = r
            self.init_cov = init_cov
            self.q_scale = q_scale or {}
            self.anchor_strength = anchor_strength
            self.ridge_lambda = ridge_lambda  # ⬅️ NEW

            self.non_negative_idx = non_negative_idx or []
            self.non_positive_idx = non_positive_idx or []
            self.max_dev = max_dev or {}

        # --------------------------------------------------

        def fit(self, X, y, base_beta=None, beta0_init=0.0):
            T, p = X.shape
            n_state = p + 1

            self.beta_ = np.zeros(n_state)
            self.beta_[0] = beta0_init
            if base_beta is not None:
                self.beta_[1:] = base_beta

            self.base_beta_ = self.beta_.copy()

            self.P_ = np.eye(n_state) * self.init_cov
            self.Q_ = self._build_Q(n_state)
            self.R_ = self.r
            self.I_ = np.eye(n_state)

            betas = np.zeros((T + 1, n_state))
            y_pred = np.zeros(T)
            betas[0] = self.beta_

            for t in range(T):
                y_pred[t] = self._kalman_step(X[t], y[t])
                betas[t + 1] = self.beta_

            self.betas_ = betas
            self.y_pred_ = y_pred
            return self

        # --------------------------------------------------

        def _build_Q(self, n_state):
            Q = np.eye(n_state) * self.q

            Q[0, 0] = self.q * 0.2

            for idx, scale in self.q_scale.items():
                Q[idx + 1, idx + 1] = self.q * scale

            return Q

        # --------------------------------------------------

        def _kalman_step(self, x_t, y_t):
            x_aug = np.concatenate(([1.0], x_t)).reshape(-1, 1)

            # ---------- PREDICT ----------
            beta_pred = self.beta_.copy()

            beta_pred[1:] += -self.anchor_strength * (
                beta_pred[1:] - self.base_beta_[1:]
            )

            P_pred = self.P_ + self.Q_

            y_hat = float(beta_pred @ x_aug)

            # ---------- UPDATE ----------
            residual = y_t - y_hat

            # ⬇️ Ridge added here
            S = float(
                x_aug.T @ P_pred @ x_aug
                + self.R_
                + self.ridge_lambda
            )

            S = max(S, 1e-12)

            K = (P_pred @ x_aug) / S
            beta_upd = beta_pred + K.flatten() * residual

            temp = self.I_ - K @ x_aug.T
            P_upd = temp @ P_pred @ temp.T + K @ K.T * self.R_

            if self.non_negative_idx or self.non_positive_idx:
                self._project_state(beta_upd, P_upd)

            if self.max_dev:
                self._limit_beta_deviation(beta_upd)

            self.beta_ = beta_upd
            self.P_ = P_upd

            return y_hat

        # --------------------------------------------------

        def _limit_beta_deviation(self, beta_vec):
            for idx, max_delta in self.max_dev.items():
                k = idx + 1
                base = self.base_beta_[k]
                lower = base - max_delta
                upper = base + max_delta

                if beta_vec[k] < lower:
                    beta_vec[k] = lower
                elif beta_vec[k] > upper:
                    beta_vec[k] = upper

        # --------------------------------------------------

        def _project_state(self, beta_vec, cov_mat):
            active = []
            seen = set()

            for idx in self.non_negative_idx:
                k = idx + 1
                if beta_vec[k] < 0 and k not in seen:
                    active.append(k)
                    seen.add(k)

            for idx in self.non_positive_idx:
                k = idx + 1
                if beta_vec[k] > 0 and k not in seen:
                    active.append(k)
                    seen.add(k)

            if not active:
                return

            active = np.array(active, dtype=int)
            P_cc = cov_mat[np.ix_(active, active)]

            jitter = 1e-9
            for _ in range(6):
                try:
                    reg = P_cc + jitter * np.eye(len(active))
                    solve_beta = np.linalg.solve(reg, beta_vec[active])
                    solve_cov = np.linalg.solve(reg, cov_mat[active, :])
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            else:
                reg = P_cc + jitter * np.eye(len(active))
                solve_beta = np.linalg.pinv(reg) @ beta_vec[active]
                solve_cov = np.linalg.pinv(reg) @ cov_mat[active, :]

            beta_vec -= cov_mat[:, active] @ solve_beta
            beta_vec[active] = 0.0

            cov_mat -= cov_mat[:, active] @ solve_cov
            cov_mat[:] = 0.5 * (cov_mat + cov_mat.T)



    # =====================================================
    # ---------------- HELPER FUNCTIONS -------------------
    # =====================================================
    def extract_model_variables(model_row):
        """
        Extract independent variables used in the model
        from beta_scaled_* columns.
        """
        beta_cols = [
            col for col in model_row.index
            if col.startswith("beta_scaled_") and pd.notna(model_row[col])
        ]

        variables = [col.replace("beta_scaled_", "") for col in beta_cols]
        return variables, beta_cols


    def build_constraint_indices(variables, positive_vars, negative_vars):
        """
        Convert variable names into index-based constraints
        for Kalman filter.
        """
        non_negative_idx = []
        non_positive_idx = []

        for i, var in enumerate(variables):
            if var in positive_vars:
                non_negative_idx.append(i)
            if var in negative_vars:
                non_positive_idx.append(i)

        return non_negative_idx, non_positive_idx



    def prepare_data(
        dataset_df,
        y_col,
        x_vars,
        standardization_method
    ):
        """
        Prepare X and y for modeling.
        """
        data = dataset_df[[y_col] + x_vars].dropna()

        y = data[y_col].values
        X_raw = data[x_vars].values

        scaler = None
        if standardization_method == "zscore":
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
        else:
            X = X_raw.copy()

        return X, y, scaler

    def prepare_dates(dataset_df, x_vars, y_col):
        """
        Returns Date series aligned with X/y after dropping NA rows.
        """
        df = dataset_df[["date", y_col] + x_vars].dropna()
        dates = pd.to_datetime(df["date"])
        return dates

    def extract_base_betas(model_row, variables):
        """
        Extract original model betas (static) for each variable.
        Returns dict: {var: beta_value}
        """
        base_betas = {}

        for var in variables:
            col = f"beta_scaled_{var}"
            if col in model_row.index:
                base_betas[var] = float(model_row[col])
            else:
                base_betas[var] = np.nan

        return base_betas



    # =====================================================
    # ---------------- STREAMLIT APP ----------------------
    # =====================================================
    
    st.title("📈 Time-Varying Kalman Regression – Beta Forecasting")

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------
    st.sidebar.header("📂 Upload Files")

    dataset_file = st.sidebar.file_uploader(
        "Dataset (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="dataset"
    )

    model_file = st.sidebar.file_uploader(
        "Model Summary (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="model"
    )

    if not dataset_file or not model_file:
        st.info("Upload both dataset and model summary files to proceed.")
        st.stop()


    # -----------------------------------------------------
    # SHEET SELECTION (COMMON)
    # -----------------------------------------------------

    common_sheet = None

    if dataset_file.name.endswith(("xlsx", "xls")) and model_file.name.endswith(("xlsx", "xls")):

        dataset_excel = pd.ExcelFile(dataset_file)
        model_excel = pd.ExcelFile(model_file)

        # Find common sheets
        common_sheets = list(
            set(dataset_excel.sheet_names).intersection(model_excel.sheet_names)
        )

        if not common_sheets:
            st.error("No common sheet names found between the two Excel files.")
            st.stop()

        common_sheet = st.sidebar.selectbox(
            "Select Sheet (applies to both files)",
            common_sheets,
            key="shared_sheet"
        )

        dataset_df = pd.read_excel(dataset_excel, sheet_name=common_sheet)
        model_df = pd.read_excel(model_excel, sheet_name=common_sheet)

    else:
        # If any file is CSV
        dataset_df = pd.read_csv(dataset_file) if dataset_file.name.endswith(".csv") else pd.read_excel(dataset_file)
        model_df = pd.read_csv(model_file) if model_file.name.endswith(".csv") else pd.read_excel(model_file)



    # st.dataframe(dataset_df)


    # dataset_df["date"] = pd.to_datetime(dataset_df["date"], format="%m/%d/%Y")
    dataset_df["date"] = pd.to_datetime(dataset_df["date"])
    dataset_df = dataset_df.sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------
    st.sidebar.header("🧠 Model Selection")

    metric = st.sidebar.selectbox("Select best model by", ["R_squared", "MAPE"])
    ascending = metric == "MAPE"

    model_row = model_df.sort_values(metric, ascending=ascending).iloc[0]

    with st.expander("View Model Summary DataFrame"):
        st.dataframe(model_df)
    # st.subheader("Selected Model")
    # st.write(model_row)

    x_vars, _ = extract_model_variables(model_row)

    base_beta = np.array(
        [model_row[f"beta_scaled_{v}"] for v in x_vars], dtype=float
    )

    beta0 = float(model_row["beta0"])
    y_col = model_row["Y"]
    std_method = model_row["Standardization_method"]

    # -----------------------------------------------------
    # CONSTRAINTS
    # -----------------------------------------------------
    st.sidebar.header("🔒 Sign Constraints")

    # Create mapping of variable → beta
    beta_dict = dict(zip(x_vars, base_beta))

    # Auto-detect signs
    auto_positive = [v for v, b in beta_dict.items() if b > 0]
    auto_negative = [v for v, b in beta_dict.items() if b < 0]

    # Multiselect with auto-default
    positive_vars = st.sidebar.multiselect(
        "Positive vars",
        x_vars,
        default=auto_positive,
        key="positive_vars"
    )

    negative_vars = st.sidebar.multiselect(
        "Negative vars",
        x_vars,
        default=auto_negative,
        key="negative_vars"
    )


    non_neg_idx, non_pos_idx = build_constraint_indices(
        x_vars, positive_vars, negative_vars
    )

    # -----------------------------------------------------
    # KALMAN PARAMETERS
    # -----------------------------------------------------
    st.sidebar.header("⚙️ Kalman Params")

    q = st.sidebar.number_input(
        "Process noise q",
        min_value=1e-5,
        max_value=1e-1,
        value=1e-3,
        format="%.5f"
    )

    r = st.sidebar.number_input(
        "Observation noise r",
        min_value=1.0,
        max_value=100.0,
        value=20.0
    )

    init_cov = st.sidebar.number_input(
        "Initial covariance",
        min_value=0.001,
        max_value=10.0,
        value=0.01
    )

    anchor = st.sidebar.number_input(
        "Anchor strength",
        min_value=0.0,
        max_value=1.0,
        value=0.001
    )

    # 🔥 NEW: Ridge Regularization
    ridge_lambda = st.sidebar.number_input(
        "Ridge regularization (λ)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

    # -----------------------------------------------------
    # MAX DEVIATION
    # -----------------------------------------------------
    st.sidebar.header("📏 Max Deviation")

    max_dev = {}
    for v in x_vars:
        if st.sidebar.checkbox(f"Limit {v}"):
            max_dev[x_vars.index(v)] = st.sidebar.slider(
                f"Max deviation for {v}",
                min_value=0.0,
                max_value=1.0,
                value=0.3
            )

    # -----------------------------------------------------
    # SESSION STATE INITIALIZATION
    # -----------------------------------------------------
    STATE_KEYS = [
        "kalman_ran", "X", "y", "kf",
        "y_static_obs", "y_kalman_obs",
        "betas_df", "final_df", "beta_stats_df"
    ]

    for k in STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None

    if st.session_state.kalman_ran is None:
        st.session_state.kalman_ran = False

    # -----------------------------------------------------
    # RUN MODEL
    # -----------------------------------------------------
    if st.button("🚀 Run Kalman"):

        X, y, _ = prepare_data(dataset_df, y_col, x_vars, std_method)

        kf = TimeVaryingKalmanRegression(
            q=q,
            r=r,
            init_cov=init_cov,
            anchor_strength=anchor,
            non_negative_idx=non_neg_idx,
            non_positive_idx=non_pos_idx,
            max_dev=max_dev,
            ridge_lambda=ridge_lambda      # 🔥 PASS RIDGE
        )

        with st.spinner("Running Kalman filter..."):
            kf.fit(X, y, base_beta, beta0)

        # ---- STORE EVERYTHING ----
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.kf = kf

        st.session_state.y_static_obs = beta0 + X @ base_beta
        st.session_state.y_kalman_obs = kf.y_pred_

        betas_df = pd.DataFrame(
            kf.betas_[1:, 1:], columns=x_vars
        )
        betas_df["beta0"] = kf.betas_[1:, 0]
        betas_df["date"] = dataset_df["date"].values

        st.session_state.betas_df = betas_df

        st.session_state.final_df = pd.DataFrame({
            "Variable": ["Intercept"] + x_vars,
            "Beta (Last Month)": kf.beta_
        })

        st.session_state.kalman_ran = True

    if not st.session_state.kalman_ran:
        st.info("Run the Kalman model to see results.")
        st.stop()



    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------
    if not st.session_state.kalman_ran:
        st.info("Run the Kalman model to see results.")
        st.stop()

    betas_df = st.session_state.betas_df
    final_df = st.session_state.final_df
    base_beta_map = dict(zip(x_vars, base_beta))


    import plotly.graph_objects as go

    st.subheader("📊 Time-Varying Betas (Per Variable)")

    betas_df = st.session_state.betas_df
    dates = betas_df["date"]

    y_series = dataset_df[y_col].values

    # -----------------------------------------------------
    # ONE EXPANDER PER VARIABLE
    # -----------------------------------------------------

    with st.expander(f"📈 Time-Varying Betas", expanded=False):


        # x_series = dataset_df[var].values

        for var in x_vars:
            x_series = dataset_df[var].values

            st.subheader(f"{var}")

            fig = go.Figure()

            # -------- Kalman beta (LEFT AXIS) --------
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=betas_df[var],
                    mode="lines",
                    name="β Kalman",
                    line=dict(width=2)
                )
            )

            # -------- Original beta (LEFT AXIS) --------
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=[base_beta_map[var]] * len(dates),
                    mode="lines",
                    name="β Original",
                    line=dict(dash="dot"),
                    opacity=0.7
                )
            )

            # -------- Y variable (RIGHT AXIS) --------
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y_series,
                    mode="lines",
                    name=y_col,
                    yaxis="y2",
                    line=dict(color="black", dash="dash")
                )
            )

            # -------- X variable itself (RIGHT AXIS) --------
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=x_series,
                    mode="lines",
                    name=var,
                    yaxis="y2",
                    line=dict(color="gray", dash="dot"),
                    opacity=0.8
                )
            )

            # -------- Layout --------
            fig.update_layout(
                height=420,
                hovermode="x unified",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Beta value"),
                yaxis2=dict(
                    title="Y & X values",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                ),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)



    # -----------------------------------------------------
    # KALMAN BETA STATISTICS
    # -----------------------------------------------------
    kalman_betas = st.session_state.betas_df[x_vars].values  # (T, p)

    stats = []
    for i, var in enumerate(x_vars):
        beta_series = kalman_betas[:, i]

        mean_beta = np.mean(beta_series)
        min_beta = np.min(beta_series)
        max_beta = np.max(beta_series)
        std_beta = np.std(beta_series)

        mean_minus_min = (
            (mean_beta - min_beta) / mean_beta * 100
            if mean_beta != 0 else np.nan
        )
        mean_minus_max = (
            (max_beta - mean_beta) / mean_beta * 100
            if mean_beta != 0 else np.nan
        )

        cv = std_beta / mean_beta if mean_beta != 0 else np.nan

        stats.append({
            "Variable": var,
            "Mean Kalman Beta": mean_beta,
            "Min Kalman Beta": min_beta,
            "Max Kalman Beta": max_beta,
            "Min from Mean (%)": mean_minus_min,
            "Max from Mean (%)": mean_minus_max,
            "Std Kalman Beta": std_beta,
            "Std / Mean (CV)": cv
        })

    beta_stats_df = pd.DataFrame(stats).round(2)

    # store for reuse
    st.session_state.beta_stats_df = beta_stats_df

    # -----------------------------------------------------
    # BETA STABILITY ANALYSIS
    # -----------------------------------------------------
    st.subheader("📋 Kalman Beta Stability Analysis")

    beta_stats_df = st.session_state.beta_stats_df

    # st.dataframe(
    #     beta_stats_df,
    #     use_container_width=True
    # )

    def highlight_unstable(val):
        if pd.isna(val):
            return ""
        return "background-color: #ffcccc" if abs(val) > 0.5 else ""

    with st.expander("Kalman Beta Stability Analysis", expanded=False):

        st.dataframe(
            beta_stats_df.style.applymap(
                highlight_unstable,
                subset=["Std / Mean (CV)"]
            ),
            use_container_width=True
        )



    # st.subheader("📉 Intercept")
    # st.line_chart(
    #     betas_df.set_index("date")[["beta0"]]
    # )

    st.subheader("🏁 Final Betas")
    with st.expander("Final Betas (Last Month)", expanded=False):
        st.dataframe(final_df)

    from sklearn.metrics import r2_score

    def rmape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    y_actual_obs = st.session_state.y
    y_static_obs = st.session_state.y_static_obs
    y_kalman_obs = st.session_state.y_kalman_obs

    if any(v is None for v in [y_actual_obs, y_static_obs, y_kalman_obs]):
        st.info("Metrics unavailable — run model.")
        st.stop()

    metrics_df = pd.DataFrame({
        "Model": ["Static Regression", "Kalman Regression"],
        "R²": [
            r2_score(y_actual_obs, y_static_obs),
            r2_score(y_actual_obs, y_kalman_obs)
        ],
        "MAPE (%)": [
            rmape(y_actual_obs, y_static_obs),
            rmape(y_actual_obs, y_kalman_obs)
        ]
    }).round(3)



    st.subheader("📊 Model Performance (Observed Period)")
    with st.expander("📊 Model Performance"):
        st.dataframe(metrics_df, use_container_width=True)

    # -----------------------------------------------------
    # FUTURE (12 MONTHS) – ACTUAL vs PREDICTED
    # -----------------------------------------------------

    # -------- SAFETY GUARD --------
    if not st.session_state.kalman_ran:
        st.info("Run the Kalman model to see forecasts.")
        st.stop()

    # -------- LOAD FROM SESSION STATE --------
    X = st.session_state.X
    y_actual_obs = st.session_state.y
    y_static_obs = st.session_state.y_static_obs
    y_kalman_obs = st.session_state.y_kalman_obs
    betas_df = st.session_state.betas_df

    # -------- FUTURE DESIGN MATRIX --------

    # -----------------------------------------------------
    # USER INPUT: BASE MODEL END DATE
    # -----------------------------------------------------
    dates_all = pd.to_datetime(dataset_df["date"]).sort_values()

    base_model_end = st.sidebar.selectbox(
        "Base model end date",
        options=dates_all.unique(),
        index=len(dates_all.unique()) - 25  # defaults to ~12 months before end
    )
    # -----------------------------------------------------
    # AUTOMATIC FORECAST HORIZON
    # -----------------------------------------------------
    forecast_mask = dates_all > base_model_end
    H = forecast_mask.sum()


    # H = 12
    X_future = X[-H:, :]

    # -------- KALMAN FUTURE PREDICTION --------
    betas_12m = betas_df.tail(H)

    y_kalman_future = (
        np.sum(X_future * betas_12m[x_vars].values, axis=1)
        + betas_12m["beta0"].values
    )

    # -------- STATIC FUTURE PREDICTION --------
    y_static_future = beta0 + X_future @ base_beta

    # -----------------------------------------------------
    # COMBINE OBSERVED + FUTURE
    # -----------------------------------------------------
    y_actual_all = y_actual_obs
    y_static_all = np.concatenate([y_static_obs])
    y_kalman_all = np.concatenate([y_kalman_obs])

    # -------- DATES --------
    dates_obs = dataset_df["date"].iloc[:len(y_actual_obs)]

    dates_future = pd.date_range(
        start=dates_obs.iloc[-1] ,#+ pd.offsets.MonthBegin(1),
        periods=H,
        freq="MS"
    )

    dates_all = pd.to_datetime(
        pd.concat([dates_obs, pd.Series(dates_future)], ignore_index=True)
    )

    # -----------------------------------------------------
    # PLOTLY: ACTUAL vs STATIC vs KALMAN
    # -----------------------------------------------------
    import plotly.graph_objects as go

    fig = go.Figure()

    # -------- Actual --------
    fig.add_trace(
        go.Scatter(
            x=dates_obs,
            y=y_actual_obs,
            mode="lines",
            name="Actual",
            line=dict(width=3)
        )
    )

    # -------- Static Model --------
    fig.add_trace(
        go.Scatter(
            x=dates_all,
            y=y_static_all,
            mode="lines",
            name="Static Model",
            line=dict(dash="dash")
        )
    )

    # -------- Kalman Model --------
    fig.add_trace(
        go.Scatter(
            x=dates_all,
            y=y_kalman_all,
            mode="lines",
            name="Kalman Model",
            line=dict(dash="dot")
        )
    )

    # -------- Forecast Split Line --------
    # forecast_start = dates_future[0]
    last_date = pd.to_datetime(dataset_df["date"].max())

    forecast_start = last_date - pd.DateOffset(months=H)


    fig.add_vline(
        x=forecast_start,
        line_width=2,
        line_dash="dot",
        line_color="black"
    )

    fig.add_annotation(
        x=forecast_start,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        xanchor="left"
    )

    # -------- Layout --------
    fig.update_layout(
        title="Actual vs Predicted: Static vs Kalman (Observed + 12M Forecast)",
        xaxis_title="Date",
        yaxis_title=y_col,
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(x=0.01, y=0.99)
    )

    year_ticks = pd.date_range(
        start=dates_all.min(),
        end=dates_all.max(),
        freq="YS"
    )

    tick_text = []
    for d in year_ticks:
        if d.year == 2025:
            tick_text.append("2024 New")
        elif d.year == 2026:
            tick_text.append("2025")
        else:
            tick_text.append(str(d.year))

    fig.update_xaxes(
        tickmode="array",
        tickvals=year_ticks,
        ticktext=tick_text
    )


    st.subheader("📈 Actual vs Predicted")
    with st.expander("📈 Actual vs Predicted (Observed + Forecast)"):
        st.plotly_chart(fig, use_container_width=True)

    

    ################# new elasticity ##################

    dates_all = pd.to_datetime(dataset_df["date"])
    base_mask = dates_all <= base_model_end
    kalman_mask = dates_all > base_model_end

    stats_rows = []

    y_full = dataset_df[y_col]
    y_base = y_full[base_mask]

    for i, var in enumerate(x_vars):

        x_full = dataset_df[var]
        x_base = x_full[base_mask]

        # ---------- Base ----------
        base_scaled_beta = base_beta[i]
        base_std = x_base.std()
        base_mean_x = x_base.mean()
        base_mean_y = y_base.mean()
        base_sum_x = x_base.sum()

        base_unscaled_beta = base_scaled_beta / base_std

        base_elasticity = (
            base_unscaled_beta * base_mean_x / base_mean_y
            if base_mean_y != 0 else np.nan
        )

        # ---------- Kalman ----------
        kalman_scaled_beta = final_df.loc[
            final_df["Variable"] == var,
            "Beta (Last Month)"
        ].values[0]

        kalman_std = x_full.std()
        kalman_mean_x = x_full.mean()
        kalman_mean_y = y_full.mean()

        kalman_unscaled_beta = kalman_scaled_beta / kalman_std

        kalman_elasticity = (
            kalman_unscaled_beta * kalman_mean_x / kalman_mean_y
            if kalman_mean_y != 0 else np.nan
        )

        stats_rows.append({
            "Variable": var,
            "Base Scaled Beta": base_scaled_beta,
            # "base sum": base_sum_x,
            "Base Std" : base_std,
            'Base mean': base_mean_x,
            'Base y mean': base_mean_y,
            "Base Unscaled Beta": base_unscaled_beta,
            "Base Elasticity": base_elasticity,
            # "Kalman Scaled Beta": kalman_scaled_beta,
            # "Kalman Unscaled Beta": kalman_unscaled_beta,
            "Kalman Elasticity": kalman_elasticity
        })

    beta_comparison_df = pd.DataFrame(stats_rows).round(4)

    st.subheader("📊 Base vs Kalman Beta Comparison")
    st.dataframe(beta_comparison_df, use_container_width=True)





################################################################# Post MOdelling ###############################################################################
################################################################# Post MOdelling ###############################################################################
################################################################# Post MOdelling ###############################################################################
################################################################# Post MOdelling ###############################################################################
################################################################# Post MOdelling ###############################################################################


elif menu == "Post-Modeling Analysis":
    st.header("Post-Modeling Analysis")

    # st.title("📊 Post-Modeling Analysis")

    dataset_file = st.sidebar.file_uploader(
        "Upload Modeling Dataset", type=["csv", "xlsx"]
    )
    elasticity_file = st.sidebar.file_uploader(
        "Upload Elasticity File", type=["csv", "xlsx"]
    )
    growth_file = st.sidebar.file_uploader(
        "Upload Growth Rate File", type=["csv", "xlsx"]
    )

    def load_file(file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)

    # -----------------------------------------------------
    # STOP if files not uploaded
    # -----------------------------------------------------
    if not (dataset_file and elasticity_file or growth_file):
        st.info("Upload all required files to proceed.")
        st.stop()

    # dataset_file = st.sidebar.file_uploader(
    #     "Upload Modeling Dataset", type=["csv", "xlsx"]
    # )

    def data_load_file(file, sheet_name=None):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file, sheet_name=sheet_name)

    dataset_df = None

    if dataset_file is not None:

        # If Excel → show sheet selector
        if dataset_file.name.endswith(".xlsx"):

            excel_file = pd.ExcelFile(dataset_file)
            sheet_names = excel_file.sheet_names

            selected_sheet = st.sidebar.selectbox(
                "Select Excel Sheet",
                sheet_names
            )

            dataset_df = pd.read_excel(dataset_file, sheet_name=selected_sheet)

        else:
            dataset_df = pd.read_csv(dataset_file)

    # Show dataframe
    # if dataset_df is not None:
        # st.dataframe(dataset_df)


    # -----------------------------------------------------
    # LOAD FILES
    # -----------------------------------------------------
    # dataset_df = load_file(dataset_file)
    elasticity_df = load_file(elasticity_file)
    growth_df = load_file(growth_file)

    # -----------------------------------------------------
    # DISPLAY DATA
    # -----------------------------------------------------
    with st.expander("Modeling Dataset"):
        # # Ensure datetime
        # dataset_df["date"] = pd.to_datetime(dataset_df["date"], errors="coerce")
        # dataset_df = dataset_df[
        #     dataset_df["date"] >= pd.Timestamp("2025-01-01")
        # ].copy()
        # # Correct year (shift back by 1 year)
        # dataset_df["date"] = dataset_df["date"] - pd.DateOffset(years=1)
        # dataset_df["Year"] = dataset_df["date"].dt.year
        # dataset_df["Month"] = dataset_df["date"].dt.month
        # dataset_df["Fiscal Year"] = (
        #     "FY" +
        #     (dataset_df["date"].dt.year + (dataset_df["date"].dt.month >= 7))
        #     .astype(str).str[-2:]
        # )
        # # dataset_df["Fiscal Year"].unique()
        # st.dataframe(dataset_df)

        # # Ensure datetime
        # dataset_df["date"] = pd.to_datetime(dataset_df["date"], errors="coerce")

        # # --------------------------------------------------
        # # 1️⃣ Remove original 2024 data
        # # --------------------------------------------------
        # dataset_df = dataset_df[
        #     dataset_df["date"].dt.year != 2024
        # ].copy()

        # # --------------------------------------------------
        # # 2️⃣ Rename years (hard replace)
        # # 2025 → 2024
        # # 2026 → 2025
        # # --------------------------------------------------
        # mask_2025 = dataset_df["date"].dt.year == 2025
        # mask_2026 = dataset_df["date"].dt.year == 2026

        # dataset_df.loc[mask_2025, "date"] = (
        #     dataset_df.loc[mask_2025, "date"] - pd.DateOffset(years=1)
        # )

        # dataset_df.loc[mask_2026, "date"] = (
        #     dataset_df.loc[mask_2026, "date"] - pd.DateOffset(years=1)
        # )

        # # --------------------------------------------------
        # # 3️⃣ Recreate date-derived columns
        # # --------------------------------------------------
        # dataset_df["Year"] = dataset_df["date"].dt.year
        # dataset_df["Month"] = dataset_df["date"].dt.month

        # dataset_df["Fiscal Year"] = (
        #     "FY" +
        #     (dataset_df["date"].dt.year + (dataset_df["date"].dt.month >= 7))
        #     .astype(str).str[-2:]
        # )

        # --------------------------------------------------
        # Display
        # --------------------------------------------------
        st.dataframe(dataset_df)


    dataset_df.rename(columns={"Region": "Segment"}, inplace=True)

    seg_list = dataset_df["Segment"].unique()

    elasticity_df = elasticity_df[
        elasticity_df["Segment"].isin(seg_list)
    ]

    with st.expander("Elasticity Data"):
        # st.dataframe(elasticity_df[["Segment","Variable","Elasticity"]])
        display_df = elasticity_df[["Segment", "Variable", "Elasticity"]].copy()
        display_df["Elasticity"] = display_df["Elasticity"].round(2)

        st.dataframe(display_df, use_container_width=True)

    # with st.expander("Growth Rate Data"):
    #     # Rename Feature column if exists
    #     if "Feature" in growth_df.columns:
    #         growth_df.rename(columns={"Feature": "Variable"}, inplace=True)
    #     segments = elasticity_df["Segment"].unique()

    #     # Create cross-join style duplication for each segment
    #     growth_df = growth_df.assign(key=1)
    #     segment_df = pd.DataFrame({"Segment": segments, "key": 1})

    #     growth_df = growth_df.merge(segment_df, on="key").drop("key", axis=1)

    #     # Function to clean variable names
    #     def clean_var_name(var):
    #         return (
    #             var.replace("Res_", "")
    #             .replace("Lag_", "")
    #             .replace("lag_", "")
    #             .strip()
    #         )

    #     # Create cleaned versions
    #     elasticity_df["CleanVar"] = elasticity_df["Variable"].apply(clean_var_name)
    #     growth_df["CleanVar"] = growth_df["Variable"].apply(clean_var_name)

    #     # Create mapping from CleanVar → original elasticity variable
    #     clean_to_original = dict(
    #         zip(elasticity_df["CleanVar"], elasticity_df["Variable"])
    #     )

    #     # Keep only growth variables that match elasticity
    #     growth_df = growth_df[
    #         growth_df["CleanVar"].isin(clean_to_original.keys())
    #     ].copy()

    #     growth_df["Variable"] = growth_df["CleanVar"].map(clean_to_original)

    #     growth_df.drop(columns=["CleanVar"], inplace=True)
    #     elasticity_df.drop(columns=["CleanVar"], inplace=True)



    #     # st.dataframe(elasticity_df)

    #     edited_growth_df = st.data_editor(growth_df)
    with st.expander("Growth Rate Data"):

        # Rename Feature column if exists
        if "Feature" in growth_df.columns:
            growth_df.rename(columns={"Feature": "Variable"}, inplace=True)

        segments = elasticity_df["Segment"].unique()

        # ---------------------------------------------
        # Split Seasonality / Trend vs Others
        # ---------------------------------------------
        special_vars = ["Seasonality", "Trend"]

        special_growth = growth_df[
            growth_df["Variable"].isin(special_vars)
        ].copy()

        other_growth = growth_df[
            ~growth_df["Variable"].isin(special_vars)
        ].copy()

        # ---------------------------------------------
        # Cross-join ONLY for other variables
        # ---------------------------------------------
        other_growth = other_growth.assign(key=1)
        segment_df = pd.DataFrame({"Segment": segments, "key": 1})

        other_growth = (
            other_growth
            .merge(segment_df, on="key")
            .drop("key", axis=1)
        )

        # ---------------------------------------------
        # For Seasonality & Trend → Keep Only Matching Segment
        # ---------------------------------------------
        special_growth = special_growth[
            special_growth["Segment"].isin(segments)
        ].copy()

        # ---------------------------------------------
        # Combine back
        # ---------------------------------------------
        growth_df = pd.concat(
            [other_growth, special_growth],
            ignore_index=True
        )

        # ---------------------------------------------
        # Clean variable names
        # ---------------------------------------------
        def clean_var_name(var):
            return (
                var.replace("Res_", "")
                .replace("Lag_", "")
                .replace("lag_", "")
                .strip()
            )

        elasticity_df["CleanVar"] = elasticity_df["Variable"].apply(clean_var_name)
        growth_df["CleanVar"] = growth_df["Variable"].apply(clean_var_name)

        clean_to_original = dict(
            zip(elasticity_df["CleanVar"], elasticity_df["Variable"])
        )

        growth_df = growth_df[
            growth_df["CleanVar"].isin(clean_to_original.keys())
        ].copy()

        growth_df["Variable"] = growth_df["CleanVar"].map(clean_to_original)
        valid_segments = elasticity_df["Segment"].unique()
        growth_df['Segment'] = valid_segments[0]

        growth_df.drop(columns=["CleanVar","Segment_x","Segment_y"], inplace=True)
        elasticity_df.drop(columns=["CleanVar"], inplace=True)

        edited_growth_df = st.data_editor(growth_df)


    # dataset_df["date"] = pd.to_datetime(dataset_df["date"])

    beta_rows = []

    for _, row in elasticity_df.iterrows():
        seg = row["Segment"]
        var = row["Variable"]
        elasticity = row["Elasticity"]

        seg_df = dataset_df[dataset_df["Segment"] == seg]

        y_mean = seg_df["Volume"].mean()
        x_mean = seg_df[var].mean()
        x_std  = seg_df[var].std()

        non_scaled_beta = elasticity * (y_mean / x_mean)
        scaled_beta = non_scaled_beta * x_std

        beta_rows.append({
            "Segment": seg,
            "Variable": var,
            "Elasticity": elasticity,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "NonScaledBeta": non_scaled_beta,
            "ScaledBeta": scaled_beta,
            "ScaledBeta0": row["Scaled beta0"]
        })

    beta_df = pd.DataFrame(beta_rows)
    # st.dataframe(beta_df)

    # Ensure datetim
    dataset_df["date"] = pd.to_datetime(dataset_df["date"], errors="coerce")

    # --------------------------------------------------
    # 1️⃣ Remove original 2024 data
    # --------------------------------------------------
    dataset_df_filter = dataset_df[
        dataset_df["date"].dt.year != 2024
    ].copy()

    # --------------------------------------------------
    # 2️⃣ Rename years (hard replace)
    # 2025 → 2024
    # 2026 → 2025
    # --------------------------------------------------
    mask_2025 = dataset_df_filter["date"].dt.year == 2025
    mask_2026 = dataset_df_filter["date"].dt.year == 2026

    dataset_df_filter.loc[mask_2025, "date"] = (
        dataset_df_filter.loc[mask_2025, "date"] - pd.DateOffset(years=1)
    )

    dataset_df_filter.loc[mask_2026, "date"] = (
        dataset_df_filter.loc[mask_2026, "date"] - pd.DateOffset(years=1)
    )

    # --------------------------------------------------
    # 3️⃣ Recreate date-derived columns
    # --------------------------------------------------
    dataset_df_filter["Year"] = dataset_df_filter["date"].dt.year
    dataset_df_filter["Month"] = dataset_df_filter["date"].dt.month

    dataset_df_filter["Fiscal Year"] = (
        "FY" +
        (dataset_df_filter["date"].dt.year + (dataset_df_filter["date"].dt.month >= 7))
        .astype(str).str[-2:]
    )

    dataset_df_filter["FY_num"] = dataset_df_filter["Fiscal Year"].str.extract(r"(\d+)").astype(int)

    recent_fy_num = dataset_df_filter["FY_num"].max() -1
    recent_fy = f"FY{recent_fy_num}"
    # st.write(f"Most recent fiscal year in dataset: {recent_fy}")

    elasticity_vars = (
        elasticity_df["Variable"]
        .dropna()
        .unique()
        .tolist()
    )

    available_vars = [
        v for v in elasticity_vars
        if v in dataset_df.columns
    ]

    cols_to_avg = ["Segment", "Volume"] + available_vars

    fy_avg = (
        dataset_df_filter
        .loc[dataset_df_filter["Fiscal Year"] == recent_fy, cols_to_avg]
        .groupby("Segment", as_index=False)
        .mean()
    )

    # st.dataframe(fy_avg)

    future_year = "A26 Growth Rate (%)"

    future_rows = []

    for _, row in edited_growth_df.iterrows():
        seg = row["Segment"]
        var = row["Variable"]
        growth = row[future_year] / 100

        base_val = fy_avg.loc[fy_avg["Segment"] == seg, var].values[0]
        future_val = base_val * (1 + growth)

        future_rows.append({
            "Segment": seg,
            "Variable": var,
            "FutureValue": future_val
        })

    future_df = pd.DataFrame(future_rows)
    # st.dataframe(future_df)

    scaled_future = []

    for _, row in future_df.iterrows():
        seg = row["Segment"]
        var = row["Variable"]

        base_df = dataset_df_filter[
            (dataset_df_filter["Segment"] == seg) &
            (dataset_df_filter["Fiscal Year"] == recent_fy)
        ]

        mean_x = dataset_df_filter[var].mean()
        std_x = dataset_df_filter[var].std()

        scaled_val = (row["FutureValue"] - mean_x) / std_x

        scaled_future.append({
            "Segment": seg,
            "Variable": var,
            "ScaledFutureValue": scaled_val
        })

    scaled_future_df = pd.DataFrame(scaled_future)
    # st.dataframe(scaled_future_df)

    predictions = []

    for seg in beta_df["Segment"].unique():
        seg_betas = beta_df[beta_df["Segment"] == seg]
        seg_x = scaled_future_df[scaled_future_df["Segment"] == seg]

        y_hat = seg_betas["ScaledBeta0"].iloc[0]

        for _, b in seg_betas.iterrows():
            x_val = seg_x.loc[
                seg_x["Variable"] == b["Variable"],
                "ScaledFutureValue"
            ].values[0]

            y_hat += b["ScaledBeta"] * x_val

        predictions.append({
            "Segment": seg,
            "PredictedVolume": y_hat
        })

    prediction_df = pd.DataFrame(predictions)

    base_vol = (
        dataset_df_filter[dataset_df_filter["Fiscal Year"] == recent_fy]
        .groupby("Segment")["Volume"]
        .mean()
        .reset_index()
    )

    final_df = prediction_df.merge(base_vol, on="Segment", suffixes=("_Future", "_Base"))

    final_df["VolumeGrowth_%"] = (
        (final_df["PredictedVolume"] - final_df["Volume"]) /
        final_df["Volume"]
    ) * 100

    # st.dataframe(final_df[["Segment", "Volume", "PredictedVolume", "VolumeGrowth_%"]])

    future_year_cols = [
        c for c in edited_growth_df.columns
        if "Growth Rate" in c
    ]

    elasticity_vars = beta_df["Variable"].unique().tolist()

    global_stats = {}

    for seg in dataset_df["Segment"].unique():
        seg_df = dataset_df[dataset_df["Segment"] == seg]

        global_stats[seg] = {
            var: {
                "mean": seg_df[var].mean(),
                "std": seg_df[var].std()
            }
            for var in elasticity_vars
            if var in seg_df.columns
        }

        global_stats[seg]["Volume"] = seg_df["Volume"].mean()

    base_fy = "FY25"

    fy24_avg = (
        dataset_df_filter[dataset_df_filter["Fiscal Year"] == base_fy]
        .groupby("Segment", as_index=False)
        .mean(numeric_only=True)
    )
    # st.dataframe(fy24_avg)

    target_years = ["A26", "A27", "A28","A29"]#
    base_vol_map = dict(
        zip(base_vol["Segment"], base_vol["Volume"])
    )

    all_forecasts = []
    all_x_contributions = []

    # track last year's volume per segment
    last_volume = base_vol_map.copy()

    # store last computed X values per segment
    last_x_values = {}

    for seg in fy24_avg["Segment"].unique():
        last_x_values[seg] = {
            var: fy24_avg.loc[fy24_avg["Segment"] == seg, var].values[0]
            for var in elasticity_vars
            if var in fy24_avg.columns
        }

    for fy in target_years:

        growth_col = f"{fy} Growth Rate (%)"

        for seg in last_x_values.keys():

            # ---- STEP 1: apply growth on last FY ----
            new_x = {}

            for var, prev_val in last_x_values[seg].items():

                g = edited_growth_df.loc[
                    (edited_growth_df["Segment"] == seg) &
                    (edited_growth_df["Variable"] == var),
                    growth_col
                ]

                if g.empty:
                    new_x[var] = prev_val
                    continue

                growth = g.values[0] / 100
                new_x[var] = prev_val * (1 + growth)
                # st.write(new_x[var],prev_val,growth)

            # ---- STEP 2: scale using GLOBAL stats ----
            y_hat = beta_df.loc[
                beta_df["Segment"] == seg,
                "ScaledBeta0"
            ].iloc[0]
            # st.write(y_hat)
            seg_betas = beta_df[beta_df["Segment"] == seg]

            beta0 = seg_betas["ScaledBeta0"].iloc[0]

            seg_betas = beta_df[beta_df["Segment"] == seg]

            var_contributions = {}

            for _, b in seg_betas.iterrows():
                var = b["Variable"]

                mean_x = dataset_df[var].mean()
                std_x  = dataset_df[var].std()

                if std_x == 0 or pd.isna(std_x):
                    continue

                x_scaled = (new_x[var] - mean_x) / std_x
                x_contrib = b["ScaledBeta"] * x_scaled
                beta = b['ScaledBeta']
                var_contributions[var] = x_contrib
                y_hat += b["ScaledBeta"] * x_scaled
                # x_contrib = x_contrib/y_hat
                # st.write(fy)
                # st.write(var)
                # st.write("new var", new_x[var])
                # st.write("mean",mean_x)
                # st.write("std",std_x)
                # st.write(x_scaled)
                # st.write(b["ScaledBeta"])
                # st.write(x_contrib)

            base_vol = fy_avg["Volume"]

            # ---- YoY growth calculation (THIS is the change) ----
            prev_vol = last_volume[seg]
            yoy_growth = ((y_hat - prev_vol) / prev_vol) * 100
            # st.write(f"Segment: {seg}, FY: {fy}, Predicted Volume: {y_hat:.2f}, previous vol: {prev_vol:.2f}, YoY Growth: {yoy_growth:.2f}%")

            all_forecasts.append({
                "FiscalYear": f"FY{fy[1:]}",
                "Segment": seg,
                "PredictedVolume": y_hat,
                "VolumeGrowth_%": yoy_growth
            })

            # ---------------------------------------------------
            # STEP 4: Store Contribution Data
            # ---------------------------------------------------
            total_prediction = y_hat if y_hat != 0 else 1

            # # Intercept contribution
            # all_x_contributions.append({
            #     "FiscalYear": f"FY{fy[1:]}",
            #     "Segment": seg,
            #     "Variable": "Intercept",
            #     "Contribution_Value": beta0,
            #     "Contribution_%": (beta0 / total_prediction) * 100,
            #     "PredictedVolume": y_hat
            # })

            # Variable contributions
            for var, contrib in var_contributions.items():

                contribution_pct = (contrib / total_prediction)

                all_x_contributions.append({
                    "FiscalYear": f"FY{fy[1:]}",
                    "Segment": seg,
                    "Variable": var,
                    # "beta": beta,
                    # 'Scaled_var': x_scaled,
                    "Contribution_Value": contrib,
                    "Contribution_%": contribution_pct,
                    "PredictedVolume": y_hat
                })

            # ---------------------------------------------------
            # STEP 5: Carry Forward Values
            # ---------------------------------------------------
            last_x_values[seg] = new_x
            last_volume[seg] = y_hat

            # # ---- STEP 3: carry forward X values ----
            # last_x_values[seg] = new_x

    forecast_df = pd.DataFrame(all_forecasts)
    x_contribution_df = pd.DataFrame(all_x_contributions)
    # st.write(x_contribution_df)
    #-----------------------------------------
    # Create Year Label (A26 format)
    # -----------------------------------------
    x_contribution_df["YearLabel"] = (
        "A" + x_contribution_df["FiscalYear"].str.replace("FY", "", regex=False)
    )

    # -----------------------------------------
    # Pivot to Wide Format
    # -----------------------------------------
    wide_contribution_df = (
        x_contribution_df
        .pivot_table(
            index=["Segment", "Variable"],
            columns="YearLabel",
            values="Contribution_%",
            aggfunc="sum"
        )
        .reset_index()
    )

    # -----------------------------------------
    # Rename columns properly
    # -----------------------------------------
    wide_contribution_df.columns = [
        f"{col} Contribution" if col.startswith("A") else col
        for col in wide_contribution_df.columns
    ]

    # st.write("Contribution of X variables:")
    # st.dataframe(wide_contribution_df)

    
    # st.dataframe(
    #     forecast_df.sort_values(["Segment", "FiscalYear"]),
    #     use_container_width=True
    # )

    dataset_df_filter["Fiscal Year"] = (
        dataset_df_filter["Fiscal Year"]
        .str.replace("FY", "A", regex=False)
    )
    recent_fy = recent_fy.replace("FY", "A")
    # st.write(recent_fy)
    # st.dataframe(dataset_df_filter)
    # Ensure date is datetime
    # dataset_df_filter["date"] = pd.to_datetime(
    #     dataset_df_filter["date"],
    #     errors="coerce"
    # )

    # Ensure datetime
    # dataset_df_filter["date"] = pd.to_datetime(
    #     dataset_df_filter["date"],
    #     errors="coerce"
    # )

    # -----------------------------
    # H1 A25 = Jul–Dec 2024
    # -----------------------------
    h1_a25 = (
        dataset_df_filter[
            (dataset_df_filter["date"].dt.year == 2024) &
            (dataset_df_filter["date"].dt.month >= 7)
        ]
        .groupby("Segment")["Volume"]
        .sum()
        .reset_index()
        .rename(columns={"Volume": "H1_A25_JulDec_2024"})
    )

    # -----------------------------
    # H1 A26 = Jul–Dec 2025
    # -----------------------------
    h1_a26 = (
        dataset_df_filter[
            (dataset_df_filter["date"].dt.year == 2025) &
            (dataset_df_filter["date"].dt.month >= 7)
        ]
        .groupby("Segment")["Volume"]
        .sum()
        .reset_index()
        .rename(columns={"Volume": "H1_A26_JulDec_2025"})
    )

    # -----------------------------
    # Merge
    # -----------------------------
    h1_growth = h1_a25.merge(h1_a26, on="Segment", how="inner")

    # -----------------------------
    # Growth Calculation
    # -----------------------------
    h1_growth["H1_A25_to_A26_Growth_%"] = (
        (h1_growth["H1_A26_JulDec_2025"] - h1_growth["H1_A25_JulDec_2024"])
        / h1_growth["H1_A25_JulDec_2024"]
    ) * 100

    h1_growth["H1_A25_to_A26_Growth_%"] = (
        h1_growth["H1_A25_to_A26_Growth_%"].round(1)
    )
    h1_growth["H1_A26_JulDec_2025"] = (
        h1_growth["H1_A26_JulDec_2025"].round(1)
    )
    h1_growth["H1_A25_JulDec_2024"] = (
        h1_growth["H1_A25_JulDec_2024"].round(1)
    )

    # -----------------------------
    # Display
    # -----------------------------
    st.subheader("📊 H1 A25 → H1 A26 Growth (Jul–Dec YoY)")

    st.dataframe(
        h1_growth[
            [
                "Segment",
                "H1_A25_JulDec_2024",
                "H1_A26_JulDec_2025",
                "H1_A25_to_A26_Growth_%"
            ]
        ],
        use_container_width=True
    )


    base_vol_df = (
        dataset_df_filter[dataset_df_filter["Fiscal Year"] == recent_fy.replace("FY", "A")]
        .groupby("Segment")["Volume"]
        .mean()
        .reset_index()
        .rename(columns={"Volume": "PredictedVolume"})
    )
    # st.write(recent_fy)
    # st.dataframe(base_vol_df)
    forecast_df["FiscalYear"] = (
        forecast_df["FiscalYear"]
        .str.replace("FY", "A", regex=False)
    )


    base_vol_df["FiscalYear"] = recent_fy.replace("FY", "A")


    tmp = pd.concat(
        [
            base_vol_df[["FiscalYear", "Segment", "PredictedVolume"]],
            forecast_df[["FiscalYear", "Segment", "PredictedVolume"]]
        ],
        ignore_index=True
    )

    # ensure correct FY order
    tmp["FY_num"] = tmp["FiscalYear"].str.extract(r"(\d+)").astype(int)

    tmp = tmp.sort_values(["Segment", "FY_num"])

    tmp["VolumeGrowth_%"] = (
        tmp
        .groupby("Segment")["PredictedVolume"]
        .pct_change() * 100
    )

    final_df = tmp[tmp["FiscalYear"] != recent_fy].drop(columns="FY_num")

    final_df["FY_num"] = final_df["FiscalYear"].str.extract(r"(\d+)").astype(int)

    forecast_years = sorted(
        final_df["FiscalYear"].unique(),
        key=lambda x: int(x[1:])
    )

    first_year = forecast_years[0]

    editable_mask = final_df["FiscalYear"] == first_year

    st.write("##### A26 Model Forecast")

    display_df = final_df.loc[
        editable_mask, 
        ["Segment", "FiscalYear", "VolumeGrowth_%"]
    ].copy()

    display_df["VolumeGrowth_%"] = display_df["VolumeGrowth_%"].round(1)

    st.dataframe(display_df, use_container_width=True)
    
    st.write("##### A26 Growth rate edit:")
    editable_display_df = (
        final_df.loc[editable_mask, ["Segment", "FiscalYear", "VolumeGrowth_%"]]
        .copy()
    )

    editable_display_df["VolumeGrowth_%"] = editable_display_df["VolumeGrowth_%"].round(1)

    edited_input = st.data_editor(
        editable_display_df,
        key="editable_fy26_growth"
    )

    for _, row in edited_input.iterrows():

        seg = row["Segment"]
        edited_growth = row["VolumeGrowth_%"] / 100

        # Base (recent actual)
        base_volume = base_vol_df.loc[
            base_vol_df["Segment"] == seg,
            "PredictedVolume"
        ].values[0]

        # -------------------------
        # 1️⃣ Recalculate FY26 Volume
        # -------------------------
        new_fy26_volume = base_volume * (1 + edited_growth)

        final_df.loc[
            (final_df["Segment"] == seg) &
            (final_df["FiscalYear"] == first_year),
            "PredictedVolume"
        ] = new_fy26_volume

        final_df.loc[
            (final_df["Segment"] == seg) &
            (final_df["FiscalYear"] == first_year),
            "VolumeGrowth_%"
        ] = edited_growth * 100

        # -------------------------
        # 2️⃣ Recalculate Growth for Remaining Years
        # -------------------------
        for i in range(1, len(forecast_years)):

            prev_year = forecast_years[i - 1]
            curr_year = forecast_years[i]

            prev_vol = final_df.loc[
                (final_df["Segment"] == seg) &
                (final_df["FiscalYear"] == prev_year),
                "PredictedVolume"
            ].values[0]

            curr_vol = final_df.loc[
                (final_df["Segment"] == seg) &
                (final_df["FiscalYear"] == curr_year),
                "PredictedVolume"
            ].values[0]

            new_growth = ((curr_vol - prev_vol) / prev_vol) * 100

            final_df.loc[
                (final_df["Segment"] == seg) &
                (final_df["FiscalYear"] == curr_year),
                "VolumeGrowth_%"
            ] = new_growth

    final_df.drop(columns="FY_num", inplace=True)

    display_df = final_df.sort_values(["Segment", "FiscalYear"]).copy()

    num_cols = display_df.select_dtypes(include="number").columns
    display_df[num_cols] = display_df[num_cols].round(1)

    st.subheader("📈 Forecasted Volumes & Growth Rates")
    st.dataframe(display_df, use_container_width=True)


    final_growth_df = final_df.copy()

    # final_growth_df["FY_short"] = final_growth_df["FiscalYear"].str.replace("FY", "A")
    final_growth_df["YearLabel"] = final_growth_df["FiscalYear"]
    final_growth_df["growth_dec"] = final_growth_df["VolumeGrowth_%"] / 100

    pivot_growth = (
        final_growth_df
        .pivot(index="Segment", columns="YearLabel", values="growth_dec")
        .reset_index()
    )

    # st.subheader("📊 CAGR Settings")
    with st.expander("📊 CAGR Settings"):

        year_cols = sorted(
            [c for c in pivot_growth.columns if c.startswith("A")],
            key=lambda x: int(x[1:])
        )

        max_years = len(year_cols)

        cagr_years = st.slider(
            "Select number of years for CAGR",
            min_value=2,
            max_value=max_years,
            value=min(3, max_years)
        )

    selected_years = year_cols[-cagr_years:]

    import numpy as np

    pivot_growth[f"{cagr_years}-yr CAGR"] = (
        np.prod(
            1 + pivot_growth[selected_years],
            axis=1
        ) ** (1 / cagr_years) - 1
    )

    display_cols = selected_years + [f"{cagr_years}-yr CAGR"]

    for col in display_cols:
        pivot_growth[col] = (pivot_growth[col] * 100).round(1)

    final_cagr_table = (
        pivot_growth[["Segment"] + display_cols]
        .set_index("Segment")
    )

    st.subheader("📈 Volume Forecast Growth Summary")

    st.dataframe(
        final_cagr_table.style.format(
            {col: "{:.1f} %" for col in display_cols}
        ),
        use_container_width=True
    )


##################### Waterfall Charts #########################

    st.subheader("🎯 Waterfall Chart")
    with st.expander("🎯 Waterfall Configuration"):

        fy_list = sorted(
            final_df["FiscalYear"].unique(),
            key=lambda x: int(x.replace("A", ""))
        )

        col1, col2 = st.columns(2)

        with col1:
            start_fy = st.selectbox(
                "Select Start Fiscal Year",
                fy_list,
                index=0
            )

            # st.write("##### 🧩 Variable Selection")

            exclude_vars = ["Seasonality", "Trend"]

            all_vars = sorted(
                v for v in elasticity_df["Variable"].unique().tolist()
                if v not in exclude_vars
            )

            selected_vars = st.multiselect(
                "Select variables to include in waterfall",
                options=all_vars,
                default=all_vars  # keep all selected by default
            )

        with col2:
            end_fy = st.selectbox(
                "Select End Fiscal Year",
                fy_list,
                index=len(fy_list) - 1
            )

            VARIABLE_SCALE = st.slider(
                "Emphasize variable contribution",
                1.0, 3.0, 2.0, 0.1
            )

    start_year = int(start_fy.replace("A", ""))
    end_year = int(end_fy.replace("A", ""))

    if end_year <= start_year:
        st.error("End Fiscal Year must be greater than Start Fiscal Year")
        st.stop()

    selected_years = [
        f"A{y}"
        for y in range(start_year + 1, end_year + 1)
    ]

    # st.subheader("🧩 Variable Selection")

    # all_vars = sorted(
    #     elasticity_df["Variable"].unique().tolist()
    # )

    # selected_vars = st.multiselect(
    #     "Select variables to include in waterfall",
    #     options=all_vars,
    #     default=all_vars  # keep all selected by default
    # )

    if not selected_vars:
        st.warning("Please select at least one variable")
        st.stop()



    contrib_rows = []

    for _, row in growth_df.iterrows():
        seg = row["Segment"]
        var = row["Variable"]

        # 🔹 NEW: variable filter
        if var not in selected_vars:
            continue

        elasticity = elasticity_df.loc[
            (elasticity_df["Segment"] == seg) &
            (elasticity_df["Variable"] == var),
            "Elasticity"
        ]

        if elasticity.empty:
            continue

        elasticity = elasticity.values[0]

        for fy in selected_years:   # e.g. ["A26", "A27", "A28"]
            growth_col = f"{fy} Growth Rate (%)"

            if growth_col not in growth_df.columns:
                continue

            g = row[growth_col]

            contrib_rows.append({
                "Segment": seg,
                "Variable": var,
                "FiscalYear": fy,
                "Contribution": elasticity * g
            })


    contrib_df = pd.DataFrame(contrib_rows)
    # st.dataframe(contrib_df)

    total_contrib = (
        contrib_df
        .groupby(["Segment", "Variable"], as_index=False)
        ["Contribution"]
        .sum()
    )

    # st.dataframe(total_contrib)

    start_vol = (
        final_df[final_df["FiscalYear"] == start_fy]
        .set_index("Segment")["PredictedVolume"]
    )

    end_vol = (
        final_df[final_df["FiscalYear"] == end_fy]
        .set_index("Segment")["PredictedVolume"]
    )

    # st.write(start_vol,end_vol)

    wf_rows = []



    segments = sorted(final_df["Segment"].unique())

    # selected_segment = st.selectbox(
    #     "Select Segment",
    #     segments
    # )


    # --- Start FY ---
    wf_rows.append({
        "Label": start_fy,
        "Value": start_vol[seg],
        "Type": "start"
    })

    # --- Variables ---
    for _, row in total_contrib[total_contrib["Segment"] == seg].iterrows():
        wf_rows.append({
            "Label": row["Variable"],
            "Value": row["Contribution"],
            "Type": "variable"
        })

    # --- End FY ---
    wf_rows.append({
        "Label": end_fy,
        "Value": end_vol[seg],
        "Type": "end"
    })

    wf_df = pd.DataFrame(wf_rows)

    start_volume = start_vol[seg]
    end_volume = end_vol[seg]

    total_variable_contrib = wf_df.loc[
        wf_df["Type"] == "variable", "Value"
    ].sum()
    # st.dataframe(wf_df)
    # st.write(total_variable_contrib)

    wf_df["Pct_Normalized"] = np.where(
        wf_df["Type"] == "variable",
        wf_df["Value"] / total_variable_contrib,
        wf_df["Value"] / start_volume
    )

    start_pct = 1.0
    end_pct = end_volume / start_volume
    delta_pct = end_pct - start_pct

    wf_df["Waterfall_Value"] = np.where(
        wf_df["Type"] == "variable",
        wf_df["Pct_Normalized"] * delta_pct,
        wf_df["Pct_Normalized"]
    )

    # st.dataframe(wf_df)


    # VARIABLE_SCALE = st.slider(
    #     "Emphasize variable contribution",
    #     1.0, 3.0, 2.0, 0.1
    # )

    START_END_SCALE = 0.25   # shrink anchors
    # VARIABLE_SCALE = 2.0    # amplify contributions

    wf_df["DisplayValue"] = np.where(
        wf_df["Type"].isin(["start", "end"]),
        wf_df["Waterfall_Value"] * START_END_SCALE,
        wf_df["Waterfall_Value"] * VARIABLE_SCALE
    )

    wf_df["LabelText"] = wf_df["Waterfall_Value"].apply(
        lambda x: f"{x*100:.1f}%"
    )

    y_max = (wf_df["DisplayValue"] * 100).max()

    import plotly.graph_objects as go

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=[
            "absolute" if t == "start"
            else "total" if t == "end"
            else "relative"
            for t in wf_df["Type"]
        ],
        x=wf_df["Label"],
        y=wf_df["DisplayValue"] * 150,   # scaled for display
        text=wf_df["LabelText"],
        textposition="outside",
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))

    fig.update_layout(
        title="Volume Growth Waterfall (Elasticity × Growth)",
        yaxis_title="Contribution (scaled for visibility)",
        showlegend=False,
        height=650,
        yaxis=dict(
            fixedrange=True,
            zeroline=True
        )
    )
    fig.update_layout(
        yaxis=dict(
            range=[0, y_max * 4.25],  # 👈 25% headroom
            fixedrange=True,
            zeroline=True
        )
    )
    fig.update_layout(
        xaxis=dict(
            showline=True,
            zeroline=False,
            showgrid=False
        ),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=True
        )
    )
    fig.update_layout(
        yaxis=dict(
            showticklabels=False
        )
    )


    st.plotly_chart(fig, use_container_width=True)